"""ACLR optimization without changing the deployed DPD architecture.

The script compares three low-risk optimization strategies:

* ``trust``: NMSE-gated spectral descent inside an anchor-weight trust region.
* ``pcgrad``: ``trust`` plus projection of spectral gradients that conflict
  with the NMSE gradient.
* ``alternating``: projected backbone spectral steps alternated with FIR-only
  NMSE recovery steps.

Every run also evaluates raw, EMA and SWA weights, then scans interpolation
between the untouched release checkpoint and the best trained candidate.  No
release checkpoint is overwritten and no model layer is added or removed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
METRIC_SCRIPT = ROOT / "steps" / "aclr_three_metric_experiment.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_low_nmse_optimization"
REFERENCE_METRICS = {
    "full": -43.73783656472544,
    "point": -45.10452191091548,
    "narrow": -44.99650320556108,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metric_exp = load_module("aclr_three_metric_experiment", METRIC_SCRIPT)


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def copy_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def update_ema(ema_state, model, decay):
    current = model.state_dict()
    with torch.no_grad():
        for key in ema_state:
            source = current[key].detach().cpu()
            if torch.is_floating_point(ema_state[key]):
                ema_state[key].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                ema_state[key].copy_(source)


def average_states(states):
    if not states:
        return None
    result = copy_state(states[0])
    for key in result:
        if not torch.is_floating_point(result[key]):
            continue
        result[key].zero_()
        for state in states:
            result[key].add_(state[key], alpha=1.0 / len(states))
    return result


def interpolate_states(anchor, candidate, alpha):
    result = {}
    for key, anchor_value in anchor.items():
        candidate_value = candidate[key]
        if torch.is_floating_point(anchor_value):
            result[key] = torch.lerp(anchor_value, candidate_value, float(alpha))
        else:
            result[key] = candidate_value.clone()
    return result


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_key(metrics, nmse_target_db, objective="stable"):
    violation = max(0.0, metrics["nmse_db"] - nmse_target_db)
    if objective == "reference":
        aclr_score = max(
            metrics["full_worst"] - REFERENCE_METRICS["full"],
            metrics["point_worst"] - REFERENCE_METRICS["point"],
            metrics["narrow_worst"] - REFERENCE_METRICS["narrow"],
        )
    elif objective == "target45":
        aclr_score = max(
            metrics["full_worst"] + 45.0,
            metrics["point_worst"] + 45.0,
            metrics["narrow_worst"] + 45.0,
        )
    else:
        aclr_score = max(metrics["full_worst"], metrics["narrow_worst"])
    return (violation > 0.0, violation, aclr_score, metrics["full_worst"], metrics["nmse_db"])


def evaluate_state(evaluator, state, x_val, y_val):
    model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE).eval()
    metrics = metric_exp.evaluate_metrics(evaluator, model, x_val, y_val)
    del model
    if evaluator.DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def trust_gate(nmse_db, target_db, margin_db):
    if nmse_db >= target_db:
        return 0.0
    if margin_db <= 0.0 or nmse_db <= target_db - margin_db:
        return 1.0
    return float((target_db - nmse_db) / margin_db)


def project_to_anchor(model, anchor_state, relative_radius):
    if relative_radius <= 0.0:
        return
    with torch.no_grad():
        displacement_sq = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
        anchor_sq = torch.zeros_like(displacement_sq)
        named = dict(model.named_parameters())
        for name, parameter in named.items():
            anchor = anchor_state[name].to(parameter.device)
            displacement_sq.add_(torch.sum((parameter - anchor).square()))
            anchor_sq.add_(torch.sum(anchor.square()))
        radius = relative_radius * torch.sqrt(anchor_sq + 1e-30)
        displacement = torch.sqrt(displacement_sq + 1e-30)
        if displacement > radius:
            scale = radius / displacement
            for name, parameter in named.items():
                anchor = anchor_state[name].to(parameter.device)
                parameter.copy_(anchor + scale * (parameter - anchor))


def batch_objectives(
    model,
    batch_x,
    batch_y,
    penalty,
    edge_trim,
    smooth_temperature,
    spectral_objective,
    narrow_weight,
):
    pred = model(batch_x)
    if edge_trim > 0:
        pred_core = pred[..., edge_trim:-edge_trim]
        target_core = batch_y[..., edge_trim:-edge_trim]
    else:
        pred_core = pred
        target_core = batch_y
    nmse_linear = torch.sum((pred_core - target_core).square()) / (
        torch.sum(target_core.square()) + 1e-30
    )
    nmse_db = 10.0 * torch.log10(nmse_linear + 1e-30)
    _, spectral_values = penalty(pred, batch_y, -45.0)
    if spectral_objective == "balanced":
        full_score = smooth_temperature * torch.logsumexp(
            spectral_values[:2] / smooth_temperature, dim=0
        )
        narrow_score = smooth_temperature * torch.logsumexp(
            spectral_values[2:] / smooth_temperature, dim=0
        )
        spectral_score = full_score + narrow_weight * narrow_score
    else:
        spectral_score = smooth_temperature * torch.logsumexp(
            spectral_values / smooth_temperature, dim=0
        )
    return nmse_db, spectral_score, spectral_values.detach()


def autograd_list(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gradients)]


def combine_gradients(nmse_grads, spectral_grads, spectral_ratio, project_conflict):
    dot = sum(torch.sum(a * b) for a, b in zip(nmse_grads, spectral_grads))
    nmse_norm_sq = sum(torch.sum(g.square()) for g in nmse_grads)
    if project_conflict and dot < 0.0:
        coefficient = dot / (nmse_norm_sq + 1e-30)
        spectral_grads = [b - coefficient * a for a, b in zip(nmse_grads, spectral_grads)]
    spectral_norm_sq = sum(torch.sum(g.square()) for g in spectral_grads)
    scale = spectral_ratio * torch.sqrt(nmse_norm_sq + 1e-30) / torch.sqrt(
        spectral_norm_sq + 1e-30
    )
    combined = [a + scale * b for a, b in zip(nmse_grads, spectral_grads)]
    cosine = dot / torch.sqrt((nmse_norm_sq + 1e-30) * (spectral_norm_sq + 1e-30))
    return combined, float(cosine.detach().cpu())


def parameter_subset(model, subset):
    named = list(model.named_parameters())
    if subset == "fir":
        return [parameter for name, parameter in named if name in {"fir_real", "fir_imag"}]
    if subset == "backbone":
        return [parameter for name, parameter in named if name not in {"fir_real", "fir_imag"}]
    return [parameter for _, parameter in named]


def train_strategy(
    args,
    evaluator,
    spec,
    strategy,
    anchor_state,
    x_train,
    y_train,
    x_val,
    y_val,
    penalty,
    output_dir,
):
    model = evaluator.AutoDiagramDPDNet(anchor_state).to(evaluator.DEVICE)
    anchor_metrics = metric_exp.evaluate_metrics(evaluator, model.eval(), x_val, y_val)
    guard_db = spec.nmse_target_db
    loader = DataLoader(
        metric_exp.Frames(x_train, y_train, y_train, args.frame_length, args.frame_stride),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    all_parameters = parameter_subset(model, "all")
    backbone_parameters = parameter_subset(model, "backbone")
    fir_parameters = parameter_subset(model, "fir")
    optimizer_all = torch.optim.AdamW(all_parameters, lr=args.lr, weight_decay=0.0)
    optimizer_backbone = torch.optim.AdamW(backbone_parameters, lr=args.lr, weight_decay=0.0)
    optimizer_fir = torch.optim.AdamW(fir_parameters, lr=args.fir_lr, weight_decay=0.0)
    ema_state = copy_state(anchor_state)
    swa_states = [copy_state(anchor_state)] if anchor_metrics["nmse_db"] <= guard_db else []
    best_states = {"raw": copy_state(anchor_state), "ema": copy_state(anchor_state)}
    best_metrics = {"raw": anchor_metrics, "ema": anchor_metrics}
    history_path = output_dir / "history.csv"

    print(
        f"[{spec.name}/{strategy}] initial NMSE={anchor_metrics['nmse_db']:.3f} "
        f"ACLR={anchor_metrics['full_worst']:.3f}/{anchor_metrics['point_worst']:.3f}/"
        f"{anchor_metrics['narrow_worst']:.3f}"
    )

    last_nmse_db = anchor_metrics["nmse_db"]
    for epoch in range(1, args.epochs + 1):
        model.train()
        gate = trust_gate(last_nmse_db, guard_db, args.nmse_margin_db)
        if strategy == "alternating" and epoch % 2 == 0:
            phase = "fir_recovery"
            parameters = fir_parameters
            optimizer = optimizer_fir
        elif strategy == "alternating":
            phase = "backbone_spectral"
            parameters = backbone_parameters
            optimizer = optimizer_backbone
        else:
            phase = "joint"
            parameters = all_parameters
            optimizer = optimizer_all

        loss_sum = 0.0
        spectral_sum = 0.0
        cosine_sum = 0.0
        cosine_count = 0
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(evaluator.DEVICE)
            batch_y = batch_y.to(evaluator.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            nmse_db, spectral_score, _ = batch_objectives(
                model,
                batch_x,
                batch_y,
                penalty,
                args.edge_trim,
                args.smooth_temperature,
                args.spectral_objective,
                args.narrow_weight,
            )
            if phase == "fir_recovery" or gate <= 0.0:
                gradients = autograd_list(nmse_db, parameters, retain_graph=False)
            else:
                nmse_grads = autograd_list(nmse_db, parameters, retain_graph=True)
                spectral_grads = autograd_list(spectral_score, parameters, retain_graph=False)
                gradients, cosine = combine_gradients(
                    nmse_grads,
                    spectral_grads,
                    args.spectral_ratio * gate,
                    project_conflict=(strategy in {"pcgrad", "alternating"}),
                )
                cosine_sum += cosine
                cosine_count += 1
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
            optimizer.step()
            project_to_anchor(model, anchor_state, args.trust_radius)
            update_ema(ema_state, model, args.ema_decay)
            loss_sum += float(nmse_db.detach().cpu())
            spectral_sum += float(spectral_score.detach().cpu())

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        raw_state = clone_state(model)
        raw_metrics = metric_exp.evaluate_metrics(evaluator, model.eval(), x_val, y_val)
        ema_metrics = evaluate_state(evaluator, ema_state, x_val, y_val)
        last_nmse_db = raw_metrics["nmse_db"]
        if raw_metrics["nmse_db"] <= guard_db:
            swa_states.append(raw_state)
        raw_improved = candidate_key(
            raw_metrics, guard_db, args.selection_objective
        ) < candidate_key(
            best_metrics["raw"], guard_db, args.selection_objective
        )
        if raw_improved:
            best_metrics["raw"] = raw_metrics
            best_states["raw"] = raw_state
        if candidate_key(ema_metrics, guard_db, args.selection_objective) < candidate_key(
            best_metrics["ema"], guard_db, args.selection_objective
        ):
            best_metrics["ema"] = ema_metrics
            best_states["ema"] = copy_state(ema_state)
        row = {
            "model": spec.name,
            "strategy": strategy,
            "epoch": epoch,
            "phase": phase,
            "gate": gate,
            "mean_batch_nmse_db": loss_sum / max(1, len(loader)),
            "mean_batch_spectral_db": spectral_sum / max(1, len(loader)),
            "mean_gradient_cosine": cosine_sum / max(1, cosine_count),
            **{f"raw_{key}": value for key, value in raw_metrics.items()},
            **{f"ema_{key}": value for key, value in ema_metrics.items()},
        }
        append_row(history_path, row)
        print(
            f"[{spec.name}/{strategy}] epoch={epoch:02d} {phase} gate={gate:.2f} "
            f"raw={raw_metrics['nmse_db']:.3f},"
            f"{raw_metrics['full_worst']:.3f}/{raw_metrics['point_worst']:.3f}/"
            f"{raw_metrics['narrow_worst']:.3f} ema={ema_metrics['nmse_db']:.3f},"
            f"{ema_metrics['full_worst']:.3f}/{ema_metrics['point_worst']:.3f}/"
            f"{ema_metrics['narrow_worst']:.3f}"
        )
        if args.rollback_on_reject and not raw_improved:
            model.load_state_dict(best_states["raw"], strict=True)
            last_nmse_db = best_metrics["raw"]["nmse_db"]
            optimizer_all.state.clear()
            optimizer_backbone.state.clear()
            optimizer_fir.state.clear()

    swa_state = average_states(swa_states)
    if swa_state is not None:
        best_states["swa"] = swa_state
        best_metrics["swa"] = evaluate_state(evaluator, swa_state, x_val, y_val)

    candidate_name = min(
        best_metrics,
        key=lambda name: candidate_key(
            best_metrics[name], guard_db, args.selection_objective
        ),
    )
    candidate_state = best_states[candidate_name]
    candidate_metrics = best_metrics[candidate_name]
    torch.save(best_states["raw"], output_dir / "best_raw.pth")
    torch.save(best_states["ema"], output_dir / "best_ema.pth")
    if "swa" in best_states:
        torch.save(best_states["swa"], output_dir / "swa.pth")

    interpolation_rows = []
    best_interp_state = candidate_state
    best_interp_metrics = candidate_metrics
    best_alpha = 1.0
    for alpha in np.arange(0.0, 1.0 + args.interpolation_step / 2.0, args.interpolation_step):
        state = interpolate_states(anchor_state, candidate_state, float(alpha))
        metrics = evaluate_state(evaluator, state, x_val, y_val)
        interpolation_rows.append(
            {"model": spec.name, "strategy": strategy, "source": candidate_name, "alpha": float(alpha), **metrics}
        )
        if candidate_key(metrics, guard_db, args.selection_objective) < candidate_key(
            best_interp_metrics, guard_db, args.selection_objective
        ):
            best_interp_state = state
            best_interp_metrics = metrics
            best_alpha = float(alpha)
    write_rows(output_dir / "interpolation.csv", interpolation_rows)
    torch.save(best_interp_state, output_dir / "best_interpolated.pth")

    candidate_rows = []
    for name, metrics in best_metrics.items():
        candidate_rows.append({"model": spec.name, "strategy": strategy, "candidate": name, "alpha": "", **metrics})
    candidate_rows.append(
        {
            "model": spec.name,
            "strategy": strategy,
            "candidate": f"interpolated_{candidate_name}",
            "alpha": best_alpha,
            **best_interp_metrics,
        }
    )
    write_rows(output_dir / "candidates.csv", candidate_rows)
    print(
        f"[{spec.name}/{strategy}] selected interpolated_{candidate_name} alpha={best_alpha:.2f} "
        f"NMSE={best_interp_metrics['nmse_db']:.3f} "
        f"ACLR={best_interp_metrics['full_worst']:.3f}/"
        f"{best_interp_metrics['point_worst']:.3f}/"
        f"{best_interp_metrics['narrow_worst']:.3f}"
    )
    return {
        "model": spec.name,
        "strategy": strategy,
        "candidate": f"interpolated_{candidate_name}",
        "alpha": best_alpha,
        "nmse_target_db": guard_db,
        **best_interp_metrics,
        "checkpoint": str(output_dir / "best_interpolated.pth"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["2284", "998", "786", "796"], default=["2284"])
    parser.add_argument("--strategies", nargs="+", choices=["trust", "pcgrad", "alternating"], default=["trust", "pcgrad", "alternating"])
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        default=None,
        help="Optional same-architecture starting checkpoint; only valid with one model.",
    )
    parser.add_argument("--tag", default="screen")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=4096)
    parser.add_argument("--frame-stride", type=int, default=2048)
    parser.add_argument("--seg-length", type=int, default=1024)
    parser.add_argument("--edge-trim", type=int, default=256)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--fir-lr", type=float, default=2e-7)
    parser.add_argument("--spectral-ratio", type=float, default=0.10)
    parser.add_argument("--nmse-margin-db", type=float, default=0.05)
    parser.add_argument("--trust-radius", type=float, default=0.005)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--smooth-temperature", type=float, default=0.5)
    parser.add_argument("--spectral-objective", choices=["global", "balanced"], default="global")
    parser.add_argument("--narrow-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-objective",
        choices=["stable", "reference", "target45"],
        default="stable",
    )
    parser.add_argument("--rollback-on-reject", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--interpolation-step", type=float, default=0.05)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluator = metric_exp.load_evaluator()
    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_index = int(len(features) * args.split)
    x_train, y_train = features[:split_index], targets[:split_index]
    x_val, y_val = features[split_index:], targets[split_index:]
    penalty = metric_exp.SpectralPenalty(
        y_train, args.seg_length, args.edge_trim, args.fs_mhz, evaluator.DEVICE
    )
    run_root = OUT_ROOT / args.tag
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False, default=str)

    rows = []
    if args.anchor_checkpoint is not None and len(args.models) != 1:
        raise ValueError("--anchor-checkpoint can only be used when exactly one model is selected")
    for model_name in args.models:
        spec = metric_exp.MODEL_SPECS[model_name]
        checkpoint = args.anchor_checkpoint if args.anchor_checkpoint is not None else Path(spec.checkpoint)
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        anchor_state = evaluator.load_state_dict(checkpoint)
        print(f"[{model_name}] anchor checkpoint: {checkpoint}")
        for strategy in args.strategies:
            output_dir = run_root / model_name / strategy
            output_dir.mkdir(parents=True, exist_ok=True)
            rows.append(
                train_strategy(
                    args,
                    evaluator,
                    spec,
                    strategy,
                    anchor_state,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    penalty,
                    output_dir,
                )
            )
            write_rows(run_root / "summary.csv", rows)
    print(f"Summary: {run_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
