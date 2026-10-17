"""Min-point ACLR optimization machinery applied to phase-spline wrapper models.

Ports steps/aclr_min_point_optimization.py (annealed soft-min surrogate,
NMSE-gated gradient surgery, trust-region projection, hard-metric epoch
acceptance with rollback, and final weight extrapolation) onto the
PhaseSplineDPD wrappers from steps/aclr_phase_spline_experiment.py.

Anchor / start = a phase-spline wrapper checkpoint (keys prefixed "base.").
All reusable pieces are imported from the original script so the surrogate,
acceptance rule and extrapolation stay bit-identical.  Metric: lowest-point
ACLR, main (20,80) MHz, adjacent +/-(120,180) MHz, Welch nperseg=4096, worst
side; NMSE limits {2284:-40, 998:-37, 786:-35, 796:-35}.
"""

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
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_wrapper(evaluator, ps_mod, wrapper_state, taps_reduction, offsets, grid, amp_max):
    inner = {k[len("base."):]: v for k, v in wrapper_state.items() if k.startswith("base.")}
    model = ps_mod.PhaseSplineDPD(evaluator, inner, taps_reduction, offsets, grid, amp_max)
    model.load_state_dict(wrapper_state, strict=True)
    return model.to(evaluator.DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["2284", "998", "786", "796"])
    parser.add_argument("--checkpoint", type=Path, required=True, help="phase-spline wrapper ckpt")
    parser.add_argument("--taps-reduction", type=int, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--tag", default="spline_minpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=16384)
    parser.add_argument("--frame-stride", type=int, default=8192)
    parser.add_argument("--segment-length", type=int, default=4096)
    parser.add_argument("--edge-trim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--spectral-ratio", type=float, default=20.0)
    parser.add_argument("--softmin-start", type=float, default=1.0)
    parser.add_argument("--softmin-end", type=float, default=0.05)
    parser.add_argument("--band-temperature", type=float, default=0.1)
    parser.add_argument("--nmse-margin-db", type=float, default=0.01)
    parser.add_argument("--trust-radius", type=float, default=0.005)
    parser.add_argument("--budget-burn", action="store_true",
                        help="while val NMSE is below target minus --burn-reserve, drop the NMSE "
                             "descent term and step along the projected spectral gradient only, "
                             "deliberately spending the NMSE margin on the point metric")
    parser.add_argument("--burn-reserve", type=float, default=0.05)
    parser.add_argument("--train-scope", choices=["all", "corr"], default="all",
                        help="corr = spline tables only (fresh params tolerate much larger lr)")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--extrapolation-max", type=float, default=3.0)
    parser.add_argument("--extrapolation-step", type=float, default=0.05)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    ps_mod = load_module("aclr_phase_spline_experiment", "steps/aclr_phase_spline_experiment.py")
    mp = load_module("aclr_min_point_optimization", "steps/aclr_min_point_optimization.py")

    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_index = int(len(features) * args.split)
    x_train, y_train = features[:split_index], targets[:split_index]
    x_val, y_val = features[split_index:], targets[split_index:]
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    anchor_state = evaluator.load_state_dict(checkpoint)
    if not any(k.startswith("base.") for k in anchor_state):
        raise SystemExit("expected a phase-spline wrapper checkpoint (base.* keys)")

    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    model = build_wrapper(evaluator, ps_mod, anchor_state, args.taps_reduction, args.offsets,
                          args.grid, amp_max)
    if args.train_scope == "corr":
        parameters = [model.corr.tables]
    else:
        parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    loader = DataLoader(
        mp.Frames(x_train, y_train, args.frame_length, args.frame_stride),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    penalty = mp.MinimumPointPenalty(
        y_train, args.segment_length, args.edge_trim, 491.52, evaluator.DEVICE
    )
    target_db = mp.NMSE_TARGETS[args.model]
    initial = mp.evaluate_minimum(evaluator, model.eval(), x_val, y_val)
    best = initial
    best_state = mp.clone_state(model)
    torch.save(best_state, output_dir / "best.pth")
    print(
        f"[{args.model}] anchor NMSE={initial['nmse_db']:.3f} "
        f"min-ACLR={initial['aclr_lower_db']:.3f}/{initial['aclr_upper_db']:.3f}/"
        f"{initial['aclr_worst_db']:.3f}"
    )

    anchor_params = {k: v.to(evaluator.DEVICE) for k, v in anchor_state.items()}
    for epoch in range(1, args.epochs + 1):
        progress = (epoch - 1) / max(1, args.epochs - 1)
        temperature = args.softmin_start * (args.softmin_end / args.softmin_start) ** progress
        gate = mp.trust_gate(best["nmse_db"], target_db, args.nmse_margin_db)
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(evaluator.DEVICE)
            batch_y = batch_y.to(evaluator.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            core_pred = pred[..., args.edge_trim : -args.edge_trim]
            core_target = batch_y[..., args.edge_trim : -args.edge_trim]
            nmse = torch.sum((core_pred - core_target).square()) / (
                torch.sum(core_target.square()) + 1e-30
            )
            nmse_db = 10.0 * torch.log10(nmse + 1e-30)
            point_score, _ = penalty(pred, batch_y, temperature, args.band_temperature)
            nmse_grads = torch.autograd.grad(nmse_db, parameters, retain_graph=True, allow_unused=True)
            point_grads = torch.autograd.grad(point_score, parameters, retain_graph=False, allow_unused=True)
            nmse_grads = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, nmse_grads)]
            point_grads = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, point_grads)]
            gradients = (
                mp.combine_gradients(nmse_grads, point_grads, args.spectral_ratio * gate)
                if gate > 0.0
                else nmse_grads
            )
            if (
                args.budget_burn
                and gate > 0.0
                and best["nmse_db"] <= target_db - args.burn_reserve
            ):
                gradients = [g - n for g, n in zip(gradients, nmse_grads)]
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
            optimizer.step()
            mp.project_to_anchor(model, anchor_params, args.trust_radius)

        metrics = mp.evaluate_minimum(evaluator, model.eval(), x_val, y_val)
        accepted = mp.candidate_key(metrics, target_db) < mp.candidate_key(best, target_db)
        if accepted:
            best = metrics
            best_state = mp.clone_state(model)
            torch.save(best_state, output_dir / "best.pth")
        else:
            model.load_state_dict(best_state, strict=True)
            optimizer.state.clear()
        mp.append_row(
            output_dir / "history.csv",
            {"epoch": epoch, "softmin_temperature": temperature, "gate": gate,
             "accepted": int(accepted), **metrics},
        )
        print(
            f"[{args.model}] ep={epoch:02d} tau={temperature:.3f} accepted={int(accepted)} "
            f"NMSE={metrics['nmse_db']:.3f} min-ACLR={metrics['aclr_lower_db']:.3f}/"
            f"{metrics['aclr_upper_db']:.3f}/{metrics['aclr_worst_db']:.3f}"
        )

    rows = []
    final_state, final_metrics, final_alpha = best_state, best, 1.0
    for alpha in np.arange(0.0, args.extrapolation_max + args.extrapolation_step / 2.0,
                           args.extrapolation_step):
        state = mp.interpolate_states(anchor_state, best_state, float(alpha))
        candidate = build_wrapper(evaluator, ps_mod, state, args.taps_reduction, args.offsets,
                                  args.grid, amp_max).eval()
        metrics = mp.evaluate_minimum(evaluator, candidate, x_val, y_val)
        rows.append({"alpha": float(alpha), **metrics})
        if mp.candidate_key(metrics, target_db) < mp.candidate_key(final_metrics, target_db):
            final_state, final_metrics, final_alpha = state, metrics, float(alpha)
    with (output_dir / "extrapolation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(final_state, output_dir / "best_extrapolated.pth")
    summary = {
        "model": args.model,
        "source_checkpoint": str(checkpoint),
        "nmse_target_db": target_db,
        "alpha": final_alpha,
        **final_metrics,
        "checkpoint": str(output_dir / "best_extrapolated.pth"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
