"""Progressive sub-band (high-pass filtered error) fine-tuning, reference convention.

Implements the reference team's training mechanism adapted to our EA-TCN models:
    L = (1 - r) * mean|e|^2 + r * mean|HPF(e)|^2 + guard * relu(NMSE_dB - target)^2
where e is the frame error with the model's frame-edge pollution cropped away,
HPF is a linear-phase FIR high-pass (passband covers the +/-(100,180) MHz
adjacent bands of the integrated ACLR metric), and r ramps up per epoch like
their schedule  r(e) = min(r_max, 1 - exp(-e / tau)).

Optionally the output FIR is re-fitted in closed form every --refit-every
epochs with adjacent-band weighting (same designer as aclr_ref_fir_sweep).
All metrics reported are steps/aclr_ref_metrics.py conventions on the val
split.  Release checkpoints are never modified.
"""

import argparse
import csv
import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.signal import firwin
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Frames(Dataset):
    def __init__(self, x, y, length, stride):
        self.x = torch.as_tensor(x, dtype=torch.float64)
        self.y = torch.as_tensor(y, dtype=torch.float64)
        self.starts = list(range(0, len(x) - length + 1, stride))
        self.length = length

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        stop = start + self.length
        return self.x[start:stop].T, self.y[start:stop].T


def build_hpf(taps, cutoff_mhz, fs_mhz, device):
    coeffs = firwin(taps, cutoff_mhz / (fs_mhz / 2.0), pass_zero=False)
    kernel = torch.as_tensor(coeffs, dtype=torch.float64, device=device)
    return kernel.view(1, 1, -1).repeat(2, 1, 1)  # groups=2, filter I and Q alike


def evaluate(evaluator, metrics_mod, model, x_val, y_val, nperseg):
    pred, target, _, _ = evaluator.evaluate_full_sequence(model.eval(), x_val, y_val)
    pred_c = evaluator.tensor_to_complex(pred)
    target_c = evaluator.tensor_to_complex(target)
    return metrics_mod.ref_metrics(evaluator, pred_c, target_c, nperseg)


def refit_fir(evaluator, sweep_mod, metrics_mod, model, x_train, target_train_c, args):
    base_train = sweep_mod.run_base_sequence(evaluator, model, x_train)
    taps = len(model.fir_real)
    designer = sweep_mod.FirDesigner(base_train, target_train_c, taps, args.fs_mhz)
    w = designer.weights(metrics_mod, args.refit_adjacent_weight, args.refit_floor_weight)
    coeffs = designer.solve(w, args.refit_ridge)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=torch.float64, device=evaluator.DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=torch.float64, device=evaluator.DEVICE))


def candidate_key(metrics, args):
    """Sort key: feasibility first, then NMSE quality."""
    aclr_ok = metrics["aclr_db"] <= args.aclr_target
    nmse_ok = args.nmse_guard is None or metrics["nmse_db"] <= args.nmse_guard
    return (not (aclr_ok and nmse_ok), metrics["nmse_db"] if aclr_ok else float("inf"),
            metrics["aclr_db"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model label used for specs and output dir")
    parser.add_argument("--checkpoint", default=None, help="start point (default: release ckpt)")
    parser.add_argument("--tag", default="subband")
    parser.add_argument("--train-mode", choices=["all", "fir"], default="all")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=4096)
    parser.add_argument("--frame-stride", type=int, default=2048)
    parser.add_argument("--edge-crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-gamma", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--r-max", type=float, default=0.99)
    parser.add_argument("--ramp-epochs", type=int, default=60)
    parser.add_argument("--hpf-taps", type=int, default=257)
    parser.add_argument("--hpf-cutoff-mhz", type=float, default=98.0)
    parser.add_argument("--nmse-guard", type=float, default=None)
    parser.add_argument("--guard-weight", type=float, default=3e-3)
    parser.add_argument("--aclr-target", type=float, default=-45.0)
    parser.add_argument("--refit-every", type=int, default=0, help="0 disables closed-form refits")
    parser.add_argument("--refit-adjacent-weight", type=float, default=10.0)
    parser.add_argument("--refit-floor-weight", type=float, default=1.0)
    parser.add_argument("--refit-ridge", type=float, default=1e-8)
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("aclr_ref_metrics", "steps/aclr_ref_metrics.py")
    sweep_mod = load_module("aclr_ref_fir_sweep", "steps/aclr_ref_fir_sweep.py")

    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_idx = int(len(features) * args.split)
    x_train, y_train = features[:split_idx], targets[:split_idx]
    x_val, y_val = features[split_idx:], targets[split_idx:]
    target_train_c = y_train[:, 0] + 1j * y_train[:, 1]

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
    else:
        checkpoint_path = ROOT / metrics_mod.MODEL_SPECS[args.model][0]
    state = evaluator.load_state_dict(checkpoint_path)
    model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad = args.train_mode == "all"
    if args.train_mode == "fir":
        model.fir_real.requires_grad = True
        model.fir_imag.requires_grad = True

    hpf = build_hpf(args.hpf_taps, args.hpf_cutoff_mhz, args.fs_mhz, evaluator.DEVICE)
    tau = args.ramp_epochs / (-math.log(1.0 - min(args.r_max, 1.0 - 1e-9)))

    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_path = output_dir / "best.pth"
    best_aclr_path = output_dir / "best_aclr.pth"

    initial = evaluate(evaluator, metrics_mod, model, x_val, y_val, args.nperseg)
    print(
        f"[{args.model}] start NMSE {initial['nmse_db']:.3f} | ACLR {initial['aclr_db']:.2f} "
        f"(from {checkpoint_path.name})"
    )
    best = initial
    best_aclr = initial
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, best_path)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, best_aclr_path)

    loader = DataLoader(
        Frames(x_train, y_train, args.frame_length, args.frame_stride),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    crop = args.edge_crop

    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "r", "loss", "nmse_db", "aclr_db", "aclr_lower_db", "aclr_upper_db"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            r = min(args.r_max, 1.0 - math.exp(-epoch / tau))
            model.train()
            loss_sum = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(evaluator.DEVICE)
                batch_y = batch_y.to(evaluator.DEVICE)
                optimizer.zero_grad()
                pred = model(batch_x)
                err = (pred - batch_y)[:, :, crop:-crop]
                tgt = batch_y[:, :, crop:-crop]
                loss_fb = err.square().mean()
                err_hp = torch.nn.functional.conv1d(err, hpf, groups=2)
                loss_sb = err_hp.square().mean()
                loss = (1.0 - r) * loss_fb + r * loss_sb
                if args.nmse_guard is not None:
                    nmse_db = 10.0 * torch.log10(
                        err.square().sum() / (tgt.square().sum() + 1e-24) + 1e-24
                    )
                    loss = loss + args.guard_weight * torch.relu(nmse_db - args.nmse_guard).square()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
            scheduler.step()

            if args.refit_every > 0 and epoch % args.refit_every == 0:
                refit_fir(evaluator, sweep_mod, metrics_mod, model.eval(), x_train, target_train_c, args)

            if epoch % args.eval_every != 0 and epoch != args.epochs:
                continue
            metrics = evaluate(evaluator, metrics_mod, model, x_val, y_val, args.nperseg)
            writer.writerow({"epoch": epoch, "r": r, "loss": loss_sum / max(1, len(loader)), **metrics})
            handle.flush()
            print(
                f"[{args.model}] ep {epoch:03d} r={r:.4f} NMSE {metrics['nmse_db']:.3f} | "
                f"ACLR {metrics['aclr_db']:.2f} (L {metrics['aclr_lower_db']:.2f} / "
                f"U {metrics['aclr_upper_db']:.2f})"
            )
            if candidate_key(metrics, args) < candidate_key(best, args):
                best = metrics
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, best_path)
            if metrics["aclr_db"] < best_aclr["aclr_db"]:
                best_aclr = metrics
                torch.save(
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, best_aclr_path
                )

    print(
        f"[{args.model}] best-feasible NMSE {best['nmse_db']:.3f} ACLR {best['aclr_db']:.2f} -> {best_path}\n"
        f"[{args.model}] best-ACLR    NMSE {best_aclr['nmse_db']:.3f} ACLR {best_aclr['aclr_db']:.2f} -> {best_aclr_path}"
    )


if __name__ == "__main__":
    main()
