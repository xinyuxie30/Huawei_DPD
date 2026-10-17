"""Optimize only the lowest-point ACLR while enforcing an NMSE constraint.

The training surrogate is an annealed soft minimum of the error PSD in the two
adjacent bands.  Checkpoint acceptance and weight extrapolation always use the
true hard minimum from the 4096-point Welch evaluator.  No mean-band or fixed
narrow-band ACLR metric is computed by this script.
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
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "steps" / "final_evaluate_unified.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_min_point_optimization"
NMSE_TARGETS = {"2284": -40.0, "998": -37.0, "786": -35.0, "796": -35.0}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("final_evaluate_unified", EVALUATOR_PATH)
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


def band_mask(freq_mhz, band):
    low, high = sorted(map(float, band))
    return (freq_mhz >= low) & (freq_mhz <= high)


class MinimumPointPenalty:
    def __init__(self, target_train, segment_length, edge_trim, fs_mhz, device):
        self.segment_length = int(segment_length)
        self.edge_trim = int(edge_trim)
        self.window = torch.hann_window(
            self.segment_length, periodic=True, dtype=torch.float64, device=device
        )
        freq = torch.fft.fftshift(
            torch.fft.fftfreq(self.segment_length, d=1.0 / fs_mhz, device=device)
        )
        self.main_mask = band_mask(freq, (20.0, 80.0))
        self.lower_mask = band_mask(freq, (-180.0, -120.0))
        self.upper_mask = band_mask(freq, (120.0, 180.0))
        target_complex = torch.as_tensor(
            target_train[:, 0] + 1j * target_train[:, 1],
            dtype=torch.complex128,
            device=device,
        )
        with torch.no_grad():
            target_power = self._average_power(target_complex.unsqueeze(0), trim=False)
            self.target_main_db = 10.0 * torch.log10(target_power[self.main_mask] + 1e-30)

    def _average_power(self, signal, trim=True):
        if trim and self.edge_trim > 0:
            signal = signal[..., self.edge_trim : -self.edge_trim]
        segments = signal.unfold(-1, self.segment_length, self.segment_length // 2)
        spectrum = torch.fft.fftshift(
            torch.fft.fft(segments * self.window, dim=-1), dim=-1
        )
        return spectrum.abs().square().mean(dim=(0, 1))

    @staticmethod
    def soft_min(values_db, temperature):
        return -temperature * torch.logsumexp(-values_db / temperature, dim=0)

    def __call__(self, pred, target, temperature, band_temperature):
        pred_complex = torch.complex(pred[:, 0], pred[:, 1])
        target_complex = torch.complex(target[:, 0], target[:, 1])
        error_power = self._average_power(target_complex - pred_complex)
        error_db = 10.0 * torch.log10(error_power + 1e-30)
        main_min = self.soft_min(self.target_main_db, temperature)
        lower = self.soft_min(error_db[self.lower_mask], temperature) - main_min
        upper = self.soft_min(error_db[self.upper_mask], temperature) - main_min
        values = torch.stack([lower, upper])
        worst = band_temperature * torch.logsumexp(values / band_temperature, dim=0)
        return worst, values


def evaluate_minimum(evaluator, model, x_val, y_val, fs_hz=491.52e6, nperseg=4096):
    pred_iq, target_iq, nmse_linear, nmse_db = evaluator.evaluate_full_sequence(
        model, x_val, y_val
    )
    target = evaluator.tensor_to_complex(target_iq)
    pred = evaluator.tensor_to_complex(pred_iq)
    error = target - pred
    freq, target_psd = evaluator.psd_db(target, fs_hz, nperseg)
    _, error_psd = evaluator.psd_db(error, fs_hz, nperseg)
    lower, upper, worst = evaluator.aclr_from_psd_minimum(
        freq,
        error_psd,
        target_psd,
        (20.0, 80.0),
        (-180.0, -120.0),
        (120.0, 180.0),
    )
    return {
        "nmse_linear": float(nmse_linear),
        "nmse_db": float(nmse_db),
        "aclr_lower_db": float(lower),
        "aclr_upper_db": float(upper),
        "aclr_worst_db": float(worst),
    }


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def interpolate_states(anchor, candidate, alpha):
    result = {}
    for key, anchor_value in anchor.items():
        if torch.is_floating_point(anchor_value):
            result[key] = torch.lerp(anchor_value, candidate[key], float(alpha))
        else:
            result[key] = candidate[key].clone()
    return result


def candidate_key(metrics, nmse_target):
    violation = max(0.0, metrics["nmse_db"] - nmse_target)
    return (violation > 0.0, violation, metrics["aclr_worst_db"], metrics["nmse_db"])


def trust_gate(nmse_db, target_db, margin_db):
    if nmse_db >= target_db:
        return 0.0
    if margin_db <= 0.0 or nmse_db <= target_db - margin_db:
        return 1.0
    return float((target_db - nmse_db) / margin_db)


def combine_gradients(nmse_grads, spectral_grads, ratio):
    dot = sum(torch.sum(a * b) for a, b in zip(nmse_grads, spectral_grads))
    nmse_norm_sq = sum(torch.sum(g.square()) for g in nmse_grads)
    if dot < 0.0:
        coefficient = dot / (nmse_norm_sq + 1e-30)
        spectral_grads = [b - coefficient * a for a, b in zip(nmse_grads, spectral_grads)]
    spectral_norm_sq = sum(torch.sum(g.square()) for g in spectral_grads)
    scale = ratio * torch.sqrt(nmse_norm_sq + 1e-30) / torch.sqrt(spectral_norm_sq + 1e-30)
    return [a + scale * b for a, b in zip(nmse_grads, spectral_grads)]


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


def append_row(path, row):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=NMSE_TARGETS)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag", default="min_point")
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
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--extrapolation-max", type=float, default=2.0)
    parser.add_argument("--extrapolation-step", type=float, default=0.05)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluator = load_evaluator()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    anchor_state = evaluator.load_state_dict(checkpoint)
    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_index = int(len(features) * args.split)
    x_train, y_train = features[:split_index], targets[:split_index]
    x_val, y_val = features[split_index:], targets[split_index:]
    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)

    model = evaluator.AutoDiagramDPDNet(anchor_state).to(evaluator.DEVICE)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    loader = DataLoader(
        Frames(x_train, y_train, args.frame_length, args.frame_stride),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    penalty = MinimumPointPenalty(
        y_train, args.segment_length, args.edge_trim, 491.52, evaluator.DEVICE
    )
    target_db = NMSE_TARGETS[args.model]
    initial = evaluate_minimum(evaluator, model.eval(), x_val, y_val)
    best = initial
    best_state = clone_state(model)
    torch.save(best_state, output_dir / "best.pth")
    print(
        f"[{args.model}] initial NMSE={initial['nmse_db']:.3f} "
        f"min-ACLR={initial['aclr_lower_db']:.3f}/"
        f"{initial['aclr_upper_db']:.3f}/{initial['aclr_worst_db']:.3f}"
    )

    for epoch in range(1, args.epochs + 1):
        progress = (epoch - 1) / max(1, args.epochs - 1)
        temperature = args.softmin_start * (
            args.softmin_end / args.softmin_start
        ) ** progress
        gate = trust_gate(best["nmse_db"], target_db, args.nmse_margin_db)
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
            point_score, _ = penalty(
                pred, batch_y, temperature, args.band_temperature
            )
            nmse_grads = torch.autograd.grad(
                nmse_db, parameters, retain_graph=True, allow_unused=True
            )
            point_grads = torch.autograd.grad(
                point_score, parameters, retain_graph=False, allow_unused=True
            )
            nmse_grads = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, nmse_grads)]
            point_grads = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, point_grads)]
            gradients = combine_gradients(
                nmse_grads, point_grads, args.spectral_ratio * gate
            ) if gate > 0.0 else nmse_grads
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad)
            optimizer.step()
            project_to_anchor(model, anchor_state, args.trust_radius)

        metrics = evaluate_minimum(evaluator, model.eval(), x_val, y_val)
        accepted = candidate_key(metrics, target_db) < candidate_key(best, target_db)
        if accepted:
            best = metrics
            best_state = clone_state(model)
            torch.save(best_state, output_dir / "best.pth")
        else:
            model.load_state_dict(best_state, strict=True)
            optimizer.state.clear()
        row = {
            "epoch": epoch,
            "softmin_temperature": temperature,
            "gate": gate,
            "accepted": int(accepted),
            **metrics,
        }
        append_row(output_dir / "history.csv", row)
        print(
            f"[{args.model}] epoch={epoch:02d} tau={temperature:.3f} "
            f"accepted={int(accepted)} NMSE={metrics['nmse_db']:.3f} "
            f"min-ACLR={metrics['aclr_lower_db']:.3f}/"
            f"{metrics['aclr_upper_db']:.3f}/{metrics['aclr_worst_db']:.3f}"
        )

    extrapolation_rows = []
    final_state = best_state
    final_metrics = best
    final_alpha = 1.0
    for alpha in np.arange(
        0.0, args.extrapolation_max + args.extrapolation_step / 2.0, args.extrapolation_step
    ):
        state = interpolate_states(anchor_state, best_state, alpha)
        candidate_model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE).eval()
        metrics = evaluate_minimum(evaluator, candidate_model, x_val, y_val)
        extrapolation_rows.append({"alpha": float(alpha), **metrics})
        if candidate_key(metrics, target_db) < candidate_key(final_metrics, target_db):
            final_state = state
            final_metrics = metrics
            final_alpha = float(alpha)
    with (output_dir / "extrapolation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(extrapolation_rows[0]))
        writer.writeheader()
        writer.writerows(extrapolation_rows)
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
