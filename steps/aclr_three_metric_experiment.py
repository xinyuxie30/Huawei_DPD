"""Experimental ACLR-constrained fine-tuning for the four final DPD models.

This script never overwrites release checkpoints.  It reports three diagnostics:
full-band mean-dB ACLR, lowest-point ACLR, and the fixed 5 MHz narrow-band ACLR.
"""

import argparse
import csv
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "steps" / "final_evaluate_unified.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_three_metric_experiments"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    nmse_target_db: float


MODEL_SPECS = {
    "teacher32": ModelSpec(
        "teacher32",
        "checkpoints/AI-DPD/final_teacher_H32_L6_NMSE42p67_ACLR34p91.pth",
        -40.0,
    ),
    "2284": ModelSpec(
        "2284",
        "checkpoints/AI-DPD/final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth",
        -40.0,
    ),
    "998": ModelSpec(
        "998",
        "checkpoints/AI-DPD/final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth",
        -37.0,
    ),
    "786": ModelSpec(
        "786",
        "checkpoints/AI-DPD/step800_scratch_h10_l4_base6_rank3_fir41_786.pth",
        -35.0,
    ),
    "796": ModelSpec(
        "796",
        "checkpoints/AI-DPD/final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth",
        -35.0,
    ),
}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("final_evaluate_unified", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Frames(Dataset):
    def __init__(self, x, y, teacher, length, stride):
        self.x = torch.as_tensor(x, dtype=torch.float64)
        self.y = torch.as_tensor(y, dtype=torch.float64)
        self.teacher = torch.as_tensor(teacher, dtype=torch.float64)
        self.starts = list(range(0, len(x) - length + 1, stride))
        self.length = length

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        stop = start + self.length
        return self.x[start:stop].T, self.y[start:stop].T, self.teacher[start:stop].T


def band_mask(freq_mhz, band):
    lo, hi = sorted(map(float, band))
    return (freq_mhz >= lo) & (freq_mhz <= hi)


class SpectralPenalty:
    """Welch-consistent adjacent-band penalty.

    The error spectrum is estimated from overlapping Hann segments averaged over
    the batch (matching the evaluator's averaged-PSD-then-dB convention), and the
    main-band references are fixed constants precomputed from the training target
    with the same estimator.
    """

    def __init__(self, target_train, seg_length, edge_trim, fs_mhz, device):
        self.seg_length = int(seg_length)
        self.edge_trim = int(edge_trim)
        self.window = torch.hann_window(
            self.seg_length, periodic=True, dtype=torch.float64, device=device
        )
        freq = torch.fft.fftshift(
            torch.fft.fftfreq(self.seg_length, d=1.0 / fs_mhz, device=device)
        )
        self.main = band_mask(freq, (20.0, 80.0))
        self.lower = band_mask(freq, (-180.0, -120.0))
        self.upper = band_mask(freq, (120.0, 180.0))
        self.narrow_main = band_mask(freq, (31.64, 36.64))
        self.narrow_lower = band_mask(freq, (-129.84, -124.84))
        self.narrow_upper = band_mask(freq, (121.08, 126.08))

        target_c = torch.as_tensor(
            target_train[:, 0] + 1j * target_train[:, 1], dtype=torch.complex128, device=device
        )
        with torch.no_grad():
            power = self._avg_power(target_c.unsqueeze(0), trim=False)
            self.main_ref_db = self._band_mean_db(power, self.main)
            self.narrow_main_ref_db = self._band_mean_db(power, self.narrow_main)

    def _avg_power(self, signal_c, trim=True):
        if trim and self.edge_trim > 0:
            signal_c = signal_c[..., self.edge_trim : signal_c.shape[-1] - self.edge_trim]
        segments = signal_c.unfold(-1, self.seg_length, self.seg_length // 2)
        spectra = torch.fft.fftshift(
            torch.fft.fft(segments * self.window, dim=-1), dim=-1
        )
        return spectra.abs().square().mean(dim=(0, 1))

    @staticmethod
    def _band_mean_db(power, mask):
        return (10.0 * torch.log10(power[mask] + 1e-30)).mean()

    def __call__(self, pred, target, aclr_target_db):
        pred_c = torch.complex(pred[:, 0], pred[:, 1])
        target_c = torch.complex(target[:, 0], target[:, 1])
        error_power = self._avg_power(target_c - pred_c)
        values = torch.stack(
            [
                self._band_mean_db(error_power, self.lower) - self.main_ref_db,
                self._band_mean_db(error_power, self.upper) - self.main_ref_db,
                self._band_mean_db(error_power, self.narrow_lower) - self.narrow_main_ref_db,
                self._band_mean_db(error_power, self.narrow_upper) - self.narrow_main_ref_db,
            ]
        )
        return torch.relu(values - aclr_target_db).square().mean(), values


def evaluate_metrics(evaluator, model, x_val, y_val, fs_hz=491.52e6, nperseg=4096):
    pred_iq, target_iq, nmse_linear, nmse_db = evaluator.evaluate_full_sequence(model, x_val, y_val)
    target = evaluator.tensor_to_complex(target_iq)
    pred = evaluator.tensor_to_complex(pred_iq)
    error = target - pred
    freq, target_psd = evaluator.psd_db(target, fs_hz, nperseg)
    _, error_psd = evaluator.psd_db(error, fs_hz, nperseg)
    full = evaluator.aclr_from_psd(
        freq, error_psd, target_psd, (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0)
    )
    point = evaluator.aclr_from_psd_minimum(
        freq, error_psd, target_psd, (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0)
    )
    narrow = evaluator.aclr_from_psd(
        freq,
        error_psd,
        target_psd,
        evaluator.NARROW_MAIN_BAND_MHZ,
        evaluator.NARROW_LOWER_BAND_MHZ,
        evaluator.NARROW_UPPER_BAND_MHZ,
    )
    return {
        "nmse_linear": float(nmse_linear),
        "nmse_db": float(nmse_db),
        "full_lower": float(full[0]),
        "full_upper": float(full[1]),
        "full_worst": float(full[2]),
        "point_lower": float(point[0]),
        "point_upper": float(point[1]),
        "point_worst": float(point[2]),
        "narrow_lower": float(narrow[0]),
        "narrow_upper": float(narrow[1]),
        "narrow_worst": float(narrow[2]),
        "three_metric_worst": float(max(full[2], point[2], narrow[2])),
    }


def candidate_key(metrics, nmse_target_db):
    nmse_violation = max(0.0, metrics["nmse_db"] - nmse_target_db)
    aclr_violation = max(0.0, metrics["three_metric_worst"] + 45.0)
    return (nmse_violation > 0.0, nmse_violation, aclr_violation, metrics["three_metric_worst"], metrics["nmse_db"])


def guard_target_db(args, spec):
    return spec.nmse_target_db if args.nmse_guard_db is None else args.nmse_guard_db


def append_row(path, row):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_model(
    args, evaluator, spec, x_train, y_train, error_train, prediction_teacher_train,
    x_val, y_val, penalty,
):
    checkpoint = ROOT / spec.checkpoint
    state = evaluator.load_state_dict(checkpoint)
    model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE)
    for parameter in model.parameters():
        parameter.requires_grad = args.train_mode == "all"
    if args.train_mode == "fir":
        model.fir_real.requires_grad = True
        model.fir_imag.requires_grad = True

    output_dir = OUT_ROOT / args.tag / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_out = output_dir / "best.pth"
    history_path = output_dir / "history.csv"
    initial = evaluate_metrics(evaluator, model.eval(), x_val, y_val)
    best = initial
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(best_state, checkpoint_out)
    print(
        f"[{spec.name}] initial NMSE {initial['nmse_db']:.3f} | "
        f"full/point/narrow {initial['full_worst']:.2f}/{initial['point_worst']:.2f}/"
        f"{initial['narrow_worst']:.2f} dBc"
    )

    teacher_train = (
        prediction_teacher_train
        if prediction_teacher_train is not None
        else y_train - args.reference_error_scale * error_train
    )
    loader = DataLoader(
        Frames(x_train, y_train, teacher_train, args.frame_length, args.frame_stride),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y, batch_teacher in loader:
            batch_x = batch_x.to(evaluator.DEVICE)
            batch_y = batch_y.to(evaluator.DEVICE)
            batch_teacher = batch_teacher.to(evaluator.DEVICE)
            optimizer.zero_grad()
            pred = model(batch_x)
            trim = args.edge_trim
            core_pred = pred[..., trim : pred.shape[-1] - trim]
            core_y = batch_y[..., trim : batch_y.shape[-1] - trim]
            time_nmse = torch.sum((core_pred - core_y).square()) / (
                torch.sum(core_y.square()) + 1e-24
            )
            core_teacher = batch_teacher[..., trim : batch_teacher.shape[-1] - trim]
            teacher_nmse = torch.sum((core_pred - core_teacher).square()) / (
                torch.sum(core_y.square()) + 1e-24
            )
            aclr_penalty, _ = penalty(pred, batch_y, args.aclr_target_db)
            batch_nmse_db = 10.0 * torch.log10(time_nmse + 1e-24)
            nmse_guard = torch.relu(batch_nmse_db - guard_target_db(args, spec)).square()
            loss = (
                args.truth_weight * time_nmse
                + args.teacher_weight * teacher_nmse
                + args.aclr_weight * aclr_penalty
                + args.nmse_guard_weight * nmse_guard
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        metrics = evaluate_metrics(evaluator, model.eval(), x_val, y_val)
        row = {"epoch": epoch, "loss": loss_sum / max(1, len(loader)), **metrics}
        append_row(history_path, row)
        print(
            f"[{spec.name}] epoch {epoch:03d} NMSE {metrics['nmse_db']:.3f} | "
            f"full/point/narrow {metrics['full_worst']:.2f}/{metrics['point_worst']:.2f}/"
            f"{metrics['narrow_worst']:.2f} | all-worst {metrics['three_metric_worst']:.2f}"
        )
        if candidate_key(metrics, guard_target_db(args, spec)) < candidate_key(
            best, guard_target_db(args, spec)
        ):
            best = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, checkpoint_out)

    print(
        f"[{spec.name}] best NMSE {best['nmse_db']:.3f} | full/point/narrow "
        f"{best['full_worst']:.2f}/{best['point_worst']:.2f}/{best['narrow_worst']:.2f} | "
        f"saved {checkpoint_out}"
    )
    return {"model": spec.name, "nmse_target_db": spec.nmse_target_db, **best, "checkpoint": str(checkpoint_out)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--train-mode", choices=["fir", "all"], default="fir")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=4096)
    parser.add_argument("--frame-stride", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--aclr-weight", type=float, default=1e-3)
    parser.add_argument("--nmse-guard-weight", type=float, default=3e-3)
    parser.add_argument("--truth-weight", type=float, default=0.1)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--reference-error-scale", type=float, default=0.5)
    parser.add_argument("--prediction-teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--aclr-target-db", type=float, default=-45.5)
    parser.add_argument("--nmse-guard-db", type=float, default=None)
    parser.add_argument("--edge-trim", type=int, default=256)
    parser.add_argument("--seg-length", type=int, default=1024)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluator = load_evaluator()
    features, targets, dataset_errors, *_ = evaluator.load_iq_arrays(
        ROOT / "data" / "DPD_data.mat", "rms"
    )
    split_idx = int(len(features) * args.split)
    x_train, y_train = features[:split_idx], targets[:split_idx]
    error_train = dataset_errors[:split_idx]
    x_val, y_val = features[split_idx:], targets[split_idx:]
    penalty = SpectralPenalty(
        y_train, args.seg_length, args.edge_trim, args.fs_mhz, evaluator.DEVICE
    )
    prediction_teacher_train = None
    if args.prediction_teacher_checkpoint is not None:
        teacher_state = evaluator.load_state_dict(args.prediction_teacher_checkpoint)
        teacher_model = evaluator.AutoDiagramDPDNet(teacher_state).to(evaluator.DEVICE).eval()
        teacher_pred, _, _, _ = evaluator.evaluate_full_sequence(
            teacher_model, x_train, y_train
        )
        prediction_teacher_train = teacher_pred.T
        print(f"Prediction teacher: {args.prediction_teacher_checkpoint}")
    results = []
    for name in args.models:
        results.append(
            run_model(
                args,
                evaluator,
                MODEL_SPECS[name],
                x_train,
                y_train,
                error_train,
                prediction_teacher_train,
                x_val,
                y_val,
                penalty,
            )
        )
    summary = OUT_ROOT / args.tag / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
