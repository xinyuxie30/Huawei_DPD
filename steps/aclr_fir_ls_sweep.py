"""Closed-form frequency-weighted ridge-LS FIR sweep for the three-metric ACLR study.

For each release checkpoint the network base (everything before the output FIR)
is cached once, then a grid of frequency weightings is solved in closed form.
Every candidate is scored with the same NMSE / full-band / lowest-point /
narrow-band metrics as steps/aclr_three_metric_experiment.py.  Release
checkpoints are never overwritten.
"""

import argparse
import csv
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "steps" / "final_evaluate_unified.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_three_metric_experiments"

MAIN_BAND = (20.0, 80.0)
LOWER_BAND = (-180.0, -120.0)
UPPER_BAND = (120.0, 180.0)

MODEL_SPECS = {
    "2284": ("checkpoints/AI-DPD/final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth", -40.0),
    "998": ("checkpoints/AI-DPD/final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth", -37.0),
    "786": ("checkpoints/AI-DPD/step800_scratch_h10_l4_base6_rank3_fir41_786.pth", -35.0),
    "796": ("checkpoints/AI-DPD/final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth", -35.0),
}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("final_evaluate_unified", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_base_sequence(evaluator, model, iq):
    tensor = torch.as_tensor(iq.T, dtype=torch.float64, device=evaluator.DEVICE).unsqueeze(0)
    had_fir = model.has_fir
    model.has_fir = False
    try:
        with torch.no_grad():
            base = model(tensor).squeeze(0).cpu().numpy()
    finally:
        model.has_fir = had_fir
    return base[0] + 1j * base[1]


def apply_fir(base_c, coeffs):
    taps = len(coeffs)
    radius = taps // 2
    out = np.zeros_like(base_c)
    n = len(base_c)
    for k, c in enumerate(coeffs):
        offset = k - radius
        lo = max(0, -offset)
        hi = min(n, n - offset)
        out[lo:hi] += c * base_c[lo + offset : hi + offset]
    return out


class FirDesigner:
    """Weighted ridge LS in the frequency domain via Toeplitz normal equations."""

    def __init__(self, base_c, target_c, taps, fs_mhz):
        self.taps = taps
        self.radius = taps // 2
        self.n = len(base_c)
        z_fft = np.fft.fft(base_c)
        t_fft = np.fft.fft(target_c)
        self.freq = np.fft.fftfreq(self.n, d=1.0 / fs_mhz)
        self.zz = np.abs(z_fft) ** 2
        self.zt = np.conj(z_fft) * t_fft

    def band_mask(self, band):
        lo, hi = sorted(map(float, band))
        return (self.freq >= lo) & (self.freq <= hi)

    def weights(self, main_weight, adjacent_weight, floor_weight, narrow_boost, evaluator):
        w = np.full(self.n, float(floor_weight))
        w[self.band_mask(MAIN_BAND)] = float(main_weight)
        adj = self.band_mask(LOWER_BAND) | self.band_mask(UPPER_BAND)
        w[adj] = float(adjacent_weight)
        if narrow_boost != 1.0:
            narrow = self.band_mask(evaluator.NARROW_LOWER_BAND_MHZ) | self.band_mask(
                evaluator.NARROW_UPPER_BAND_MHZ
            )
            w[narrow] = float(adjacent_weight) * float(narrow_boost)
        return w

    def solve(self, w, ridge):
        a = np.fft.ifft(w * self.zz) * self.n
        b = np.fft.fft(w * self.zt)
        offsets = np.arange(-self.radius, self.radius + 1)
        lags = offsets[None, :] - offsets[:, None]
        lhs = a[np.mod(lags, self.n)] + float(ridge) * np.eye(self.taps)
        rhs = b[np.mod(offsets, self.n)]
        return np.linalg.solve(lhs, rhs)


def metrics_from_pred(evaluator, pred_c, target_c, fs_hz, nperseg):
    error = target_c - pred_c
    nmse = np.sum(np.abs(error) ** 2) / np.sum(np.abs(target_c) ** 2)
    freq, target_psd = evaluator.psd_db(target_c, fs_hz, nperseg)
    _, error_psd = evaluator.psd_db(error, fs_hz, nperseg)
    full = evaluator.aclr_from_psd(freq, error_psd, target_psd, MAIN_BAND, LOWER_BAND, UPPER_BAND)
    point = evaluator.aclr_from_psd_minimum(
        freq, error_psd, target_psd, MAIN_BAND, LOWER_BAND, UPPER_BAND
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
        "nmse_db": float(10 * np.log10(nmse)),
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


def run_model(args, evaluator, name, x_train, y_train, x_val, y_val):
    checkpoint_rel, nmse_target = MODEL_SPECS[name]
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
    else:
        checkpoint_path = ROOT / checkpoint_rel
    if args.nmse_target is not None:
        nmse_target = args.nmse_target
    state = evaluator.load_state_dict(checkpoint_path)
    model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE)
    model.eval()
    taps = len(state["fir_real"])
    fs_hz = args.fs_mhz * 1e6

    base_train = run_base_sequence(evaluator, model, x_train)
    base_val = run_base_sequence(evaluator, model, x_val)
    target_train = y_train[:, 0] + 1j * y_train[:, 1]
    target_val = y_val[:, 0] + 1j * y_val[:, 1]

    original = state["fir_real"].numpy() + 1j * state["fir_imag"].numpy()
    reference = metrics_from_pred(
        evaluator, apply_fir(base_val, original), target_val, fs_hz, args.nperseg
    )
    check = evaluator.evaluate_full_sequence(model, x_val, y_val)
    if abs(check[3] - reference["nmse_db"]) > 1e-6:
        raise RuntimeError(
            f"[{name}] cached-base NMSE {reference['nmse_db']:.6f} dB does not match "
            f"model forward {check[3]:.6f} dB"
        )
    print(
        f"[{name}] checkpoint FIR: NMSE {reference['nmse_db']:.3f} | full/point/narrow "
        f"{reference['full_worst']:.2f}/{reference['point_worst']:.2f}/{reference['narrow_worst']:.2f}"
    )

    designer = FirDesigner(base_train, target_train, taps, args.fs_mhz)
    output_dir = OUT_ROOT / args.tag / name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "model": name,
            "main_weight": np.nan,
            "adjacent_weight": np.nan,
            "floor_weight": np.nan,
            "narrow_boost": np.nan,
            "ridge": np.nan,
            **reference,
            "feasible": reference["nmse_db"] <= nmse_target,
            "source": "checkpoint",
        }
    ]
    best = None
    grid = list(
        itertools.product(args.adjacent_weights, args.floor_weights, args.narrow_boosts, args.ridges)
    )
    for adjacent_weight, floor_weight, narrow_boost, ridge in grid:
        w = designer.weights(args.main_weight, adjacent_weight, floor_weight, narrow_boost, evaluator)
        coeffs = designer.solve(w, ridge)
        metrics = metrics_from_pred(
            evaluator, apply_fir(base_val, coeffs), target_val, fs_hz, args.nperseg
        )
        feasible = metrics["nmse_db"] <= nmse_target
        rows.append(
            {
                "model": name,
                "main_weight": args.main_weight,
                "adjacent_weight": adjacent_weight,
                "floor_weight": floor_weight,
                "narrow_boost": narrow_boost,
                "ridge": ridge,
                **metrics,
                "feasible": feasible,
                "source": "ls_sweep",
            }
        )
        if feasible and (best is None or metrics["three_metric_worst"] < best[0]["three_metric_worst"]):
            best = (metrics, coeffs, (adjacent_weight, floor_weight, narrow_boost, ridge))

    sweep_path = output_dir / "sweep.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if best is None:
        print(f"[{name}] no feasible candidate (NMSE <= {nmse_target}) in the grid")
        summary = {"model": name, "nmse_target_db": nmse_target, **reference, "source": "checkpoint"}
    else:
        metrics, coeffs, params = best
        new_state = {key: value.clone() for key, value in state.items()}
        new_state["fir_real"] = torch.as_tensor(coeffs.real, dtype=torch.float64)
        new_state["fir_imag"] = torch.as_tensor(coeffs.imag, dtype=torch.float64)
        torch.save(new_state, output_dir / "best.pth")
        print(
            f"[{name}] best adj/floor/narrow/ridge {params}: NMSE {metrics['nmse_db']:.3f} | "
            f"full/point/narrow {metrics['full_worst']:.2f}/{metrics['point_worst']:.2f}/"
            f"{metrics['narrow_worst']:.2f} | all-worst {metrics['three_metric_worst']:.2f}"
        )
        summary = {
            "model": name,
            "nmse_target_db": nmse_target,
            "adjacent_weight": params[0],
            "floor_weight": params[1],
            "narrow_boost": params[2],
            "ridge": params[3],
            **metrics,
            "source": "ls_sweep",
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--tag", default="fir_ls_sweep")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the checkpoint path (only valid with a single --models entry).",
    )
    parser.add_argument("--nmse-target", type=float, default=None)
    parser.add_argument("--main-weight", type=float, default=1.0)
    parser.add_argument(
        "--adjacent-weights",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0],
    )
    parser.add_argument("--floor-weights", type=float, nargs="+", default=[0.05, 0.3, 1.0])
    parser.add_argument("--narrow-boosts", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    parser.add_argument("--ridges", type=float, nargs="+", default=[1e-8, 1e-4])
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = load_evaluator()
    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_idx = int(len(features) * args.split)
    x_train, y_train = features[:split_idx], targets[:split_idx]
    x_val, y_val = features[split_idx:], targets[split_idx:]
    results = []
    for name in args.models:
        results.append(run_model(args, evaluator, name, x_train, y_train, x_val, y_val))
    summary_path = OUT_ROOT / args.tag / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in results for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
