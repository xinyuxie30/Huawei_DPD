"""Closed-form weighted ridge-LS FIR sweep under the reference-team convention.

Fits the output FIR on the train split with frequency weights aligned to the
integrated ACLR metric (uniform weight on the +/-(100,180) MHz adjacent bands),
then scores every candidate on the val split with steps/aclr_ref_metrics.py
conventions.  Selection: best (lowest) NMSE subject to ACLR <= --aclr-target
and, when given, NMSE <= --nmse-target.  Release checkpoints are never
modified; the winning FIR is written to a new best.pth.
"""

import argparse
import csv
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
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
    radius = len(coeffs) // 2
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

    def weights(self, metrics_mod, adjacent_weight, floor_weight):
        w = np.full(self.n, float(floor_weight))
        w[self.band_mask(metrics_mod.MAIN_BAND_MHZ)] = 1.0
        adj = self.band_mask(metrics_mod.ADJ_LOWER_MHZ) | self.band_mask(metrics_mod.ADJ_UPPER_MHZ)
        w[adj] = float(adjacent_weight)
        return w

    def solve(self, w, ridge):
        a = np.fft.ifft(w * self.zz) * self.n
        b = np.fft.fft(w * self.zt)
        offsets = np.arange(-self.radius, self.radius + 1)
        lags = offsets[None, :] - offsets[:, None]
        lhs = a[np.mod(lags, self.n)] + float(ridge) * np.eye(self.taps)
        rhs = b[np.mod(offsets, self.n)]
        return np.linalg.solve(lhs, rhs)


def run_model(args, evaluator, metrics_mod, name, checkpoint_path, nmse_target, data):
    x_train, y_train, x_val, y_val = data
    state = evaluator.load_state_dict(checkpoint_path)
    model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE)
    model.eval()
    taps = len(state["fir_real"])

    base_train = run_base_sequence(evaluator, model, x_train)
    base_val = run_base_sequence(evaluator, model, x_val)
    target_train = y_train[:, 0] + 1j * y_train[:, 1]
    target_val = y_val[:, 0] + 1j * y_val[:, 1]

    original = state["fir_real"].numpy() + 1j * state["fir_imag"].numpy()
    reference = metrics_mod.ref_metrics(
        evaluator, apply_fir(base_val, original), target_val, args.nperseg
    )
    check = evaluator.evaluate_full_sequence(model, x_val, y_val)
    if abs(check[3] - reference["nmse_db"]) > 1e-6:
        raise RuntimeError(f"[{name}] cached base does not reproduce the model forward pass")
    print(
        f"[{name}] checkpoint FIR: NMSE {reference['nmse_db']:.3f} | "
        f"ACLR {reference['aclr_db']:.2f} (L {reference['aclr_lower_db']:.2f} / "
        f"U {reference['aclr_upper_db']:.2f})"
    )

    designer = FirDesigner(base_train, target_train, taps, args.fs_mhz)
    rows = [
        {
            "model": name,
            "adjacent_weight": np.nan,
            "floor_weight": np.nan,
            "ridge": np.nan,
            **reference,
            "source": "checkpoint",
        }
    ]
    candidates = []
    for adjacent_weight, floor_weight, ridge in itertools.product(
        args.adjacent_weights, args.floor_weights, args.ridges
    ):
        w = designer.weights(metrics_mod, adjacent_weight, floor_weight)
        coeffs = designer.solve(w, ridge)
        metrics = metrics_mod.ref_metrics(
            evaluator, apply_fir(base_val, coeffs), target_val, args.nperseg
        )
        params = {"adjacent_weight": adjacent_weight, "floor_weight": floor_weight, "ridge": ridge}
        rows.append({"model": name, **params, **metrics, "source": "ls_sweep"})
        candidates.append((metrics, coeffs, params))

    output_dir = OUT_ROOT / args.tag / name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def nmse_ok(metrics):
        return nmse_target is None or metrics["nmse_db"] <= nmse_target

    feasible = [c for c in candidates if c[0]["aclr_db"] <= args.aclr_target and nmse_ok(c[0])]
    if feasible:
        best = min(feasible, key=lambda c: c[0]["nmse_db"])
        status = "feasible"
    else:
        pool = [c for c in candidates if nmse_ok(c[0])] or candidates
        best = min(pool, key=lambda c: c[0]["aclr_db"])
        status = "aclr_gap"
    metrics, coeffs, params = best
    new_state = {key: value.clone() for key, value in state.items()}
    new_state["fir_real"] = torch.as_tensor(coeffs.real, dtype=torch.float64)
    new_state["fir_imag"] = torch.as_tensor(coeffs.imag, dtype=torch.float64)
    torch.save(new_state, output_dir / "best.pth")
    print(
        f"[{name}] {status} best adj={params['adjacent_weight']} floor={params['floor_weight']} "
        f"ridge={params['ridge']}: NMSE {metrics['nmse_db']:.3f} | ACLR {metrics['aclr_db']:.2f} "
        f"(L {metrics['aclr_lower_db']:.2f} / U {metrics['aclr_upper_db']:.2f})"
    )
    return {
        "model": name,
        "status": status,
        "nmse_target_db": nmse_target if nmse_target is not None else np.nan,
        **params,
        **metrics,
        "checkpoint_nmse_db": reference["nmse_db"],
        "checkpoint_aclr_db": reference["aclr_db"],
        "best_path": str(output_dir / "best.pth"),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["2284", "998", "786", "796"])
    parser.add_argument("--tag", default="fir_sweep")
    parser.add_argument("--checkpoint", default=None, help="override (single --models entry only)")
    parser.add_argument("--nmse-target", type=float, default=None, help="override NMSE constraint")
    parser.add_argument("--aclr-target", type=float, default=-45.0)
    parser.add_argument(
        "--adjacent-weights",
        type=float,
        nargs="+",
        default=[1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 17.0, 22.0,
                 30.0, 40.0, 55.0, 75.0, 100.0, 140.0, 200.0, 300.0, 500.0, 800.0],
    )
    parser.add_argument("--floor-weights", type=float, nargs="+", default=[1.0, 0.3, 0.05])
    parser.add_argument("--ridges", type=float, nargs="+", default=[1e-8, 1e-4])
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("aclr_ref_metrics", "steps/aclr_ref_metrics.py")
    if args.checkpoint is not None and len(args.models) != 1:
        raise SystemExit("--checkpoint requires exactly one --models entry")
    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_idx = int(len(features) * args.split)
    data = (features[:split_idx], targets[:split_idx], features[split_idx:], targets[split_idx:])
    results = []
    for name in args.models:
        if args.checkpoint is not None:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = ROOT / checkpoint_path
            nmse_target = args.nmse_target
        else:
            rel, default_target = metrics_mod.MODEL_SPECS[name]
            checkpoint_path = ROOT / rel
            nmse_target = args.nmse_target if args.nmse_target is not None else default_target
        results.append(
            run_model(args, evaluator, metrics_mod, name, checkpoint_path, nmse_target, data)
        )
    summary_path = OUT_ROOT / args.tag / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
