"""Closed-form FIR weight-sweep on a phase-spline wrapper, selected by lowest-point ACLR.

The output FIR is re-solved on the train split over a grid of frequency
weightings (adjacent/floor/ridge); every candidate is scored on the val split
with the hard lowest-point ACLR (main (20,80), adjacent +/-(120,180)) and the
best point-metric candidate satisfying the NMSE limit is saved.  Motivated by
the observation that FIR variants shuffle the Welch minimum location by
several tenths of a dB at negligible NMSE cost.
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

from importlib import util as _util


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"


def load_module(name, relative):
    spec = _util.spec_from_file_location(name, ROOT / relative)
    module = _util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--taps-reduction", type=int, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--tag", default="fir_pointsweep")
    parser.add_argument("--nmse-limit", type=float, required=True)
    parser.add_argument("--adjacent-weights", type=float, nargs="+",
                        default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0])
    parser.add_argument("--floor-weights", type=float, nargs="+", default=[0.05, 0.3, 1.0, 2.0])
    parser.add_argument("--ridges", type=float, nargs="+", default=[1e-8, 1e-4])
    parser.add_argument("--random-profiles", type=int, default=0,
                        help="additionally try N random piecewise adjacent-band weight profiles")
    parser.add_argument("--profile-segments", type=int, default=6)
    parser.add_argument("--profile-seed", type=int, default=20260714)
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("aclr_ref_metrics", "steps/aclr_ref_metrics.py")
    ps_mod = load_module("aclr_phase_spline_experiment", "steps/aclr_phase_spline_experiment.py")
    sweep_mod = load_module("aclr_ref_fir_sweep", "steps/aclr_ref_fir_sweep.py")

    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_idx = int(len(features) * args.split)
    x_train, y_train = features[:split_idx], targets[:split_idx]
    x_val, y_val = features[split_idx:], targets[split_idx:]
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))
    fs_hz = args.fs_mhz * 1e6

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    state = evaluator.load_state_dict(ckpt)
    inner = {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
    model = ps_mod.PhaseSplineDPD(evaluator, inner, args.taps_reduction, args.offsets, args.grid, amp_max)
    model.load_state_dict(state, strict=True)
    model = model.to(evaluator.DEVICE).eval()

    base_train = sweep_mod.run_base_sequence(evaluator, model, x_train)
    base_val = sweep_mod.run_base_sequence(evaluator, model, x_val)
    target_train = y_train[:, 0] + 1j * y_train[:, 1]
    target_val = y_val[:, 0] + 1j * y_val[:, 1]
    taps = len(model.fir_real)
    designer = sweep_mod.FirDesigner(base_train, target_train, taps, args.fs_mhz)

    def point_metrics(pred_c):
        error = target_val - pred_c
        nmse = float(10 * np.log10(np.sum(np.abs(error) ** 2) / np.sum(np.abs(target_val) ** 2)))
        freq, tp = evaluator.psd_db(target_val, fs_hz, args.nperseg)
        _, ep_ = evaluator.psd_db(error, fs_hz, args.nperseg)
        lower, upper, worst = evaluator.aclr_from_psd_minimum(
            freq, ep_, tp, (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0)
        )
        return nmse, float(lower), float(upper), float(worst)

    incumbent = state["fir_real"].numpy() + 1j * state["fir_imag"].numpy()
    ref = point_metrics(sweep_mod.apply_fir(base_val, incumbent))
    print(f"[{args.model}] incumbent FIR: NMSE {ref[0]:.3f} | point {ref[1]:.2f}/{ref[2]:.2f}/{ref[3]:.2f}")

    rows = [{"adjacent": np.nan, "floor": np.nan, "ridge": np.nan,
             "nmse_db": ref[0], "point_lower": ref[1], "point_upper": ref[2], "point_worst": ref[3],
             "source": "incumbent"}]
    best = (ref, incumbent, None) if ref[0] <= args.nmse_limit else None
    for adjacent, floor, ridge in itertools.product(args.adjacent_weights, args.floor_weights, args.ridges):
        w = designer.weights(metrics_mod, adjacent, floor)
        coeffs = designer.solve(w, ridge)
        m = point_metrics(sweep_mod.apply_fir(base_val, coeffs))
        rows.append({"adjacent": adjacent, "floor": floor, "ridge": ridge,
                     "nmse_db": m[0], "point_lower": m[1], "point_upper": m[2], "point_worst": m[3],
                     "source": "sweep"})
        if m[0] <= args.nmse_limit and (best is None or m[3] < best[0][3]):
            best = (m, coeffs, (adjacent, floor, ridge))

    if args.random_profiles > 0:
        rng = np.random.default_rng(args.profile_seed)
        seg = args.profile_segments
        lower_edges = np.linspace(-180.0, -120.0, seg + 1)
        upper_edges = np.linspace(120.0, 180.0, seg + 1)
        for trial in range(args.random_profiles):
            w = np.ones(designer.n)
            floor = float(10 ** rng.uniform(np.log10(0.05), np.log10(2.0)))
            w[:] = floor
            w[designer.band_mask(metrics_mod.MAIN_BAND_MHZ)] = 1.0
            for edges in (lower_edges, upper_edges):
                for k in range(seg):
                    weight = float(10 ** rng.uniform(np.log10(0.3), np.log10(30.0)))
                    w[designer.band_mask((edges[k], edges[k + 1]))] = weight
            ridge = float(rng.choice(args.ridges))
            coeffs = designer.solve(w, ridge)
            m = point_metrics(sweep_mod.apply_fir(base_val, coeffs))
            rows.append({"adjacent": np.nan, "floor": floor, "ridge": ridge,
                         "nmse_db": m[0], "point_lower": m[1], "point_upper": m[2],
                         "point_worst": m[3], "source": f"random_{trial}"})
            if m[0] <= args.nmse_limit and (best is None or m[3] < best[0][3]):
                best = (m, coeffs, (f"random_{trial}", floor, ridge))

    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    m, coeffs, params = best
    new_state = {k: v.clone() for k, v in state.items()}
    new_state["fir_real"] = torch.as_tensor(np.real(coeffs), dtype=torch.float64)
    new_state["fir_imag"] = torch.as_tensor(np.imag(coeffs), dtype=torch.float64)
    torch.save(new_state, output_dir / "best.pth")
    print(
        f"[{args.model}] best {params}: NMSE {m[0]:.3f} | point {m[1]:.2f}/{m[2]:.2f}/{m[3]:.2f} "
        f"({m[3]-ref[3]:+.2f} vs incumbent) -> {output_dir / 'best.pth'}"
    )


if __name__ == "__main__":
    main()
