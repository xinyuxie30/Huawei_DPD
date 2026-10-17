"""Shared metrics for the reference-team ("APNKAN report") evaluation convention.

Conventions implemented here (from the UESTC APNKAN design report):
- NMSE  = 10*log10( sum|y - y_hat|^2 / sum|y|^2 )              (identical to ours)
- ACLR  = 10*log10( [P_err(-180,-100) + P_err(100,180)] / P_target(-100,100) )
  where P are Welch-PSD band-integrated powers of the residual (numerator) and
  the target signal (denominator).  Verified against the report: the dataset
  reference error scores -44.75 dBc here vs. -44.59 dBc in their Fig. 8.

Both metrics are ratios, so they are invariant to the data normalization
(rms vs. max); `--check-normalization` demonstrates this numerically.

Run as a CLI to print the baseline table for the four release checkpoints.
This file must not modify any release checkpoint.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "steps" / "final_evaluate_unified.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"

FS_HZ = 491.52e6
MAIN_BAND_MHZ = (-100.0, 100.0)
ADJ_LOWER_MHZ = (-180.0, -100.0)
ADJ_UPPER_MHZ = (100.0, 180.0)

MODEL_SPECS = {
    "2284": ("checkpoints/AI-DPD/final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth", -40.0),
    "998": ("checkpoints/AI-DPD/final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth", None),
    "786": ("checkpoints/AI-DPD/step800_scratch_h10_l4_base6_rank3_fir41_786.pth", None),
    "796": ("checkpoints/AI-DPD/final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth", None),
}


def load_evaluator():
    spec = importlib.util.spec_from_file_location("final_evaluate_unified", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def band_power(freq_hz, psd_db, band_mhz):
    lo, hi = sorted(float(v) * 1e6 for v in band_mhz)
    mask = (freq_hz >= lo) & (freq_hz <= hi)
    if not np.any(mask):
        raise ValueError(f"Band {band_mhz} MHz has no PSD bins.")
    return float(np.sum(10.0 ** (psd_db[mask] / 10.0)))


def ref_metrics(evaluator, pred_c, target_c, nperseg=4096):
    """NMSE + integrated-power ACLR exactly in the reference team's convention."""
    error = target_c - pred_c
    nmse = np.sum(np.abs(error) ** 2) / np.sum(np.abs(target_c) ** 2)
    freq, target_psd = evaluator.psd_db(target_c, FS_HZ, nperseg)
    _, error_psd = evaluator.psd_db(error, FS_HZ, nperseg)
    p_main = band_power(freq, target_psd, MAIN_BAND_MHZ)
    p_lower = band_power(freq, error_psd, ADJ_LOWER_MHZ)
    p_upper = band_power(freq, error_psd, ADJ_UPPER_MHZ)
    return {
        "nmse_db": float(10 * np.log10(nmse)),
        "aclr_db": float(10 * np.log10((p_lower + p_upper) / p_main)),
        "aclr_lower_db": float(10 * np.log10(p_lower / p_main)),
        "aclr_upper_db": float(10 * np.log10(p_upper / p_main)),
    }


def model_pred_complex(evaluator, model, x, y):
    pred, target, _, _ = evaluator.evaluate_full_sequence(model, x, y)
    return evaluator.tensor_to_complex(pred), evaluator.tensor_to_complex(target)


def load_model(evaluator, checkpoint_path):
    state = evaluator.load_state_dict(checkpoint_path)
    model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE)
    model.eval()
    return model, state


def check_normalization_invariance(evaluator, nperseg):
    """Same model, metrics computed in rms- and max-normalized units."""
    feats_rms, targs_rms, _, rms_x, rms_y, max_x, max_y = evaluator.load_iq_arrays(
        ROOT / "data" / "DPD_data.mat", "rms"
    )
    split = int(len(feats_rms) * 0.8)
    model, _ = load_model(evaluator, ROOT / MODEL_SPECS["2284"][0])
    pred_c, target_c = model_pred_complex(evaluator, model, feats_rms[split:], targs_rms[split:])
    in_rms = ref_metrics(evaluator, pred_c, target_c, nperseg)
    scale = rms_y / max_y  # convert rms-normalized units to max-normalized units
    in_max = ref_metrics(evaluator, pred_c * scale, target_c * scale, nperseg)
    print("normalization invariance check (2284, val):")
    print(f"  rms units: NMSE {in_rms['nmse_db']:.6f} | ACLR {in_rms['aclr_db']:.6f}")
    print(f"  max units: NMSE {in_max['nmse_db']:.6f} | ACLR {in_max['aclr_db']:.6f}")
    mse_max_units = 10 ** (in_max["nmse_db"] / 10) * np.mean(np.abs(target_c * scale) ** 2)
    print(
        f"  (their MSE threshold view: mean|y|^2 = {np.mean(np.abs(target_c * scale) ** 2):.4f} "
        f"in max units -> model MSE = {mse_max_units:.3e}, spec 1e-4 corresponds to "
        f"NMSE {10 * np.log10(1e-4 / np.mean(np.abs(target_c * scale) ** 2)):.2f} dB)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--checkpoint", default=None, help="score one explicit checkpoint instead")
    parser.add_argument("--segment", choices=["val", "train", "full"], default="val")
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--check-normalization", action="store_true")
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    evaluator = load_evaluator()
    features, targets, errors, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split = int(len(features) * args.split)
    segment = {
        "val": slice(split, None),
        "train": slice(None, split),
        "full": slice(None),
    }[args.segment]
    x_seg, y_seg, e_seg = features[segment], targets[segment], errors[segment]

    if args.check_normalization:
        check_normalization_invariance(evaluator, args.nperseg)

    rows = []
    target_c = y_seg[:, 0] + 1j * y_seg[:, 1]
    dataset_err = e_seg[:, 0] + 1j * e_seg[:, 1]
    ref_row = ref_metrics(evaluator, target_c - dataset_err, target_c, args.nperseg)
    rows.append({"model": "dataset_error", "segment": args.segment, **ref_row})

    if args.checkpoint is not None:
        specs = [("custom", Path(args.checkpoint))]
    else:
        specs = [(name, ROOT / MODEL_SPECS[name][0]) for name in args.models]
    for name, ckpt in specs:
        model, _ = load_model(evaluator, ckpt)
        pred_c, tc = model_pred_complex(evaluator, model, x_seg, y_seg)
        rows.append({"model": name, "segment": args.segment, **ref_metrics(evaluator, pred_c, tc, args.nperseg)})

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"metrics_{args.segment}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nreference-convention metrics ({args.segment} segment):")
    print(f"{'model':>14} | {'NMSE dB':>9} | {'ACLR dBc':>9} | {'lower':>8} | {'upper':>8}")
    for row in rows:
        print(
            f"{row['model']:>14} | {row['nmse_db']:9.3f} | {row['aclr_db']:9.2f} | "
            f"{row['aclr_lower_db']:8.2f} | {row['aclr_upper_db']:8.2f}"
        )
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
