"""PSD-shape evaluation for phase-spline wrapper checkpoints.

The review criterion is no longer the lowest-point number alone: the residual
PSD must LOOK like the dataset reference error's PSD (smooth adjacent band,
no artificial notches).  This tool scores and plots exactly that:

* Welch PSD (nperseg=4096) of dataset y, reference error, and model residual;
* shape closeness = mean |model - reference| dB deviation over the adjacent
  bands, plus a notch indicator (band median - band min of the model residual,
  compared with the same quantity for the reference error);
* the usual numbers (NMSE, integrated ACLR, lowest-point ACLR old bands);
* an overlay PNG per checkpoint for visual inspection.

Wrapper geometry (offsets / taps reduction) is read from the checkpoint itself
plus --offsets; MAC is reported against --mac-limit.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"

MAIN_BAND = (20.0, 80.0)
ADJ_BANDS = ((-180.0, -120.0), (120.0, 180.0))
WIDE_ADJ_BANDS = ((-180.0, -100.0), (100.0, 180.0))


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def band_mask(freq_hz, band_mhz):
    lo, hi = sorted(float(v) * 1e6 for v in band_mhz)
    return (freq_hz >= lo) & (freq_hz <= hi)


def shape_metrics(freq, model_psd, ref_psd):
    rows = {}
    for label, bands in (("adj", ADJ_BANDS), ("wide", WIDE_ADJ_BANDS)):
        dev = []
        notch_model = []
        notch_ref = []
        for band in bands:
            mask = band_mask(freq, band)
            dev.append(np.abs(model_psd[mask] - ref_psd[mask]))
            notch_model.append(np.median(model_psd[mask]) - np.min(model_psd[mask]))
            notch_ref.append(np.median(ref_psd[mask]) - np.min(ref_psd[mask]))
        rows[f"{label}_mean_abs_dev_db"] = float(np.mean(np.concatenate(dev)))
        rows[f"{label}_max_notch_db"] = float(np.max(notch_model))
        rows[f"{label}_ref_notch_db"] = float(np.max(notch_ref))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--label", default=None)
    parser.add_argument("--mac-limit", type=int, default=None)
    parser.add_argument("--out-dir", default="psd_shape")
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()

    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("aclr_ref_metrics", "steps/aclr_ref_metrics.py")
    ps_mod = load_module("aclr_phase_spline_experiment", "steps/aclr_phase_spline_experiment.py")

    features, targets, errors, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split = int(len(features) * args.split)
    x_val, y_val, e_val = features[split:], targets[split:], errors[split:]
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))
    fs_hz = 491.52e6

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    state = evaluator.load_state_dict(checkpoint)
    inner = {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
    taps_reduction = int(inner["fir_real"].numel()) - int(state["fir_real"].numel())
    model = ps_mod.PhaseSplineDPD(evaluator, inner, taps_reduction, args.offsets, args.grid, amp_max)
    model.load_state_dict(state, strict=True)
    model = model.to(evaluator.DEVICE).eval()
    mac = model.mac_report(evaluator, inner, strict=True)["total"]

    pred, target, _, nmse_db = evaluator.evaluate_full_sequence(model, x_val, y_val)
    pred_c = evaluator.tensor_to_complex(pred)
    target_c = evaluator.tensor_to_complex(target)
    residual = target_c - pred_c
    ref_error = e_val[:, 0] + 1j * e_val[:, 1]

    freq, y_psd = evaluator.psd_db(target_c, fs_hz, args.nperseg)
    _, ref_psd = evaluator.psd_db(ref_error, fs_hz, args.nperseg)
    _, model_psd = evaluator.psd_db(residual, fs_hz, args.nperseg)

    point = evaluator.aclr_from_psd_minimum(freq, model_psd, y_psd, MAIN_BAND, *ADJ_BANDS)
    ref_point = evaluator.aclr_from_psd_minimum(freq, ref_psd, y_psd, MAIN_BAND, *ADJ_BANDS)
    integrated = metrics_mod.ref_metrics(evaluator, pred_c, target_c, args.nperseg)["aclr_db"]
    shape = shape_metrics(freq, model_psd, ref_psd)

    label = args.label or checkpoint.parent.name + "/" + checkpoint.stem
    summary = {
        "label": label,
        "checkpoint": str(checkpoint),
        "mac": mac,
        "mac_limit": args.mac_limit,
        "nmse_db": float(nmse_db),
        "integrated_aclr_db": float(integrated),
        "point_worst_db": float(max(point[0], point[1])),
        "ref_point_worst_db": float(max(ref_point[0], ref_point[1])),
        **shape,
    }

    out_dir = OUT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = label.replace("/", "_")
    with (out_dir / f"{safe}.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, ax = plt.subplots(figsize=(11, 6))
    f_mhz = freq / 1e6
    order = np.argsort(f_mhz)
    ax.plot(f_mhz[order], y_psd[order], color="#7fd4d4", lw=0.8, label="Dataset y")
    ax.plot(f_mhz[order], ref_psd[order], color="#e377c2", lw=0.9, label="Reference error")
    ax.plot(f_mhz[order], model_psd[order], color="#ff7f0e", lw=0.9, label="EA-TCN error")
    for band in ADJ_BANDS:
        ax.axvspan(band[0], band[1], color="grey", alpha=0.10)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-245, 245)
    fig.tight_layout()
    fig.savefig(out_dir / f"{safe}.png", dpi=110)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"plot: {out_dir / (safe + '.png')}")


if __name__ == "__main__":
    main()
