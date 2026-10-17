"""Pareto blend a shape-first checkpoint toward a point-optimized checkpoint.

The two endpoints must have identical wrapper topology and tensor shapes.  The
script scans interpolation of the full state, the deployable correction head,
tables only, and FIR only.  Selection remains shape-first: a candidate must
recover the original fir_pointsweep minimum-point number while respecting
NMSE, integrated ACLR, shape-score, and natural-notch guards.  Both the
closest-shape qualifying candidate and the best point inside the same guards
are saved and plotted for visual PSD review.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def verify_compatible(shape_state, point_state):
    if set(shape_state) != set(point_state):
        missing_shape = sorted(set(point_state) - set(shape_state))
        missing_point = sorted(set(shape_state) - set(point_state))
        raise ValueError(
            f"state keys differ; shape missing={missing_shape}, point missing={missing_point}"
        )
    mismatches = [
        key for key in shape_state if tuple(shape_state[key].shape) != tuple(point_state[key].shape)
    ]
    if mismatches:
        raise ValueError(f"tensor shapes differ: {mismatches}")


def interpolation_keys(state, mode):
    if mode == "full":
        return {
            key
            for key, value in state.items()
            if torch.is_floating_point(value) and key not in {"corr.gmin", "corr.gmax"}
        }
    if mode == "head":
        return {"fir_real", "fir_imag", "corr.tables"}
    if mode == "tables":
        return {"corr.tables"}
    if mode == "fir":
        return {"fir_real", "fir_imag"}
    raise ValueError(mode)


def interpolate_state(shape_state, point_state, alpha, mode):
    selected = interpolation_keys(shape_state, mode)
    result = {}
    for key, shape_value in shape_state.items():
        point_value = point_state[key]
        if key in selected:
            result[key] = torch.lerp(shape_value, point_value, float(alpha))
        else:
            result[key] = shape_value.clone()
    return result


def build_wrapper(evaluator, ps_mod, state, offsets):
    inner = {key[len("base."):]: value for key, value in state.items() if key.startswith("base.")}
    if not inner:
        raise ValueError("expected phase-spline wrapper state")
    taps_reduction = int(inner["fir_real"].numel() - state["fir_real"].numel())
    grid = int(state["corr.tables"].shape[-1] - 1)
    amp_max = float(torch.max(state["corr.gmax"]).item())
    model = ps_mod.PhaseSplineDPD(
        evaluator, inner, taps_reduction, offsets, grid, amp_max
    )
    model.load_state_dict(state, strict=True)
    model = model.to(evaluator.DEVICE).eval()
    mac = int(model.mac_report(evaluator, inner, strict=True)["total"])
    return model, mac


def score_state(audit, evaluator, metrics_mod, ps_mod, state, offsets, arrays, reference_psd, sigma):
    model, mac = build_wrapper(evaluator, ps_mod, state, offsets)
    metrics, payload = audit.evaluate_candidate(
        evaluator, metrics_mod, model, arrays, reference_psd, sigma
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics["mac"] = mac
    return metrics, payload


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_pareto(rows, baseline, shape_anchor, output):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"full": "tab:purple", "head": "tab:blue", "tables": "tab:green", "fir": "tab:orange"}
    for mode in colors:
        selected = [row for row in rows if row["mode"] == mode]
        ax.scatter(
            [row["shape_score_db"] for row in selected],
            [row["point_worst_4096_db"] for row in selected],
            s=18,
            alpha=0.7,
            color=colors[mode],
            label=mode,
        )
    ax.axhline(baseline["point_worst_4096_db"], color="black", linestyle="--", linewidth=1.0, label="fir_pointsweep point target")
    ax.axvline(shape_anchor["shape_score_db"], color="0.4", linestyle=":", linewidth=1.0, label="shape anchor")
    ax.invert_yaxis()
    ax.set_xlabel("shape score (dB, lower is better)")
    ax.set_ylabel("4096 minimum-point ACLR (dB, more negative is better)")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_psd(audit, reference, target_psd, payloads, records, output):
    freq_mhz, reference_raw = reference
    reference_smooth = audit.smooth_psd_db(reference_raw, 4.0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    colors = {
        "shape anchor": "tab:green",
        "selected": "tab:blue",
        "guarded point": "tab:orange",
        "point anchor": "tab:red",
    }
    for axis, smooth in zip(axes, (False, True)):
        curves = [reference_smooth if smooth else reference_raw]
        axis.plot(freq_mhz, curves[0], color="black", linewidth=1.5, label="dataset reference error")
        seen = set()
        for label, record in records:
            signature = (record["mode"], record["alpha"])
            if signature in seen:
                continue
            seen.add(signature)
            raw, smoothed = payloads[signature][1], payloads[signature][2]
            curve = smoothed if smooth else raw
            curves.append(curve)
            axis.plot(freq_mhz, curve, color=colors[label], linewidth=1.2, label=f"{label}: {record['mode']} a={record['alpha']:.3f}")
        if not smooth:
            axis.plot(freq_mhz, target_psd, color="0.7", linestyle="--", linewidth=0.9, label="dataset target y")
        else:
            view = audit.band_mask(freq_mhz, (-200.0, 200.0))
            stacked = np.concatenate([curve[view] for curve in curves])
            axis.set_ylim(float(np.percentile(stacked, 0.5) - 2.0), float(np.percentile(stacked, 99.5) + 2.0))
        axis.axvspan(-180, -100, color="tab:purple", alpha=0.05)
        axis.axvspan(100, 180, color="tab:purple", alpha=0.05)
        axis.set_xlim(-200, 200)
        axis.set_ylabel("PSD (dB/Hz)")
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.legend(loc="best", fontsize=8)
        axis.set_title("smoothed error-shape view" if smooth else "raw 4096-Welch view")
    axes[-1].set_xlabel("Frequency (MHz)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["2284", "998", "786", "796"])
    parser.add_argument("--shape-checkpoint", required=True)
    parser.add_argument("--point-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--mac-limit", type=int, required=True)
    parser.add_argument("--nmse-limit", type=float, required=True)
    parser.add_argument("--tag", default="shape_point_blend")
    parser.add_argument("--alpha-step", type=float, default=0.02)
    parser.add_argument("--modes", nargs="+", choices=["full", "head", "tables", "fir"], default=["full", "head", "tables", "fir"])
    parser.add_argument("--shape-slack", type=float, default=0.03)
    parser.add_argument("--notch-slack", type=float, default=0.50)
    parser.add_argument("--integrated-slack", type=float, default=0.05)
    parser.add_argument("--smooth-sigma-bins", type=float, default=4.0)
    parser.add_argument("--split", type=float, default=0.8)
    args = parser.parse_args()
    if not 0.0 < args.alpha_step <= 1.0:
        parser.error("--alpha-step must be in (0,1]")
    if args.offsets.count(0) != 1 or len(args.offsets) != len(set(args.offsets)):
        parser.error("--offsets must contain one zero and no duplicates")

    audit = load_module("blend_shape_audit", "steps/aclr_psd_shape_audit.py")
    evaluator = load_module("blend_evaluator", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("blend_metrics", "steps/aclr_ref_metrics.py")
    ps_mod = load_module("blend_phase_spline", "steps/aclr_phase_spline_experiment.py")
    features, targets, errors, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split = int(len(features) * args.split)
    arrays = (features[split:], targets[split:])
    target_c = targets[split:, 0] + 1j * targets[split:, 1]
    reference_c = errors[split:, 0] + 1j * errors[split:, 1]
    freq_hz, target_psd = evaluator.psd_db(target_c, 491.52e6, 4096)
    _, reference_psd = evaluator.psd_db(reference_c, 491.52e6, 4096)

    shape_path = absolute(args.shape_checkpoint)
    point_path = absolute(args.point_checkpoint)
    baseline_path = absolute(args.baseline_checkpoint)
    shape_state = evaluator.load_state_dict(shape_path)
    point_state = evaluator.load_state_dict(point_path)
    verify_compatible(shape_state, point_state)

    shape_metrics, shape_payload = score_state(
        audit, evaluator, metrics_mod, ps_mod, shape_state, args.offsets, arrays, reference_psd, args.smooth_sigma_bins
    )
    point_metrics, point_payload = score_state(
        audit, evaluator, metrics_mod, ps_mod, point_state, args.offsets, arrays, reference_psd, args.smooth_sigma_bins
    )
    baseline_model, baseline_mac, _, _ = audit.build_model(
        evaluator, ps_mod, baseline_path, args.model
    )
    baseline_metrics, _ = audit.evaluate_candidate(
        evaluator, metrics_mod, baseline_model, arrays, reference_psd, args.smooth_sigma_bins
    )
    del baseline_model
    baseline_metrics["mac"] = baseline_mac

    alphas = np.unique(
        np.clip(np.append(np.arange(0.0, 1.0 + args.alpha_step * 0.5, args.alpha_step), 1.0), 0.0, 1.0)
    )
    rows = []
    states = {}
    payloads = {}
    for mode in args.modes:
        for alpha in alphas:
            state = interpolate_state(shape_state, point_state, float(alpha), mode)
            metrics, payload = score_state(
                audit, evaluator, metrics_mod, ps_mod, state, args.offsets, arrays, reference_psd, args.smooth_sigma_bins
            )
            row = {"model": args.model, "mode": mode, "alpha": float(alpha), **metrics}
            row["nmse_pass"] = row["nmse_db"] <= args.nmse_limit
            row["mac_pass"] = row["mac"] <= args.mac_limit
            row["point_pass"] = row["point_worst_4096_db"] <= baseline_metrics["point_worst_4096_db"]
            row["integrated_pass"] = row["integrated_aclr_db"] <= baseline_metrics["integrated_aclr_db"] + args.integrated_slack
            row["shape_pass"] = row["shape_score_db"] <= shape_metrics["shape_score_db"] + args.shape_slack
            row["notch_pass"] = row["notch_depth_db"] <= shape_metrics["notch_depth_db"] + args.notch_slack
            row["all_pass"] = all(
                row[key]
                for key in ("nmse_pass", "mac_pass", "point_pass", "integrated_pass", "shape_pass", "notch_pass")
            )
            key = (mode, float(alpha))
            rows.append(row)
            states[key] = state
            payloads[key] = payload

    output = OUT_ROOT / args.tag / args.model
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "scan.csv", rows)
    feasible = [row for row in rows if row["all_pass"]]
    if not feasible:
        counts = {
            key: sum(bool(row[key]) for row in rows)
            for key in ("nmse_pass", "mac_pass", "point_pass", "integrated_pass", "shape_pass", "notch_pass", "all_pass")
        }
        (output / "no_feasible.json").write_text(
            json.dumps({"counts": counts, "rows": len(rows)}, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"no blend satisfies every guard; inspect {output / 'scan.csv'}; guard counts={counts}"
        )
    selected = min(
        feasible,
        key=lambda row: (row["shape_score_db"], row["point_worst_4096_db"], row["alpha"]),
    )
    guarded_point = min(
        feasible,
        key=lambda row: (row["point_worst_4096_db"], row["shape_score_db"], row["alpha"]),
    )
    shape_record = {"mode": args.modes[0], "alpha": 0.0, **shape_metrics}
    point_record = {"mode": args.modes[0], "alpha": 1.0, **point_metrics}
    payloads[(shape_record["mode"], 0.0)] = shape_payload
    payloads[(point_record["mode"], 1.0)] = point_payload

    torch.save(states[(selected["mode"], selected["alpha"])], output / "best.pth")
    torch.save(states[(guarded_point["mode"], guarded_point["alpha"])], output / "best_point_guarded.pth")
    metadata = {
        "model": args.model,
        "shape_checkpoint": str(shape_path.resolve()),
        "point_checkpoint": str(point_path.resolve()),
        "baseline_checkpoint": str(baseline_path.resolve()),
        "offsets": args.offsets,
        "mac_limit": args.mac_limit,
        "nmse_limit": args.nmse_limit,
        "guards": {
            "shape_slack": args.shape_slack,
            "notch_slack": args.notch_slack,
            "integrated_slack": args.integrated_slack,
            "point_target_db": baseline_metrics["point_worst_4096_db"],
        },
        "shape_anchor": shape_metrics,
        "point_anchor": point_metrics,
        "baseline": baseline_metrics,
        "selected": selected,
        "guarded_point": guarded_point,
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "offsets.json").write_text(
        json.dumps({"model": args.model, "selected_offsets": args.offsets, "mac": int(selected["mac"])}, indent=2),
        encoding="utf-8",
    )
    plot_pareto(rows, baseline_metrics, shape_metrics, output / "pareto.png")
    plot_psd(
        audit,
        (freq_hz / 1e6, reference_psd),
        target_psd,
        payloads,
        [
            ("shape anchor", shape_record),
            ("selected", selected),
            ("guarded point", guarded_point),
            ("point anchor", point_record),
        ],
        output / "psd_comparison.png",
    )
    print(json.dumps({"selected": selected, "guarded_point": guarded_point}, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
