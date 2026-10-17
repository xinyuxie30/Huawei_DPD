"""Audit ACLR checkpoints against the dataset reference-error PSD.

This script re-evaluates every discovered ``best*.pth`` on the same validation
split and produces one row per checkpoint.  It deliberately keeps the two
selection goals separate:

* point line: lowest official 4096-bin minimum-point ACLR under NMSE/MAC;
* shape line: closest adjacent-band reference-error shape while retaining at
  least the original ``fir_pointsweep`` point score and integrated ACLR.

The PSD shape score is computed after light linear-power smoothing.  A centered
  RMSE measures shape independently of vertical level.  The balanced score is
  the equal-weight mean of adjacent-band and -180..180 MHz centered RMSE,
  while absolute RMSE,
median level gap, correlation, and excess notch depth remain visible so a
single artificial Welch-bin notch cannot masquerade as a good spectrum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCAN_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"
MODEL_LIMITS = {"2284": -40.0, "998": -37.0, "786": -35.0, "796": -35.0}
MODEL_MAC_LIMITS = {"2284": 2399, "998": 998, "786": 800, "796": 796}
DEFAULT_OFFSETS = {
    "2284": [-2, -1, 0, 1, 2],
    "998": [-2, -1, 0, 1, 2],
    "786": [-1, 0, 1],
    "796": [-2, -1, 0, 1, 2],
}
POINT_MAIN = (20.0, 80.0)
POINT_LOWER = (-180.0, -120.0)
POINT_UPPER = (120.0, 180.0)
SHAPE_ADJACENT = ((-180.0, -100.0), (100.0, 180.0))
SHAPE_FULL = (-180.0, 180.0)
NPERSEGS = (2048, 4096, 8192)
SIDECARS = ("offsets.json", "model_config.json", "ranges.json", "summary.json", "config.json")


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_offsets(value):
    if not isinstance(value, list) or not value:
        return None
    if any(isinstance(item, (list, dict, bool)) for item in value):
        return None
    try:
        offsets = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    if offsets.count(0) != 1 or len(offsets) != len(set(offsets)):
        return None
    return offsets


def json_offsets(metadata):
    if not isinstance(metadata, dict):
        return None, None
    for key in ("selected_offsets", "target_offsets", "offsets", "source_offsets"):
        offsets = valid_offsets(metadata.get(key))
        if offsets is not None:
            return offsets, key
    return None, None


def referenced_checkpoint(metadata, sidecar: Path):
    if not isinstance(metadata, dict):
        return None
    for key in ("source_checkpoint", "checkpoint"):
        raw = metadata.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            root_candidate = ROOT / candidate
            local_candidate = sidecar.parent / candidate
            candidate = root_candidate if root_candidate.exists() else local_candidate
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_offsets(checkpoint: Path, state, model_name: str, visited=None):
    """Resolve feature semantics, recording every heuristic fallback."""
    if "corr.tables" not in state:
        return None, "not_wrapper"
    visited = set() if visited is None else set(visited)
    checkpoint = checkpoint.resolve()
    if checkpoint in visited:
        return None, "metadata_cycle"
    visited.add(checkpoint)

    references = []
    for filename in SIDECARS:
        sidecar = checkpoint.parent / filename
        if not sidecar.exists():
            continue
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        offsets, key = json_offsets(metadata)
        if offsets is not None:
            return offsets, f"{filename}:{key}"
        reference = referenced_checkpoint(metadata, sidecar)
        if reference is not None and reference not in visited:
            references.append(reference)

    for reference in references:
        try:
            ref_state = torch.load(reference, map_location="cpu", weights_only=False)
            if isinstance(ref_state, dict) and "state_dict" in ref_state:
                ref_state = ref_state["state_dict"]
            offsets, source = resolve_offsets(reference, ref_state, model_name, visited)
        except Exception:
            continue
        if offsets is not None:
            return offsets, f"reference:{source}"

    n_features = int(state["corr.tables"].shape[0])
    expected_default = 1 + 2 * (len(DEFAULT_OFFSETS[model_name]) - 1)
    tag = checkpoint.parent.parent.name.lower()
    if model_name == "2284" and n_features == 17 and tag in {"budget2399_train", "shape_2284"}:
        return list(range(-4, 5)), "path_heuristic_2284_pm4"
    if model_name == "786" and n_features == 7:
        if tag in {"budget800_train_l1", "budget800_train_l2"}:
            added = 1 if tag.endswith("l1") else 2
            return [-1, 0, added, 3], f"path_heuristic_786_add_{added}"
        if tag in {"budget800_train_lm2", "shape_786"}:
            return [-2, -1, 0, 3], "path_heuristic_786_add_m2"
    if n_features == 5 and tag in {"phase_spline", "phase_spline_all"}:
        return [-1, 0, 1], "path_heuristic_phase_spline_pm1"
    if model_name == "786" and n_features == 9 and "786m2" in tag:
        return [-2, -1, 0, 1, 2], "path_heuristic_786_pm2"
    if n_features == expected_default:
        path_text = checkpoint.as_posix().lower()
        custom_786 = re.search(
            r"(lag_topology_search_r2|lag_topology_fir_rand|feature_range_786|"
            r"final_optimized|final_verified|final_optimized_robust)",
            path_text,
        )
        if model_name == "786" and custom_786:
            return [-1, 0, 3], "path_heuristic_786_r3"
        return list(DEFAULT_OFFSETS[model_name]), "feature_count_default"
    return None, f"unresolved_{n_features}_features"


def build_model(evaluator, ps_mod, checkpoint: Path, model_name: str):
    state = evaluator.load_state_dict(checkpoint)
    if not any(key.startswith("base.") for key in state):
        model = evaluator.AutoDiagramDPDNet(state).to(evaluator.DEVICE).eval()
        return model, evaluator.real_multiplier_count(state), None, "raw"

    offsets, offset_source = resolve_offsets(checkpoint, state, model_name)
    if offsets is None:
        raise ValueError(f"cannot resolve offsets ({offset_source})")
    n_features = int(state["corr.tables"].shape[0])
    if n_features != 1 + 2 * (len(offsets) - 1):
        raise ValueError(f"offsets {offsets} do not match {n_features} table rows")
    inner = {key[len("base."):]: value for key, value in state.items() if key.startswith("base.")}
    if "fir_real" not in inner or "fir_real" not in state:
        raise ValueError("wrapper checkpoint lacks inner or outer FIR")
    taps_reduction = int(inner["fir_real"].numel() - state["fir_real"].numel())
    grid = int(state["corr.tables"].shape[-1] - 1)
    amp_max = float(torch.max(state["corr.gmax"]).item())
    model = ps_mod.PhaseSplineDPD(
        evaluator, inner, taps_reduction, offsets, grid, amp_max
    )
    model.load_state_dict(state, strict=True)
    model = model.to(evaluator.DEVICE).eval()
    mac = int(model.mac_report(evaluator, inner, strict=True)["total"])
    return model, mac, offsets, offset_source


def band_mask(freq_mhz, bands):
    if isinstance(bands[0], (int, float)):
        bands = (bands,)
    mask = np.zeros(freq_mhz.shape, dtype=bool)
    for lo, hi in bands:
        lo, hi = sorted((float(lo), float(hi)))
        mask |= (freq_mhz >= lo) & (freq_mhz <= hi)
    return mask


def smooth_psd_db(psd_db, sigma_bins):
    power = np.power(10.0, np.asarray(psd_db, dtype=np.float64) / 10.0)
    smooth = gaussian_filter1d(power, float(sigma_bins), mode="nearest")
    return 10.0 * np.log10(np.maximum(smooth, 1e-300))


def centered_rmse(a, b):
    ac = a - np.median(a)
    bc = b - np.median(b)
    return float(np.sqrt(np.mean(np.square(ac - bc))))


def safe_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def notch_depth(psd_db, freq_mhz):
    depths = []
    for band in (POINT_LOWER, POINT_UPPER):
        values = psd_db[band_mask(freq_mhz, band)]
        depths.append(float(np.median(values) - np.min(values)))
    return max(depths)


def point_metrics(evaluator, pred_c, target_c, nperseg):
    error_c = target_c - pred_c
    freq_hz, target_psd = evaluator.psd_db(target_c, 491.52e6, nperseg)
    _, error_psd = evaluator.psd_db(error_c, 491.52e6, nperseg)
    lower, upper, worst = evaluator.aclr_from_psd_minimum(
        freq_hz, error_psd, target_psd, POINT_MAIN, POINT_LOWER, POINT_UPPER
    )
    return float(lower), float(upper), float(worst), freq_hz, target_psd, error_psd


def evaluate_candidate(evaluator, metrics_mod, model, arrays, reference_psd, sigma_bins):
    x_val, y_val = arrays
    pred_iq, target_iq, _, _ = evaluator.evaluate_full_sequence(model, x_val, y_val)
    pred_c = evaluator.tensor_to_complex(pred_iq)
    target_c = evaluator.tensor_to_complex(target_iq)
    error_c = target_c - pred_c
    nmse_linear = float(np.sum(np.abs(error_c) ** 2) / np.sum(np.abs(target_c) ** 2))
    row = {
        "nmse_db": float(10.0 * np.log10(nmse_linear)),
        "nmse_linear": nmse_linear,
    }
    psd_payload = None
    for nperseg in NPERSEGS:
        lower, upper, worst, freq_hz, target_psd, error_psd = point_metrics(
            evaluator, pred_c, target_c, nperseg
        )
        row[f"point_lower_{nperseg}_db"] = lower
        row[f"point_upper_{nperseg}_db"] = upper
        row[f"point_worst_{nperseg}_db"] = worst
        if nperseg == 4096:
            psd_payload = (freq_hz / 1e6, target_psd, error_psd)

    integrated = metrics_mod.ref_metrics(evaluator, pred_c, target_c, 4096)
    row["integrated_aclr_db"] = float(integrated["aclr_db"])
    row["integrated_lower_db"] = float(integrated["aclr_lower_db"])
    row["integrated_upper_db"] = float(integrated["aclr_upper_db"])

    freq_mhz, _, error_psd = psd_payload
    reference_error_psd = reference_psd
    error_smooth = smooth_psd_db(error_psd, sigma_bins)
    reference_smooth = smooth_psd_db(reference_error_psd, sigma_bins)
    adjacent = band_mask(freq_mhz, SHAPE_ADJACENT)
    full = band_mask(freq_mhz, SHAPE_FULL)
    residual_adj = error_smooth[adjacent]
    reference_adj = reference_smooth[adjacent]
    difference = residual_adj - reference_adj
    row.update(
        {
            "shape_rmse_adj_db": centered_rmse(residual_adj, reference_adj),
            "shape_rmse_full_db": centered_rmse(error_smooth[full], reference_smooth[full]),
            "shape_corr_adj": safe_corr(residual_adj, reference_adj),
            "shape_abs_rmse_adj_db": float(np.sqrt(np.mean(np.square(difference)))),
            "shape_median_gap_adj_db": float(np.median(difference)),
            "shape_excess_rmse_adj_db": float(
                np.sqrt(np.mean(np.square(np.maximum(difference, 0.0))))
            ),
            "notch_depth_db": notch_depth(error_psd, freq_mhz),
            "reference_notch_depth_db": notch_depth(reference_error_psd, freq_mhz),
        }
    )
    row["shape_score_db"] = 0.5 * (
        row["shape_rmse_adj_db"] + row["shape_rmse_full_db"]
    )
    row["notch_excess_db"] = row["notch_depth_db"] - row["reference_notch_depth_db"]
    return row, (freq_mhz, error_psd, error_smooth)


def discover_checkpoints(scan_root: Path, tags):
    checkpoints = []
    tag_filter = set(tags or [])
    for checkpoint in sorted(scan_root.glob("*/[0-9]*/best*.pth")):
        tag = checkpoint.parent.parent.name
        model = checkpoint.parent.name
        if model not in MODEL_LIMITS:
            continue
        if tag_filter and tag not in tag_filter:
            continue
        checkpoints.append(checkpoint)
    return checkpoints


def baseline_rows(rows):
    baselines = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("tag") != "fir_pointsweep":
            continue
        if row.get("checkpoint_name") != "best.pth":
            continue
        baselines[row["model"]] = row
    missing = sorted(set(MODEL_LIMITS) - set(baselines))
    if missing:
        raise RuntimeError(f"missing fir_pointsweep baselines for {missing}")
    return baselines


def feasible(row):
    return (
        row.get("status") == "ok"
        and int(row["mac"]) <= MODEL_MAC_LIMITS[row["model"]]
        and float(row["nmse_db"]) <= MODEL_LIMITS[row["model"]]
    )


def select_champions(rows):
    baselines = baseline_rows(rows)
    champions = {}
    for model in MODEL_LIMITS:
        candidates = [row for row in rows if row.get("model") == model and feasible(row)]
        if not candidates:
            raise RuntimeError(f"no feasible candidates for {model}")
        point = min(candidates, key=lambda row: (float(row["point_worst_4096_db"]), float(row["shape_rmse_adj_db"])))
        baseline = baselines[model]
        shape_pool = [
            row
            for row in candidates
            if float(row["point_worst_4096_db"]) <= float(baseline["point_worst_4096_db"]) + 1e-9
            and float(row["integrated_aclr_db"]) <= float(baseline["integrated_aclr_db"]) + 0.05
        ]
        if not shape_pool:
            shape_pool = [baseline]
        shape = min(
            shape_pool,
            key=lambda row: (
                float(row["shape_score_db"]),
                float(row["shape_abs_rmse_adj_db"]),
                float(row["point_worst_4096_db"]),
            ),
        )
        champions[model] = {"baseline": baseline, "point": point, "shape": shape}
    return champions


def plot_champions(model, selected, reference, target, payloads, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    freq_mhz, reference_raw, reference_smooth = reference
    target_smooth = smooth_psd_db(target, 4.0)
    colors = {"baseline": "tab:orange", "point": "tab:blue", "shape": "tab:green"}
    for axis, smooth in zip(axes, (False, True)):
        labels_seen = set()
        visible_error_curves = [reference_smooth if smooth else reference_raw]
        axis.plot(freq_mhz, reference_smooth if smooth else reference_raw, color="black", linewidth=1.5, label="dataset reference error")
        for role in ("baseline", "point", "shape"):
            row = selected[role]
            key = row["candidate_key"]
            raw, smoothed = payloads[key][1], payloads[key][2]
            label = f"{role}: {row['tag']}/{row['checkpoint_name']}"
            signature = (role, row["sha256"], row.get("offsets", ""))
            if signature in labels_seen:
                continue
            labels_seen.add(signature)
            curve = smoothed if smooth else raw
            visible_error_curves.append(curve)
            axis.plot(freq_mhz, curve, color=colors[role], linewidth=1.2, label=label)
        if not smooth:
            axis.plot(freq_mhz, target, color="0.65", linestyle="--", linewidth=0.9, label="dataset target y")
        else:
            view = band_mask(freq_mhz, (-200.0, 200.0))
            stacked = np.concatenate([curve[view] for curve in visible_error_curves])
            axis.set_ylim(float(np.percentile(stacked, 0.5) - 2.0), float(np.percentile(stacked, 99.5) + 2.0))
        axis.axvspan(-180, -100, color="tab:purple", alpha=0.05)
        axis.axvspan(100, 180, color="tab:purple", alpha=0.05)
        axis.set_xlim(-200, 200)
        axis.grid(True, linestyle=":", alpha=0.55)
        axis.set_ylabel("PSD (dB/Hz)")
        axis.set_title(f"{model}: {'smoothed shape view' if smooth else 'raw 4096-Welch view'}")
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Frequency (MHz)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_champion(role, model, row, output_dir):
    destination = output_dir / role / model
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(row["checkpoint"])
    for stale_name in (*SIDECARS, "audit_summary.json"):
        stale = destination / stale_name
        if stale.exists():
            stale.unlink()
    shutil.copy2(source, destination / "best.pth")
    for sidecar_name in SIDECARS:
        sidecar = source.parent / sidecar_name
        if sidecar.exists():
            shutil.copy2(sidecar, destination / sidecar_name)
    offsets = [int(value) for value in str(row.get("offsets", "")).split(",") if value]
    if offsets and not (destination / "offsets.json").exists():
        (destination / "offsets.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "selected_offsets": offsets,
                    "mac": int(row["mac"]),
                    "source_checkpoint": str(source.resolve()),
                    "offset_source": row.get("offset_source", "audit"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    metadata = dict(row)
    metadata["selection_role"] = role
    (destination / "audit_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SCAN_ROOT / "shape_audit")
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--smooth-sigma-bins", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None, help="debug: evaluate only first N files")
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.split < 1.0:
        parser.error("--split must be between zero and one")
    if args.smooth_sigma_bins <= 0.0:
        parser.error("--smooth-sigma-bins must be positive")

    scan_root = args.scan_root if args.scan_root.is_absolute() else ROOT / args.scan_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = load_module("shape_audit_evaluator", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("shape_audit_metrics", "steps/aclr_ref_metrics.py")
    ps_mod = load_module("shape_audit_phase_spline", "steps/aclr_phase_spline_experiment.py")
    features, targets, errors, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split = int(len(features) * args.split)
    arrays = (features[split:], targets[split:])
    target_c = targets[split:, 0] + 1j * targets[split:, 1]
    dataset_error_c = errors[split:, 0] + 1j * errors[split:, 1]
    freq_hz, target_psd = evaluator.psd_db(target_c, 491.52e6, 4096)
    _, reference_psd = evaluator.psd_db(dataset_error_c, 491.52e6, 4096)
    freq_mhz = freq_hz / 1e6
    reference_smooth = smooth_psd_db(reference_psd, args.smooth_sigma_bins)

    checkpoints = discover_checkpoints(scan_root, args.tags)
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    print(f"Device: {evaluator.DEVICE}; checkpoints: {len(checkpoints)}")
    rows = []
    cache = {}
    payloads = {}
    for index, checkpoint in enumerate(checkpoints, start=1):
        model_name = checkpoint.parent.name
        tag = checkpoint.parent.parent.name
        digest = file_sha256(checkpoint)
        row = {
            "model": model_name,
            "tag": tag,
            "checkpoint_name": checkpoint.name,
            "checkpoint": str(checkpoint.resolve()),
            "sha256": digest,
            "status": "error",
        }
        model = None
        try:
            state = evaluator.load_state_dict(checkpoint)
            offsets, offset_source = resolve_offsets(checkpoint, state, model_name)
            cache_key = (digest, tuple(offsets or []))
            row["candidate_key"] = f"{digest}:{','.join(map(str, offsets or []))}"
            if cache_key in cache:
                cached_row, payload = cache[cache_key]
                for key, value in cached_row.items():
                    if key not in {"model", "tag", "checkpoint_name", "checkpoint", "sha256"}:
                        row[key] = value
                row["deduplicated_from"] = cached_row["checkpoint"]
            else:
                model, mac, offsets, offset_source = build_model(
                    evaluator, ps_mod, checkpoint, model_name
                )
                metrics, payload = evaluate_candidate(
                    evaluator,
                    metrics_mod,
                    model,
                    arrays,
                    reference_psd,
                    args.smooth_sigma_bins,
                )
                row.update(metrics)
                row["mac"] = int(mac)
                row["offsets"] = "" if offsets is None else ",".join(map(str, offsets))
                row["offset_source"] = offset_source
                row["status"] = "ok"
                cache[cache_key] = (dict(row), payload)
            payloads[row["candidate_key"]] = payload
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(row)
        if index == 1 or index % 10 == 0 or index == len(checkpoints):
            ok = sum(candidate.get("status") == "ok" for candidate in rows)
            print(f"[{index:>3}/{len(checkpoints)}] ok={ok} errors={index-ok} {tag}/{model_name}/{checkpoint.name}")

    write_csv(out_dir / "audit.csv", rows)
    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        write_csv(out_dir / "failures.csv", failures)
    champions = select_champions(rows)
    champion_json = {}
    for model, selected in champions.items():
        champion_json[model] = {}
        for role, row in selected.items():
            champion_json[model][role] = dict(row)
            if role in {"point", "shape"} and not args.no_copy:
                copy_champion(role, model, row, out_dir)
        plot_champions(
            model,
            selected,
            (freq_mhz, reference_psd, reference_smooth),
            target_psd,
            payloads,
            out_dir / f"{model}_psd_comparison.png",
        )
    (out_dir / "champions.json").write_text(
        json.dumps(champion_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nmodel  role      NMSE       point4096  integrated  shapeScore  candidate")
    for model, selected in champions.items():
        for role in ("baseline", "point", "shape"):
            row = selected[role]
            print(
                f"{model:>5}  {role:<8} {float(row['nmse_db']):>9.3f}  "
                f"{float(row['point_worst_4096_db']):>10.3f}  "
                f"{float(row['integrated_aclr_db']):>10.3f}  "
                f"{float(row['shape_score_db']):>10.3f}  "
                f"{row['tag']}/{row['checkpoint_name']}"
            )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
