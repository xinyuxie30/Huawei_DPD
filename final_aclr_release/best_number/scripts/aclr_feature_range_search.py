"""Search phase-spline feature ranges at fixed topology, grid, and MAC count.

The phase-spline checkpoints use uniformly spaced piecewise-linear lookup
tables over per-feature ranges stored in ``corr.gmin`` and ``corr.gmax``.
Their original common range is deliberately conservative; in particular it
wastes knots when a rotated real/imaginary feature occupies only a narrow,
asymmetric part of that interval.  This experiment changes only those ranges:

* ``--range-scales`` scales every source bound about physical zero; or
* ``--quantiles`` derives a separate range for every feature from the complete
  training split.

For every candidate, the source tables are evaluated at the new physical knot
coordinates (queries outside the source range clamp to its endpoint), giving a
function-preserving/interpolated warm start rather than copying table indices.
The EA-TCN base, output FIR, offsets, table grid, and deployed MAC count remain
fixed.  Only ``corr.tables`` is then fit on the complete training sequence with
the same convex quadratic used by ``aclr_active_bin_table_optimization.py``:
time-domain MSE plus the powers at the two validation-selected active Welch
bins and an optional L2 trust term.  The fitted direction is scanned on
validation, with hard NMSE feasibility and exact minimum-point ACLR selection.

Outputs are ``best.pth``, ``summary.json``, ``ranges.json``, and ``search.csv``
under ``eval_outputs_direct/aclr_ref_convention/<tag>/<model>/``.  Release and
source checkpoints are never modified.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"
ACTIVE_BIN_SCRIPT = ROOT / "steps" / "aclr_active_bin_table_optimization.py"
MODEL_OFFSETS = {
    "2284": (-2, -1, 0, 1, 2),
    "998": (-2, -1, 0, 1, 2),
    "796": (-2, -1, 0, 1, 2),
    "786": (-1, 0, 1),
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def feature_semantics(offsets):
    semantics = [{"row": 0, "kind": "amplitude", "offset": 0}]
    row = 1
    for offset in offsets:
        if int(offset) == 0:
            continue
        semantics.append({"row": row, "kind": "real", "offset": int(offset)})
        semantics.append({"row": row + 1, "kind": "imag", "offset": int(offset)})
        row += 2
    return semantics


def phase_features_numpy(x_iq, offsets):
    """Reproduce ``PhaseSplineCorrection`` features as ``(n_feat, samples)``."""
    x_iq = np.asarray(x_iq)
    if x_iq.ndim != 2 or x_iq.shape[1] != 2:
        raise ValueError(f"Expected x_iq shape (samples, 2), got {x_iq.shape}")
    xc = x_iq[:, 0] + 1j * x_iq[:, 1]
    amplitude = np.abs(xc)
    rotation = np.conj(xc) / (amplitude + 1e-12)
    features = [amplitude]
    for offset in offsets:
        offset = int(offset)
        if offset == 0:
            continue
        rotated = np.roll(xc, -offset) * rotation
        features.extend((rotated.real, rotated.imag))
    return np.stack(features, axis=0).astype(np.float64, copy=False)


def infer_wrapper_layout(state):
    required = {"corr.tables", "corr.gmin", "corr.gmax", "fir_real", "fir_imag"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Source checkpoint is missing wrapper keys: {missing}")
    inner = {
        key[len("base.") :]: value
        for key, value in state.items()
        if key.startswith("base.")
    }
    if not inner:
        raise ValueError("Expected a phase-spline wrapper checkpoint with base.* keys")
    if "fir_real" not in inner or "fir_imag" not in inner:
        raise ValueError("Wrapped base checkpoint has no output FIR")
    original_taps = int(inner["fir_real"].numel())
    wrapper_taps = int(state["fir_real"].numel())
    if state["fir_real"].shape != state["fir_imag"].shape:
        raise ValueError("Top-level real/imaginary FIR shapes differ")
    taps_reduction = original_taps - wrapper_taps
    if taps_reduction < 0:
        raise ValueError(
            f"Wrapper FIR ({wrapper_taps}) exceeds base FIR ({original_taps})"
        )
    tables = state["corr.tables"]
    if tables.ndim != 3 or int(tables.shape[1]) != 2:
        raise ValueError(f"Expected corr.tables shape (features, 2, knots), got {tables.shape}")
    grid = int(tables.shape[-1] - 1)
    if grid < 1:
        raise ValueError(f"Invalid spline grid inferred from checkpoint: {grid}")
    n_feat = int(tables.shape[0])
    if tuple(state["corr.gmin"].shape) != (n_feat,):
        raise ValueError("corr.gmin shape does not match corr.tables")
    if tuple(state["corr.gmax"].shape) != (n_feat,):
        raise ValueError("corr.gmax shape does not match corr.tables")
    if not torch.all(state["corr.gmax"] > state["corr.gmin"]):
        raise ValueError("Every source corr.gmax must be greater than corr.gmin")
    return inner, taps_reduction, grid


def validate_feature_count(state, offsets):
    expected = 1 + 2 * sum(int(offset) != 0 for offset in offsets)
    actual = int(state["corr.tables"].shape[0])
    if actual != expected:
        raise ValueError(
            f"Offsets imply {expected} phase features, but checkpoint has {actual} rows"
        )


def table_values_at_physical_coordinates(tables, gmin, gmax, coordinates):
    """Evaluate per-row piecewise-linear tables with endpoint clamping.

    Args:
        tables: ``(features, 2, grid + 1)``.
        gmin/gmax: ``(features,)`` source physical bounds.
        coordinates: ``(features, points)`` physical feature coordinates.
    """
    if coordinates.ndim != 2 or coordinates.shape[0] != tables.shape[0]:
        raise ValueError(
            "coordinates must have shape (features, points); "
            f"got {tuple(coordinates.shape)}"
        )
    grid = int(tables.shape[-1] - 1)
    scaled = (coordinates - gmin[:, None]) / (gmax - gmin)[:, None] * grid
    scaled = scaled.clamp(0.0, float(grid))
    lower = scaled.floor().clamp(0.0, float(grid - 1))
    fraction = scaled - lower
    index = lower.long().unsqueeze(1).expand(-1, tables.shape[1], -1)
    first = torch.gather(tables, dim=2, index=index)
    second = torch.gather(tables, dim=2, index=index + 1)
    return first + fraction.unsqueeze(1) * (second - first)


def remap_feature_ranges(state, target_gmin, target_gmax):
    """Warm-start tables by interpolation in physical feature coordinates."""
    infer_wrapper_layout(state)
    source_tables = state["corr.tables"].detach().cpu()
    source_gmin = state["corr.gmin"].detach().cpu().to(dtype=source_tables.dtype)
    source_gmax = state["corr.gmax"].detach().cpu().to(dtype=source_tables.dtype)
    target_gmin = torch.as_tensor(target_gmin, dtype=source_tables.dtype).cpu()
    target_gmax = torch.as_tensor(target_gmax, dtype=source_tables.dtype).cpu()
    if target_gmin.shape != source_gmin.shape or target_gmax.shape != source_gmax.shape:
        raise ValueError("Target range shapes must match source corr.gmin/corr.gmax")
    if not torch.isfinite(target_gmin).all() or not torch.isfinite(target_gmax).all():
        raise ValueError("Target feature ranges must be finite")
    if not torch.all(target_gmax > target_gmin):
        raise ValueError("Every target gmax must be greater than target gmin")

    unit = torch.linspace(
        0.0,
        1.0,
        source_tables.shape[-1],
        dtype=source_tables.dtype,
    )
    target_knots = target_gmin[:, None] + (
        target_gmax - target_gmin
    )[:, None] * unit[None, :]
    target_tables = table_values_at_physical_coordinates(
        source_tables, source_gmin, source_gmax, target_knots
    )
    result = clone_state(state)
    result["corr.tables"] = target_tables
    result["corr.gmin"] = target_gmin
    result["corr.gmax"] = target_gmax
    return result


def parse_quantile_spec(text):
    """Parse ``q`` as [q, 1-q], or ``low:high`` as explicit quantiles."""
    text = str(text).strip()
    pieces = [part for part in re.split(r"[:,/]", text) if part]
    try:
        values = [float(part) for part in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid quantile specification: {text}") from exc
    if len(values) == 1:
        tail = values[0]
        if not 0.0 <= tail < 0.5:
            raise argparse.ArgumentTypeError(
                "a scalar quantile is a symmetric tail probability and must be in [0, 0.5)"
            )
        low, high = tail, 1.0 - tail
    elif len(values) == 2:
        low, high = values
    else:
        raise argparse.ArgumentTypeError(
            "use q for symmetric [q,1-q], or low:high for an explicit pair"
        )
    if not (0.0 <= low < high <= 1.0):
        raise argparse.ArgumentTypeError(
            f"quantiles must satisfy 0 <= low < high <= 1, got {low}:{high}"
        )
    return float(low), float(high)


def range_label(mode, value):
    if mode == "scale":
        return f"scale={float(value):g}"
    low, high = value
    return f"quantiles={low:g}:{high:g}"


def make_range_candidates(state, train_features, range_scales=None, quantiles=None):
    source_gmin = state["corr.gmin"].detach().cpu().double()
    source_gmax = state["corr.gmax"].detach().cpu().double()
    if train_features.shape[0] != source_gmin.numel():
        raise ValueError(
            f"Training feature rows {train_features.shape[0]} do not match checkpoint "
            f"rows {source_gmin.numel()}"
        )
    candidates = []
    if range_scales is not None:
        for scale in range_scales:
            scale = float(scale)
            target_gmin = source_gmin * scale
            target_gmax = source_gmax * scale
            candidates.append(
                {
                    "label": range_label("scale", scale),
                    "mode": "scale",
                    "scale": scale,
                    "quantile_low": None,
                    "quantile_high": None,
                    "gmin": target_gmin,
                    "gmax": target_gmax,
                }
            )
    elif quantiles is not None:
        for low, high in quantiles:
            values = np.quantile(train_features, [low, high], axis=1)
            target_gmin = torch.as_tensor(values[0], dtype=torch.float64)
            target_gmax = torch.as_tensor(values[1], dtype=torch.float64)
            if not torch.all(target_gmax > target_gmin):
                bad = torch.nonzero(target_gmax <= target_gmin).flatten().tolist()
                raise ValueError(
                    f"Degenerate train-quantile ranges for rows {bad} at {low}:{high}"
                )
            candidates.append(
                {
                    "label": range_label("quantile", (low, high)),
                    "mode": "quantile",
                    "scale": None,
                    "quantile_low": float(low),
                    "quantile_high": float(high),
                    "gmin": target_gmin,
                    "gmax": target_gmax,
                }
            )
    else:
        raise ValueError("Either range_scales or quantiles must be supplied")
    return candidates


def build_wrapper(evaluator, ps_mod, state, offsets, amp_max):
    inner, taps_reduction, grid = infer_wrapper_layout(state)
    model = ps_mod.PhaseSplineDPD(
        evaluator,
        inner,
        taps_reduction,
        list(offsets),
        grid,
        amp_max,
    )
    model.load_state_dict(state, strict=True)
    return model.to(evaluator.DEVICE)


def validation_active_bins(active_mod, welch, error_complex):
    with torch.no_grad():
        power = welch.power(error_complex).detach().cpu().numpy()
    lower = active_mod.separated_low_bins(
        power, welch.freq_mhz, (-180.0, -120.0), count=1, separation=1
    )[0]
    upper = active_mod.separated_low_bins(
        power, welch.freq_mhz, (120.0, 180.0), count=1, separation=1
    )[0]
    return int(lower), int(upper)


def feasible_better(active_mod, metrics, best_metrics, nmse_limit):
    if metrics["nmse_db"] > nmse_limit:
        return False
    return active_mod.candidate_key(metrics, nmse_limit) < active_mod.candidate_key(
        best_metrics, nmse_limit
    )


def candidate_public_record(candidate, train_features):
    gmin = candidate["gmin"].detach().cpu().numpy()
    gmax = candidate["gmax"].detach().cpu().numpy()
    return {
        "label": candidate["label"],
        "mode": candidate["mode"],
        "scale": candidate["scale"],
        "quantile_low": candidate["quantile_low"],
        "quantile_high": candidate["quantile_high"],
        "gmin": gmin.tolist(),
        "gmax": gmax.tolist(),
        "train_fraction_below": np.mean(train_features < gmin[:, None], axis=1).tolist(),
        "train_fraction_above": np.mean(train_features > gmax[:, None], axis=1).tolist(),
    }


def search_row(
    model_name,
    range_index,
    candidate,
    source,
    active_mhz,
    spectral_weight,
    alpha,
    loss,
    closure_calls,
    metrics,
):
    return {
        "model": model_name,
        "range_index": range_index,
        "range_label": candidate["label"],
        "range_mode": candidate["mode"],
        "range_scale": "" if candidate["scale"] is None else candidate["scale"],
        "quantile_low": (
            "" if candidate["quantile_low"] is None else candidate["quantile_low"]
        ),
        "quantile_high": (
            "" if candidate["quantile_high"] is None else candidate["quantile_high"]
        ),
        "source": source,
        "active_lower_mhz": "" if active_mhz is None else active_mhz[0],
        "active_upper_mhz": "" if active_mhz is None else active_mhz[1],
        "spectral_weight": spectral_weight,
        "alpha": alpha,
        "lbfgs_loss": loss,
        "closure_calls": closure_calls,
        **metrics,
    }


def source_candidate(state):
    return {
        "label": "source",
        "mode": "source",
        "scale": None,
        "quantile_low": None,
        "quantile_high": None,
        "gmin": state["corr.gmin"].detach().cpu().double().clone(),
        "gmax": state["corr.gmax"].detach().cpu().double().clone(),
    }


def run_model(args, model_name, evaluator, metrics_mod, ps_mod, active_mod, arrays):
    x_train, y_train, x_val, y_val, amp_max = arrays
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(args.checkpoint_root) / model_name / "best.pth"
    )
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    offsets, _ = active_mod.resolve_checkpoint_offsets(
        checkpoint,
        model_name,
        args.offsets,
        require_metadata=args.checkpoint is not None,
    )
    offsets = tuple(offsets)
    source_state = evaluator.load_state_dict(checkpoint)
    validate_feature_count(source_state, offsets)
    inner, taps_reduction, grid = infer_wrapper_layout(source_state)
    train_features = phase_features_numpy(x_train, offsets)
    candidates = make_range_candidates(
        source_state,
        train_features,
        range_scales=args.range_scales,
        quantiles=args.quantiles,
    )
    nmse_limit = float(active_mod.MODEL_LAYOUTS[model_name]["nmse"])

    x_train_t = torch.as_tensor(
        x_train.T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    y_train_t = torch.as_tensor(
        y_train.T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    x_val_t = torch.as_tensor(
        x_val.T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    y_val_t = torch.as_tensor(
        y_val.T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    target_train_c = torch.complex(y_train_t[:, 0], y_train_t[:, 1])
    target_val_c = torch.complex(y_val_t[:, 0], y_val_t[:, 1])
    target_val = y_val[:, 0] + 1j * y_val[:, 1]
    welch = active_mod.TorchWelch(args.nperseg, args.fs_mhz, evaluator.DEVICE)

    source_model = build_wrapper(
        evaluator, ps_mod, source_state, offsets, amp_max
    ).eval()
    source_macs = source_model.mac_report(evaluator, inner, strict=True)
    if source_macs["total"] != int(model_name):
        raise ValueError(
            f"Source wrapper MAC {source_macs['total']} does not match model label {model_name}"
        )
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    base_train = active_mod.cached_base(source_model, x_train_t)
    base_val = active_mod.cached_base(source_model, x_val_t)
    with torch.no_grad():
        source_val_iq = active_mod.forward_from_base(source_model, x_val_t, base_val)
        source_val = (
            source_val_iq[0, 0].cpu().numpy()
            + 1j * source_val_iq[0, 1].cpu().numpy()
        )
    source_metrics = active_mod.numpy_metrics(
        evaluator, metrics_mod, source_val, target_val, args.nperseg
    )
    if source_metrics["nmse_db"] > nmse_limit:
        raise ValueError(
            f"Source checkpoint violates hard NMSE limit {nmse_limit}: {source_metrics}"
        )

    source_range = source_candidate(source_state)
    best_state = clone_state(source_state)
    best_metrics = source_metrics
    best_range = source_range
    best_description = "source checkpoint"
    rows = [
        search_row(
            model_name,
            -1,
            source_range,
            "source_checkpoint",
            None,
            "",
            0.0,
            "",
            0,
            source_metrics,
        )
    ]
    print(
        f"[{model_name}] source grid={grid} offsets={offsets} MAC={source_macs['total']} "
        f"NMSE={source_metrics['nmse_db']:.3f} "
        f"point={source_metrics['point_lower_db']:.3f}/"
        f"{source_metrics['point_upper_db']:.3f}/"
        f"{source_metrics['point_worst_db']:.3f}"
    )

    range_records = []
    for range_index, candidate in enumerate(candidates):
        range_records.append(candidate_public_record(candidate, train_features))
        anchor_state = remap_feature_ranges(
            source_state, candidate["gmin"], candidate["gmax"]
        )
        model = build_wrapper(evaluator, ps_mod, anchor_state, offsets, amp_max).eval()
        candidate_macs = model.mac_report(evaluator, inner, strict=True)
        if candidate_macs["total"] != source_macs["total"]:
            raise RuntimeError(
                f"Range {candidate['label']} changed MAC: "
                f"{source_macs['total']} -> {candidate_macs['total']}"
            )
        if int(model.corr.grid) != grid or tuple(model.corr.offsets) != offsets:
            raise RuntimeError(
                f"Range {candidate['label']} changed grid or offsets unexpectedly"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.corr.tables.requires_grad_(True)

        with torch.no_grad():
            anchor_train_iq = active_mod.forward_from_base(
                model, x_train_t, base_train
            )
            anchor_val_iq = active_mod.forward_from_base(model, x_val_t, base_val)
            anchor_val_c = torch.complex(anchor_val_iq[:, 0], anchor_val_iq[:, 1])
            lower_bin, upper_bin = validation_active_bins(
                active_mod, welch, target_val_c - anchor_val_c
            )
            anchor_train_error = target_train_c - torch.complex(
                anchor_train_iq[:, 0], anchor_train_iq[:, 1]
            )
            anchor_train_power = welch.power(anchor_train_error)
            active_scale = anchor_train_power[[lower_bin, upper_bin]].clamp_min(1e-30)
            anchor_mse = torch.mean((anchor_train_iq - y_train_t).square()).clamp_min(
                1e-30
            )
            anchor_tables = model.corr.tables.detach().clone()
            table_scale = torch.mean(anchor_tables.square()).clamp_min(1e-12)
            anchor_val = (
                anchor_val_iq[0, 0].cpu().numpy()
                + 1j * anchor_val_iq[0, 1].cpu().numpy()
            )
        anchor_metrics = active_mod.numpy_metrics(
            evaluator, metrics_mod, anchor_val, target_val, args.nperseg
        )
        active_mhz = (
            float(welch.freq_mhz[lower_bin]),
            float(welch.freq_mhz[upper_bin]),
        )
        rows.append(
            search_row(
                model_name,
                range_index,
                candidate,
                "physical_interpolation_anchor",
                active_mhz,
                "",
                0.0,
                "",
                0,
                anchor_metrics,
            )
        )
        if feasible_better(active_mod, anchor_metrics, best_metrics, nmse_limit):
            best_state = anchor_state
            best_metrics = anchor_metrics
            best_range = candidate
            best_description = f"interpolated anchor {candidate['label']}"

        max_clip = max(
            max(range_records[-1]["train_fraction_below"]),
            max(range_records[-1]["train_fraction_above"]),
        )
        print(
            f"[{model_name}] range {range_index} {candidate['label']} "
            f"active=({active_mhz[0]:.2f},{active_mhz[1]:.2f}) MHz "
            f"max-train-clip={max_clip:.4f} "
            f"anchor NMSE={anchor_metrics['nmse_db']:.3f} "
            f"point={anchor_metrics['point_worst_db']:.3f}"
        )

        for spectral_weight in args.spectral_weights:
            with torch.no_grad():
                model.corr.tables.copy_(anchor_tables)
            optimizer = torch.optim.LBFGS(
                [model.corr.tables],
                lr=args.lbfgs_lr,
                max_iter=args.lbfgs_iters,
                history_size=args.lbfgs_history,
                tolerance_grad=args.lbfgs_tolerance_grad,
                tolerance_change=args.lbfgs_tolerance_change,
                line_search_fn="strong_wolfe",
            )
            closure_calls = 0

            def closure():
                nonlocal closure_calls
                closure_calls += 1
                optimizer.zero_grad(set_to_none=True)
                pred = active_mod.forward_from_base(model, x_train_t, base_train)
                error_iq = y_train_t - pred
                mse_ratio = torch.mean(error_iq.square()) / anchor_mse
                error_c = torch.complex(error_iq[:, 0], error_iq[:, 1])
                spectra = welch.spectra(error_c)
                active_power = spectra[..., [lower_bin, upper_bin]].abs().square().mean(
                    dim=(0, 1)
                )
                spectral_ratio = torch.mean(active_power / active_scale)
                trust = torch.mean(
                    (model.corr.tables - anchor_tables).square()
                ) / table_scale
                loss = (
                    mse_ratio
                    + float(spectral_weight) * spectral_ratio
                    + args.trust_weight * trust
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            fitted_tables = model.corr.tables.detach().clone()
            if not torch.isfinite(fitted_tables).all():
                raise FloatingPointError(
                    f"Non-finite tables for {candidate['label']}, "
                    f"weight={spectral_weight}"
                )
            with torch.no_grad():
                fitted_train_iq = active_mod.forward_from_base(
                    model, x_train_t, base_train
                )
                fitted_error_iq = y_train_t - fitted_train_iq
                fitted_mse_ratio = torch.mean(fitted_error_iq.square()) / anchor_mse
                fitted_error_c = torch.complex(
                    fitted_error_iq[:, 0], fitted_error_iq[:, 1]
                )
                fitted_spectra = welch.spectra(fitted_error_c)
                fitted_active_power = fitted_spectra[
                    ..., [lower_bin, upper_bin]
                ].abs().square().mean(dim=(0, 1))
                fitted_spectral_ratio = torch.mean(fitted_active_power / active_scale)
                fitted_trust = torch.mean(
                    (model.corr.tables - anchor_tables).square()
                ) / table_scale
                final_loss = float(
                    (
                        fitted_mse_ratio
                        + float(spectral_weight) * fitted_spectral_ratio
                        + args.trust_weight * fitted_trust
                    ).cpu()
                )
                fitted_val_iq = active_mod.forward_from_base(model, x_val_t, base_val)
                fitted_val = (
                    fitted_val_iq[0, 0].cpu().numpy()
                    + 1j * fitted_val_iq[0, 1].cpu().numpy()
                )
            direction_val = fitted_val - anchor_val

            for alpha in np.arange(
                args.alpha_min,
                args.alpha_max + args.alpha_step / 2.0,
                args.alpha_step,
            ):
                metrics = active_mod.numpy_metrics(
                    evaluator,
                    metrics_mod,
                    anchor_val + float(alpha) * direction_val,
                    target_val,
                    args.nperseg,
                )
                rows.append(
                    search_row(
                        model_name,
                        range_index,
                        candidate,
                        "lbfgs_alpha_scan",
                        active_mhz,
                        float(spectral_weight),
                        float(alpha),
                        final_loss,
                        closure_calls,
                        metrics,
                    )
                )
                if feasible_better(active_mod, metrics, best_metrics, nmse_limit):
                    best_state = active_mod.interpolate_table_state(
                        anchor_state, fitted_tables, float(alpha)
                    )
                    best_metrics = metrics
                    best_range = candidate
                    best_description = (
                        f"{candidate['label']} active=({active_mhz[0]:.2f},"
                        f"{active_mhz[1]:.2f}) weight={spectral_weight:g} "
                        f"alpha={alpha:.3f}"
                    )

    output_dir = OUT_ROOT / args.tag / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(best_state, output_dir / "best.pth")

    ranges_sidecar = {
        "model": model_name,
        "source_checkpoint": str(checkpoint),
        "offsets": list(offsets),
        "grid": grid,
        "taps_reduction": taps_reduction,
        "mac": source_macs["total"],
        "feature_semantics": feature_semantics(offsets),
        "source": candidate_public_record(source_range, train_features),
        "candidates": range_records,
        "selected_label": best_range["label"],
        "selected_gmin": best_state["corr.gmin"].detach().cpu().tolist(),
        "selected_gmax": best_state["corr.gmax"].detach().cpu().tolist(),
        "interpolation": (
            "source piecewise-linear tables evaluated at target physical knots; "
            "queries outside source range clamp to endpoint"
        ),
    }
    with (output_dir / "ranges.json").open("w", encoding="utf-8") as handle:
        json.dump(ranges_sidecar, handle, indent=2)

    final_model = build_wrapper(
        evaluator, ps_mod, best_state, offsets, amp_max
    ).eval()
    final_macs = final_model.mac_report(evaluator, inner, strict=True)
    if final_macs["total"] != int(model_name):
        raise RuntimeError(f"Serialized winner MAC mismatch: {final_macs}")
    pred_iq, target_iq, _, _ = evaluator.evaluate_full_sequence(
        final_model, x_val, y_val
    )
    final_metrics = active_mod.numpy_metrics(
        evaluator,
        metrics_mod,
        evaluator.tensor_to_complex(pred_iq),
        evaluator.tensor_to_complex(target_iq),
        args.nperseg,
    )
    if final_metrics["nmse_db"] > nmse_limit:
        raise RuntimeError(f"Serialized winner violates NMSE: {final_metrics}")
    for key in ("nmse_db", "point_lower_db", "point_upper_db", "point_worst_db"):
        if not np.isclose(final_metrics[key], best_metrics[key], atol=1e-9, rtol=0.0):
            raise RuntimeError(
                f"Serialized winner mismatch for {key}: "
                f"selected={best_metrics[key]}, reconstructed={final_metrics[key]}"
            )

    summary = {
        "model": model_name,
        "source_checkpoint": str(checkpoint),
        "selection": best_description,
        "selected_range": best_range["label"],
        "nmse_limit_db": nmse_limit,
        "source_point_worst_db": source_metrics["point_worst_db"],
        "point_gain_db": source_metrics["point_worst_db"]
        - final_metrics["point_worst_db"],
        "offsets": list(offsets),
        "grid": grid,
        "mac": final_macs["total"],
        **final_metrics,
        "checkpoint": str(output_dir / "best.pth"),
        "ranges_sidecar": str(output_dir / "ranges.json"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def smoke_model(args, model_name, evaluator, ps_mod, active_mod, x_train, y_train, amp_max):
    """Check physical remapping, wrapper reconstruction, MAC, and gradients."""
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(args.checkpoint_root) / model_name / "best.pth"
    )
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    offsets, _ = active_mod.resolve_checkpoint_offsets(
        checkpoint,
        model_name,
        args.offsets,
        require_metadata=args.checkpoint is not None,
    )
    offsets = tuple(offsets)
    state = evaluator.load_state_dict(checkpoint)
    validate_feature_count(state, offsets)
    inner, _, grid = infer_wrapper_layout(state)
    train_features = phase_features_numpy(x_train, offsets)
    candidate = make_range_candidates(
        state,
        train_features,
        range_scales=args.range_scales,
        quantiles=args.quantiles,
    )[0]
    remapped = remap_feature_ranges(state, candidate["gmin"], candidate["gmax"])

    unit = torch.linspace(0.0, 1.0, grid + 1, dtype=torch.float64)
    knots = candidate["gmin"][:, None] + (
        candidate["gmax"] - candidate["gmin"]
    )[:, None] * unit[None, :]
    expected = table_values_at_physical_coordinates(
        state["corr.tables"].detach().cpu().double(),
        state["corr.gmin"].detach().cpu().double(),
        state["corr.gmax"].detach().cpu().double(),
        knots,
    )
    torch.testing.assert_close(
        remapped["corr.tables"], expected, rtol=0.0, atol=2e-15
    )

    model = build_wrapper(evaluator, ps_mod, remapped, offsets, amp_max).eval()
    macs = model.mac_report(evaluator, inner, strict=True)
    if macs["total"] != int(model_name):
        raise RuntimeError(f"Smoke MAC mismatch: {macs['total']} != {model_name}")
    if int(model.corr.grid) != grid or tuple(model.corr.offsets) != offsets:
        raise RuntimeError("Smoke changed grid or offsets")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.corr.tables.requires_grad_(True)

    length = min(len(x_train), max(2 * args.nperseg, args.nperseg + 1))
    if length < args.nperseg:
        raise ValueError("Dataset is shorter than nperseg")
    x = torch.as_tensor(
        x_train[:length].T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    y = torch.as_tensor(
        y_train[:length].T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    base = active_mod.cached_base(model, x)
    pred = active_mod.forward_from_base(model, x, base)
    error_iq = y - pred
    error_c = torch.complex(error_iq[:, 0], error_iq[:, 1])
    welch = active_mod.TorchWelch(args.nperseg, args.fs_mhz, evaluator.DEVICE)
    lower, upper = validation_active_bins(active_mod, welch, error_c.detach())
    spectra = welch.spectra(error_c)
    loss = torch.mean(error_iq.square()) + 1e-12 * spectra[
        ..., [lower, upper]
    ].abs().square().mean()
    loss.backward()
    gradient = model.corr.tables.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("Smoke backward produced no finite corr.tables gradient")
    frozen_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if name != "corr.tables" and parameter.grad is not None
    ]
    if frozen_with_grad:
        raise RuntimeError(f"Frozen parameters unexpectedly received gradients: {frozen_with_grad}")
    print(
        "smoke-ok "
        f"model={model_name} candidate={candidate['label']} grid={grid} "
        f"offsets={offsets} MAC={macs['total']} "
        f"active=({welch.freq_mhz[lower]:.2f},{welch.freq_mhz[upper]:.2f})MHz "
        f"loss={float(loss.detach().cpu()):.6e}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_OFFSETS), default=list(MODEL_OFFSETS)
    )
    parser.add_argument(
        "--checkpoint-root",
        default="eval_outputs_direct/aclr_ref_convention/fir_pointsweep",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="single-model checkpoint override (requires exactly one --models entry)",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs="+",
        default=None,
        help="true ordered offsets for --checkpoint; otherwise read a sidecar",
    )
    parser.add_argument("--tag", default="feature_range_search")
    ranges = parser.add_mutually_exclusive_group()
    ranges.add_argument(
        "--range-scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "candidate multipliers applied to source gmin/gmax about physical zero; "
            "default: 0.5 0.65 0.8 1.0"
        ),
    )
    ranges.add_argument(
        "--quantiles",
        type=parse_quantile_spec,
        nargs="+",
        default=None,
        metavar="Q",
        help=(
            "per-feature train-quantile candidates; q means [q,1-q], while "
            "low:high gives an explicit pair (example: 0.001 0.005:0.995)"
        ),
    )
    parser.add_argument(
        "--spectral-weights", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0]
    )
    parser.add_argument("--trust-weight", type=float, default=1e-4)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-iters", type=int, default=30)
    parser.add_argument("--lbfgs-history", type=int, default=20)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="check first range candidate on a short sequence and write no outputs",
    )
    args = parser.parse_args()
    if args.checkpoint is not None and len(args.models) != 1:
        parser.error("--checkpoint requires exactly one --models entry")
    if args.offsets is not None and len(args.models) != 1:
        parser.error("--offsets requires exactly one --models entry")
    if args.range_scales is None and args.quantiles is None:
        args.range_scales = [0.5, 0.65, 0.8, 1.0]
    if args.range_scales is not None and any(
        not math.isfinite(scale) or scale <= 0.0 for scale in args.range_scales
    ):
        parser.error("--range-scales must contain finite positive values")
    if any(weight < 0.0 or not math.isfinite(weight) for weight in args.spectral_weights):
        parser.error("--spectral-weights must contain finite non-negative values")
    if args.trust_weight < 0.0 or not math.isfinite(args.trust_weight):
        parser.error("--trust-weight must be finite and non-negative")
    if args.lbfgs_iters < 1 or args.lbfgs_history < 1:
        parser.error("--lbfgs-iters and --lbfgs-history must be positive")
    if args.alpha_step <= 0.0:
        parser.error("--alpha-step must be positive")
    if args.alpha_max < args.alpha_min:
        parser.error("--alpha-max must be >= --alpha-min")
    if args.nperseg < 2:
        parser.error("--nperseg must be at least 2")
    if not 0.0 < args.split < 1.0:
        parser.error("--split must be between zero and one")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    active_mod = load_module(
        "aclr_active_bin_table_optimization_range_search", ACTIVE_BIN_SCRIPT
    )
    evaluator = load_module(
        "final_evaluate_unified_range_search", ROOT / "steps" / "final_evaluate_unified.py"
    )
    metrics_mod = load_module(
        "aclr_ref_metrics_range_search", ROOT / "steps" / "aclr_ref_metrics.py"
    )
    if not np.isclose(args.fs_mhz * 1e6, metrics_mod.FS_HZ, rtol=0.0, atol=1e-6):
        raise SystemExit(
            f"--fs-mhz={args.fs_mhz} conflicts with the fixed evaluator rate "
            f"{metrics_mod.FS_HZ / 1e6} MHz"
        )
    ps_mod = load_module(
        "aclr_phase_spline_experiment_range_search",
        ROOT / "steps" / "aclr_phase_spline_experiment.py",
    )
    features, targets, _, *_ = evaluator.load_iq_arrays(
        ROOT / "data" / "DPD_data.mat", "rms"
    )
    split_index = int(len(features) * args.split)
    x_train, y_train = features[:split_index], targets[:split_index]
    x_val, y_val = features[split_index:], targets[split_index:]
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))

    if args.smoke:
        for model_name in args.models:
            smoke_model(
                args,
                model_name,
                evaluator,
                ps_mod,
                active_mod,
                x_train,
                y_train,
                amp_max,
            )
        return

    arrays = (x_train, y_train, x_val, y_val, amp_max)
    config_dir = OUT_ROOT / args.tag
    config_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    summaries = [
        run_model(args, name, evaluator, metrics_mod, ps_mod, active_mod, arrays)
        for name in args.models
    ]
    with (config_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
