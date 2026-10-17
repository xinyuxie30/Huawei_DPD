"""Multi-resolution active-bin optimization for one phase-spline wrapper.

The loaded EA-TCN base and both complex FIRs stay frozen; only
``corr.tables`` is fitted.  At every fitting Welch resolution, lower/upper
active bins come from the validation residual of the current incumbent.  The
table coefficients themselves are fitted exclusively on the complete train
split with normalized MSE, normalized active-bin power, and table trust.

The one-dimensional fitted-table direction is scanned on validation.  The
winner is selected only by official 4096-point ACLR under the model's hard
validation-NMSE limit.  This script reuses the differentiable Welch, cached
forward, metrics, interpolation, and bin-selection helpers from
``aclr_active_bin_table_optimization.py``.

Exactly one raw wrapper checkpoint is accepted.  Offsets must be explicit
because their semantics cannot be recovered from tensor shapes; spline grid
and FIR-tap reduction are inferred from checkpoint state.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"
ACTIVE_BIN_SCRIPT = ROOT / "steps" / "aclr_active_bin_table_optimization.py"
OFFICIAL_NPERSEG = 4096
NMSE_LIMITS_DB = {"2284": -40.0, "998": -37.0, "796": -35.0, "786": -35.0}
MAC_LIMITS = {"2284": 2399, "998": 998, "786": 800, "796": 796}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clone_state(state):
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def normalize_offsets(offsets):
    values = tuple(int(value) for value in offsets)
    if not values:
        raise ValueError("--offsets must not be empty")
    if values.count(0) != 1:
        raise ValueError(f"--offsets must contain zero exactly once: {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"--offsets contains duplicates: {values}")
    return values


def infer_wrapper_layout(state, offsets):
    """Validate a raw wrapper state and infer its grid/FIR-tap trade."""
    required = {
        "corr.tables", "corr.gmin", "corr.gmax", "fir_real", "fir_imag",
        "base.fir_real", "base.fir_imag",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Wrapper checkpoint is missing keys: {missing}")
    inner = {
        key[len("base.") :]: value
        for key, value in state.items()
        if key.startswith("base.")
    }
    if not inner:
        raise ValueError("Expected a raw phase-spline wrapper with base.* keys")

    tables = state["corr.tables"]
    expected_features = 1 + 2 * (len(offsets) - 1)
    if tables.ndim != 3 or int(tables.shape[1]) != 2:
        raise ValueError(
            "corr.tables must have shape (features, 2, grid + 1); "
            f"got {tuple(tables.shape)}"
        )
    if int(tables.shape[0]) != expected_features:
        raise ValueError(
            "Explicit offsets do not match corr.tables: "
            f"offsets imply {expected_features} rows, state has {tables.shape[0]}"
        )
    if tuple(state["corr.gmin"].shape) != (expected_features,):
        raise ValueError("corr.gmin shape does not match corr.tables")
    if tuple(state["corr.gmax"].shape) != (expected_features,):
        raise ValueError("corr.gmax shape does not match corr.tables")
    if state["fir_real"].shape != state["fir_imag"].shape:
        raise ValueError("Wrapper real/imag FIR shapes differ")
    if inner["fir_real"].shape != inner["fir_imag"].shape:
        raise ValueError("Base real/imag FIR shapes differ")

    base_fir_taps = int(inner["fir_real"].numel())
    output_fir_taps = int(state["fir_real"].numel())
    taps_reduction = base_fir_taps - output_fir_taps
    if taps_reduction < 0:
        raise ValueError(
            f"Wrapper FIR ({output_fir_taps}) is longer than base FIR ({base_fir_taps})"
        )
    if output_fir_taps < 5 or output_fir_taps % 2 == 0:
        raise ValueError(f"Invalid inferred wrapper FIR length: {output_fir_taps}")
    grid = int(tables.shape[-1] - 1)
    if grid < 1:
        raise ValueError(f"Invalid inferred spline grid: {grid}")
    return {
        "inner_state": inner,
        "grid": grid,
        "base_fir_taps": base_fir_taps,
        "output_fir_taps": output_fir_taps,
        "taps_reduction": taps_reduction,
        "table_features": expected_features,
    }


def build_wrapper(evaluator, ps_mod, state, offsets, amp_max):
    layout = infer_wrapper_layout(state, offsets)
    model = ps_mod.PhaseSplineDPD(
        evaluator, layout["inner_state"], layout["taps_reduction"],
        list(offsets), layout["grid"], amp_max,
    )
    model.load_state_dict(state, strict=True)
    return model.to(evaluator.DEVICE), layout


def configure_trainables(model, optimize_ranges=False):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.corr.tables.requires_grad_(True)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if trainable != ["corr.tables"]:
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")
    model.corr.gmin.requires_grad_(bool(optimize_ranges))
    model.corr.gmax.requires_grad_(bool(optimize_ranges))
    tensors = [model.corr.tables]
    if optimize_ranges:
        tensors.extend([model.corr.gmin, model.corr.gmax])
    return tensors


def iq_tensor(values, device):
    return torch.as_tensor(values.T, dtype=torch.float64, device=device).unsqueeze(0)


def iq_to_numpy_complex(values):
    return (
        values[0, 0].detach().cpu().numpy()
        + 1j * values[0, 1].detach().cpu().numpy()
    )


def extended_numpy_metrics(active_mod, evaluator, metrics_mod, pred, target, nperseg):
    """Point metrics plus lower/upper/worst reference integrated ACLR."""
    result = active_mod.numpy_metrics(evaluator, metrics_mod, pred, target, nperseg)
    integrated = metrics_mod.ref_metrics(evaluator, pred, target, nperseg)
    result["integrated_aclr_db"] = float(integrated["aclr_db"])
    result["integrated_lower_db"] = float(integrated["aclr_lower_db"])
    result["integrated_upper_db"] = float(integrated["aclr_upper_db"])
    return result


def evaluation_resolutions(fit_npersegs):
    return tuple(dict.fromkeys([*(int(value) for value in fit_npersegs), OFFICIAL_NPERSEG]))


def metrics_at_resolutions(active_mod, evaluator, metrics_mod, pred, target, npersegs):
    return {
        str(int(nperseg)): extended_numpy_metrics(
            active_mod, evaluator, metrics_mod, pred, target, int(nperseg)
        )
        for nperseg in npersegs
    }


def flatten_metrics(metrics_by_nperseg):
    flattened = {}
    for nperseg, metrics in metrics_by_nperseg.items():
        prefix = f"n{nperseg}_"
        flattened.update({prefix + key: value for key, value in metrics.items()})
    return flattened


def official_metrics(metrics_by_nperseg):
    return metrics_by_nperseg[str(OFFICIAL_NPERSEG)]


def select_active_bins(active_mod, welch, validation_error, bins_per_side, separation):
    """Select low-power adjacent bins exclusively from validation residual."""
    with torch.no_grad():
        power = welch.power(validation_error).detach().cpu().numpy()
    lower = active_mod.separated_low_bins(
        power, welch.freq_mhz, (-180.0, -120.0),
        count=bins_per_side, separation=separation,
    )
    upper = active_mod.separated_low_bins(
        power, welch.freq_mhz, (120.0, 180.0),
        count=bins_per_side, separation=separation,
    )
    return tuple(int(value) for value in [*lower, *upper]), power


def active_bin_record(welch, indices, validation_power, train_scale, bins_per_side):
    rows = []
    for position, index in enumerate(indices):
        rows.append(
            {
                "side": "lower" if position < bins_per_side else "upper",
                "index": int(index),
                "frequency_mhz": float(welch.freq_mhz[index]),
                "validation_incumbent_power": float(validation_power[index]),
                "train_incumbent_scale_power": float(train_scale[position].detach().cpu()),
            }
        )
    return rows


def choose_bins_and_scales(
    active_mod,
    welch_by_nperseg,
    train_error,
    validation_error,
    bins_per_side,
    separation,
):
    selections = {}
    for nperseg, welch in welch_by_nperseg.items():
        indices, validation_power = select_active_bins(
            active_mod, welch, validation_error, bins_per_side, separation
        )
        with torch.no_grad():
            train_power = welch.power(train_error)
            scale = train_power[list(indices)].detach().clamp_min(1e-30)
        selections[nperseg] = {
            "indices": indices,
            "scale": scale,
            "record": active_bin_record(
                welch, indices, validation_power, scale, bins_per_side
            ),
        }
    return selections


def choose_band_scales(welch_by_nperseg, train_error):
    """Use complete adjacent bands; no validation-selected frequency hyperparameter."""
    selections = {}
    for nperseg, welch in welch_by_nperseg.items():
        lower = np.flatnonzero(
            (welch.freq_mhz >= -180.0) & (welch.freq_mhz <= -100.0)
        )
        upper = np.flatnonzero(
            (welch.freq_mhz >= 100.0) & (welch.freq_mhz <= 180.0)
        )
        with torch.no_grad():
            power = welch.power(train_error)
            scale = torch.stack(
                [power[lower].mean(), power[upper].mean()]
            ).clamp_min(1e-30)
        selections[nperseg] = {
            "lower_indices": tuple(int(value) for value in lower),
            "upper_indices": tuple(int(value) for value in upper),
            "scale": scale,
            "record": [
                {
                    "side": "lower",
                    "band_mhz": [-180.0, -100.0],
                    "bins": int(len(lower)),
                    "train_incumbent_scale_power": float(scale[0].cpu()),
                },
                {
                    "side": "upper",
                    "band_mhz": [100.0, 180.0],
                    "bins": int(len(upper)),
                    "train_incumbent_scale_power": float(scale[1].cpu()),
                },
            ],
        }
    return selections


def active_bins_json(selections):
    return json.dumps(
        {str(nperseg): item["record"] for nperseg, item in selections.items()},
        separators=(",", ":"),
    )


def objective_components(
    active_mod,
    model,
    x_train_t,
    y_train_t,
    base_train,
    welch_by_nperseg,
    selections,
    anchor_mse,
    anchor_tables,
    table_scale,
    spectral_mode,
    anchor_gmin,
    anchor_gmax,
    range_trust_weight,
    min_range_width_ratio,
):
    pred = active_mod.forward_from_base(model, x_train_t, base_train)
    error_iq = y_train_t - pred
    mse_ratio = torch.mean(error_iq.square()) / anchor_mse
    error_complex = torch.complex(error_iq[:, 0], error_iq[:, 1])
    resolution_ratios = []
    for nperseg, welch in welch_by_nperseg.items():
        spectra = welch.spectra(error_complex)
        selection = selections[nperseg]
        if spectral_mode == "active":
            power = (
                spectra[..., list(selection["indices"])]
                .abs().square().mean(dim=(0, 1))
            )
            resolution_ratios.append(torch.mean(power / selection["scale"]))
        elif spectral_mode == "band":
            power = spectra.abs().square().mean(dim=(0, 1))
            side_power = torch.stack(
                [
                    power[list(selection["lower_indices"])].mean(),
                    power[list(selection["upper_indices"])].mean(),
                ]
            )
            resolution_ratios.append(torch.mean(side_power / selection["scale"]))
        else:
            raise ValueError(spectral_mode)
    spectral_ratio = torch.stack(resolution_ratios).mean()
    table_trust = torch.mean((model.corr.tables - anchor_tables).square()) / table_scale
    anchor_width = (anchor_gmax - anchor_gmin).clamp_min(1e-12)
    range_trust = 0.5 * (
        torch.mean(((model.corr.gmin - anchor_gmin) / anchor_width).square())
        + torch.mean(((model.corr.gmax - anchor_gmax) / anchor_width).square())
    )
    width = model.corr.gmax - model.corr.gmin
    width_barrier = torch.mean(
        torch.relu(float(min_range_width_ratio) * anchor_width - width).square()
        / anchor_width.square()
    )
    trust_ratio = (
        table_trust
        + float(range_trust_weight) * range_trust
        + 100.0 * width_barrier
    )
    return mse_ratio, spectral_ratio, trust_ratio


def selection_key(metrics_by_nperseg, nmse_limit, objective):
    official = official_metrics(metrics_by_nperseg)
    violation = max(0.0, float(official["nmse_db"]) - float(nmse_limit))
    robust_point = max(
        float(metrics["point_worst_db"])
        for metrics in metrics_by_nperseg.values()
    )
    if objective == "point":
        score = float(official["point_worst_db"])
    elif objective == "stable":
        score = robust_point
    elif objective == "integrated":
        score = float(official["integrated_aclr_db"])
    else:
        raise ValueError(objective)
    return (
        violation > 0.0,
        violation,
        score,
        robust_point,
        float(official["integrated_aclr_db"]),
        float(official["nmse_db"]),
    )


def passes_point_guard(candidate, source, max_regression_db):
    for nperseg, source_metrics in source.items():
        candidate_point = float(candidate[nperseg]["point_worst_db"])
        source_point = float(source_metrics["point_worst_db"])
        if candidate_point > source_point + float(max_regression_db) + 1e-12:
            return False
    return True


def candidate_better(
    candidate,
    incumbent,
    source,
    nmse_limit,
    objective,
    max_point_regression_db,
):
    official = official_metrics(candidate)
    if float(official["nmse_db"]) > float(nmse_limit):
        return False
    if not passes_point_guard(candidate, source, max_point_regression_db):
        return False
    return selection_key(candidate, nmse_limit, objective) < selection_key(
        incumbent, nmse_limit, objective
    )


def make_row(
    candidate_id,
    round_index,
    source,
    spectral_weight,
    alpha,
    loss,
    mse_ratio,
    spectral_ratio,
    trust_ratio,
    closure_calls,
    selections,
    metrics_by_nperseg,
    nmse_limit,
):
    official = official_metrics(metrics_by_nperseg)
    return {
        "candidate_id": candidate_id,
        "selected": False,
        "round": round_index,
        "source": source,
        "spectral_weight": spectral_weight,
        "alpha": alpha,
        "lbfgs_loss": loss,
        "train_mse_ratio": mse_ratio,
        "train_multires_active_ratio": spectral_ratio,
        "table_trust_ratio": trust_ratio,
        "closure_calls": closure_calls,
        "active_bins_from": "validation_incumbent",
        "coefficients_fitted_on": "complete_training_split_only",
        "active_bins_json": active_bins_json(selections),
        "official_nperseg": OFFICIAL_NPERSEG,
        "official_feasible": official["nmse_db"] <= nmse_limit,
        **flatten_metrics(metrics_by_nperseg),
    }


def alpha_grid(alpha_min, alpha_max, alpha_step):
    return np.arange(alpha_min, alpha_max + alpha_step / 2.0, alpha_step)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smoothed_psd(psd_db, sigma_bins=4.0):
    power = np.power(10.0, np.asarray(psd_db, dtype=np.float64) / 10.0)
    smooth = gaussian_filter1d(power, float(sigma_bins), mode="nearest")
    return 10.0 * np.log10(np.maximum(smooth, 1e-300))


def plot_psd_comparison(
    evaluator,
    metrics_mod,
    source_pred,
    final_pred,
    target,
    dataset_error,
    source_metrics,
    final_metrics,
    output,
):
    freq_hz, target_psd = evaluator.psd_db(target, metrics_mod.FS_HZ, OFFICIAL_NPERSEG)
    _, reference_psd = evaluator.psd_db(dataset_error, metrics_mod.FS_HZ, OFFICIAL_NPERSEG)
    _, source_psd = evaluator.psd_db(target - source_pred, metrics_mod.FS_HZ, OFFICIAL_NPERSEG)
    _, final_psd = evaluator.psd_db(target - final_pred, metrics_mod.FS_HZ, OFFICIAL_NPERSEG)
    freq_mhz = freq_hz / 1e6
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    traces = (
        (
            "dataset reference error",
            reference_psd,
            "black",
            1.5,
        ),
        (
            "source residual "
            f"(NMSE {source_metrics['nmse_db']:.3f}, "
            f"ACLR {source_metrics['integrated_aclr_db']:.3f})",
            source_psd,
            "tab:orange",
            1.15,
        ),
        (
            "optimized residual "
            f"(NMSE {final_metrics['nmse_db']:.3f}, "
            f"ACLR {final_metrics['integrated_aclr_db']:.3f})",
            final_psd,
            "tab:blue",
            1.25,
        ),
    )
    for axis, smooth in zip(axes, (False, True)):
        for label, values, color, width in traces:
            curve = smoothed_psd(values) if smooth else values
            axis.plot(freq_mhz, curve, label=label, color=color, linewidth=width)
        target_curve = smoothed_psd(target_psd) if smooth else target_psd
        axis.plot(
            freq_mhz,
            target_curve,
            label="target y",
            color="0.65",
            linestyle="--",
            linewidth=0.9,
        )
        for lo, hi in ((-180.0, -100.0), (100.0, 180.0)):
            axis.axvspan(lo, hi, color="tab:red", alpha=0.055)
        axis.axvline(-100.0, color="0.45", linestyle=":", linewidth=0.8)
        axis.axvline(100.0, color="0.45", linestyle=":", linewidth=0.8)
        axis.set_xlim(-245.76, 245.76)
        axis.set_ylabel("PSD (dB)")
        axis.set_title("Smoothed linear-power PSD" if smooth else "Raw Welch PSD (nperseg=4096)")
        axis.grid(True, linestyle=":", alpha=0.45)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Frequency (MHz)")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_nmse_convergence(closure_rows, candidate_rows, nmse_limit, output):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    if closure_rows:
        axes[0].plot(
            [row["global_closure"] for row in closure_rows],
            [row["train_nmse_db"] for row in closure_rows],
            color="tab:blue",
            linewidth=1.0,
        )
    axes[0].axhline(nmse_limit, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_title("Training NMSE convergence (L-BFGS closure calls)")
    axes[0].set_xlabel("Closure call")
    axes[0].set_ylabel("Train NMSE (dB)")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    numeric = [row for row in candidate_rows if row.get("candidate_id", "").startswith("candidate_")]
    if numeric:
        y = [float(row[f"n{OFFICIAL_NPERSEG}_nmse_db"]) for row in numeric]
        axes[1].plot(range(1, len(numeric) + 1), y, color="tab:green", linewidth=0.9)
        selected = [index for index, row in enumerate(numeric, start=1) if row.get("selected")]
        if selected:
            axes[1].scatter(
                selected,
                [y[index - 1] for index in selected],
                color="tab:red",
                marker="*",
                s=100,
                label="selected",
                zorder=4,
            )
            axes[1].legend(loc="best")
    axes[1].axhline(nmse_limit, color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Validation NMSE along candidate search")
    axes[1].set_xlabel("Candidate")
    axes[1].set_ylabel("Validation NMSE (dB)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def run_search(args, evaluator, metrics_mod, ps_mod, active_mod, arrays):
    x_train, y_train, x_val, y_val, dataset_error_val, amp_max = arrays
    offsets = normalize_offsets(args.offsets)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    source_state = clone_state(evaluator.load_state_dict(checkpoint))
    source_model, inferred = build_wrapper(
        evaluator, ps_mod, source_state, offsets, amp_max
    )
    source_macs = source_model.mac_report(
        evaluator, inferred["inner_state"], strict=True
    )
    mac_limit = int(args.mac_limit) if args.mac_limit is not None else MAC_LIMITS[args.model]
    if source_macs["total"] > mac_limit:
        raise ValueError(
            f"Wrapper MAC {source_macs['total']} exceeds limit {mac_limit} for "
            f"--model {args.model}; check --offsets and checkpoint"
        )
    nmse_limit = (
        float(args.nmse_limit)
        if args.nmse_limit is not None
        else NMSE_LIMITS_DB[args.model]
    )
    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_npersegs = tuple(int(value) for value in args.fit_npersegs)
    eval_npersegs = evaluation_resolutions(fit_npersegs)
    if len(x_train) < max(fit_npersegs) or len(x_val) < max(eval_npersegs):
        raise ValueError(
            "Train/validation split is shorter than a requested Welch resolution: "
            f"train={len(x_train)}, val={len(x_val)}, requested={eval_npersegs}"
        )

    x_train_t = iq_tensor(x_train, evaluator.DEVICE)
    y_train_t = iq_tensor(y_train, evaluator.DEVICE)
    x_val_t = iq_tensor(x_val, evaluator.DEVICE)
    y_val_t = iq_tensor(y_val, evaluator.DEVICE)
    target_val = y_val[:, 0] + 1j * y_val[:, 1]
    target_train_c = torch.complex(y_train_t[:, 0], y_train_t[:, 1])
    target_val_c = torch.complex(y_val_t[:, 0], y_val_t[:, 1])
    welch_by_nperseg = {
        nperseg: active_mod.TorchWelch(nperseg, args.fs_mhz, evaluator.DEVICE)
        for nperseg in fit_npersegs
    }

    configure_trainables(source_model, optimize_ranges=False)
    source_model.eval()
    base_train = active_mod.cached_base(source_model, x_train_t)
    base_val = active_mod.cached_base(source_model, x_val_t)
    with torch.no_grad():
        source_val_iq = active_mod.forward_from_base(source_model, x_val_t, base_val)
        source_val = iq_to_numpy_complex(source_val_iq)
    source_metrics_by_resolution = metrics_at_resolutions(
        active_mod, evaluator, metrics_mod, source_val, target_val, eval_npersegs
    )
    source_official = official_metrics(source_metrics_by_resolution)
    if source_official["nmse_db"] > nmse_limit:
        raise ValueError(
            f"Source violates hard NMSE <= {nmse_limit:.3f} dB: "
            f"{source_official['nmse_db']:.6f} dB"
        )

    current_state = source_state
    current_metrics_by_resolution = source_metrics_by_resolution
    current_official = source_official
    selected_candidate_id = "source"
    selected_description = "source checkpoint"
    selected_active_bins = None
    rows = []
    closure_rows = []
    global_closure = 0
    active_bin_history = []
    candidate_number = 0

    print(
        f"[{args.model}] source MAC={source_macs['total']}/{mac_limit} grid={inferred['grid']} "
        f"FIR={inferred['base_fir_taps']}->{inferred['output_fir_taps']} "
        f"NMSE={source_official['nmse_db']:.3f} point="
        f"{source_official['point_lower_db']:.3f}/"
        f"{source_official['point_upper_db']:.3f}/"
        f"{source_official['point_worst_db']:.3f}"
    )

    for round_index in range(1, args.rounds + 1):
        model, _ = build_wrapper(evaluator, ps_mod, current_state, offsets, amp_max)
        trainable_tensors = configure_trainables(model, args.optimize_ranges)
        model.eval()
        with torch.no_grad():
            anchor_train_iq = active_mod.forward_from_base(model, x_train_t, base_train)
            anchor_val_iq = active_mod.forward_from_base(model, x_val_t, base_val)
            anchor_train_error = target_train_c - torch.complex(
                anchor_train_iq[:, 0], anchor_train_iq[:, 1]
            )
            anchor_val_error = target_val_c - torch.complex(
                anchor_val_iq[:, 0], anchor_val_iq[:, 1]
            )
            anchor_mse = torch.mean(
                (anchor_train_iq - y_train_t).square()
            ).clamp_min(1e-30)
            anchor_train_nmse = anchor_mse / torch.mean(y_train_t.square()).clamp_min(1e-30)
            anchor_tables = model.corr.tables.detach().clone()
            anchor_gmin = model.corr.gmin.detach().clone()
            anchor_gmax = model.corr.gmax.detach().clone()
            table_scale = torch.mean(anchor_tables.square()).clamp_min(1e-12)
            anchor_val = iq_to_numpy_complex(anchor_val_iq)

        if args.spectral_mode == "active":
            selections = choose_bins_and_scales(
                active_mod, welch_by_nperseg, anchor_train_error, anchor_val_error,
                args.bins_per_side, args.bin_separation,
            )
            selection_origin = "validation incumbent residual"
        else:
            selections = choose_band_scales(
                welch_by_nperseg, anchor_train_error
            )
            selection_origin = "fixed complete adjacent bands"
        active_records = {
            str(nperseg): selection["record"]
            for nperseg, selection in selections.items()
        }
        active_bin_history.append(
            {
                "round": round_index,
                "selection_sequence": selection_origin,
                "coefficient_fit_sequence": "complete training split only",
                "bins": active_records,
            }
        )
        anchor_metrics_by_resolution = metrics_at_resolutions(
            active_mod, evaluator, metrics_mod, anchor_val, target_val, eval_npersegs
        )
        anchor_official = official_metrics(anchor_metrics_by_resolution)
        rows.append(
            make_row(
                f"round{round_index}_anchor", round_index, "incumbent_anchor",
                "", 0.0, "", 1.0, 1.0, 0.0, 0, selections,
                anchor_metrics_by_resolution, nmse_limit,
            )
        )
        if args.spectral_mode == "active":
            bins_text = "; ".join(
                f"n{n}:" + ",".join(
                    f"{row['frequency_mhz']:.2f}" for row in item["record"]
                )
                for n, item in selections.items()
            )
        else:
            bins_text = "; ".join(
                f"n{n}:full-adjacent({sum(row['bins'] for row in item['record'])} bins)"
                for n, item in selections.items()
            )
        print(
            f"[{args.model}] round {round_index} validation-active [{bins_text}] MHz | "
            f"anchor NMSE={anchor_official['nmse_db']:.3f} "
            f"point={anchor_official['point_worst_db']:.3f}"
        )

        round_best_state = current_state
        round_best_metrics_by_resolution = anchor_metrics_by_resolution
        round_best_official = anchor_official
        round_best_candidate_id = selected_candidate_id
        round_best_description = selected_description
        round_best_active_bins = selected_active_bins

        for spectral_weight in args.spectral_weights:
            with torch.no_grad():
                model.corr.tables.copy_(anchor_tables)
                model.corr.gmin.copy_(anchor_gmin)
                model.corr.gmax.copy_(anchor_gmax)
            optimizer = torch.optim.LBFGS(
                trainable_tensors,
                lr=args.lbfgs_lr,
                max_iter=args.lbfgs_iters,
                history_size=args.lbfgs_history,
                tolerance_grad=args.lbfgs_tolerance_grad,
                tolerance_change=args.lbfgs_tolerance_change,
                line_search_fn="strong_wolfe",
            )
            closure_calls = 0

            def closure():
                nonlocal closure_calls, global_closure
                closure_calls += 1
                global_closure += 1
                optimizer.zero_grad(set_to_none=True)
                mse_ratio, spectral_ratio, trust_ratio = objective_components(
                    active_mod, model, x_train_t, y_train_t, base_train,
                    welch_by_nperseg, selections, anchor_mse, anchor_tables, table_scale,
                    args.spectral_mode,
                    anchor_gmin,
                    anchor_gmax,
                    args.range_trust_weight if args.optimize_ranges else 0.0,
                    args.min_range_width_ratio,
                )
                loss = (
                    mse_ratio
                    + float(spectral_weight) * spectral_ratio
                    + args.trust_weight * trust_ratio
                )
                closure_rows.append(
                    {
                        "global_closure": global_closure,
                        "round": round_index,
                        "spectral_weight": float(spectral_weight),
                        "closure_in_trial": closure_calls,
                        "train_nmse_db": float(
                            10.0
                            * torch.log10(
                                (mse_ratio.detach() * anchor_train_nmse).clamp_min(1e-30)
                            ).cpu()
                        ),
                        "mse_ratio": float(mse_ratio.detach().cpu()),
                        "spectral_ratio": float(spectral_ratio.detach().cpu()),
                        "trust_ratio": float(trust_ratio.detach().cpu()),
                        "loss": float(loss.detach().cpu()),
                    }
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            fitted_tables = model.corr.tables.detach().clone()
            fitted_gmin = model.corr.gmin.detach().clone()
            fitted_gmax = model.corr.gmax.detach().clone()
            if not all(
                torch.isfinite(value).all()
                for value in (fitted_tables, fitted_gmin, fitted_gmax)
            ):
                raise FloatingPointError(
                    f"Non-finite fitted table/ranges for weight={spectral_weight}"
                )
            with torch.no_grad():
                fitted_mse, fitted_spectral, fitted_trust = objective_components(
                    active_mod, model, x_train_t, y_train_t, base_train,
                    welch_by_nperseg, selections, anchor_mse, anchor_tables, table_scale,
                    args.spectral_mode,
                    anchor_gmin,
                    anchor_gmax,
                    args.range_trust_weight if args.optimize_ranges else 0.0,
                    args.min_range_width_ratio,
                )
                final_loss = (
                    fitted_mse
                    + float(spectral_weight) * fitted_spectral
                    + args.trust_weight * fitted_trust
                )
                fitted_val_iq = active_mod.forward_from_base(model, x_val_t, base_val)
                fitted_val = iq_to_numpy_complex(fitted_val_iq)
            direction_val = fitted_val - anchor_val

            for alpha in alpha_grid(args.alpha_min, args.alpha_max, args.alpha_step):
                candidate_number += 1
                candidate_id = f"candidate_{candidate_number:06d}"
                if args.optimize_ranges:
                    with torch.no_grad():
                        model.corr.tables.copy_(
                            torch.lerp(anchor_tables, fitted_tables, float(alpha))
                        )
                        model.corr.gmin.copy_(
                            torch.lerp(anchor_gmin, fitted_gmin, float(alpha))
                        )
                        model.corr.gmax.copy_(
                            torch.lerp(anchor_gmax, fitted_gmax, float(alpha))
                        )
                        candidate_width = model.corr.gmax - model.corr.gmin
                        minimum_width = args.min_range_width_ratio * (
                            anchor_gmax - anchor_gmin
                        )
                        ranges_valid = bool(torch.all(candidate_width >= minimum_width).cpu())
                        candidate_val_iq = active_mod.forward_from_base(
                            model, x_val_t, base_val
                        )
                        candidate_val = iq_to_numpy_complex(candidate_val_iq)
                else:
                    ranges_valid = True
                    candidate_val = anchor_val + float(alpha) * direction_val
                metrics_by_resolution = metrics_at_resolutions(
                    active_mod,
                    evaluator,
                    metrics_mod,
                    candidate_val,
                    target_val,
                    eval_npersegs,
                )
                candidate_official = official_metrics(metrics_by_resolution)
                rows.append(
                    make_row(
                        candidate_id,
                        round_index,
                        "lbfgs_table_direction",
                        float(spectral_weight),
                        float(alpha),
                        float(final_loss.detach().cpu()),
                        float(fitted_mse.detach().cpu()),
                        float(fitted_spectral.detach().cpu()),
                        float(fitted_trust.detach().cpu()),
                        closure_calls,
                        selections,
                        metrics_by_resolution,
                        nmse_limit,
                    )
                )
                if ranges_valid and candidate_better(
                    metrics_by_resolution,
                    round_best_metrics_by_resolution,
                    source_metrics_by_resolution,
                    nmse_limit,
                    args.objective,
                    args.max_point_regression_db,
                ):
                    round_best_state = active_mod.interpolate_table_state(
                        current_state, fitted_tables, float(alpha)
                    )
                    if args.optimize_ranges:
                        round_best_state["corr.gmin"] = torch.lerp(
                            current_state["corr.gmin"], fitted_gmin.cpu(), float(alpha)
                        )
                        round_best_state["corr.gmax"] = torch.lerp(
                            current_state["corr.gmax"], fitted_gmax.cpu(), float(alpha)
                        )
                    round_best_metrics_by_resolution = metrics_by_resolution
                    round_best_official = candidate_official
                    round_best_candidate_id = candidate_id
                    round_best_description = (
                        f"round={round_index} weight={spectral_weight:g} "
                        f"alpha={alpha:.6g}"
                    )
                    round_best_active_bins = active_records

        anchor_key = selection_key(anchor_metrics_by_resolution, nmse_limit, args.objective)
        round_key = selection_key(round_best_metrics_by_resolution, nmse_limit, args.objective)
        gain = float(anchor_key[2]) - float(round_key[2])
        print(
            f"[{args.model}] round {round_index} best {round_best_description}: "
            f"NMSE={round_best_official['nmse_db']:.3f} "
            f"point={round_best_official['point_worst_db']:.3f} "
            f"{args.objective}_gain={gain:+.3f} dB"
        )
        if gain <= args.minimum_gain:
            break
        current_state = clone_state(round_best_state)
        current_metrics_by_resolution = round_best_metrics_by_resolution
        current_official = round_best_official
        selected_candidate_id = round_best_candidate_id
        selected_description = round_best_description
        selected_active_bins = round_best_active_bins

    if current_official["nmse_db"] > nmse_limit:
        raise RuntimeError(
            f"No serialized candidate can satisfy NMSE <= {nmse_limit}: "
            f"{current_official}"
        )

    checkpoint_out = output_dir / "best.pth"
    torch.save(current_state, checkpoint_out)

    # Independently reload/reconstruct the serialized winner before metadata.
    serialized_state = clone_state(evaluator.load_state_dict(checkpoint_out))
    final_model, final_layout = build_wrapper(
        evaluator, ps_mod, serialized_state, offsets, amp_max
    )
    final_model.eval()
    pred_iq, target_iq, _, _ = evaluator.evaluate_full_sequence(
        final_model, x_val, y_val
    )
    final_metrics_by_resolution = metrics_at_resolutions(
        active_mod,
        evaluator,
        metrics_mod,
        evaluator.tensor_to_complex(pred_iq),
        evaluator.tensor_to_complex(target_iq),
        eval_npersegs,
    )
    final_official = official_metrics(final_metrics_by_resolution)
    if final_official["nmse_db"] > nmse_limit:
        raise RuntimeError(
            "Serialized winner violates the hard official NMSE constraint: "
            f"{final_official['nmse_db']:.6f} > {nmse_limit:.6f}"
        )

    for row in rows:
        row["selected"] = row["candidate_id"] == selected_candidate_id
    write_csv(output_dir / "search.csv", rows)
    write_csv(output_dir / "nmse_history.csv", closure_rows)

    final_pred = evaluator.tensor_to_complex(pred_iq)
    final_target = evaluator.tensor_to_complex(target_iq)
    dataset_error_complex = (
        dataset_error_val[:, 0] + 1j * dataset_error_val[:, 1]
    )
    psd_path = output_dir / "psd_comparison.png"
    nmse_curve_path = output_dir / "nmse_convergence.png"
    plot_psd_comparison(
        evaluator,
        metrics_mod,
        source_val,
        final_pred,
        final_target,
        dataset_error_complex,
        source_official,
        final_official,
        psd_path,
    )
    plot_nmse_convergence(closure_rows, rows, nmse_limit, nmse_curve_path)

    protocol = {
        "spectral_mode": args.spectral_mode,
        "optimize_ranges": args.optimize_ranges,
        "frequency_selection": (
            "lowest lower/upper residual-power bins of validation incumbent"
            if args.spectral_mode == "active"
            else "fixed complete lower/upper adjacent bands"
        ),
        "frequency_selection_uses_validation": args.spectral_mode == "active",
        "coefficient_fit": "L-BFGS on complete training split only",
        "validation_samples_used_in_lbfgs": False,
        "trainable_parameters": (
            ["corr.tables", "corr.gmin", "corr.gmax"]
            if args.optimize_ranges
            else ["corr.tables"]
        ),
        "frozen_components": [
            "base", "base FIR", "wrapper output FIR", "corr.gmin", "corr.gmax"
        ],
        "alpha_and_checkpoint_selection": (
            "validation official 4096 point ACLR under hard NMSE"
        ),
        "objective": (
            "normalized train MSE + spectral_weight * mean normalized spectral "
            "power over all fit resolutions + trust_weight * normalized table distance"
        ),
    }
    sidecar = {
        "format": (
            "PhaseSplineDPD raw state_dict plus explicit reconstruction metadata"
        ),
        "model": args.model,
        "source_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint_out),
        "offsets": list(offsets),
        "state_inference": {
            key: final_layout[key]
            for key in (
                "grid", "base_fir_taps", "output_fir_taps",
                "taps_reduction", "table_features",
            )
        },
        "mac": source_macs,
        "fit_npersegs": list(fit_npersegs),
        "official_selection_nperseg": OFFICIAL_NPERSEG,
        "bins_per_side": args.bins_per_side,
        "protocol": protocol,
        "active_bin_history": active_bin_history,
        "selected_active_bins": selected_active_bins,
        "selection": {
            "candidate_id": selected_candidate_id,
            "description": selected_description,
            "objective": args.objective,
            "max_point_regression_db": args.max_point_regression_db,
            "nmse_limit_db": nmse_limit,
        },
        "metrics_by_nperseg": final_metrics_by_resolution,
    }
    sidecar_path = output_dir / "best.sidecar.json"
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)

    source_point = source_official["point_worst_db"]
    source_integrated = source_official["integrated_aclr_db"]
    summary = {
        "model": args.model,
        "source_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint_out),
        "sidecar": str(sidecar_path),
        "selection": selected_description,
        "objective": args.objective,
        "spectral_mode": args.spectral_mode,
        "optimize_ranges": args.optimize_ranges,
        "official_nperseg": OFFICIAL_NPERSEG,
        "nmse_limit_db": nmse_limit,
        "frequency_selection": (
            "validation incumbent residual"
            if args.spectral_mode == "active"
            else "fixed complete adjacent bands"
        ),
        "coefficients_fitted_on": "complete training split only",
        "source_official_point_worst_db": source_point,
        "official_point_gain_db": source_point - final_official["point_worst_db"],
        "source_integrated_aclr_db": source_integrated,
        "integrated_aclr_gain_db": source_integrated - final_official["integrated_aclr_db"],
        "psd_plot": str(psd_path),
        "nmse_curve": str(nmse_curve_path),
        **final_official,
        "metrics_by_nperseg": final_metrics_by_resolution,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    config = vars(args).copy()
    config["resolved_nmse_limit_db"] = nmse_limit
    config["official_nperseg"] = OFFICIAL_NPERSEG
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with (output_dir / "offsets.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"model": args.model, "selected_offsets": list(offsets), "mac": source_macs["total"]},
            handle,
            indent=2,
        )
    required_artifacts = {
        "checkpoint": checkpoint_out,
        "summary": output_dir / "summary.json",
        "search": output_dir / "search.csv",
        "nmse_history": output_dir / "nmse_history.csv",
        "psd_plot": psd_path,
        "nmse_curve": nmse_curve_path,
        "sidecar": sidecar_path,
        "offsets": output_dir / "offsets.json",
    }
    manifest = {
        name: {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        for name, path in required_artifacts.items()
    }
    missing = [name for name, item in manifest.items() if not item["exists"] or item["bytes"] <= 0]
    if missing:
        raise RuntimeError(f"Missing/empty required artifacts: {missing}")
    with (output_dir / "artifacts_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-resolution spline-table fine-tuning with hard NMSE/MAC and "
            "cross-resolution point guards."
        )
    )
    parser.add_argument("--model", required=True, choices=NMSE_LIMITS_DB)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--tag", default="multires_active_bin")
    parser.add_argument("--spectral-mode", choices=["active", "band"], default="active")
    parser.add_argument(
        "--optimize-ranges",
        action="store_true",
        help="jointly optimize corr.gmin/gmax constants with corr.tables; deployment MAC is unchanged",
    )
    parser.add_argument("--objective", choices=["integrated", "stable", "point"], default="integrated")
    parser.add_argument("--nmse-limit", type=float, default=None)
    parser.add_argument("--mac-limit", type=int, default=None)
    parser.add_argument("--fit-npersegs", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--bins-per-side", type=int, default=3)
    parser.add_argument("--bin-separation", type=int, default=4)
    parser.add_argument(
        "--spectral-weights", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0]
    )
    parser.add_argument("--trust-weight", type=float, default=1e-4)
    parser.add_argument("--range-trust-weight", type=float, default=10.0)
    parser.add_argument("--min-range-width-ratio", type=float, default=0.5)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-iters", type=int, default=30)
    parser.add_argument("--lbfgs-history", type=int, default=20)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--minimum-gain", type=float, default=1e-4)
    parser.add_argument(
        "--max-point-regression-db",
        type=float,
        default=0.10,
        help="maximum allowed point-ACLR regression at each fitted Welch resolution",
    )
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if args.rounds < 1 or args.bins_per_side < 1 or args.bin_separation < 1:
        parser.error("--rounds, --bins-per-side and --bin-separation must be positive")
    if any(value < 2 for value in args.fit_npersegs):
        parser.error("all --fit-npersegs must be at least 2")
    if len(set(args.fit_npersegs)) != len(args.fit_npersegs):
        parser.error("--fit-npersegs must not contain duplicates")
    if args.lbfgs_iters < 1 or args.lbfgs_history < 1 or args.lbfgs_lr <= 0.0:
        parser.error("L-BFGS iteration/history counts and learning rate must be positive")
    if args.alpha_step <= 0.0 or args.alpha_max < args.alpha_min:
        parser.error("require --alpha-step > 0 and --alpha-max >= --alpha-min")
    if args.max_point_regression_db < 0.0:
        parser.error("--max-point-regression-db must be non-negative")
    if (
        args.trust_weight < 0.0
        or args.range_trust_weight < 0.0
        or any(value < 0.0 for value in args.spectral_weights)
    ):
        parser.error("trust and spectral weights must be non-negative")
    if not 0.0 < args.min_range_width_ratio <= 1.0:
        parser.error("--min-range-width-ratio must lie in (0, 1]")
    if not 0.0 < args.split < 1.0:
        parser.error("--split must lie strictly between zero and one")
    normalize_offsets(args.offsets)
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    evaluator = load_module(
        "final_evaluate_unified_multires", ROOT / "steps" / "final_evaluate_unified.py"
    )
    metrics_mod = load_module(
        "aclr_ref_metrics_multires", ROOT / "steps" / "aclr_ref_metrics.py"
    )
    ps_mod = load_module(
        "aclr_phase_spline_multires", ROOT / "steps" / "aclr_phase_spline_experiment.py"
    )
    active_mod = load_module(
        "aclr_active_bin_multires", ACTIVE_BIN_SCRIPT
    )
    if not np.isclose(args.fs_mhz * 1e6, metrics_mod.FS_HZ, rtol=0.0, atol=1e-6):
        raise SystemExit(
            f"--fs-mhz={args.fs_mhz} conflicts with evaluator rate "
            f"{metrics_mod.FS_HZ / 1e6} MHz"
        )

    features, targets, errors, *_ = evaluator.load_iq_arrays(
        ROOT / "data" / "DPD_data.mat", "rms"
    )
    split_index = int(len(features) * args.split)
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))
    arrays = (
        features[:split_index],
        targets[:split_index],
        features[split_index:],
        targets[split_index:],
        errors[split_index:],
        amp_max,
    )
    summary = run_search(
        args, evaluator, metrics_mod, ps_mod, active_mod, arrays
    )
    tag_dir = OUT_ROOT / args.tag
    aggregate_path = tag_dir / "summary.csv"
    existing = []
    if aggregate_path.exists():
        with aggregate_path.open("r", newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    existing = [row for row in existing if row.get("model") != args.model]
    existing.append(
        {
            key: value
            for key, value in summary.items()
            if not isinstance(value, (dict, list))
        }
    )
    write_csv(aggregate_path, existing)


if __name__ == "__main__":
    main()
