"""Search phase-spline lag topologies without changing the MAC budget.

The phase-spline correction stores one table row for ``|x|`` and two rows
(real/imaginary) for every non-zero lag.  Consequently, lag sets with the same
number of entries have exactly the same deployed MAC count.  This experiment
remaps a wrapper checkpoint to one or more alternative lag sets by semantic
row identity:

* the amplitude row is always copied;
* real/imaginary rows for lags shared with the source topology are copied; and
* rows belonging to new lags are zero initialized.

For every target topology the EA-TCN base and output FIR are frozen.  The
complete training sequence is then used to fit only ``corr.tables`` with a
convex quadratic objective: time-domain MSE plus the Welch powers at the two
active adjacent-band bins selected on validation, plus an optional L2 trust
term.  The fitted table direction is scanned on validation and only candidates
satisfying the model's hard NMSE limit can replace the source checkpoint.

The script deliberately imports the Welch, cached-forward, hard-evaluation and
candidate-ranking utilities from ``aclr_active_bin_table_optimization.py`` so
both experiments use exactly the same numerical conventions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"
ACTIVE_BIN_SCRIPT = ROOT / "steps" / "aclr_active_bin_table_optimization.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_offsets(offsets, label):
    values = tuple(int(value) for value in offsets)
    if not values:
        raise ValueError(f"{label} must not be empty")
    if values.count(0) != 1:
        raise ValueError(f"{label} must contain zero exactly once: {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate offsets: {values}")
    return values


def feature_rows(offsets):
    """Map phase-spline table rows to their semantic feature identities."""
    offsets = normalize_offsets(offsets, "offsets")
    rows = {("amplitude", 0): 0}
    row = 1
    for offset in offsets:
        if offset == 0:
            continue
        rows[("real", offset)] = row
        rows[("imag", offset)] = row + 1
        row += 2
    return rows


def validate_source_state(state, source_offsets):
    required = {"corr.tables", "corr.gmin", "corr.gmax", "fir_real", "fir_imag"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Source checkpoint is missing wrapper keys: {missing}")
    if not any(key.startswith("base.") for key in state):
        raise ValueError("Expected a phase-spline wrapper checkpoint with base.* keys")
    rows = feature_rows(source_offsets)
    expected_features = len(rows)
    actual_features = int(state["corr.tables"].shape[0])
    if actual_features != expected_features:
        raise ValueError(
            "Source offsets do not match corr.tables: "
            f"offsets imply {expected_features} rows, checkpoint has {actual_features}"
        )
    if tuple(state["corr.gmin"].shape) != (expected_features,):
        raise ValueError("corr.gmin shape does not match corr.tables")
    if tuple(state["corr.gmax"].shape) != (expected_features,):
        raise ValueError("corr.gmax shape does not match corr.tables")
    if state["fir_real"].shape != state["fir_imag"].shape:
        raise ValueError("Top-level real/imag FIR shapes differ")


def infer_wrapper_layout(state):
    inner = {
        key[len("base.") :]: value
        for key, value in state.items()
        if key.startswith("base.")
    }
    if "fir_real" not in inner or "fir_imag" not in inner:
        raise ValueError("Base checkpoint has no FIR from which to infer the MAC trade")
    original_taps = int(inner["fir_real"].numel())
    wrapper_taps = int(state["fir_real"].numel())
    taps_reduction = original_taps - wrapper_taps
    if taps_reduction < 0:
        raise ValueError(
            f"Wrapper FIR ({wrapper_taps}) is longer than base FIR ({original_taps})"
        )
    grid = int(state["corr.tables"].shape[-1] - 1)
    if grid < 1:
        raise ValueError(f"Invalid spline grid inferred from checkpoint: {grid}")
    return inner, taps_reduction, grid


def remap_table_rows(state, source_offsets, target_offsets):
    """Clone ``state`` and remap table rows by lag semantics."""
    source_offsets = normalize_offsets(source_offsets, "source offsets")
    target_offsets = normalize_offsets(target_offsets, "target offsets")
    if len(source_offsets) != len(target_offsets):
        raise ValueError(
            "Target topology must keep the same offset count/MAC: "
            f"source={source_offsets}, target={target_offsets}"
        )
    validate_source_state(state, source_offsets)
    source_rows = feature_rows(source_offsets)
    target_rows = feature_rows(target_offsets)
    source_tables = state["corr.tables"]
    target_tables = torch.zeros_like(source_tables)
    target_gmin = torch.empty_like(state["corr.gmin"])
    target_gmax = torch.empty_like(state["corr.gmax"])
    copied = []
    initialized = []
    for semantic, target_row in target_rows.items():
        if semantic in source_rows:
            source_row = source_rows[semantic]
            target_tables[target_row].copy_(source_tables[source_row])
            target_gmin[target_row].copy_(state["corr.gmin"][source_row])
            target_gmax[target_row].copy_(state["corr.gmax"][source_row])
            copied.append(semantic)
        else:
            # New lags inherit the existing range for the same feature kind.
            exemplar = next(
                row for key, row in source_rows.items() if key[0] == semantic[0]
            )
            target_gmin[target_row].copy_(state["corr.gmin"][exemplar])
            target_gmax[target_row].copy_(state["corr.gmax"][exemplar])
            initialized.append(semantic)
    result = {key: value.detach().cpu().clone() for key, value in state.items()}
    result["corr.tables"] = target_tables.detach().cpu()
    result["corr.gmin"] = target_gmin.detach().cpu()
    result["corr.gmax"] = target_gmax.detach().cpu()
    return result, copied, initialized


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


def topology_label(offsets):
    return ",".join(str(int(value)) for value in offsets)


def validation_active_bins(active_mod, welch, error_complex):
    with torch.no_grad():
        power = welch.power(error_complex).detach().cpu().numpy()
    lower = active_mod.separated_low_bins(
        power, welch.freq_mhz, (-180.0, -120.0), count=1, separation=1
    )[0]
    upper = active_mod.separated_low_bins(
        power, welch.freq_mhz, (120.0, 180.0), count=1, separation=1
    )[0]
    return int(lower), int(upper), power


def feasible_better(active_mod, metrics, best_metrics, nmse_limit):
    if metrics["nmse_db"] > nmse_limit:
        return False
    return active_mod.candidate_key(metrics, nmse_limit) < active_mod.candidate_key(
        best_metrics, nmse_limit
    )


def row_from_metrics(
    model_name,
    topology_index,
    offsets,
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
        "topology_index": topology_index,
        "offsets": topology_label(offsets),
        "source": source,
        "active_lower_mhz": active_mhz[0] if active_mhz else "",
        "active_upper_mhz": active_mhz[1] if active_mhz else "",
        "spectral_weight": spectral_weight,
        "alpha": alpha,
        "lbfgs_loss": loss,
        "closure_calls": closure_calls,
        **metrics,
    }


def run_search(args, evaluator, metrics_mod, ps_mod, active_mod, arrays):
    x_train, y_train, x_val, y_val, amp_max = arrays
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    source_offsets = normalize_offsets(args.source_offsets, "source offsets")
    target_sets = [
        normalize_offsets(offsets, f"target offsets #{index}")
        for index, offsets in enumerate(args.target_offsets, start=1)
    ]
    if len(set(target_sets)) != len(target_sets):
        raise ValueError("Duplicate target offset sets were provided")

    source_state = evaluator.load_state_dict(checkpoint)
    validate_source_state(source_state, source_offsets)
    inner, taps_reduction, grid = infer_wrapper_layout(source_state)
    nmse_limit = float(active_mod.MODEL_LAYOUTS[args.model]["nmse"])

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
    target_val = y_val[:, 0] + 1j * y_val[:, 1]
    target_train_c = torch.complex(y_train_t[:, 0], y_train_t[:, 1])
    target_val_c = torch.complex(y_val_t[:, 0], y_val_t[:, 1])
    welch = active_mod.TorchWelch(args.nperseg, args.fs_mhz, evaluator.DEVICE)

    source_model = build_wrapper(evaluator, ps_mod, source_state, source_offsets, amp_max).eval()
    source_macs = source_model.mac_report(evaluator, inner, strict=True)
    if source_macs["total"] != int(args.model):
        raise ValueError(
            f"Source wrapper MAC {source_macs['total']} does not match model label {args.model}"
        )
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    base_train = active_mod.cached_base(source_model, x_train_t)
    base_val = active_mod.cached_base(source_model, x_val_t)
    with torch.no_grad():
        source_val_iq = active_mod.forward_from_base(
            source_model, x_val_t, base_val
        )
        source_val = (
            source_val_iq[0, 0].cpu().numpy()
            + 1j * source_val_iq[0, 1].cpu().numpy()
        )
    source_metrics = active_mod.numpy_metrics(
        evaluator, metrics_mod, source_val, target_val, args.nperseg
    )
    if source_metrics["nmse_db"] > nmse_limit:
        raise ValueError(
            f"Source checkpoint violates NMSE limit {nmse_limit}: {source_metrics}"
        )

    best_state = {key: value.detach().cpu().clone() for key, value in source_state.items()}
    best_offsets = source_offsets
    best_metrics = source_metrics
    best_description = "source checkpoint"
    rows = [
        row_from_metrics(
            args.model,
            -1,
            source_offsets,
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
        f"[{args.model}] source offsets={source_offsets} MAC={source_macs['total']} "
        f"NMSE={source_metrics['nmse_db']:.3f} "
        f"point={source_metrics['point_lower_db']:.3f}/"
        f"{source_metrics['point_upper_db']:.3f}/"
        f"{source_metrics['point_worst_db']:.3f}"
    )

    for topology_index, target_offsets in enumerate(target_sets):
        if len(target_offsets) != len(source_offsets):
            raise ValueError(
                f"Target offsets must preserve feature count/MAC: {target_offsets}"
            )
        anchor_state, copied, initialized = remap_table_rows(
            source_state, source_offsets, target_offsets
        )
        model = build_wrapper(evaluator, ps_mod, anchor_state, target_offsets, amp_max).eval()
        target_macs = model.mac_report(evaluator, inner, strict=True)
        if target_macs["total"] != source_macs["total"]:
            raise RuntimeError(
                f"Topology {target_offsets} changed MAC: "
                f"{source_macs['total']} -> {target_macs['total']}"
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
            lower_bin, upper_bin, _ = validation_active_bins(
                active_mod, welch, target_val_c - anchor_val_c
            )
            anchor_train_error = target_train_c - torch.complex(
                anchor_train_iq[:, 0], anchor_train_iq[:, 1]
            )
            anchor_train_power = welch.power(anchor_train_error)
            active_scale = anchor_train_power[[lower_bin, upper_bin]].clamp_min(1e-30)
            anchor_mse = torch.mean((anchor_train_iq - y_train_t).square()).clamp_min(1e-30)
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
            row_from_metrics(
                args.model,
                topology_index,
                target_offsets,
                "remapped_anchor",
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
            best_offsets = target_offsets
            best_metrics = anchor_metrics
            best_description = f"remapped anchor offsets={target_offsets}"

        copied_lags = sorted(
            {semantic[1] for semantic in copied if semantic[0] != "amplitude"}
        )
        new_lags = sorted({semantic[1] for semantic in initialized})
        print(
            f"[{args.model}] topology {topology_index} offsets={target_offsets} "
            f"copied_lags={copied_lags} new_lags={new_lags} "
            f"active=({active_mhz[0]:.2f},{active_mhz[1]:.2f}) MHz "
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
                    f"Non-finite fitted tables for offsets={target_offsets}, "
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
                    row_from_metrics(
                        args.model,
                        topology_index,
                        target_offsets,
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
                    best_offsets = target_offsets
                    best_metrics = metrics
                    best_description = (
                        f"offsets={target_offsets} active=({active_mhz[0]:.2f},"
                        f"{active_mhz[1]:.2f}) weight={spectral_weight:g} "
                        f"alpha={alpha:.3f}"
                    )

    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(best_state, output_dir / "best.pth")

    offsets_sidecar = {
        "model": args.model,
        "source_checkpoint": str(checkpoint),
        "source_offsets": list(source_offsets),
        "selected_offsets": list(best_offsets),
        "target_offset_sets": [list(values) for values in target_sets],
        "grid": grid,
        "taps_reduction": taps_reduction,
        "mac": source_macs["total"],
        "table_row_semantics": [
            {"kind": semantic[0], "offset": semantic[1], "row": row}
            for semantic, row in sorted(
                feature_rows(best_offsets).items(), key=lambda item: item[1]
            )
        ],
    }
    with (output_dir / "offsets.json").open("w", encoding="utf-8") as handle:
        json.dump(offsets_sidecar, handle, indent=2)

    final_model = build_wrapper(
        evaluator, ps_mod, best_state, best_offsets, amp_max
    ).eval()
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
    for key in (
        "nmse_db",
        "point_lower_db",
        "point_upper_db",
        "point_worst_db",
    ):
        if not np.isclose(final_metrics[key], best_metrics[key], atol=1e-9, rtol=0.0):
            raise RuntimeError(
                f"Serialized winner mismatch for {key}: "
                f"selected={best_metrics[key]}, reconstructed={final_metrics[key]}"
            )

    summary = {
        "model": args.model,
        "source_checkpoint": str(checkpoint),
        "source_offsets": list(source_offsets),
        "selected_offsets": list(best_offsets),
        "selection": best_description,
        "nmse_limit_db": nmse_limit,
        "source_point_worst_db": source_metrics["point_worst_db"],
        "point_gain_db": source_metrics["point_worst_db"]
        - final_metrics["point_worst_db"],
        "mac": source_macs["total"],
        **final_metrics,
        "checkpoint": str(output_dir / "best.pth"),
        "offsets_sidecar": str(output_dir / "offsets.json"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def smoke_check(args, evaluator, ps_mod, active_mod, features, targets, amp_max):
    """Build/remap/backpropagate one small batch without writing any output."""
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    source_offsets = normalize_offsets(args.source_offsets, "source offsets")
    target_offsets = normalize_offsets(args.target_offsets[0], "target offsets #1")
    source_state = evaluator.load_state_dict(checkpoint)
    validate_source_state(source_state, source_offsets)
    target_state, copied, initialized = remap_table_rows(
        source_state, source_offsets, target_offsets
    )
    inner, _, _ = infer_wrapper_layout(source_state)
    model = build_wrapper(evaluator, ps_mod, target_state, target_offsets, amp_max).eval()
    macs = model.mac_report(evaluator, inner, strict=True)
    if macs["total"] != int(args.model):
        raise RuntimeError(f"Smoke MAC mismatch: {macs['total']} != {args.model}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.corr.tables.requires_grad_(True)
    length = min(len(features), max(2 * args.nperseg, args.nperseg + 1))
    if length < args.nperseg:
        raise ValueError("Dataset is shorter than nperseg")
    x = torch.as_tensor(
        features[:length].T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    y = torch.as_tensor(
        targets[:length].T, dtype=torch.float64, device=evaluator.DEVICE
    ).unsqueeze(0)
    base = active_mod.cached_base(model, x)
    pred = active_mod.forward_from_base(model, x, base)
    error_iq = y - pred
    error_c = torch.complex(error_iq[:, 0], error_iq[:, 1])
    welch = active_mod.TorchWelch(args.nperseg, args.fs_mhz, evaluator.DEVICE)
    lower, upper, _ = validation_active_bins(active_mod, welch, error_c.detach())
    spectra = welch.spectra(error_c)
    loss = torch.mean(error_iq.square()) + 1e-12 * spectra[
        ..., [lower, upper]
    ].abs().square().mean()
    loss.backward()
    gradient = model.corr.tables.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("Smoke backward produced no finite corr.tables gradient")
    print(
        "smoke-ok "
        f"model={args.model} source={source_offsets} target={target_offsets} "
        f"MAC={macs['total']} copied_rows={len(copied)} new_rows={len(initialized)} "
        f"active=({welch.freq_mhz[lower]:.2f},{welch.freq_mhz[upper]:.2f})MHz "
        f"loss={float(loss.detach().cpu()):.6e}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["2284", "998", "786", "796"])
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="source phase-spline wrapper checkpoint",
    )
    parser.add_argument(
        "--source-offsets",
        type=int,
        nargs="+",
        required=True,
        help="ordered offsets used to create the source checkpoint",
    )
    parser.add_argument(
        "--target-offsets",
        type=int,
        nargs="+",
        action="append",
        required=True,
        help="ordered target set; repeat this option to search multiple topologies",
    )
    parser.add_argument("--tag", default="lag_topology_search")
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
        help="remap/build/backpropagate a short sequence and exit without writing outputs",
    )
    args = parser.parse_args()
    if args.alpha_step <= 0.0:
        parser.error("--alpha-step must be positive")
    if args.alpha_max < args.alpha_min:
        parser.error("--alpha-max must be >= --alpha-min")
    if args.nperseg < 2:
        parser.error("--nperseg must be at least 2")
    if any(weight < 0.0 for weight in args.spectral_weights):
        parser.error("--spectral-weights must be non-negative")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    active_mod = load_module("aclr_active_bin_table_optimization", ACTIVE_BIN_SCRIPT)
    evaluator = load_module(
        "final_evaluate_unified_lag_search", ROOT / "steps" / "final_evaluate_unified.py"
    )
    metrics_mod = load_module(
        "aclr_ref_metrics_lag_search", ROOT / "steps" / "aclr_ref_metrics.py"
    )
    if not np.isclose(args.fs_mhz * 1e6, metrics_mod.FS_HZ, rtol=0.0, atol=1e-6):
        raise SystemExit(
            f"--fs-mhz={args.fs_mhz} conflicts with the fixed evaluator rate "
            f"{metrics_mod.FS_HZ / 1e6} MHz"
        )
    ps_mod = load_module(
        "aclr_phase_spline_experiment_lag_search",
        ROOT / "steps" / "aclr_phase_spline_experiment.py",
    )
    features, targets, _, *_ = evaluator.load_iq_arrays(
        ROOT / "data" / "DPD_data.mat", "rms"
    )
    split_index = int(len(features) * args.split)
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))
    if args.smoke:
        smoke_check(args, evaluator, ps_mod, active_mod, features, targets, amp_max)
        return
    arrays = (
        features[:split_index],
        targets[:split_index],
        features[split_index:],
        targets[split_index:],
        amp_max,
    )
    run_search(args, evaluator, metrics_mod, ps_mod, active_mod, arrays)


if __name__ == "__main__":
    main()
