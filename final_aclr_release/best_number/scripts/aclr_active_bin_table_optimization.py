"""Full-sequence active-bin optimization for phase-spline table checkpoints.

The minimum-point ACLR metric is controlled by one Welch bin on each adjacent
side.  The previous optimizer uses a mini-batch soft minimum over every bin,
which is deliberately robust but can dilute the gradient and move the active
bin between batches.  This experiment instead:

1. identifies separated low-power bins on a declared selection split
   (validation by default, or training with ``--bin-source train``);
2. freezes the EA-TCN base and output FIR;
3. solves the phase-spline tables on the complete training sequence with
   full-batch L-BFGS for each active-bin pair and spectral trade-off; and
4. scans the exact linear table-update direction on validation, accepting only
   checkpoints that satisfy the model's hard NMSE limit.

With a fixed FIR the deployed output is linear in every spline-table entry, so
each fit objective (time-domain MSE plus two fixed-bin powers and L2 trust) is a
convex quadratic.  No layer, table, tap, or MAC count is changed.
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
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"
MODEL_LAYOUTS = {
    "2284": {"taps_reduction": 14, "offsets": [-2, -1, 0, 1, 2], "nmse": -40.0},
    "998": {"taps_reduction": 14, "offsets": [-2, -1, 0, 1, 2], "nmse": -37.0},
    "786": {"taps_reduction": 8, "offsets": [-1, 0, 1], "nmse": -35.0},
    "796": {"taps_reduction": 14, "offsets": [-2, -1, 0, 1, 2], "nmse": -35.0},
}


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def resample_table_grid(state, target_grid):
    """Resample a piecewise-linear table without changing its represented function."""
    source = state["corr.tables"]
    source_grid = int(source.shape[-1] - 1)
    if source_grid == int(target_grid):
        return {key: value.clone() for key, value in state.items()}
    if int(target_grid) % source_grid != 0:
        raise ValueError(
            f"Exact warm start requires target grid to be a multiple of {source_grid}; "
            f"got {target_grid}"
        )
    flat = source.reshape(-1, 1, source.shape[-1])
    resized = nn.functional.interpolate(
        flat,
        size=int(target_grid) + 1,
        mode="linear",
        align_corners=True,
    ).reshape(source.shape[0], source.shape[1], int(target_grid) + 1)
    result = {key: value.clone() for key, value in state.items()}
    result["corr.tables"] = resized
    return result


def resolve_checkpoint_offsets(checkpoint, model_name, explicit_offsets, require_metadata):
    if explicit_offsets is not None:
        offsets = [int(value) for value in explicit_offsets]
    else:
        offsets = None
        for filename in ("offsets.json", "model_config.json"):
            sidecar = checkpoint.parent / filename
            if not sidecar.exists():
                continue
            with sidecar.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            for key in ("selected_offsets", "offsets", "source_offsets"):
                if key in metadata:
                    offsets = [int(value) for value in metadata[key]]
                    break
            if offsets is not None:
                break
        if offsets is None:
            if require_metadata:
                raise ValueError(
                    "--checkpoint is not self-describing: pass its true --offsets or place "
                    "offsets.json/model_config.json beside it"
                )
            offsets = list(MODEL_LAYOUTS[model_name]["offsets"])
    if offsets.count(0) != 1 or len(set(offsets)) != len(offsets):
        raise ValueError(f"Invalid offsets for {model_name}: {offsets}")
    expected_features = 1 + 2 * (len(offsets) - 1)
    return offsets, expected_features


def build_wrapper(evaluator, ps_mod, state, model_name, grid, amp_max, offsets=None):
    layout = MODEL_LAYOUTS[model_name]
    offsets = list(layout["offsets"] if offsets is None else offsets)
    inner = {key[len("base."):]: value for key, value in state.items() if key.startswith("base.")}
    if not inner:
        raise ValueError("Expected a phase-spline wrapper checkpoint with base.* keys")
    # Infer the FIR trade from the checkpoint itself so non-default tap counts load.
    taps_reduction = int(inner["fir_real"].numel()) - int(state["fir_real"].numel())
    model = ps_mod.PhaseSplineDPD(
        evaluator,
        inner,
        taps_reduction,
        offsets,
        grid,
        amp_max,
    )
    model.load_state_dict(state, strict=True)
    return model.to(evaluator.DEVICE)


def cached_base(model, x_iq):
    with torch.no_grad():
        return model.base(x_iq).detach()


def forward_from_base(model, x_iq, base_iq):
    """Wrapper forward with its frozen EA-TCN output supplied by the caller."""
    y = base_iq + model.corr(x_iq)
    pad = len(model.fir_real) // 2
    yi, yq = y[:, 0:1, :], y[:, 1:2, :]
    kr = model.fir_real.view(1, 1, -1)
    ki = model.fir_imag.view(1, 1, -1)
    out_i = nn.functional.conv1d(yi, kr, padding=pad) - nn.functional.conv1d(
        yq, ki, padding=pad
    )
    out_q = nn.functional.conv1d(yi, ki, padding=pad) + nn.functional.conv1d(
        yq, kr, padding=pad
    )
    return torch.cat([out_i, out_q], dim=1)


class TorchWelch:
    """Differentiable counterpart of scipy.signal.welch's complex defaults."""

    def __init__(self, nperseg, fs_mhz, device):
        self.nperseg = int(nperseg)
        # scipy.signal.welch defaults to noverlap=nperseg//2, hence the step
        # is nperseg-noverlap (different from nperseg//2 for odd lengths).
        self.hop = self.nperseg - self.nperseg // 2
        self.window = torch.hann_window(
            self.nperseg, periodic=True, dtype=torch.float64, device=device
        )
        self.freq_mhz = np.fft.fftfreq(self.nperseg, d=1.0 / float(fs_mhz))

    def spectra(self, signal):
        """Return segment spectra after SciPy's default constant detrending."""
        segments = signal.unfold(-1, self.nperseg, self.hop)
        segments = segments - segments.mean(dim=-1, keepdim=True)
        return torch.fft.fft(segments * self.window, dim=-1)

    def power(self, signal):
        return self.spectra(signal).abs().square().mean(dim=tuple(range(signal.ndim)))


def separated_low_bins(power, freq_mhz, band, count, separation):
    lo, hi = sorted(map(float, band))
    candidates = np.flatnonzero((freq_mhz >= lo) & (freq_mhz <= hi))
    ordered = candidates[np.argsort(power[candidates])]
    chosen = []
    for index in ordered:
        if all(abs(int(index) - old) >= separation for old in chosen):
            chosen.append(int(index))
            if len(chosen) == count:
                break
    if len(chosen) < count:
        raise RuntimeError(f"Could only find {len(chosen)} separated bins in band {band}")
    return chosen


def numpy_metrics(evaluator, metrics_mod, pred, target, nperseg):
    error = target - pred
    nmse_linear = float(np.sum(np.abs(error) ** 2) / np.sum(np.abs(target) ** 2))
    freq, target_psd = evaluator.psd_db(target, metrics_mod.FS_HZ, nperseg)
    _, error_psd = evaluator.psd_db(error, metrics_mod.FS_HZ, nperseg)
    lower, upper, worst = evaluator.aclr_from_psd_minimum(
        freq, error_psd, target_psd, (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0)
    )
    lower_mask = (freq >= -180e6) & (freq <= -120e6)
    upper_mask = (freq >= 120e6) & (freq <= 180e6)
    lower_index = np.flatnonzero(lower_mask)[np.argmin(error_psd[lower_mask])]
    upper_index = np.flatnonzero(upper_mask)[np.argmin(error_psd[upper_mask])]
    reference = metrics_mod.ref_metrics(evaluator, pred, target, nperseg)
    return {
        "nmse_linear": nmse_linear,
        "nmse_db": float(10.0 * np.log10(nmse_linear)),
        "point_lower_db": float(lower),
        "point_upper_db": float(upper),
        "point_worst_db": float(worst),
        "point_lower_mhz": float(freq[lower_index] / 1e6),
        "point_upper_mhz": float(freq[upper_index] / 1e6),
        "integrated_aclr_db": float(reference["aclr_db"]),
    }


def candidate_key(metrics, nmse_limit):
    violation = max(0.0, metrics["nmse_db"] - nmse_limit)
    return (
        violation > 0.0,
        violation,
        metrics["point_worst_db"],
        metrics["nmse_db"],
    )


def interpolate_table_state(anchor_state, fitted_tables, alpha):
    result = {key: value.clone() for key, value in anchor_state.items()}
    result["corr.tables"] = torch.lerp(
        anchor_state["corr.tables"], fitted_tables.cpu(), float(alpha)
    )
    return result


def run_model(args, model_name, evaluator, metrics_mod, ps_mod, arrays):
    x_train, y_train, x_val, y_val, amp_max = arrays
    layout = MODEL_LAYOUTS[model_name]
    nmse_limit = float(layout["nmse"])
    mac_limit = args.mac_limit
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(args.checkpoint_root) / model_name / "best.pth"
    )
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    current_state = resample_table_grid(evaluator.load_state_dict(checkpoint), args.grid)
    offsets, expected_features = resolve_checkpoint_offsets(
        checkpoint,
        model_name,
        args.offsets,
        require_metadata=args.checkpoint is not None,
    )
    if int(current_state["corr.tables"].shape[0]) != expected_features:
        raise ValueError(
            f"Offsets {offsets} imply {expected_features} features, but checkpoint has "
            f"{current_state['corr.tables'].shape[0]}"
        )
    output_dir = OUT_ROOT / args.tag / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model": model_name,
        "real_multipliers": int(model_name),
        "checkpoint_format": "PhaseSplineDPD raw state_dict",
        "source_checkpoint": str(checkpoint),
        "offsets": offsets,
        "grid": int(args.grid),
        "taps_reduction": int(current_state["base.fir_real"].numel())
        - int(current_state["fir_real"].numel()),
        "base_fir_taps": int(current_state["base.fir_real"].numel()),
        "output_fir_taps": int(current_state["fir_real"].numel()),
        "metric": "minimum-point ACLR; main=(20,80), adjacent=(-180,-120)/(120,180) MHz",
        "nmse_limit_db": nmse_limit,
        "normalization": "rms",
        "split": float(args.split),
        "fs_mhz": float(args.fs_mhz),
        "nperseg": int(args.nperseg),
    }
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
    welch = TorchWelch(args.nperseg, args.fs_mhz, evaluator.DEVICE)
    rows = []

    for round_index in range(1, args.rounds + 1):
        model = build_wrapper(
            evaluator, ps_mod, current_state, model_name, args.grid, amp_max, offsets
        )
        inner_state = {
            key[len("base."):]: value
            for key, value in current_state.items()
            if key.startswith("base.")
        }
        macs = model.mac_report(evaluator, inner_state, strict=True)
        if mac_limit is None:
            if macs["total"] != int(model_name):
                raise ValueError(
                    f"Checkpoint/model mismatch: reconstructed MAC={macs['total']} but "
                    f"--model is {model_name}"
                )
        elif macs["total"] > mac_limit:
            raise ValueError(
                f"Reconstructed MAC={macs['total']} exceeds --mac-limit {mac_limit}"
            )
        if round_index == 1:
            with (output_dir / "model_config.json").open("w", encoding="utf-8") as handle:
                json.dump(model_config, handle, indent=2)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.corr.tables.requires_grad_(True)
        base_train = cached_base(model, x_train_t)
        base_val = cached_base(model, x_val_t)

        with torch.no_grad():
            anchor_train_iq = forward_from_base(model, x_train_t, base_train)
            anchor_val_iq = forward_from_base(model, x_val_t, base_val)
            anchor_train_c = torch.complex(anchor_train_iq[:, 0], anchor_train_iq[:, 1])
            target_train_c = torch.complex(y_train_t[:, 0], y_train_t[:, 1])
            error_train = target_train_c - anchor_train_c
            anchor_power = welch.power(error_train)
            anchor_val_c = torch.complex(anchor_val_iq[:, 0], anchor_val_iq[:, 1])
            target_val_c = torch.complex(y_val_t[:, 0], y_val_t[:, 1])
            anchor_val_power = welch.power(target_val_c - anchor_val_c)
            anchor_mse = torch.mean((anchor_train_iq - y_train_t).square())
            anchor_tables = model.corr.tables.detach().clone()
            anchor_val = (
                anchor_val_iq[0, 0].cpu().numpy()
                + 1j * anchor_val_iq[0, 1].cpu().numpy()
            )
        anchor_metrics = numpy_metrics(evaluator, metrics_mod, anchor_val, target_val, args.nperseg)
        best_metrics = anchor_metrics
        best_state = current_state
        best_label = "anchor"
        print(
            f"[{model_name}] round {round_index} anchor NMSE {anchor_metrics['nmse_db']:.3f} | "
            f"point {anchor_metrics['point_lower_db']:.3f}/{anchor_metrics['point_upper_db']:.3f}/"
            f"{anchor_metrics['point_worst_db']:.3f}"
        )

        if args.bin_source == "train":
            power_np = anchor_power.detach().cpu().numpy()
        else:
            # This only chooses the two frequency hyperparameters.  All fitted
            # table coefficients still come exclusively from the train split.
            power_np = anchor_val_power.detach().cpu().numpy()
        lower_bins = separated_low_bins(
            power_np, welch.freq_mhz, (-180.0, -120.0), args.active_candidates, args.bin_separation
        )
        upper_bins = separated_low_bins(
            power_np, welch.freq_mhz, (120.0, 180.0), args.active_candidates, args.bin_separation
        )
        pairs = [(li, ui) for li in lower_bins for ui in upper_bins]
        table_scale = torch.mean(anchor_tables.square()).clamp_min(1e-12)

        for pair_index, (lower_bin, upper_bin) in enumerate(pairs):
            active_scale = anchor_power[[lower_bin, upper_bin]].detach().clamp_min(1e-30)
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
                    pred = forward_from_base(model, x_train_t, base_train)
                    error_iq = y_train_t - pred
                    mse_ratio = torch.mean(error_iq.square()) / anchor_mse
                    error_c = torch.complex(error_iq[:, 0], error_iq[:, 1])
                    spectra = welch.spectra(error_c)
                    active_power = spectra[..., [lower_bin, upper_bin]].abs().square().mean(dim=(0, 1))
                    spectral_ratio = torch.mean(active_power / active_scale)
                    trust = torch.mean((model.corr.tables - anchor_tables).square()) / table_scale
                    loss = mse_ratio + float(spectral_weight) * spectral_ratio + args.trust_weight * trust
                    loss.backward()
                    return loss

                optimizer.step(closure)
                fitted_tables = model.corr.tables.detach().clone()
                with torch.no_grad():
                    fitted_train_iq = forward_from_base(model, x_train_t, base_train)
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
                    fitted_val_iq = forward_from_base(model, x_val_t, base_val)
                    fitted_val = (
                        fitted_val_iq[0, 0].cpu().numpy()
                        + 1j * fitted_val_iq[0, 1].cpu().numpy()
                    )
                direction_val = fitted_val - anchor_val

                for alpha in np.arange(
                    0.0,
                    args.alpha_max + args.alpha_step / 2.0,
                    args.alpha_step,
                ):
                    metrics = numpy_metrics(
                        evaluator,
                        metrics_mod,
                        anchor_val + float(alpha) * direction_val,
                        target_val,
                        args.nperseg,
                    )
                    row = {
                        "model": model_name,
                        "round": round_index,
                        "pair": pair_index,
                        "bin_source": args.bin_source,
                        "active_lower_mhz": float(welch.freq_mhz[lower_bin]),
                        "active_upper_mhz": float(welch.freq_mhz[upper_bin]),
                        "spectral_weight": float(spectral_weight),
                        "alpha": float(alpha),
                        "lbfgs_loss": final_loss,
                        "closure_calls": closure_calls,
                        **metrics,
                    }
                    rows.append(row)
                    if candidate_key(metrics, nmse_limit) < candidate_key(best_metrics, nmse_limit):
                        best_metrics = metrics
                        best_state = interpolate_table_state(current_state, fitted_tables, float(alpha))
                        best_label = (
                            f"pair={pair_index} "
                            f"({welch.freq_mhz[lower_bin]:.2f},{welch.freq_mhz[upper_bin]:.2f}) "
                            f"weight={spectral_weight:g} alpha={alpha:.3f}"
                        )

        improvement = anchor_metrics["point_worst_db"] - best_metrics["point_worst_db"]
        print(
            f"[{model_name}] round {round_index} best {best_label}: NMSE {best_metrics['nmse_db']:.3f} | "
            f"point {best_metrics['point_lower_db']:.3f}/{best_metrics['point_upper_db']:.3f}/"
            f"{best_metrics['point_worst_db']:.3f} (gain {improvement:+.3f} dB)"
        )
        if improvement <= args.minimum_gain:
            break
        current_state = best_state

    with (output_dir / "search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(current_state, output_dir / "best.pth")

    # Independent reconstruction/forward check of the serialized winner.
    final_model = build_wrapper(
        evaluator, ps_mod, current_state, model_name, args.grid, amp_max, offsets
    ).eval()
    pred_iq, target_iq, _, _ = evaluator.evaluate_full_sequence(final_model, x_val, y_val)
    final_metrics = numpy_metrics(
        evaluator,
        metrics_mod,
        evaluator.tensor_to_complex(pred_iq),
        evaluator.tensor_to_complex(target_iq),
        args.nperseg,
    )
    if candidate_key(final_metrics, nmse_limit)[0]:
        raise RuntimeError(f"Serialized winner violates NMSE: {final_metrics}")
    summary = {
        "model": model_name,
        "source_checkpoint": str(checkpoint),
        "nmse_limit_db": nmse_limit,
        **final_metrics,
        "checkpoint": str(output_dir / "best.pth"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_LAYOUTS, default=list(MODEL_LAYOUTS))
    parser.add_argument(
        "--checkpoint-root",
        default="eval_outputs_direct/aclr_ref_convention/fir_pointsweep_rand",
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
    parser.add_argument("--tag", default="active_bin_tables")
    parser.add_argument("--mac-limit", type=int, default=None,
                        help="allow reconstructed MAC up to this ceiling instead of == model label")
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--active-candidates", type=int, default=2)
    parser.add_argument(
        "--bin-source",
        choices=["train", "val"],
        default="val",
        help="sequence used only to identify active frequency-bin hyperparameters",
    )
    parser.add_argument("--bin-separation", type=int, default=4)
    parser.add_argument(
        "--spectral-weights", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0]
    )
    parser.add_argument("--trust-weight", type=float, default=1e-4)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-iters", type=int, default=30)
    parser.add_argument("--lbfgs-history", type=int, default=20)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-9)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1e-12)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--minimum-gain", type=float, default=1e-4)
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.grid < 1 or args.rounds < 1 or args.active_candidates < 1:
        parser.error("--grid, --rounds and --active-candidates must be positive")
    if args.bin_separation < 1 or args.nperseg < 2:
        parser.error("--bin-separation must be positive and --nperseg must be at least 2")
    if args.alpha_step <= 0.0 or args.alpha_max < 0.0:
        parser.error("--alpha-step must be positive and --alpha-max non-negative")
    if not 0.0 < args.split < 1.0:
        parser.error("--split must lie strictly between zero and one")
    if args.trust_weight < 0.0 or any(weight < 0.0 for weight in args.spectral_weights):
        parser.error("trust and spectral weights must be non-negative")
    if args.lbfgs_iters < 1 or args.lbfgs_history < 1 or args.lbfgs_lr <= 0.0:
        parser.error("L-BFGS iteration/history counts and learning rate must be positive")
    return args


def main():
    args = parse_args()
    if args.checkpoint is not None and len(args.models) != 1:
        raise SystemExit("--checkpoint requires exactly one --models entry")
    if args.offsets is not None and len(args.models) != 1:
        raise SystemExit("--offsets requires exactly one --models entry")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("aclr_ref_metrics", "steps/aclr_ref_metrics.py")
    if not np.isclose(args.fs_mhz * 1e6, metrics_mod.FS_HZ, rtol=0.0, atol=1e-6):
        raise SystemExit(
            f"--fs-mhz={args.fs_mhz} conflicts with the fixed evaluator rate "
            f"{metrics_mod.FS_HZ / 1e6} MHz"
        )
    ps_mod = load_module("aclr_phase_spline_experiment", "steps/aclr_phase_spline_experiment.py")
    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_index = int(len(features) * args.split)
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))
    arrays = (
        features[:split_index],
        targets[:split_index],
        features[split_index:],
        targets[split_index:],
        amp_max,
    )

    config_dir = OUT_ROOT / args.tag
    config_dir.mkdir(parents=True, exist_ok=True)
    with (config_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    summaries = [
        run_model(args, name, evaluator, metrics_mod, ps_mod, arrays) for name in args.models
    ]
    with (config_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
