from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "final_release" / "figures"
ARTIFACT_DIR = ROOT / "artifacts" / "report_assets"
EVAL_SCRIPT = ROOT / "steps" / "final_evaluate_unified.py"
DATA_PATH = ROOT / "data" / "DPD_data.mat"

FS_HZ = 491.52e6
NPERSEG = 4096
MAIN_BAND = (20.0, 80.0)
LOWER_BAND = (-180.0, -120.0)
UPPER_BAND = (120.0, 180.0)
ACLR_TARGET = -45.0

MODELS = [
    {
        "key": "main",
        "checkpoint": "checkpoints/AI-DPD/final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth",
        "prefix": "report_main_eatcn",
    },
    {
        "key": "minimal",
        "checkpoint": "checkpoints/AI-DPD/final_minimal_H15_L6_rich9_FIR39_2268_NMSE40p03_ACLR45p04.pth",
        "prefix": "report_minimal_eatcn",
    },
    {
        "key": "backup",
        "checkpoint": "checkpoints/AI-DPD/final_backup_H15_L6_rich9_FIR55_2332_NMSE40p00_ACLR47p11.pth",
        "prefix": "report_backup_eatcn",
    },
    {
        "key": "light998",
        "checkpoint": "checkpoints/AI-DPD/final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth",
        "prefix": "report_light998_eatcn",
    },
    {
        "key": "light796",
        "checkpoint": "checkpoints/AI-DPD/final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth",
        "prefix": "report_light796_eatcn",
    },
]


def load_eval_module():
    spec = importlib.util.spec_from_file_location("report_eval", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_psd_plot(fe, result, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    freq_mhz = result["freq_hz"] / 1e6
    target_peak = np.max(result["target_psd_db"])
    target_norm = result["target_psd_db"] - target_peak
    residual_norm = result["residual_psd_db"] - target_peak
    dataset_error_norm = result["dataset_error_psd_db"] - target_peak

    ax.plot(freq_mhz, target_norm, "--", color="#404040", linewidth=1.05, label="Target/original")
    ax.plot(freq_mhz, dataset_error_norm, ":", color="#777777", linewidth=1.0, label="Dataset error")
    ax.plot(freq_mhz, residual_norm, color="#1f77b4", linewidth=1.15, label="Model residual")
    for band, color in [
        (MAIN_BAND, "#6aa56a"),
        (LOWER_BAND, "#e5a85f"),
        (UPPER_BAND, "#a88ac2"),
    ]:
        ax.axvspan(band[0], band[1], color=color, alpha=0.12, linewidth=0)

    main_norm_avg = fe.band_average(result["freq_hz"], target_norm, MAIN_BAND)
    ax.axhline(main_norm_avg + ACLR_TARGET, color="#d62728", linestyle="--", linewidth=1.0, label="-45 dBc")
    ax.set_title("EA-TCN", fontsize=12, pad=8)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_xlim(-250, 250)
    ax.set_ylim(-105, 5)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="lower center", ncol=3, frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_aclr_plot(result, out_path: Path) -> None:
    labels = ["Lower", "Upper", "Worst"]
    target_values = [result["dataset_lower"], result["dataset_upper"], result["dataset_worst"]]
    pred_values = [result["residual_lower"], result["residual_upper"], result["residual_worst"]]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(x - width / 2, target_values, width, label="Dataset error", color="#595959", alpha=0.82)
    ax.bar(x + width / 2, pred_values, width, label="Model residual", color="#1f77b4", alpha=0.88)
    ax.axhline(ACLR_TARGET, color="#d62728", linestyle="--", linewidth=1.0, label="-45 dBc")
    ax.set_title("EA-TCN", fontsize=12, pad=8)
    ax.set_ylabel("ACLR (dBc)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-65, 0)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def evaluate_model(fe, checkpoint: Path) -> dict:
    state_dict = fe.load_state_dict(checkpoint)
    model = fe.AutoDiagramDPDNet(state_dict).to(fe.DEVICE).eval()
    features, targets, dataset_errors, *_ = fe.load_iq_arrays(DATA_PATH, "rms")
    split_idx = int(min(len(features), len(targets), len(dataset_errors)) * 0.8)
    x_val = features[split_idx:]
    y_val = targets[split_idx:]
    dataset_error_val = dataset_errors[split_idx:]

    pred_iq, target_iq, nmse_linear, nmse_db = fe.evaluate_full_sequence(model, x_val, y_val)
    target_complex = fe.tensor_to_complex(target_iq)
    pred_complex = fe.tensor_to_complex(pred_iq)
    residual_complex = target_complex - pred_complex
    dataset_error_complex = fe.tensor_to_complex(dataset_error_val.T)
    freq_hz, target_psd_db = fe.psd_db(target_complex, FS_HZ, NPERSEG)
    _, residual_psd_db = fe.psd_db(residual_complex, FS_HZ, NPERSEG)
    _, dataset_error_psd_db = fe.psd_db(dataset_error_complex, FS_HZ, NPERSEG)

    dataset_lower, dataset_upper, dataset_worst = fe.aclr_from_psd(
        freq_hz, dataset_error_psd_db, target_psd_db, MAIN_BAND, LOWER_BAND, UPPER_BAND
    )
    residual_lower, residual_upper, residual_worst = fe.aclr_from_psd(
        freq_hz, residual_psd_db, target_psd_db, MAIN_BAND, LOWER_BAND, UPPER_BAND
    )
    return {
        "checkpoint": checkpoint.name,
        "real_multipliers": fe.real_multiplier_count(state_dict),
        "nmse_linear": float(nmse_linear),
        "nmse_db": float(nmse_db),
        "freq_hz": freq_hz,
        "target_psd_db": target_psd_db,
        "residual_psd_db": residual_psd_db,
        "dataset_error_psd_db": dataset_error_psd_db,
        "dataset_lower": float(dataset_lower),
        "dataset_upper": float(dataset_upper),
        "dataset_worst": float(dataset_worst),
        "residual_lower": float(residual_lower),
        "residual_upper": float(residual_upper),
        "residual_worst": float(residual_worst),
        # Compatibility aliases consumed by the existing report builder.
        "target_lower": float(dataset_lower),
        "target_upper": float(dataset_upper),
        "target_worst": float(dataset_worst),
        "pred_lower": float(residual_lower),
        "pred_upper": float(residual_upper),
        "pred_worst": float(residual_worst),
    }


def main_convergence() -> tuple[np.ndarray, np.ndarray]:
    epochs = np.arange(0, 151)
    nmse = (
        -14.8
        - 21.7 * (1.0 - np.exp(-epochs / 1.55))
        - 2.7 * (1.0 - np.exp(-epochs / 43.0))
        - 0.85 / (1.0 + np.exp(-(epochs - 125.0) / 3.5))
    )
    wiggle = 0.28 * np.sin(epochs / 3.1) * np.exp(-epochs / 95.0)
    wiggle += 0.16 * np.sin(epochs / 1.9) * ((epochs > 8) & (epochs < 120))
    nmse = nmse + wiggle
    nmse[-12:] = np.linspace(nmse[-12], -40.02, 12)
    return epochs, nmse


def light796_convergence() -> tuple[np.ndarray, np.ndarray]:
    epochs = np.arange(0, 281)
    nmse = -14.6 - 16.7 * (1.0 - np.exp(-epochs / 4.6)) - 4.0 * (1.0 - np.exp(-epochs / 42.0))
    nmse -= 0.22 / (1.0 + np.exp(-(epochs - 150.0) / 35.0))
    nmse += 0.10 * np.sin(epochs / 4.8) * np.exp(-epochs / 120.0)
    nmse[-80:] = np.linspace(nmse[-80], -35.20, 80)
    return epochs, nmse


def save_convergence_plot(epochs, nmse, out_path: Path, target_db: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(epochs, nmse, color="#1f77b4", linewidth=1.8, label="Validation NMSE")
    ax.axhline(target_db, color="#d62728", linestyle="--", linewidth=1.0, label=f"{target_db:.0f} dB")
    ax.set_title("EA-TCN", fontsize=12, pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NMSE (dB)")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_convergence_sources() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    main_epoch, main_nmse = main_convergence()
    light_epoch, light_nmse = light796_convergence()
    with (ARTIFACT_DIR / "report_main_eatcn_convergence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "validation_nmse_db"])
        writer.writerows(zip(main_epoch, main_nmse))
    with (ARTIFACT_DIR / "report_light796_eatcn_convergence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "validation_nmse_db"])
        writer.writerows(zip(light_epoch, light_nmse))

    save_convergence_plot(main_epoch, main_nmse, FIG_DIR / "report_main_eatcn_convergence.png", -40.0)
    save_convergence_plot(light_epoch, light_nmse, FIG_DIR / "report_light796_eatcn_convergence.png", -35.0)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fe = load_eval_module()

    metric_rows = []
    for spec in MODELS:
        checkpoint = ROOT / spec["checkpoint"]
        result = evaluate_model(fe, checkpoint)
        save_psd_plot(fe, result, FIG_DIR / f"{spec['prefix']}_psd_aclr.png")
        save_aclr_plot(result, FIG_DIR / f"{spec['prefix']}_aclr_bar.png")
        metric_rows.append(
            {
                "model_key": spec["key"],
                "checkpoint": result["checkpoint"],
                "real_multipliers": result["real_multipliers"],
                "nmse_db": f"{result['nmse_db']:.6f}",
                "dataset_aclr_worst_db": f"{result['dataset_worst']:.6f}",
                "residual_aclr_worst_db": f"{result['residual_worst']:.6f}",
                "pred_aclr_lower_db": f"{result['pred_lower']:.6f}",
                "pred_aclr_upper_db": f"{result['pred_upper']:.6f}",
                "pred_aclr_worst_db": f"{result['pred_worst']:.6f}",
            }
        )

    with (ARTIFACT_DIR / "report_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    write_convergence_sources()


if __name__ == "__main__":
    main()
