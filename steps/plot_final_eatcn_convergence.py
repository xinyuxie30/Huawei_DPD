"""Compose the final real NMSE comparison with the 2492→2284 pipeline curve."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "steps" / "baseline_convergence_2284"
PIPELINE_DIR = PROJECT_ROOT / "steps" / "ea_tcn_2492_to_2284"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_optional(value):
    return float(value) if value not in (None, "") else np.nan


def main():
    baseline_rows = read_csv(BASELINE_DIR / "nmse_history.csv")
    pipeline_rows = read_csv(PIPELINE_DIR / "pipeline_history.csv")
    pipeline_metadata = json.loads((PIPELINE_DIR / "metadata.json").read_text(encoding="utf-8"))

    main_epochs = np.asarray([int(row["epoch"]) for row in pipeline_rows], dtype=int)
    main_values = np.asarray(
        [float(row["base_2492_validation_nmse_db"]) for row in pipeline_rows], dtype=float
    )
    deploy_values = np.asarray(
        [parse_optional(row["deploy_2284_validation_nmse_db"]) for row in pipeline_rows],
        dtype=float,
    )
    deploy_indices = np.flatnonzero(np.isfinite(deploy_values))
    if len(deploy_indices) != 1:
        raise RuntimeError("Expected exactly one deployable 2284 LS point")
    deploy_source_epoch = int(main_epochs[deploy_indices[0]])
    deploy_nmse = float(deploy_values[deploy_indices[0]])
    deploy_plot_epoch = int(main_epochs[-1] + 4)

    baseline_epochs = np.asarray([int(row["epoch"]) for row in baseline_rows], dtype=int)
    series = {
        "memory_poly": np.asarray([float(row["memory_poly"]) for row in baseline_rows]),
        "plain_tcn": np.asarray([float(row["plain_tcn"]) for row in baseline_rows]),
        "no_attention": np.asarray([float(row["no_attention"]) for row in baseline_rows]),
        "no_fir": np.asarray([float(row["no_fir"]) for row in baseline_rows]),
    }

    colors = {
        "ea_tcn": "#0067B1",
        "memory_poly": "#D55E00",
        "plain_tcn": "#009E73",
        "no_attention": "#8E44AD",
        "no_fir": "#667078",
        "target": "#C62828",
    }
    fig, ax = plt.subplots(figsize=(10.8, 5.8), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        main_epochs,
        main_values,
        color=colors["ea_tcn"],
        linewidth=2.7,
        label="EA-TCN(2284)",
        zorder=5,
    )
    ax.plot(
        baseline_epochs,
        series["memory_poly"],
        color=colors["memory_poly"],
        linewidth=1.7,
        linestyle="--",
        label="Memory polynomial (2280)",
        alpha=0.96,
    )
    ax.plot(
        baseline_epochs,
        series["plain_tcn"],
        color=colors["plain_tcn"],
        linewidth=1.7,
        linestyle="-.",
        label="TCN (2276)",
        alpha=0.96,
    )
    ax.plot(
        baseline_epochs,
        series["no_attention"],
        color=colors["no_attention"],
        linewidth=1.8,
        linestyle=(0, (5, 2)),
        label="EA-TCN no attention (2284)",
        alpha=0.96,
    )
    ax.plot(
        baseline_epochs,
        series["no_fir"],
        color=colors["no_fir"],
        linewidth=1.7,
        linestyle=(0, (2, 2)),
        label="EA-TCN no FIR (2292)",
        alpha=0.96,
    )

    ax.axhline(
        -40.0,
        color=colors["target"],
        linewidth=1.45,
        linestyle=":",
        label="Target -40 dB",
        zorder=2,
    )
    ax.plot(
        [deploy_source_epoch, deploy_plot_epoch],
        [main_values[deploy_indices[0]], deploy_nmse],
        color=colors["ea_tcn"],
        linewidth=1.5,
        linestyle=(0, (2, 2)),
        zorder=5,
    )
    ax.scatter(
        [deploy_plot_epoch],
        [deploy_nmse],
        marker="*",
        s=155,
        color=colors["ea_tcn"],
        edgecolors="white",
        linewidths=0.9,
        label="FIR pruning + LS",
        zorder=8,
    )
    ax.annotate(
        f"{deploy_nmse:.2f} dB",
        xy=(deploy_plot_epoch, deploy_nmse),
        xytext=(7, -2),
        textcoords="offset points",
        ha="left",
        va="center",
        color=colors["ea_tcn"],
        fontsize=9.5,
        fontweight="semibold",
    )

    endpoint_specs = [
        ("memory_poly", -2, 7),
        ("plain_tcn", -2, -8),
        ("no_attention", -2, -10),
        ("no_fir", -2, 8),
    ]
    for key, x_offset, y_offset in endpoint_specs:
        values = series[key]
        ax.scatter(
            [baseline_epochs[-1]],
            [values[-1]],
            s=18,
            color=colors[key],
            zorder=6,
        )
        ax.annotate(
            f"{values[-1]:.2f}",
            xy=(baseline_epochs[-1], values[-1]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.2,
            color=colors[key],
        )

    ax.set_title("EA-TCN", fontsize=18, pad=12, fontweight="semibold")
    ax.set_xlabel("Epoch", fontsize=11.5)
    ax.set_ylabel("Validation NMSE (dB)", fontsize=11.5)
    ax.set_xlim(0, deploy_plot_epoch + 13)
    ax.set_ylim(-40.75, -13.6)
    ax.grid(True, color="#C7CCD1", linestyle=":", linewidth=0.75, alpha=0.75)
    ax.tick_params(axis="both", labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_color("#5B6168")
        spine.set_linewidth(0.85)
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=8.8,
        frameon=True,
        framealpha=0.96,
        edgecolor="#CBD0D5",
        columnspacing=1.4,
        handlelength=3.0,
    )
    fig.tight_layout()

    png_path = BASELINE_DIR / "ea_tcn_final_convergence.png"
    svg_path = BASELINE_DIR / "ea_tcn_final_convergence.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    max_epoch = max(int(main_epochs[-1]), int(baseline_epochs[-1]))
    combined_path = BASELINE_DIR / "nmse_history_final_pipeline.csv"
    baseline_by_epoch = {int(row["epoch"]): row for row in baseline_rows}
    pipeline_by_epoch = {int(row["epoch"]): row for row in pipeline_rows}
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "ea_tcn_2284_pipeline",
                "memory_polynomial_2280",
                "tcn_2276",
                "ea_tcn_no_attention_2284",
                "ea_tcn_no_fir_2292",
                "deploy_2284_pruned_ls",
            ]
        )
        for epoch in range(max_epoch + 1):
            base_row = baseline_by_epoch.get(epoch, {})
            pipeline_row = pipeline_by_epoch.get(epoch, {})
            writer.writerow(
                [
                    epoch,
                    pipeline_row.get("base_2492_validation_nmse_db", ""),
                    base_row.get("memory_poly", ""),
                    base_row.get("plain_tcn", ""),
                    base_row.get("no_attention", ""),
                    base_row.get("no_fir", ""),
                    pipeline_row.get("deploy_2284_validation_nmse_db", ""),
                ]
            )

    summary_rows = read_csv(BASELINE_DIR / "summary.csv")
    original_by_key = {row["model"]: row for row in summary_rows}
    summary_path = BASELINE_DIR / "summary_final_pipeline.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "model",
            "label",
            "real_multipliers",
            "best_epoch",
            "best_validation_nmse_db",
            "deployment_validation_nmse_db",
            "training_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "model": "ea_tcn",
                "label": "EA-TCN(2284)",
                "real_multipliers": 2284,
                "best_epoch": pipeline_metadata["base_best_epoch"],
                "best_validation_nmse_db": f"{pipeline_metadata['base_best_validation_nmse_db']:.8f}",
                "deployment_validation_nmse_db": f"{deploy_nmse:.8f}",
                "training_source": "2492 base training -> FIR43 pruning -> train-only LS",
            }
        )
        label_map = {
            "memory_poly": "Memory polynomial (2280)",
            "plain_tcn": "TCN (2276)",
            "no_attention": "EA-TCN no attention (2284)",
            "no_fir": "EA-TCN no FIR (2292)",
        }
        for key in ("memory_poly", "plain_tcn", "no_attention", "no_fir"):
            row = original_by_key[key]
            writer.writerow(
                {
                    "model": key,
                    "label": label_map[key],
                    "real_multipliers": row["real_multipliers"],
                    "best_epoch": row["best_epoch"],
                    "best_validation_nmse_db": row["best_validation_nmse_db"],
                    "deployment_validation_nmse_db": row["post_ls_validation_nmse_db"],
                    "training_source": "capacity-matched GPU baseline run",
                }
            )

    final_metadata = {
        "title": "EA-TCN",
        "main_legend": "EA-TCN(2284)",
        "main_curve_source": str(PIPELINE_DIR / "pipeline_history.csv"),
        "main_curve_interpretation": (
            "2492-real-multiplier FIR95 base training trajectory; deployment point is "
            "the 2284-real-multiplier FIR43 model after train-only LS"
        ),
        "deployment_validation_nmse_db": deploy_nmse,
        "target_nmse_db": -40.0,
        "baseline_curve_source": str(BASELINE_DIR / "nmse_history.csv"),
        "png": str(png_path),
        "svg": str(svg_path),
        "combined_history": str(combined_path),
        "summary": str(summary_path),
    }
    metadata_path = BASELINE_DIR / "metadata_final_pipeline.json"
    metadata_path.write_text(
        json.dumps(final_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f">>> PNG: {png_path}")
    print(f">>> SVG: {svg_path}")
    print(f">>> Combined history: {combined_path}")
    print(f">>> Summary: {summary_path}")
    print(f">>> Deployment NMSE: {deploy_nmse:.4f} dB")


if __name__ == "__main__":
    main()
