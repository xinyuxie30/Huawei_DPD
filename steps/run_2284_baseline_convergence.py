"""Train capacity-matched AI-DPD baselines and plot real NMSE histories.

The runner intentionally records validation NMSE after every epoch.  It is
separate from ``tools/refresh_report_assets.py`` because that report helper
uses illustrative (synthetic) convergence curves rather than training logs.

Models compared under the same data split, optimizer, batches and seed:

* EA-TCN: the 2284-real-multiplier final architecture.
* Memory polynomial: six odd orders and 95 memory taps (2280 multipliers).
* Plain TCN: raw I/Q, six K=7 TCN blocks, no prior/attention/FIR (2276).
* No attention: the 135-multiplier gate is replaced by a conventional
  depthwise K=9 convolution (2284).
* No FIR: per-layer kernels are enlarged while preserving both the 2284-class
  compute budget and the original end-to-end receptive field (2292).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DEFAULT_DATA = PROJECT_ROOT / "data" / "DPD_data.mat"
DEFAULT_OUTPUT = SCRIPT_DIR / "baseline_convergence_2284"


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd_ablation_module", AI_DPD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {AI_DPD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aidpd_ablation_module"] = module
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


class FramedDataset(Dataset):
    def __init__(self, x_np, y_np, frame_length, stride_length, dtype):
        self.x = torch.as_tensor(x_np, dtype=dtype)
        self.y = torch.as_tensor(y_np, dtype=dtype)
        self.frame_length = int(frame_length)
        self.stride_length = int(stride_length)
        self.num_frames = (len(self.x) - self.frame_length) // self.stride_length + 1
        if self.num_frames <= 0:
            raise ValueError("frame_length exceeds the training sequence")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        start = index * self.stride_length
        end = start + self.frame_length
        return self.x[start:end].T, self.y[start:end].T


class ConventionalConvControl(nn.Module):
    """Compute-matched non-attentive control replacing the 135-MAC gate."""

    def __init__(self, channels, dtype):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=9,
            padding=4,
            groups=channels,
            bias=False,
            dtype=dtype,
        )
        nn.init.zeros_(self.conv.weight)

    def forward(self, x, *_args, **_kwargs):
        return x + self.conv(x)


class PlainTCN(nn.Module):
    """Raw-I/Q residual TCN without polynomial prior, attention, or FIR."""

    def __init__(self, g_real, g_imag, dtype):
        super().__init__()
        hidden = 16
        self.hidden = hidden
        self.g_real = nn.Parameter(torch.tensor(g_real, dtype=dtype))
        self.g_imag = nn.Parameter(torch.tensor(g_imag, dtype=dtype))
        self.in_conv = nn.Conv1d(2, hidden, 1, bias=False, dtype=dtype)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList()
        # Raw-I/Q TCN matches the full rich9+TCN+FIR 297-sample context.
        for kernel, dilation in zip([9, 9, 5, 9, 5, 5], [1, 2, 4, 8, 16, 32]):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden,
                        hidden,
                        kernel,
                        padding=(kernel // 2) * dilation,
                        dilation=dilation,
                        groups=hidden,
                        bias=False,
                        dtype=dtype,
                    ),
                    nn.Conv1d(hidden, hidden, 1, bias=False, dtype=dtype),
                    nn.SiLU(),
                )
            )
        self.out_conv = nn.Conv1d(hidden, 2, 1, bias=False, dtype=dtype)
        nn.init.zeros_(self.out_conv.weight)

    @staticmethod
    def multiplier_count():
        # linear skip + raw-I/Q projection + variable-kernel blocks + output
        return 4 + 2 * 16 + 42 * 16 + 6 * 16 * 16 + 2 * 16

    def forward(self, x_iq):
        i_x = x_iq[:, 0:1]
        q_x = x_iq[:, 1:2]
        feat = self.act(self.in_conv(x_iq))
        for block in self.blocks:
            feat = feat + block(feat)
        residual = self.out_conv(feat)
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real
        return torch.cat([lin_i, lin_q], dim=1) + residual


class MemoryPolynomial(nn.Module):
    """Complex memory polynomial with six odd orders and 95 taps per order."""

    def __init__(
        self,
        g_real,
        g_imag,
        dtype,
        basis_scales,
        orders=(1, 3, 5, 7, 9, 11),
        taps=95,
    ):
        super().__init__()
        self.orders = tuple(int(order) for order in orders)
        self.taps = int(taps)
        if self.taps % 2 != 1:
            raise ValueError("Memory-polynomial taps must be odd")
        real = torch.zeros(len(self.orders), self.taps, dtype=dtype)
        imag = torch.zeros(len(self.orders), self.taps, dtype=dtype)
        center = self.taps // 2
        scales = torch.as_tensor(basis_scales, dtype=dtype)
        if scales.shape != (len(self.orders),):
            raise ValueError("basis_scales must provide one value per polynomial order")
        self.register_buffer("basis_scales", scales)
        # Scaling is folded into coefficients, so it adds no inference MAC.
        real[0, center] = float(g_real) * float(scales[0])
        imag[0, center] = float(g_imag) * float(scales[0])
        self.coeff_real = nn.Parameter(real)
        self.coeff_imag = nn.Parameter(imag)

    def multiplier_count(self):
        return 4 * len(self.orders) * self.taps

    def forward(self, x_iq):
        i_x = x_iq[:, 0:1]
        q_x = x_iq[:, 1:2]
        amp2 = i_x.square() + q_x.square() + 1e-12
        feat_i = []
        feat_q = []
        for index, order in enumerate(self.orders):
            scale = amp2.pow((order - 1) // 2)
            feat_i.append(i_x * scale / self.basis_scales[index])
            feat_q.append(q_x * scale / self.basis_scales[index])
        basis_i = torch.cat(feat_i, dim=1)
        basis_q = torch.cat(feat_q, dim=1)
        real = self.coeff_real.unsqueeze(1)
        imag = self.coeff_imag.unsqueeze(1)
        padding = self.taps // 2
        groups = len(self.orders)
        rr = F.conv1d(basis_i, real, padding=padding, groups=groups)
        qi = F.conv1d(basis_q, imag, padding=padding, groups=groups)
        ri = F.conv1d(basis_i, imag, padding=padding, groups=groups)
        qr = F.conv1d(basis_q, real, padding=padding, groups=groups)
        return torch.cat([(rr - qi).sum(dim=1, keepdim=True), (ri + qr).sum(dim=1, keepdim=True)], dim=1)


@dataclass
class Experiment:
    key: str
    label: str
    model: nn.Module
    multipliers: int
    optimizer: torch.optim.Optimizer | None = None
    scheduler: object | None = None
    history: list[float] | None = None
    loss_history: list[float] | None = None
    lr_history: list[float] | None = None
    best_nmse_db: float = math.inf
    best_epoch: int = 0
    elapsed_seconds: float = 0.0
    post_ls_nmse_db: float = math.nan


def load_split(data_path):
    _, _, x_c, y_c = aidpd.load_normalized_data(data_path)
    x_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    y_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)
    length = min(len(x_np), len(y_np))
    split = int(length * 0.8)
    x_train = x_np[:split]
    y_train = y_np[:split]
    x_val = x_np[split:length]
    y_val = y_np[split:length]
    g_opt = np.vdot(x_c[:split], y_c[:split]) / np.vdot(x_c[:split], x_c[:split])
    return x_train, y_train, x_val, y_val, float(np.real(g_opt)), float(np.imag(g_opt))


def replace_no_fir_blocks(model, dtype):
    """Match full-model receptive field after removing the 43-tap FIR.

    With dilations [1,2,4,8,16,32], kernels [11,7,5,9,5,5] have a total
    receptive field of 295 samples: the original K=5 TCN's 253 samples plus
    the FIR's additional 42 samples.
    """
    hidden = int(model.hidden)
    kernels = [11, 7, 5, 9, 5, 5]
    blocks = []
    for kernel, dilation in zip(kernels, model.dilations):
        blocks.append(
            nn.Sequential(
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel,
                    padding=(kernel // 2) * dilation,
                    dilation=dilation,
                    groups=hidden,
                    bias=False,
                    dtype=dtype,
                ),
                nn.Conv1d(hidden, hidden, 1, bias=False, dtype=dtype),
                nn.SiLU(),
            )
        )
    model.tcn_blocks = nn.ModuleList(blocks)


def make_experiments(g_real, g_imag, dtype, selected, mp_scales):
    experiments = []

    if "ea_tcn" in selected:
        config = aidpd.preset_config("rich9_h15_l6_fir43_2284")
        model = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(dtype=dtype)
        experiments.append(
            Experiment("ea_tcn", "EA-TCN", model, aidpd.real_multiplier_count_from_config(config))
        )

    if "memory_poly" in selected:
        model = MemoryPolynomial(g_real, g_imag, dtype=dtype, basis_scales=mp_scales)
        experiments.append(Experiment("memory_poly", "Memory polynomial", model, model.multiplier_count()))

    if "plain_tcn" in selected:
        model = PlainTCN(g_real, g_imag, dtype=dtype)
        experiments.append(Experiment("plain_tcn", "Plain TCN", model, model.multiplier_count()))

    if "no_attention" in selected:
        config = aidpd.preset_config("rich9_h15_l6_fir43_2284")
        config["preset"] = "rich9_h15_l6_no_attention_conv_control_2284"
        model = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(dtype=dtype)
        model.attn = ConventionalConvControl(15, dtype)
        count = aidpd.real_multiplier_count_from_config(config)
        experiments.append(Experiment("no_attention", "EA-TCN w/o attention", model, count))

    if "no_fir" in selected:
        config = aidpd.preset_config("rich9_h15_l6_fir43_2284")
        config["preset"] = "rich9_h15_l6_variable_kernel_no_fir_2292"
        config["output_fir_taps"] = 1
        model = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(dtype=dtype)
        replace_no_fir_blocks(model, dtype)
        count = aidpd.real_multiplier_count_from_config(config) + 15 * (42 - 30)
        experiments.append(Experiment("no_fir", "EA-TCN w/o FIR", model, count))

    return experiments


def evaluate_nmse_db(model, x_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor)
        nmse = torch.sum((pred - y_tensor).square()) / torch.sum(y_tensor.square())
    return float(10.0 * torch.log10(nmse).item())


def memory_polynomial_scales(x_train, orders=(1, 3, 5, 7, 9, 11)):
    x_complex = x_train[:, 0] + 1j * x_train[:, 1]
    amplitude = np.abs(x_complex)
    scales = []
    for order in orders:
        basis = x_complex * amplitude ** (order - 1)
        scales.append(max(float(np.sqrt(np.mean(np.abs(basis) ** 2))), 1e-12))
    return np.asarray(scales, dtype=np.float64)


def history_csv_path(output_dir):
    return output_dir / "nmse_history.csv"


def write_history_csv(output_dir, experiments):
    max_len = max(len(exp.history or []) for exp in experiments)
    path = history_csv_path(output_dir)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *[exp.key for exp in experiments]])
        for epoch in range(max_len):
            row = [epoch]
            for exp in experiments:
                values = exp.history or []
                row.append(f"{values[epoch]:.8f}" if epoch < len(values) else "")
            writer.writerow(row)
    return path


def write_long_history_csv(output_dir, experiments):
    path = output_dir / "training_history.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "epoch",
                "validation_nmse_db",
                "train_mse",
                "learning_rate",
                "real_multipliers",
            ],
        )
        writer.writeheader()
        for exp in experiments:
            for epoch, nmse_db in enumerate(exp.history or []):
                loss = exp.loss_history[epoch]
                writer.writerow(
                    {
                        "model": exp.key,
                        "epoch": epoch,
                        "validation_nmse_db": f"{nmse_db:.8f}",
                        "train_mse": "" if not math.isfinite(loss) else f"{loss:.10e}",
                        "learning_rate": f"{exp.lr_history[epoch]:.10e}",
                        "real_multipliers": exp.multipliers,
                    }
                )
    return path


def write_summary(output_dir, experiments):
    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "label",
                "real_multipliers",
                "trainable_parameters",
                "best_epoch",
                "best_validation_nmse_db",
                "final_validation_nmse_db",
                "post_ls_validation_nmse_db",
                "training_seconds",
            ],
        )
        writer.writeheader()
        for exp in experiments:
            writer.writerow(
                {
                    "model": exp.key,
                    "label": exp.label,
                    "real_multipliers": exp.multipliers,
                    "trainable_parameters": sum(p.numel() for p in exp.model.parameters()),
                    "best_epoch": exp.best_epoch,
                    "best_validation_nmse_db": f"{exp.best_nmse_db:.8f}",
                    "final_validation_nmse_db": f"{exp.history[-1]:.8f}",
                    "post_ls_validation_nmse_db": (
                        f"{exp.post_ls_nmse_db:.8f}" if math.isfinite(exp.post_ls_nmse_db) else ""
                    ),
                    "training_seconds": f"{exp.elapsed_seconds:.3f}",
                }
            )
    return path


def plot_histories(output_dir, experiments, target_db):
    colors = {
        "ea_tcn": "#0067B1",
        "memory_poly": "#D55E00",
        "plain_tcn": "#009E73",
        "no_attention": "#8E44AD",
        "no_fir": "#7F8C8D",
    }
    styles = {
        "ea_tcn": "-",
        "memory_poly": "--",
        "plain_tcn": "-.",
        "no_attention": (0, (5, 2)),
        "no_fir": (0, (2, 2)),
    }
    fig, ax = plt.subplots(figsize=(10.6, 5.7))
    max_epoch = max(len(exp.history) - 1 for exp in experiments)
    for exp in experiments:
        epochs = np.arange(len(exp.history))
        linewidth = 2.5 if exp.key == "ea_tcn" else 1.8
        ax.plot(
            epochs,
            exp.history,
            label=f"{exp.label} ({exp.multipliers})",
            color=colors[exp.key],
            linestyle=styles[exp.key],
            linewidth=linewidth,
        )
        ax.scatter(
            [exp.best_epoch],
            [exp.best_nmse_db],
            s=25 if exp.key == "ea_tcn" else 18,
            color=colors[exp.key],
            zorder=4,
        )
        display_nmse = exp.post_ls_nmse_db if math.isfinite(exp.post_ls_nmse_db) else exp.best_nmse_db
        display_epoch = max_epoch + 1 if math.isfinite(exp.post_ls_nmse_db) else exp.best_epoch
        if math.isfinite(exp.post_ls_nmse_db):
            ax.scatter(
                [display_epoch],
                [display_nmse],
                marker="*",
                s=95,
                color=colors[exp.key],
                edgecolors="white",
                linewidths=0.7,
                zorder=6,
            )
        y_offset = -9 if exp.key == "ea_tcn" else (8 if exp.key == "no_attention" else 0)
        value_label = f"LS {display_nmse:.2f}" if math.isfinite(exp.post_ls_nmse_db) else f"{display_nmse:.2f}"
        ax.annotate(
            value_label,
            xy=(display_epoch, display_nmse),
            xytext=(6, y_offset),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=colors[exp.key],
        )
    if any(math.isfinite(exp.post_ls_nmse_db) for exp in experiments):
        ax.scatter([], [], marker="*", s=80, color="#555555", label="Post-training FIR LS")
    ax.axhline(target_db, color="#C62828", linestyle=":", linewidth=1.4, label=f"Target {target_db:.0f} dB")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NMSE (dB)")
    ax.set_title("EA-TCN vs. capacity-matched baselines")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="best", fontsize=9, ncol=2, frameon=True)
    ax.set_xlim(0, max_epoch + 13)
    fig.tight_layout()
    png_path = output_dir / "nmse_convergence_2284_baselines.png"
    svg_path = output_dir / "nmse_convergence_2284_baselines.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def fit_output_fir_ls(model, x_train, y_train, device, dtype):
    taps = int(getattr(model, "output_fir_taps", 1))
    if taps <= 1 or not hasattr(model, "forward_base"):
        return False
    radius = taps // 2
    x_tensor = torch.as_tensor(x_train.T, dtype=dtype, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        base = model.forward_base(x_tensor).squeeze(0).detach().cpu().numpy().T
    z = base[:, 0].astype(np.float64) + 1j * base[:, 1].astype(np.float64)
    target = y_train[:, 0].astype(np.float64) + 1j * y_train[:, 1].astype(np.float64)
    design = np.stack(
        [z[radius + offset : len(z) - radius + offset] for offset in range(-radius, radius + 1)],
        axis=1,
    )
    rhs = target[radius : len(target) - radius]
    coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=dtype, device=device))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=dtype, device=device))
    return True


def save_runner_state(path, epoch, experiments):
    state = {
        "epoch": epoch,
        "experiments": {
            exp.key: {
                "model": exp.model.state_dict(),
                "optimizer": exp.optimizer.state_dict(),
                "scheduler": exp.scheduler.state_dict(),
                "history": exp.history,
                "loss_history": exp.loss_history,
                "lr_history": exp.lr_history,
                "best_nmse_db": exp.best_nmse_db,
                "best_epoch": exp.best_epoch,
                "elapsed_seconds": exp.elapsed_seconds,
            }
            for exp in experiments
        },
    }
    torch.save(state, path)


def load_runner_state(path, experiments, device):
    state = torch.load(path, map_location=device, weights_only=False)
    by_key = {exp.key: exp for exp in experiments}
    if set(state["experiments"]) != set(by_key):
        raise ValueError("Resume variants do not match the saved runner state")
    for key, saved in state["experiments"].items():
        exp = by_key[key]
        exp.model.load_state_dict(saved["model"], strict=True)
        exp.optimizer.load_state_dict(saved["optimizer"])
        exp.scheduler.load_state_dict(saved["scheduler"])
        exp.history = list(saved["history"])
        exp.loss_history = list(saved["loss_history"])
        exp.lr_history = list(saved["lr_history"])
        exp.best_nmse_db = float(saved["best_nmse_db"])
        exp.best_epoch = int(saved["best_epoch"])
        exp.elapsed_seconds = float(saved["elapsed_seconds"])
    return int(state["epoch"])


def parse_args():
    parser = argparse.ArgumentParser(description="Run real 2284-budget NMSE baseline/ablation experiments")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-db", type=float, default=-40.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["ea_tcn", "memory_poly", "plain_tcn", "no_attention", "no_fir"],
        default=["ea_tcn", "memory_poly", "plain_tcn", "no_attention", "no_fir"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "runner_state.pth"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    x_train, y_train, x_val, y_val, g_real, g_imag = load_split(Path(args.data).resolve())

    dataset = FramedDataset(x_train, y_train, args.frame_length, args.stride_length, dtype)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    x_val_tensor = torch.as_tensor(x_val.T, dtype=dtype, device=device).unsqueeze(0)
    y_val_tensor = torch.as_tensor(y_val.T, dtype=dtype, device=device).unsqueeze(0)

    mp_scales = memory_polynomial_scales(x_train)
    experiments = make_experiments(g_real, g_imag, dtype, args.variants, mp_scales)
    for exp in experiments:
        exp.model = exp.model.to(device=device, dtype=dtype)
        exp.optimizer = torch.optim.AdamW(exp.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        exp.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            exp.optimizer,
            mode="min",
            factor=0.5,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
        exp.history = [evaluate_nmse_db(exp.model, x_val_tensor, y_val_tensor)]
        exp.loss_history = [math.nan]
        exp.lr_history = [args.lr]
        exp.best_nmse_db = exp.history[0]
        exp.best_epoch = 0
        if not args.resume:
            torch.save(exp.model.state_dict(), checkpoint_dir / f"{exp.key}_best.pth")

    start_epoch = 0
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(f"Resume state not found: {state_path}")
        start_epoch = load_runner_state(state_path, experiments, device)

    print(f">>> Device: {device}")
    if device.type == "cuda":
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Dtype: {args.dtype}")
    print(f">>> Train/val samples: {len(x_train)}/{len(x_val)}")
    print(f">>> Frames per epoch: {len(dataset)} | batch size: {args.batch_size}")
    for exp in experiments:
        print(
            f">>> {exp.key}: {exp.label} | real multipliers={exp.multipliers} | "
            f"parameters={sum(p.numel() for p in exp.model.parameters())} | initial={exp.history[0]:.3f} dB"
        )

    criterion = nn.MSELoss()
    total_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start = time.time()
        loss_sums = {exp.key: 0.0 for exp in experiments}
        loss_counts = {exp.key: 0 for exp in experiments}
        for exp in experiments:
            exp.model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            for exp in experiments:
                model_start = time.time()
                exp.optimizer.zero_grad(set_to_none=True)
                pred = exp.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(exp.model.parameters(), max_norm=args.grad_clip)
                exp.optimizer.step()
                loss_sums[exp.key] += float(loss.item())
                loss_counts[exp.key] += 1
                exp.elapsed_seconds += time.time() - model_start

        values = []
        for exp in experiments:
            nmse_db = evaluate_nmse_db(exp.model, x_val_tensor, y_val_tensor)
            exp.history.append(nmse_db)
            exp.scheduler.step(nmse_db)
            exp.loss_history.append(loss_sums[exp.key] / max(loss_counts[exp.key], 1))
            exp.lr_history.append(float(exp.optimizer.param_groups[0]["lr"]))
            if nmse_db < exp.best_nmse_db:
                exp.best_nmse_db = nmse_db
                exp.best_epoch = epoch
                torch.save(exp.model.state_dict(), checkpoint_dir / f"{exp.key}_best.pth")
            values.append(f"{exp.key}={nmse_db:.3f}")

        write_history_csv(output_dir, experiments)
        write_long_history_csv(output_dir, experiments)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_runner_state(state_path, epoch, experiments)
        print(
            f"Epoch [{epoch}/{args.epochs}] | {time.time() - epoch_start:.1f}s | "
            + " | ".join(values),
            flush=True,
        )

    # Re-load each best epoch.  For FIR-based models, reproduce the final
    # deployment step with a train-only closed-form FIR refit and report it as a
    # separate marker rather than pretending it is another gradient epoch.
    for exp in experiments:
        best_path = checkpoint_dir / f"{exp.key}_best.pth"
        exp.model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
        if fit_output_fir_ls(exp.model, x_train, y_train, device, dtype):
            exp.post_ls_nmse_db = evaluate_nmse_db(exp.model, x_val_tensor, y_val_tensor)
            torch.save(exp.model.state_dict(), checkpoint_dir / f"{exp.key}_best_post_ls.pth")

    history_path = write_history_csv(output_dir, experiments)
    long_history_path = write_long_history_csv(output_dir, experiments)
    summary_path = write_summary(output_dir, experiments)
    png_path, svg_path = plot_histories(output_dir, experiments, args.target_db)

    metadata = {
        "created_unix": time.time(),
        "data": str(Path(args.data).resolve()),
        "train_samples": len(x_train),
        "validation_samples": len(x_val),
        "split": "first 80% train / last 20% validation",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frame_length": args.frame_length,
        "stride_length": args.stride_length,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(factor=0.5)",
        "post_training_step": "train-only complex FIR least-squares refit where applicable",
        "seed": args.seed,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "finalization_seconds_this_invocation": time.time() - total_start,
        "recorded_model_training_seconds": {
            exp.key: round(exp.elapsed_seconds, 3) for exp in experiments
        },
        "sum_recorded_model_training_seconds": round(
            sum(exp.elapsed_seconds for exp in experiments), 3
        ),
        "models": [
            {
                "key": exp.key,
                "label": exp.label,
                "real_multipliers": exp.multipliers,
                "trainable_parameters": sum(p.numel() for p in exp.model.parameters()),
            }
            for exp in experiments
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f">>> History: {history_path}")
    print(f">>> Long history: {long_history_path}")
    print(f">>> Summary: {summary_path}")
    print(f">>> Figure PNG: {png_path}")
    print(f">>> Figure SVG: {svg_path}")
    print(f">>> Metadata: {metadata_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
