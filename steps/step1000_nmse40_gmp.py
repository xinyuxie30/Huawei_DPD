"""
<=1000-real-multiplier NMSE -40 dB assault script.

Strategy: the project multiplier rule counts weight multiplications only, so
GMP-style bypass capacity is extremely cheap:

1. Per-order complex FIR branches on x*|x|^(o-1) (filtered memory polynomial).
   Each tap costs 4 real multipliers, and all branch coefficients admit a
   frequency-weighted ridge LS closed-form refit (train split only).
2. Optional scalar GMP cross terms x[n] * |x[n+s]|^(o-1), 4 multipliers each.
3. Optional lagged-envelope input channels, H multipliers per channel.
4. Configurable TCN dilations (free) to widen the nonlinear receptive field.

The remaining EA-TCN backbone, the dataset split, and the NMSE/ACLR evaluation
math are reused verbatim from steps/AI-DPD.py (imported, never modified).
Alternating closed-form refits (GMP branches <-> output FIR) run during
training, generalizing the output-FIR refit that previously rescued ACLR.

Artifacts live under steps/step1000_nmse40/. Existing scripts are untouched.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "data" / "DPD_data.mat"
ATTEMPT_ROOT = SCRIPT_DIR / "step1000_nmse40"
CHECKPOINT_DIR = ATTEMPT_ROOT / "checkpoints"
FIGURE_DIR = ATTEMPT_ROOT / "figures"
SUMMARY_CSV = ATTEMPT_ROOT / "summary.csv"

DEFAULT_TEACHER_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "AI-DPD"
    / "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd_step1000_nmse40", AI_DPD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {AI_DPD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aidpd_step1000_nmse40"] = module
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()
# Route shared figure helpers to this attempt's folder without touching AI-DPD.py.
aidpd.FIGURE_DIR = FIGURE_DIR


# ---------------------------------------------------------------------------
# Variant registry. All configs must satisfy real_multiplier_count(cfg) <= 1000
# under the project counting rule:
#   4 * (sum of complex bypass taps + cross terms)
#   + Cin * H + L * (5H + H^2) + 2H + 3 * A * H + 4 * output_fir_taps
# ---------------------------------------------------------------------------
VARIANTS = {
    # GMP branch FIRs on x|x|^2 / x|x|^4, keep the proven H10/L4 backbone.
    "gmp_l4_fir35_1000": {
        "hidden": 10,
        "input_mode": "rich9",
        "extra_env_lags": [],
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 9, 5: 5},
        "cross_terms": [],
        "output_fir_taps": 35,
    },
    # Trade one TCN block for heavy GMP + a long output FIR, wide dilations.
    "gmp_l3_fir67_994": {
        "hidden": 10,
        "input_mode": "rich9",
        "extra_env_lags": [],
        "dilations": [1, 3, 9],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 11, 5: 7},
        "cross_terms": [],
        "output_fir_taps": 67,
    },
    # Lagged-envelope input channels instead of GMP branches.
    "envlag_l4_fir41_996": {
        "hidden": 10,
        "input_mode": "rich9",
        "extra_env_lags": [["amp", 1], ["amp", 2]],
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 1, 5: 1},
        "cross_terms": [],
        "output_fir_taps": 41,
    },
    # Wider hidden, three blocks, 4-channel attention.
    "h11_l3_fir51_997": {
        "hidden": 11,
        "input_mode": "rich9",
        "extra_env_lags": [],
        "dilations": [1, 2, 6],
        "attention_channels": 4,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 1, 5: 1},
        "cross_terms": [],
        "output_fir_taps": 51,
    },
    # Maximum GMP bank incl. 7th order, short output FIR.
    "gmp_heavy_l4_fir27_998": {
        "hidden": 10,
        "input_mode": "base6",
        "extra_env_lags": [],
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 15, 5: 9, 7: 5},
        "cross_terms": [],
        "output_fir_taps": 27,
    },
    # H12 with three blocks.
    "h12_l3_fir33_996": {
        "hidden": 12,
        "input_mode": "rich9",
        "extra_env_lags": [],
        "dilations": [1, 2, 4],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 1, 5: 1},
        "cross_terms": [],
        "output_fir_taps": 33,
    },
    # Baseline replica of the 998 recommended model, for A/B sanity.
    "base_l4_fir39_998": {
        "hidden": 10,
        "input_mode": "rich9",
        "extra_env_lags": [],
        "dilations": [1, 2, 4, 8],
        "attention_channels": 4,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 1, 5: 1},
        "cross_terms": [],
        "output_fir_taps": 39,
    },
    # Free GMP-product input channels (data*data products cost only H each).
    "gmpfeat13_l4_fir37_1000": {
        "hidden": 10,
        "input_mode": "rich9",
        "extra_env_lags": [["ixampl", 1], ["qxampl", 1], ["ixampl", -1], ["qxampl", -1]],
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 1, 5: 1},
        "cross_terms": [],
        "output_fir_taps": 37,
    },
    "gmpfeat15_l3_fir69_998": {
        "hidden": 10,
        "input_mode": "rich9",
        "extra_env_lags": [
            ["ixampl", 1],
            ["qxampl", 1],
            ["ixampl", -1],
            ["qxampl", -1],
            ["ampprod", 1],
            ["damp", 1],
        ],
        "dilations": [1, 3, 9],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 1, 5: 1},
        "cross_terms": [],
        "output_fir_taps": 69,
    },
    # h11 winner plus small GMP branch FIRs.
    "h11_l3_gmp_fir45_997": {
        "hidden": 11,
        "input_mode": "rich9",
        "extra_env_lags": [],
        "dilations": [1, 2, 6],
        "attention_channels": 4,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 5, 5: 3},
        "cross_terms": [],
        "output_fir_taps": 45,
    },
    # h12 NMSE leader with widened dilations (free).
    "h12_l3_d139_fir33_996": {
        "hidden": 12,
        "input_mode": "rich9",
        "extra_env_lags": [],
        "dilations": [1, 3, 9],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 1, 5: 1},
        "cross_terms": [],
        "output_fir_taps": 33,
    },
    # Combo slot: GMP branches + envelope lags + cross terms (post-probe tuning).
    "combo_gmp_envlag_1000": {
        "hidden": 10,
        "input_mode": "rich9",
        "extra_env_lags": [["amp", 1]],
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_taps": {1: 1, 3: 7, 5: 3},
        "cross_terms": [[3, 2], [3, -1]],
        "output_fir_taps": 33,
    },
}


def input_channels_for_config(config):
    return aidpd.input_channels_for_mode(config["input_mode"]) + len(
        config.get("extra_env_lags", [])
    )


def real_multiplier_count(config):
    hidden = int(config["hidden"])
    cin = input_channels_for_config(config)
    attention_channels = int(config["attention_channels"])
    blocks = len(config["dilations"])
    poly_taps = {int(k): int(v) for k, v in config["poly_taps"].items()}
    bypass_taps = sum(poly_taps.values()) + len(config.get("cross_terms", []))
    block_multipliers = hidden * 5 + hidden * hidden
    fir_taps = int(config.get("output_fir_taps", 1))
    fir_multipliers = 4 * fir_taps if fir_taps > 1 else 0
    return int(
        4 * bypass_taps
        + cin * hidden
        + blocks * block_multipliers
        + 2 * hidden
        + hidden * attention_channels * 3
        + fir_multipliers
    )


ENV_FEATURES = ("amp", "amp2", "amp3", "amp5")
PRODUCT_FEATURES = ("ixampl", "qxampl", "ilagamp", "qlagamp", "ampprod", "damp")
EXTRA_FEATURES = ENV_FEATURES + PRODUCT_FEATURES


class GmpEaTcnNet(nn.Module):
    """EA-TCN with GMP branch FIRs, cross terms, and lagged-envelope inputs.

    Exposes forward_base / apply_output_fir / output_fir_taps / fir_real /
    fir_imag so the unchanged AI-DPD.py output-FIR LS refit and evaluation
    helpers work on it directly.
    """

    def __init__(self, init_g_real, init_g_imag, config):
        super().__init__()
        self.config = dict(config)
        self.hidden = int(self.config["hidden"])
        self.input_mode = self.config["input_mode"]
        self.extra_env_lags = [
            (str(name), int(shift)) for name, shift in self.config.get("extra_env_lags", [])
        ]
        for name, _ in self.extra_env_lags:
            if name not in EXTRA_FEATURES:
                raise ValueError(f"unsupported env-lag feature: {name}")
        self.input_channels = input_channels_for_config(self.config)
        self.attention_channels = int(self.config["attention_channels"])
        self.attention_mode = self.config.get("attention_mode", "amp37")
        self.dilations = list(self.config["dilations"])
        self.poly_taps = {int(k): int(v) for k, v in self.config["poly_taps"].items()}
        for order, taps in self.poly_taps.items():
            if taps < 1 or taps % 2 == 0:
                raise ValueError(f"poly taps must be odd and >= 1, got order {order}: {taps}")
        self.cross_terms = [
            (int(order), int(shift)) for order, shift in self.config.get("cross_terms", [])
        ]

        # Complex FIR per polynomial order; order 1 center tap starts at the
        # train-split LS gain, higher orders start at zero (same convention as
        # the g/g3/g5 scalars in AI-DPD.py).
        self.branch_real = nn.ParameterDict()
        self.branch_imag = nn.ParameterDict()
        for order in sorted(self.poly_taps):
            taps = self.poly_taps[order]
            real = torch.zeros(taps, dtype=DTYPE)
            imag = torch.zeros(taps, dtype=DTYPE)
            if order == 1:
                real[taps // 2] = float(init_g_real)
                imag[taps // 2] = float(init_g_imag)
            self.branch_real[str(order)] = nn.Parameter(real)
            self.branch_imag[str(order)] = nn.Parameter(imag)

        self.cross_real = nn.Parameter(torch.zeros(len(self.cross_terms), dtype=DTYPE))
        self.cross_imag = nn.Parameter(torch.zeros(len(self.cross_terms), dtype=DTYPE))

        self.in_conv = nn.Conv1d(self.input_channels, self.hidden, 1, bias=False, dtype=DTYPE)
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(
                    self.hidden,
                    self.hidden,
                    5,
                    padding=2 * dilation,
                    dilation=dilation,
                    groups=self.hidden,
                    bias=False,
                    dtype=DTYPE,
                ),
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=DTYPE),
                nn.SiLU(),
            )
            for dilation in self.dilations
        )
        self.attn = aidpd.OptimalSpectrumAttention(
            self.hidden, self.attention_channels, mode=self.attention_mode
        )
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=DTYPE)
        nn.init.constant_(self.out_conv.weight, 0.0)

        self.output_fir_taps = int(self.config.get("output_fir_taps", 1))
        if self.output_fir_taps > 1:
            center = self.output_fir_taps // 2
            fir_real = torch.zeros(self.output_fir_taps, dtype=DTYPE)
            fir_imag = torch.zeros(self.output_fir_taps, dtype=DTYPE)
            fir_real[center] = 1.0
            self.fir_real = nn.Parameter(fir_real)
            self.fir_imag = nn.Parameter(fir_imag)

    # -- shared feature computation -------------------------------------
    @staticmethod
    def envelope_powers(i_x, q_x):
        amp = torch.sqrt(i_x * i_x + q_x * q_x + 1e-12)
        amp2 = amp * amp
        amp3 = amp * amp2
        amp4 = amp2 * amp2
        amp5 = amp * amp4
        amp7 = amp5 * amp2
        return {"amp": amp, "amp2": amp2, "amp3": amp3, "amp4": amp4, "amp5": amp5, "amp7": amp7}

    def order_basis(self, i_x, q_x, env, order):
        if order == 1:
            return i_x, q_x
        if order == 3:
            factor = env["amp2"]
        elif order == 5:
            factor = env["amp4"]
        elif order == 7:
            factor = env["amp4"] * env["amp2"]
        elif order == 9:
            factor = env["amp4"] * env["amp4"]
        else:
            factor = env["amp"] ** (order - 1)
        return i_x * factor, q_x * factor

    @staticmethod
    def complex_fir(sig_i, sig_q, w_real, w_imag):
        taps = w_real.shape[0]
        if taps == 1:
            out_i = sig_i * w_real - sig_q * w_imag
            out_q = sig_i * w_imag + sig_q * w_real
            return out_i, out_q
        pad = taps // 2
        real = w_real.view(1, 1, -1)
        imag = w_imag.view(1, 1, -1)
        out_i = F.conv1d(sig_i, real, padding=pad) - F.conv1d(sig_q, imag, padding=pad)
        out_q = F.conv1d(sig_i, imag, padding=pad) + F.conv1d(sig_q, real, padding=pad)
        return out_i, out_q

    def gmp_bypass(self, i_x, q_x, env):
        total_i = None
        total_q = None
        for order in sorted(self.poly_taps):
            base_i, base_q = self.order_basis(i_x, q_x, env, order)
            out_i, out_q = self.complex_fir(
                base_i, base_q, self.branch_real[str(order)], self.branch_imag[str(order)]
            )
            total_i = out_i if total_i is None else total_i + out_i
            total_q = out_q if total_q is None else total_q + out_q
        for idx, (order, shift) in enumerate(self.cross_terms):
            env_shift = torch.roll(env["amp"], shifts=shift, dims=2) ** (order - 1)
            base_i = i_x * env_shift
            base_q = q_x * env_shift
            wr = self.cross_real[idx]
            wi = self.cross_imag[idx]
            total_i = total_i + base_i * wr - base_q * wi
            total_q = total_q + base_i * wi + base_q * wr
        return torch.cat([total_i, total_q], dim=1)

    def extra_channel(self, name, shift, i_x, q_x, env):
        if name in ENV_FEATURES:
            return torch.roll(env[name], shifts=shift, dims=2)
        amp = env["amp"]
        if name == "ixampl":
            return i_x * torch.roll(amp, shifts=shift, dims=2)
        if name == "qxampl":
            return q_x * torch.roll(amp, shifts=shift, dims=2)
        if name == "ilagamp":
            return torch.roll(i_x, shifts=shift, dims=2) * amp
        if name == "qlagamp":
            return torch.roll(q_x, shifts=shift, dims=2) * amp
        if name == "ampprod":
            return amp * torch.roll(amp, shifts=shift, dims=2)
        if name == "damp":
            return amp - torch.roll(amp, shifts=shift, dims=2)
        raise ValueError(f"unsupported extra feature: {name}")

    def backbone_features(self, i_x, q_x, env):
        features = aidpd.build_features(i_x, q_x, env["amp"], env["amp3"], env["amp5"], self.input_mode)
        if self.extra_env_lags:
            extra = [
                self.extra_channel(name, shift, i_x, q_x, env)
                for name, shift in self.extra_env_lags
            ]
            features = torch.cat([features] + extra, dim=1)
        return features

    def forward_residual(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]
        env = self.envelope_powers(i_x, q_x)
        feat = self.act(self.in_conv(self.backbone_features(i_x, q_x, env)))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        feat = self.attn(feat, env["amp"], env["amp3"], env["amp7"], amp5=env["amp5"])
        return self.out_conv(feat)

    def forward_base(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]
        env = self.envelope_powers(i_x, q_x)
        bypass = self.gmp_bypass(i_x, q_x, env)
        feat = self.act(self.in_conv(self.backbone_features(i_x, q_x, env)))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        feat = self.attn(feat, env["amp"], env["amp3"], env["amp7"], amp5=env["amp5"])
        return bypass + self.out_conv(feat)

    def apply_output_fir(self, y):
        if self.output_fir_taps <= 1:
            return y
        pad = self.output_fir_taps // 2
        real = self.fir_real.view(1, 1, -1)
        imag = self.fir_imag.view(1, 1, -1)
        y_i = y[:, 0:1, :]
        y_q = y[:, 1:2, :]
        out_i = F.conv1d(y_i, real, padding=pad) - F.conv1d(y_q, imag, padding=pad)
        out_q = F.conv1d(y_i, imag, padding=pad) + F.conv1d(y_q, real, padding=pad)
        return torch.cat([out_i, out_q], dim=1)

    def forward(self, x_iq):
        return self.apply_output_fir(self.forward_base(x_iq))


# ---------------------------------------------------------------------------
# Closed-form GMP refit: given the current output FIR and the frozen NN
# residual, solve all bypass coefficients by frequency-weighted ridge LS on
# the training split. Mirrors the math of fit_output_fir_freq_from_train.
# ---------------------------------------------------------------------------
def band_weights(n, fs_mhz, args):
    freq = np.fft.fftfreq(n, d=1.0 / fs_mhz)
    weights = np.full(n, float(args.fir_floor_weight), dtype=np.float64)
    main = (freq >= args.main_band[0]) & (freq <= args.main_band[1])
    lower = (freq >= args.lower_band[0]) & (freq <= args.lower_band[1])
    upper = (freq >= args.upper_band[0]) & (freq <= args.upper_band[1])
    weights[main] = float(args.fir_main_weight)
    weights[lower | upper] = float(args.fir_adjacent_weight)
    return weights


def output_fir_response(model, n):
    taps = int(model.output_fir_taps)
    if taps <= 1:
        return np.ones(n, dtype=np.complex128)
    coeffs = (
        model.fir_real.detach().cpu().numpy() + 1j * model.fir_imag.detach().cpu().numpy()
    )
    radius = taps // 2
    omega = 2 * np.pi * np.arange(n) / n
    offsets = np.arange(-radius, radius + 1)
    return (coeffs[None, :] * np.exp(1j * omega[:, None] * offsets[None, :])).sum(axis=1)


def gmp_basis_signals(model, x_np):
    """Complex basis signals for every bypass column, on the full sequence."""
    x_tensor = torch.as_tensor(x_np.T, dtype=DTYPE, device=DEVICE).unsqueeze(0)
    i_x, q_x = x_tensor[:, 0:1, :], x_tensor[:, 1:2, :]
    with torch.no_grad():
        env = model.envelope_powers(i_x, q_x)
        columns = []
        layout = []
        for order in sorted(model.poly_taps):
            base_i, base_q = model.order_basis(i_x, q_x, env, order)
            base = (base_i + 1j * base_q).squeeze().cpu().numpy()
            radius = model.poly_taps[order] // 2
            layout.append(("branch", order, radius))
            columns.append((base, radius))
        for idx, (order, shift) in enumerate(model.cross_terms):
            env_shift = torch.roll(env["amp"], shifts=shift, dims=2) ** (order - 1)
            base = ((i_x * env_shift) + 1j * (q_x * env_shift)).squeeze().cpu().numpy()
            layout.append(("cross", idx, 0))
            columns.append((base, 0))
    return columns, layout


def refit_gmp_freq(model, x_train, y_train, args):
    """Weighted ridge LS for all bypass coefficients, train split only."""
    n = len(x_train)
    target = y_train[:, 0] + 1j * y_train[:, 1]
    with torch.no_grad():
        x_tensor = torch.as_tensor(x_train.T, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        resid = model.forward_residual(x_tensor).squeeze(0).cpu().numpy()
    resid_c = resid[0] + 1j * resid[1]

    h_response = output_fir_response(model, n)
    weights = band_weights(n, args.fs_mhz, args)
    sqrt_w = np.sqrt(weights)

    target_fft = np.fft.fft(target)
    resid_fft = np.fft.fft(resid_c)
    rhs = (target_fft - h_response * resid_fft) * sqrt_w

    columns, layout = gmp_basis_signals(model, x_train)
    omega = 2 * np.pi * np.arange(n) / n
    design_cols = []
    for base, radius in columns:
        base_fft = np.fft.fft(base)
        offsets = np.arange(-radius, radius + 1)
        phase = np.exp(1j * omega[:, None] * offsets[None, :])
        design_cols.append(base_fft[:, None] * phase * (h_response * sqrt_w)[:, None])
    design = np.concatenate(design_cols, axis=1)

    lhs = design.conj().T @ design + float(args.fir_ridge) * n * np.eye(design.shape[1])
    coeffs = np.linalg.solve(lhs, design.conj().T @ rhs)

    cursor = 0
    with torch.no_grad():
        for kind, key, radius in layout:
            if kind == "branch":
                taps = 2 * radius + 1
                segment = coeffs[cursor : cursor + taps]
                cursor += taps
                model.branch_real[str(key)].copy_(
                    torch.as_tensor(segment.real, dtype=DTYPE, device=DEVICE)
                )
                model.branch_imag[str(key)].copy_(
                    torch.as_tensor(segment.imag, dtype=DTYPE, device=DEVICE)
                )
            else:
                value = coeffs[cursor]
                cursor += 1
                model.cross_real[key] = float(value.real)
                model.cross_imag[key] = float(value.imag)
    return coeffs


def fit_output_fir(model, x_train, y_train, args):
    if int(model.output_fir_taps) <= 1:
        return None
    if args.fir_fit_domain == "time":
        return aidpd.fit_output_fir_from_train(model, x_train, y_train)
    return aidpd.fit_output_fir_freq_from_train(
        model,
        x_train,
        y_train,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        main_weight=args.fir_main_weight,
        adjacent_weight=args.fir_adjacent_weight,
        floor_weight=args.fir_floor_weight,
        ridge=args.fir_ridge,
    )


def alternating_refit(model, x_train, y_train, args, rounds=1):
    for _ in range(max(1, int(rounds))):
        if args.refit_gmp:
            refit_gmp_freq(model, x_train, y_train, args)
        fit_output_fir(model, x_train, y_train, args)


# ---------------------------------------------------------------------------
# Data / loss / evaluation (metric math delegated to AI-DPD.py helpers)
# ---------------------------------------------------------------------------
class DistillFrameDataset(Dataset):
    def __init__(self, x_np, y_np, d_np, frame_length, stride_length):
        self.x = torch.as_tensor(x_np, dtype=DTYPE)
        self.y = torch.as_tensor(y_np, dtype=DTYPE)
        self.d = torch.as_tensor(d_np, dtype=DTYPE)
        self.frame_length = int(frame_length)
        self.starts = list(range(0, len(self.x) - self.frame_length + 1, int(stride_length)))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.frame_length
        return self.x[start:end].T, self.y[start:end].T, self.d[start:end].T


def split_data(data_path):
    _, _, x_c, y_c = aidpd.load_normalized_data(data_path)
    x_np = aidpd.iq_array(x_c)
    y_np = aidpd.iq_array(y_c)
    split_idx = int(min(len(x_np), len(y_np)) * 0.8)
    g_opt = np.vdot(x_c[:split_idx], y_c[:split_idx]) / np.vdot(x_c[:split_idx], x_c[:split_idx])
    return (
        x_np[:split_idx],
        y_np[:split_idx],
        x_np[split_idx:],
        y_np[split_idx:],
        float(g_opt.real),
        float(g_opt.imag),
    )


def evaluate_aclr(model, x_val, y_val, args):
    pred_iq = aidpd.run_sequence(model, x_val)
    pred_complex = aidpd.tensor_to_complex(pred_iq)
    target_complex = aidpd.tensor_to_complex(y_val.T)
    error_complex = target_complex - pred_complex
    freq_hz, error_psd = aidpd.psd_db(error_complex, fs_hz=args.fs_mhz * 1e6)
    _, target_psd = aidpd.psd_db(target_complex, fs_hz=args.fs_mhz * 1e6)
    residual_aclr = aidpd.aclr_from_psd(
        freq_hz,
        error_psd,
        tuple(args.main_band),
        tuple(args.lower_band),
        tuple(args.upper_band),
        original_values_db=target_psd,
    )
    return {
        # Keep the pred_* aliases because the search/checkpoint code consumes them.
        "pred_worst": float(residual_aclr[2]),
        "pred_lower": float(residual_aclr[0]),
        "pred_upper": float(residual_aclr[1]),
    }


def evaluate_full(model, x_val, y_val, args):
    nmse, nmse_db = aidpd.evaluate_nmse(model, x_val, y_val)
    metrics = evaluate_aclr(model, x_val, y_val, args)
    metrics.update({"nmse_linear": float(nmse), "nmse_db": float(nmse_db)})
    return metrics


def score_metrics(metrics, target_dbc):
    return metrics["nmse_db"] + 0.35 * max(0.0, metrics["pred_worst"] - target_dbc)


def crop_if_needed(*items, margin):
    if margin <= 0 or items[0].shape[-1] <= 2 * margin:
        return items
    return tuple(item[:, :, margin:-margin] for item in items)


def training_loss(pred, truth, distill, args):
    pred_c, truth_c, distill_c = crop_if_needed(pred, truth, distill, margin=args.crop_margin)
    loss = F.mse_loss(pred_c, truth_c)
    if args.diff_loss_weight > 0:
        loss = loss + args.diff_loss_weight * F.mse_loss(
            pred_c[:, :, 1:] - pred_c[:, :, :-1],
            truth_c[:, :, 1:] - truth_c[:, :, :-1],
        )
    if args.spectral_truth_weight > 0:
        loss = loss + args.spectral_truth_weight * aidpd.weighted_spectral_nmse(
            pred_c,
            truth_c,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            main_weight=args.spectral_main_weight,
            adjacent_weight=args.spectral_adjacent_weight,
            floor_weight=args.spectral_floor_weight,
        )
    if args.distill_loss_weight > 0:
        distill_loss = F.mse_loss(pred_c, distill_c)
        if args.distill_diff_weight > 0:
            distill_loss = distill_loss + args.distill_diff_weight * F.mse_loss(
                pred_c[:, :, 1:] - pred_c[:, :, :-1],
                distill_c[:, :, 1:] - distill_c[:, :, :-1],
            )
        loss = loss + args.distill_loss_weight * distill_loss
    return loss


class EmaTracker:
    """EMA over NN weights only; closed-form-refitted params stay live."""

    def __init__(self, model, decay, skip_prefixes=("fir_", "branch_", "cross_")):
        self.decay = float(decay)
        self.skip_prefixes = skip_prefixes
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if not name.startswith(skip_prefixes)
        }
        self.backup = {}

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply(self, model):
        # Snapshot everything: the closed-form refits that follow will also
        # rewrite fir_/branch_/cross_ params, so restore must roll back fully.
        self.backup = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name])

    def restore(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.backup:
                    param.copy_(self.backup[name])
        self.backup = {}


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def append_summary(row):
    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    is_new = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                ["timestamp", "variant", "multipliers", "stage", "nmse_db", "aclr_worst", "score", "checkpoint"]
            )
        writer.writerow(row)


def train_stage(
    model,
    x_train,
    y_train,
    d_train,
    x_val,
    y_val,
    args,
    *,
    stage_name,
    epochs,
    lr,
    frame_length,
    stride_length,
    batch_size,
    checkpoint_path,
    ema=None,
):
    if epochs <= 0:
        metrics = evaluate_full(model, x_val, y_val, args)
        return metrics, [(0, metrics["nmse_db"], metrics["pred_worst"])]

    loader = DataLoader(
        DistillFrameDataset(x_train, y_train, d_train, frame_length, stride_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, min_lr=args.min_lr
    )

    alternating_refit(model, x_train, y_train, args)
    best_metrics = evaluate_full(model, x_val, y_val, args)
    best_score = score_metrics(best_metrics, args.aclr_target_dbc)
    save_checkpoint(model, checkpoint_path)
    history = [(0, best_metrics["nmse_db"], best_metrics["pred_worst"])]
    print(
        f">>> {stage_name} initial | NMSE {best_metrics['nmse_db']:.2f} dB | "
        f"ACLR {best_metrics['pred_worst']:.2f} dBc | score {best_score:.2f}"
    )

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, batch_d in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            batch_d = batch_d.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = training_loss(pred, batch_y, batch_d, args)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        model.eval()
        nmse, nmse_db = aidpd.evaluate_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        aclr_value = float("nan")
        extra = ""

        if args.refit_every > 0 and epoch % args.refit_every == 0:
            alternating_refit(model, x_train, y_train, args)
            nmse, nmse_db = aidpd.evaluate_nmse(model, x_val, y_val)

        candidates = [("live", None)]
        if ema is not None and epoch % args.aclr_every == 0:
            candidates.append(("ema", ema))
        if epoch % args.aclr_every == 0 or epoch == epochs:
            for tag, tracker in candidates:
                if tracker is not None:
                    tracker.apply(model)
                    alternating_refit(model, x_train, y_train, args)
                metrics = evaluate_full(model, x_val, y_val, args)
                current_score = score_metrics(metrics, args.aclr_target_dbc)
                if tag == "live":
                    aclr_value = metrics["pred_worst"]
                    extra = f" | ACLR {aclr_value:.2f} dBc | score {current_score:.2f}"
                else:
                    extra += f" | EMA {metrics['nmse_db']:.2f}/{metrics['pred_worst']:.2f}"
                if current_score < best_score:
                    best_score = current_score
                    best_metrics = metrics
                    save_checkpoint(model, checkpoint_path)
                if tracker is not None:
                    tracker.restore(model)

        history.append((epoch, float(nmse_db), aclr_value))
        elapsed = time.time() - start
        print(
            f"{stage_name} Epoch [{epoch}/{epochs}] | "
            f"Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"NMSE {nmse_db:.2f} dB{extra}",
            flush=True,
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=True)
    alternating_refit(model, x_train, y_train, args, rounds=2)
    final_metrics = evaluate_full(model, x_val, y_val, args)
    if score_metrics(final_metrics, args.aclr_target_dbc) <= score_metrics(
        best_metrics, args.aclr_target_dbc
    ):
        save_checkpoint(model, checkpoint_path)
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=True)
        final_metrics = evaluate_full(model, x_val, y_val, args)
    return final_metrics, history


def plot_history(all_history, output_prefix):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    offset = 0
    for label, history in all_history:
        xs = [offset + row[0] for row in history]
        ys = [row[1] for row in history]
        ax.plot(xs, ys, label=label)
        offset = xs[-1]
    ax.axhline(-37.37, color="gray", linestyle="--", linewidth=1.0, label="998-MAC baseline")
    ax.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.2, label="-40 dB target")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NMSE (dB)")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="best")
    fig.tight_layout()
    path = FIGURE_DIR / f"{output_prefix}_convergence.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def build_teacher(args, g_real, g_imag):
    config = aidpd.preset_config("rich9_h15_l6_fir43_2284")
    teacher = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(DEVICE)
    teacher_path = Path(args.teacher_checkpoint)
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher_path}")
    teacher.load_state_dict(aidpd.load_state_dict(teacher_path), strict=True)
    teacher.eval()
    return teacher


def report(metrics, multipliers, checkpoint_path, args, label="RESULT"):
    print(f"\n>>> {label}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Real multipliers: {multipliers}")
    print(f"Linear NMSE: {metrics['nmse_linear']:.6e}")
    print(f"dB NMSE: {metrics['nmse_db']:.2f} dB")
    print(
        f"Residual ACLR worst (error adjacent - target main): {metrics['pred_worst']:.2f} dBc | "
        f"margin {args.aclr_target_dbc - metrics['pred_worst']:.2f} dB"
    )
    print(
        f"Target check: NMSE <= {args.nmse_target_db:.2f} dB "
        f"({'PASS' if metrics['nmse_db'] <= args.nmse_target_db else 'FAIL'}), "
        f"ACLR <= {args.aclr_target_dbc:.2f} dBc "
        f"({'PASS' if metrics['pred_worst'] <= args.aclr_target_dbc else 'FAIL'})"
    )


def refit_scan(model, x_train, y_train, x_val, y_val, args, checkpoint_path):
    """Grid-scan FIR/GMP refit weights on the loaded checkpoint, keep best score."""
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    grid_adjacent = args.scan_adjacent_weights
    grid_floor = args.scan_floor_weights
    grid_ridge = args.scan_ridges
    best = None
    for adjacent in grid_adjacent:
        for floor in grid_floor:
            for ridge in grid_ridge:
                model.load_state_dict(base_state, strict=True)
                args.fir_adjacent_weight = float(adjacent)
                args.fir_floor_weight = float(floor)
                args.fir_ridge = float(ridge)
                alternating_refit(model, x_train, y_train, args, rounds=2)
                metrics = evaluate_full(model, x_val, y_val, args)
                score = score_metrics(metrics, args.aclr_target_dbc)
                print(
                    f"scan adjacent={adjacent:g} floor={floor:g} ridge={ridge:g} | "
                    f"NMSE {metrics['nmse_db']:.3f} dB | ACLR {metrics['pred_worst']:.2f} dBc | "
                    f"score {score:.3f}"
                )
                if best is None or score < best[0]:
                    best = (score, adjacent, floor, ridge, metrics)
    score, adjacent, floor, ridge, metrics = best
    model.load_state_dict(base_state, strict=True)
    args.fir_adjacent_weight = float(adjacent)
    args.fir_floor_weight = float(floor)
    args.fir_ridge = float(ridge)
    alternating_refit(model, x_train, y_train, args, rounds=2)
    save_checkpoint(model, checkpoint_path)
    print(
        f">>> refit-scan best: adjacent={adjacent:g} floor={floor:g} ridge={ridge:g} | "
        f"NMSE {metrics['nmse_db']:.3f} dB | ACLR {metrics['pred_worst']:.2f} dBc"
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="<=1000-real-multiplier GMP-EA-TCN assault on NMSE -40 dB."
    )
    parser.add_argument("--variant", default="gmp_l4_fir35_1000", choices=sorted(VARIANTS))
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-multipliers", type=int, default=1000)

    parser.add_argument("--teacher-mode", choices=["load", "none"], default="load")
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER_CHECKPOINT))
    parser.add_argument("--init-checkpoint", default=None, help="Optional partial-init checkpoint.")

    parser.add_argument("--stage1-epochs", type=int, default=60)
    parser.add_argument("--stage2-epochs", type=int, default=30)
    parser.add_argument("--stage3-epochs", type=int, default=0)
    parser.add_argument("--polish-epochs", type=int, default=0)
    parser.add_argument("--stage1-lr", type=float, default=8e-4)
    parser.add_argument("--stage2-lr", type=float, default=2e-4)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--polish-lr", type=float, default=1e-5)
    parser.add_argument("--polish-frame-length", type=int, default=4096)
    parser.add_argument("--polish-stride-length", type=int, default=256)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=64)
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=15)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--grad-clip", type=float, default=0.5)

    parser.add_argument("--distill-loss-weight", type=float, default=0.04)
    parser.add_argument("--distill-diff-weight", type=float, default=0.02)
    parser.add_argument("--diff-loss-weight", type=float, default=0.005)
    parser.add_argument("--spectral-truth-weight", type=float, default=2e-5)
    parser.add_argument("--spectral-main-weight", type=float, default=1.0)
    parser.add_argument("--spectral-adjacent-weight", type=float, default=4.0)
    parser.add_argument("--spectral-floor-weight", type=float, default=0.04)

    parser.add_argument("--refit-every", type=int, default=10)
    parser.add_argument("--aclr-every", type=int, default=10)
    parser.add_argument("--refit-gmp", dest="refit_gmp", action="store_true", default=True)
    parser.add_argument("--no-refit-gmp", dest="refit_gmp", action="store_false")
    parser.add_argument("--fir-fit-domain", choices=["time", "freq"], default="freq")
    parser.add_argument("--fir-main-weight", type=float, default=1.0)
    parser.add_argument("--fir-adjacent-weight", type=float, default=1.1)
    parser.add_argument("--fir-floor-weight", type=float, default=0.03)
    parser.add_argument("--fir-ridge", type=float, default=1e-7)

    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(20.0, 80.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-180.0, -120.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(120.0, 180.0))
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0)
    parser.add_argument("--nmse-target-db", type=float, default=-40.0)

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--refit-scan", action="store_true")
    parser.add_argument(
        "--scan-adjacent-weights", type=float, nargs="+", default=[0.03, 0.3, 1.1, 2.0]
    )
    parser.add_argument("--scan-floor-weights", type=float, nargs="+", default=[0.02, 0.04, 0.08])
    parser.add_argument("--scan-ridges", type=float, nargs="+", default=[1e-8, 1e-7, 1e-6])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_variants:
        for name, config in sorted(VARIANTS.items()):
            print(f"{name}: {real_multiplier_count(config)} multipliers | {config}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = dict(VARIANTS[args.variant])
    multipliers = real_multiplier_count(config)
    if multipliers > args.max_multipliers:
        raise RuntimeError(
            f"multiplier budget exceeded for {args.variant}: {multipliers} > {args.max_multipliers}"
        )

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else CHECKPOINT_DIR / f"{args.variant}.pth"
    )
    output_prefix = args.output_prefix or f"step1000_{args.variant}"

    data_path = Path(args.data)
    x_train, y_train, x_val, y_val, g_real, g_imag = split_data(data_path)

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Variant: {args.variant}")
    print(f">>> Config: {config}")
    print(f">>> Real multipliers: {multipliers} (budget {args.max_multipliers})")
    print(f">>> Checkpoint: {checkpoint_path}")
    print(f">>> Train/val samples: {len(x_train)} / {len(x_val)}")

    model = GmpEaTcnNet(g_real, g_imag, config).to(DEVICE)

    if args.evaluate_only or args.refit_scan:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=True)
        if args.refit_scan:
            metrics = refit_scan(model, x_train, y_train, x_val, y_val, args, checkpoint_path)
        else:
            metrics = evaluate_full(model, x_val, y_val, args)
        aidpd.evaluate_aclr_visualization(
            model,
            x_val,
            y_val,
            output_prefix=output_prefix,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            target_dbc=args.aclr_target_dbc,
        )
        report(metrics, multipliers, checkpoint_path, args)
        append_summary(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                args.variant,
                multipliers,
                "refit-scan" if args.refit_scan else "evaluate",
                f"{metrics['nmse_db']:.4f}",
                f"{metrics['pred_worst']:.4f}",
                f"{score_metrics(metrics, args.aclr_target_dbc):.4f}",
                str(checkpoint_path),
            ]
        )
        return

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        loaded, partial, skipped = aidpd.load_partial_state(model, init_path)
        print(f">>> Partial init from: {init_path}")
        print(f">>> Loaded tensors: exact={len(loaded)}, partial={len(partial)}, skipped={len(skipped)}")

    if args.teacher_mode == "load":
        teacher = build_teacher(args, g_real, g_imag)
        d_train = np.asarray(aidpd.run_sequence(teacher, x_train)).T
        teacher_nmse = float(np.sum((d_train - y_train) ** 2) / np.sum(y_train**2))
        print(f">>> Teacher train-split NMSE vs y: {10 * np.log10(teacher_nmse):.2f} dB")
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        d_train = y_train.copy()
        print(">>> Teacher mode none: distillation target is dataset y.")

    ema = EmaTracker(model, args.ema_decay) if args.ema else None

    stages = [
        ("stage1", args.stage1_epochs, args.stage1_lr, args.frame_length, args.stride_length),
        ("stage2", args.stage2_epochs, args.stage2_lr, args.frame_length, args.stride_length),
        ("stage3", args.stage3_epochs, args.stage3_lr, args.frame_length, args.stride_length),
        (
            "polish",
            args.polish_epochs,
            args.polish_lr,
            args.polish_frame_length,
            args.polish_stride_length,
        ),
    ]

    all_history = []
    final_metrics = None
    for stage_name, epochs, lr, frame_length, stride_length in stages:
        if epochs <= 0:
            continue
        stage_ckpt = (
            checkpoint_path
            if stage_name == "polish" or stage_name == "stage3"
            else checkpoint_path.with_name(f"{checkpoint_path.stem}_{stage_name}.pth")
        )
        if stage_name in ("stage3", "polish"):
            stage_ckpt = checkpoint_path
        batch_size = max(2, args.batch_size // 4) if stage_name == "polish" else args.batch_size
        metrics, history = train_stage(
            model,
            x_train,
            y_train,
            d_train,
            x_val,
            y_val,
            args,
            stage_name=stage_name,
            epochs=epochs,
            lr=lr,
            frame_length=frame_length,
            stride_length=stride_length,
            batch_size=batch_size,
            checkpoint_path=stage_ckpt,
            ema=ema,
        )
        all_history.append((stage_name, history))
        final_metrics = metrics
        print(
            f">>> {stage_name} done | NMSE {metrics['nmse_db']:.2f} dB | "
            f"ACLR {metrics['pred_worst']:.2f} dBc"
        )
        append_summary(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                args.variant,
                multipliers,
                stage_name,
                f"{metrics['nmse_db']:.4f}",
                f"{metrics['pred_worst']:.4f}",
                f"{score_metrics(metrics, args.aclr_target_dbc):.4f}",
                str(stage_ckpt),
            ]
        )

    if final_metrics is None:
        raise RuntimeError("no training stages were run; check --stage*-epochs")

    save_checkpoint(model, checkpoint_path)
    convergence_path = plot_history(all_history, output_prefix)
    aidpd.evaluate_aclr_visualization(
        model,
        x_val,
        y_val,
        output_prefix=output_prefix,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        target_dbc=args.aclr_target_dbc,
    )
    print(f">>> Convergence figure: {convergence_path}")
    report(final_metrics, multipliers, checkpoint_path, args, label="FINAL RESULT")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
