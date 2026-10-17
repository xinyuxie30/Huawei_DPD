"""
Final <=999-real-multiplier DPD training script.

Default route:
1. Load the normalized Huawei DPD_data.mat.
2. Load a verified high-capacity teacher checkpoint.
3. Train a <=999-real-multiplier student from random
   initialization with teacher-assisted supervision.
4. Refit the train-only output FIR, evaluate validation NMSE and ACLR, and save
   PSD/ACLR figures plus the final checkpoint.

For a fully self-contained rebuild, pass --teacher-mode train. That first trains
the H15/L6/rich9/FIR43 teacher, then trains the lightweight student from scratch.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"

DEFAULT_TEACHER_CHECKPOINT = (
    CHECKPOINT_DIR
    / "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth"
)
DEFAULT_FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "final_lightweight_992_rich9_H10_L4_FIR45_scratch_distill.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


STUDENT_CONFIGS = {
    "h10_l3_base6_fir41_796": {
        "name": "H10_L3_base6_FIR41_796",
        "hidden": 10,
        "input_mode": "base6",
        "dilations": [1, 2, 4],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 41,
    },
    "h10_l4_rich9_fir45_992": {
        "name": "H10_L4_rich9_FIR45_992",
        "hidden": 10,
        "input_mode": "rich9",
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 45,
    },
    "h10_l4_base6_fir53_994": {
        "name": "H10_L4_base6_FIR53_994",
        "hidden": 10,
        "input_mode": "base6",
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 53,
    },
    "h10_l4_rich9_attn4_fir39_998": {
        "name": "H10_L4_rich9_attn4_FIR39_998",
        "hidden": 10,
        "input_mode": "rich9",
        "dilations": [1, 2, 4, 8],
        "attention_channels": 4,
        "attention_mode": "amp37",
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 39,
    },
    "h16_l6_base6_rank1_fir9_992": {
        "name": "H16_L6_base6_rank1_FIR9_992",
        "hidden": 16,
        "input_mode": "base6",
        "dilations": [1, 2, 4, 8, 16, 32],
        "attention_channels": 3,
        "attention_mode": "amp37",
        "poly_orders": [1, 3, 5],
        "pointwise_rank": 1,
        "output_fir_taps": 9,
    },
}
DEFAULT_STUDENT_PRESET = "h10_l4_rich9_fir45_992"

TEACHER_CONFIG = {
    "name": "H15_L6_rich9_FIR43_2284",
    "hidden": 15,
    "input_mode": "rich9",
    "dilations": [1, 2, 4, 8, 16, 32],
    "attention_channels": 3,
    "attention_mode": "amp37",
    "poly_orders": [1, 3, 5],
    "output_fir_taps": 43,
}


class OptimalSpectrumAttention(nn.Module):
    def __init__(self, channels, input_channels=3, mode="amp37"):
        super().__init__()
        self.input_channels = int(input_channels)
        self.mode = mode
        self.env_conv = nn.Conv1d(
            self.input_channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
            dtype=DTYPE,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp, amp3, amp7, amp5=None, thermal=None):
        if self.mode == "thermal16":
            if thermal is None:
                raise ValueError("thermal16 attention requires thermal input")
            env = torch.cat([amp, amp3, thermal], dim=1)
        elif self.mode == "amp37_thermal16":
            if thermal is None:
                raise ValueError("amp37_thermal16 attention requires thermal input")
            env = torch.cat([amp, amp3, amp7, thermal], dim=1)
        elif self.input_channels == 3:
            env = torch.cat([amp, amp3, amp7], dim=1)
        elif self.input_channels == 4:
            if amp5 is None:
                raise ValueError("4-channel attention requires amp5")
            env = torch.cat([amp, amp3, amp5, amp7], dim=1)
        else:
            raise ValueError(f"unsupported attention channels: {self.input_channels}")
        return x * self.sigmoid(self.env_conv(env))


def input_channels_for_mode(input_mode):
    if input_mode == "base6":
        return 6
    if input_mode == "rich9":
        return 9
    if input_mode == "rich11":
        return 11
    raise ValueError(f"unsupported input mode: {input_mode}")


def real_multiplier_count(config):
    hidden = int(config["hidden"])
    input_channels = input_channels_for_mode(config["input_mode"])
    attention_channels = int(config["attention_channels"])
    blocks = len(config["dilations"])
    polynomial_orders = len(config.get("poly_orders", [1, 3, 5]))
    pointwise_rank = config.get("pointwise_rank")
    if pointwise_rank is None:
        block_mults = hidden * 5 + hidden * hidden
    else:
        rank = int(pointwise_rank)
        block_mults = hidden * 5 + 2 * hidden * rank
    fir_taps = int(config.get("output_fir_taps", 1))
    fir_mults = 4 * fir_taps if fir_taps > 1 else 0
    return int(
        4 * polynomial_orders
        + input_channels * hidden
        + blocks * block_mults
        + 2 * hidden
        + hidden * attention_channels * 3
        + fir_mults
    )


def build_features(i_x, q_x, amp, amp3, amp5, input_mode):
    i_future = torch.roll(i_x, shifts=-1, dims=2)
    q_future = torch.roll(q_x, shifts=-1, dims=2)
    if input_mode == "base6":
        return torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)

    i_past = torch.roll(i_x, shifts=1, dims=2)
    q_past = torch.roll(q_x, shifts=1, dims=2)
    if input_mode == "rich9":
        return torch.cat(
            [i_x, q_x, i_future, q_future, i_past, q_past, amp, amp3, amp5],
            dim=1,
        )
    if input_mode == "rich11":
        return torch.cat(
            [
                i_x,
                q_x,
                i_future,
                q_future,
                i_past,
                q_past,
                i_x - i_past,
                q_x - q_past,
                amp,
                amp3,
                amp5,
            ],
            dim=1,
        )
    raise ValueError(f"unsupported input mode: {input_mode}")


class DiagramDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag, config):
        super().__init__()
        self.config = dict(config)
        self.hidden = int(self.config["hidden"])
        self.input_mode = self.config["input_mode"]
        self.input_channels = input_channels_for_mode(self.input_mode)
        self.attention_channels = int(self.config.get("attention_channels", 3))
        self.attention_mode = self.config.get("attention_mode", "amp37")
        self.thermal_window = int(self.config.get("thermal_window", 16))
        self.dilations = list(self.config["dilations"])
        self.poly_orders = list(self.config.get("poly_orders", [1, 3, 5]))
        self.pointwise_rank = self.config.get("pointwise_rank")

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=DTYPE))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=DTYPE))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.extra_poly_real = nn.ParameterDict()
        self.extra_poly_imag = nn.ParameterDict()
        for order in self.poly_orders:
            if order > 5:
                self.extra_poly_real[str(order)] = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
                self.extra_poly_imag[str(order)] = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))

        self.in_conv = nn.Conv1d(self.input_channels, self.hidden, 1, bias=False, dtype=DTYPE)
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList()
        for dilation in self.dilations:
            if self.pointwise_rank is None:
                pointwise = [nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=DTYPE)]
            else:
                rank = int(self.pointwise_rank)
                pointwise = [
                    nn.Conv1d(self.hidden, rank, 1, bias=False, dtype=DTYPE),
                    nn.Conv1d(rank, self.hidden, 1, bias=False, dtype=DTYPE),
                ]
            self.tcn_blocks.append(
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
                    *pointwise,
                    nn.SiLU(),
                )
            )

        self.attn = OptimalSpectrumAttention(
            self.hidden,
            self.attention_channels,
            mode=self.attention_mode,
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

    def forward_base(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        amp = torch.sqrt(i_x * i_x + q_x * q_x + 1e-12)
        amp2 = amp * amp
        amp3 = amp * amp2
        amp4 = amp2 * amp2
        amp5 = amp * amp4
        amp7 = amp5 * amp2

        x3_i = i_x * amp2
        x3_q = q_x * amp2
        lin3_i = x3_i * self.g3_real - x3_q * self.g3_imag
        lin3_q = x3_i * self.g3_imag + x3_q * self.g3_real

        x5_i = i_x * amp4
        x5_q = q_x * amp4
        lin5_i = x5_i * self.g5_real - x5_q * self.g5_imag
        lin5_q = x5_i * self.g5_imag + x5_q * self.g5_real

        for order in self.poly_orders:
            if order <= 5:
                continue
            amp_power = amp ** (order - 1)
            xo_i = i_x * amp_power
            xo_q = q_x * amp_power
            gr = self.extra_poly_real[str(order)]
            gi = self.extra_poly_imag[str(order)]
            lin5_i = lin5_i + xo_i * gr - xo_q * gi
            lin5_q = lin5_q + xo_i * gi + xo_q * gr

        linear_out = torch.cat([lin_i + lin3_i + lin5_i, lin_q + lin3_q + lin5_q], dim=1)

        features = build_features(i_x, q_x, amp, amp3, amp5, self.input_mode)
        feat = self.act(self.in_conv(features))
        for block in self.tcn_blocks:
            feat = feat + block(feat)

        thermal = None
        if self.attention_mode in ("thermal16", "amp37_thermal16"):
            thermal = F.avg_pool1d(
                F.pad(amp2, (self.thermal_window - 1, 0), mode="replicate"),
                kernel_size=self.thermal_window,
                stride=1,
            )
        feat = self.attn(feat, amp, amp3, amp7, amp5=amp5, thermal=thermal)
        return linear_out + self.out_conv(feat)

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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_normalized_data(data_path):
    data = scipy.io.loadmat(data_path)
    raw_x = data["x"].flatten()
    raw_y = data["y"].flatten()
    x_c = raw_x / np.sqrt(np.mean(np.abs(raw_x) ** 2))
    y_c = raw_y / np.sqrt(np.mean(np.abs(raw_y) ** 2))
    return raw_x, raw_y, x_c, y_c


def iq_array(complex_signal):
    return np.stack([complex_signal.real, complex_signal.imag], axis=1)


def split_data(data_path):
    _, _, x_c, y_c = load_normalized_data(data_path)
    x_np = iq_array(x_c)
    y_np = iq_array(y_c)
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


def load_checkpoint(path):
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def run_sequence(model, x_np):
    x_tensor = torch.as_tensor(x_np.T, dtype=DTYPE, device=DEVICE).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        return model(x_tensor).squeeze(0).cpu().numpy()


def run_base_sequence(model, x_np):
    x_tensor = torch.as_tensor(x_np.T, dtype=DTYPE, device=DEVICE).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        return model.forward_base(x_tensor).squeeze(0).cpu().numpy()


def as_sample_iq(iq):
    arr = np.asarray(iq)
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        return arr.T
    return arr


def nmse_metrics(pred_iq, target_iq):
    pred = as_sample_iq(pred_iq)
    target = as_sample_iq(target_iq)
    nmse = float(np.sum((pred - target) ** 2) / np.sum(target ** 2))
    return nmse, float(10 * np.log10(nmse))


def evaluate_full_nmse(model, x_val, y_val):
    pred = run_sequence(model, x_val)
    return nmse_metrics(pred, y_val)


def fit_output_fir_time(model, x_train, y_train):
    taps = int(model.output_fir_taps)
    if taps <= 1:
        return None
    radius = taps // 2
    base = run_base_sequence(model, x_train).T
    z = base[:, 0] + 1j * base[:, 1]
    target = y_train[:, 0] + 1j * y_train[:, 1]
    cols = []
    for offset in range(-radius, radius + 1):
        cols.append(z[radius + offset : len(z) - radius + offset])
    design = np.stack(cols, axis=1)
    rhs = target[radius : len(target) - radius]
    coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=DTYPE, device=DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=DTYPE, device=DEVICE))
    return coeffs


def fit_output_fir_freq(model, x_train, y_train, args):
    taps = int(model.output_fir_taps)
    if taps <= 1:
        return None
    radius = taps // 2
    base = run_base_sequence(model, x_train).T
    z = base[:, 0] + 1j * base[:, 1]
    target = y_train[:, 0] + 1j * y_train[:, 1]

    n = len(z)
    z_fft = np.fft.fft(z)
    target_fft = np.fft.fft(target)
    freq = np.fft.fftfreq(n, d=1.0 / args.fs_mhz)
    weights = np.full(n, float(args.fir_floor_weight), dtype=np.float64)
    main_mask = (freq >= args.main_band[0]) & (freq <= args.main_band[1])
    lower_mask = (freq >= args.lower_band[0]) & (freq <= args.lower_band[1])
    upper_mask = (freq >= args.upper_band[0]) & (freq <= args.upper_band[1])
    weights[main_mask] = float(args.fir_main_weight)
    weights[lower_mask | upper_mask] = float(args.fir_adjacent_weight)

    omega = 2 * np.pi * np.arange(n) / n
    offsets = np.arange(-radius, radius + 1)
    phase = np.exp(1j * omega[:, None] * offsets[None, :])
    design = z_fft[:, None] * phase
    weighted = design * np.sqrt(weights)[:, None]
    rhs = target_fft * np.sqrt(weights)
    lhs = weighted.conj().T @ weighted + float(args.fir_ridge) * np.eye(taps)
    coeffs = np.linalg.solve(lhs, weighted.conj().T @ rhs)

    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=DTYPE, device=DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=DTYPE, device=DEVICE))
    return coeffs


def fit_output_fir(model, x_train, y_train, args):
    if args.fir_fit_domain == "time":
        return fit_output_fir_time(model, x_train, y_train)
    return fit_output_fir_freq(model, x_train, y_train, args)


def tensor_to_complex(iq):
    arr = np.squeeze(np.asarray(iq))
    if arr.shape[0] == 2 and arr.shape[-1] != 2:
        return arr[0] + 1j * arr[1]
    return arr[:, 0] + 1j * arr[:, 1]


def psd_db(signal, fs_hz=491.52e6, nperseg=4096):
    freq, density = welch(
        signal,
        fs=fs_hz,
        nperseg=min(nperseg, len(signal)),
        return_onesided=False,
    )
    freq = np.fft.fftshift(freq)
    density = np.fft.fftshift(density)
    return freq, 10 * np.log10(np.maximum(density, 1e-300))


def band_average(freq_hz, values_db, band_mhz):
    low_hz = float(band_mhz[0]) * 1e6
    high_hz = float(band_mhz[1]) * 1e6
    mask = (freq_hz >= low_hz) & (freq_hz <= high_hz)
    if not np.any(mask):
        raise ValueError(f"no PSD bins in band {band_mhz} MHz")
    return float(np.mean(values_db[mask]))


def aclr_from_psd(freq_hz, values_db, main_band, lower_band, upper_band):
    main_avg = band_average(freq_hz, values_db, main_band)
    lower_avg = band_average(freq_hz, values_db, lower_band)
    upper_avg = band_average(freq_hz, values_db, upper_band)
    lower_aclr = lower_avg - main_avg
    upper_aclr = upper_avg - main_avg
    return lower_aclr, upper_aclr, max(lower_aclr, upper_aclr)


def compute_aclr_metrics(pred_iq, target_iq, args):
    pred_complex = tensor_to_complex(pred_iq)
    target_complex = tensor_to_complex(target_iq)
    freq_hz, pred_psd = psd_db(pred_complex, fs_hz=args.fs_mhz * 1e6)
    _, target_psd = psd_db(target_complex, fs_hz=args.fs_mhz * 1e6)
    pred_aclr = aclr_from_psd(freq_hz, pred_psd, args.main_band, args.lower_band, args.upper_band)
    target_aclr = aclr_from_psd(freq_hz, target_psd, args.main_band, args.lower_band, args.upper_band)
    return {
        "pred_lower": float(pred_aclr[0]),
        "pred_upper": float(pred_aclr[1]),
        "pred_worst": float(pred_aclr[2]),
        "target_lower": float(target_aclr[0]),
        "target_upper": float(target_aclr[1]),
        "target_worst": float(target_aclr[2]),
        "freq_hz": freq_hz,
        "pred_psd": pred_psd,
        "target_psd": target_psd,
    }


def evaluate_model(model, x_val, y_val, args):
    pred = run_sequence(model, x_val)
    nmse, db = nmse_metrics(pred, y_val)
    aclr = compute_aclr_metrics(pred, y_val, args)
    aclr.update({"nmse_linear": nmse, "nmse_db": db})
    return aclr


def save_eval_figures(metrics, output_prefix, args):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    freq_mhz = metrics["freq_hz"] / 1e6
    target_norm = metrics["target_psd"] - np.max(metrics["target_psd"])
    pred_norm = metrics["pred_psd"] - np.max(metrics["pred_psd"])

    psd_path = FIGURE_DIR / f"{output_prefix}_psd_aclr.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(freq_mhz, target_norm, "--", linewidth=1.1, label="Target")
    ax.plot(freq_mhz, pred_norm, linewidth=1.1, label="EA-TCN")
    for band, label, color in [
        (args.main_band, "Main band", "tab:green"),
        (args.lower_band, "Lower adjacent", "tab:orange"),
        (args.upper_band, "Upper adjacent", "tab:purple"),
    ]:
        ax.axvspan(band[0], band[1], color=color, alpha=0.12, label=label)
    pred_main_avg = band_average(metrics["freq_hz"], pred_norm, args.main_band)
    ax.axhline(
        pred_main_avg + args.aclr_target_dbc,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"{args.aclr_target_dbc:.0f} dBc target",
    )
    ax.set_title("EA-TCN")
    ax.set_xlabel("Frequency offset (MHz)")
    ax.set_ylabel("Normalized PSD (dB)")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(psd_path, dpi=300)
    plt.close(fig)

    aclr_path = FIGURE_DIR / f"{output_prefix}_aclr_bar.png"
    labels = ["Lower", "Upper", "Worst"]
    target_values = [metrics["target_lower"], metrics["target_upper"], metrics["target_worst"]]
    pred_values = [metrics["pred_lower"], metrics["pred_upper"], metrics["pred_worst"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, target_values, width, label="Target", color="tab:red", alpha=0.82)
    ax.bar(x + width / 2, pred_values, width, label="EA-TCN", color="tab:blue", alpha=0.82)
    ax.axhline(args.aclr_target_dbc, color="black", linestyle="--", linewidth=1.2, label="Target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ACLR (dBc)")
    ax.set_title("EA-TCN")
    ax.grid(True, axis="y", linestyle=":", alpha=0.65)
    ax.legend()
    fig.tight_layout()
    fig.savefig(aclr_path, dpi=300)
    plt.close(fig)
    return psd_path, aclr_path


def weighted_spectral_nmse(pred, target, args):
    n = pred.shape[-1]
    pred_complex = torch.complex(pred[:, 0, :], pred[:, 1, :])
    target_complex = torch.complex(target[:, 0, :], target[:, 1, :])
    err_fft = torch.fft.fftshift(
        torch.fft.fft(pred_complex - target_complex, dim=-1, norm="ortho"),
        dim=-1,
    )
    target_fft = torch.fft.fftshift(torch.fft.fft(target_complex, dim=-1, norm="ortho"), dim=-1)
    freq = torch.fft.fftshift(torch.fft.fftfreq(n, d=1.0 / args.fs_mhz, device=pred.device))
    weights = torch.full((n,), float(args.spectral_floor_weight), dtype=pred.dtype, device=pred.device)
    main = (freq >= args.main_band[0]) & (freq <= args.main_band[1])
    lower = (freq >= args.lower_band[0]) & (freq <= args.lower_band[1])
    upper = (freq >= args.upper_band[0]) & (freq <= args.upper_band[1])
    weights = torch.where(main, torch.as_tensor(args.spectral_main_weight, dtype=pred.dtype, device=pred.device), weights)
    adjacent_value = torch.as_tensor(args.spectral_adjacent_weight, dtype=pred.dtype, device=pred.device)
    weights = torch.where(lower | upper, adjacent_value, weights).unsqueeze(0)
    err_power = (err_fft.real * err_fft.real + err_fft.imag * err_fft.imag) * weights
    target_power = (target_fft.real * target_fft.real + target_fft.imag * target_fft.imag) * weights
    return err_power.mean() / (target_power.mean() + torch.as_tensor(1e-24, dtype=pred.dtype, device=pred.device))


def crop_if_needed(*items, margin):
    if margin <= 0 or items[0].shape[-1] <= 2 * margin:
        return items
    return tuple(item[:, :, margin:-margin] for item in items)


def training_loss(pred, truth, distill, args):
    pred_c, truth_c, distill_c = crop_if_needed(pred, truth, distill, margin=args.crop_margin)
    truth_loss = F.mse_loss(pred_c, truth_c)
    if args.diff_loss_weight > 0:
        truth_loss = truth_loss + args.diff_loss_weight * F.mse_loss(
            pred_c[:, :, 1:] - pred_c[:, :, :-1],
            truth_c[:, :, 1:] - truth_c[:, :, :-1],
        )
    if args.spectral_truth_weight > 0:
        truth_loss = truth_loss + args.spectral_truth_weight * weighted_spectral_nmse(pred_c, truth_c, args)

    distill_loss = F.mse_loss(pred_c, distill_c)
    if args.distill_diff_weight > 0:
        distill_loss = distill_loss + args.distill_diff_weight * F.mse_loss(
            pred_c[:, :, 1:] - pred_c[:, :, :-1],
            distill_c[:, :, 1:] - distill_c[:, :, :-1],
        )
    return args.truth_loss_weight * truth_loss + args.distill_loss_weight * distill_loss


def make_loader(x_train, y_train, d_train, args, *, teacher=False):
    frame_length = args.teacher_frame_length if teacher else args.frame_length
    stride_length = args.teacher_stride_length if teacher else args.stride_length
    batch_size = args.teacher_batch_size if teacher else args.batch_size
    return DataLoader(
        DistillFrameDataset(x_train, y_train, d_train, frame_length, stride_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def score_metrics(metrics, target_dbc):
    aclr_penalty = max(0.0, metrics["pred_worst"] - target_dbc)
    return metrics["nmse_db"] + 0.35 * aclr_penalty


def train_model(
    model,
    x_train,
    y_train,
    d_train,
    x_val,
    y_val,
    args,
    *,
    epochs,
    lr,
    checkpoint_path,
    stage_name,
    teacher=False,
):
    loader = make_loader(x_train, y_train, d_train, args, teacher=teacher)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    fit_output_fir(model, x_train, y_train, args)
    best_metrics = evaluate_model(model, x_val, y_val, args)
    best_score = score_metrics(best_metrics, args.aclr_target_dbc)
    save_checkpoint(model, checkpoint_path)

    history = [(0, best_metrics["nmse_db"], best_metrics["pred_worst"])]
    print(
        f">>> {stage_name} initial | NMSE {best_metrics['nmse_db']:.2f} dB | "
        f"ACLR {best_metrics['pred_worst']:.2f} dBc"
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

        nmse, nmse_db = evaluate_full_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        aclr_value = float("nan")
        extra = ""

        if epoch % args.refit_every == 0:
            fit_output_fir(model, x_train, y_train, args)
        if epoch % args.aclr_every == 0 or epoch == epochs:
            metrics = evaluate_model(model, x_val, y_val, args)
            current_score = score_metrics(metrics, args.aclr_target_dbc)
            aclr_value = metrics["pred_worst"]
            if current_score < best_score:
                best_score = current_score
                best_metrics = metrics
                save_checkpoint(model, checkpoint_path)
            extra = f" | ACLR {aclr_value:.2f} dBc | score {current_score:.2f}"

        history.append((epoch, float(nmse_db), aclr_value))
        elapsed = time.time() - start
        print(
            f"{stage_name} Epoch [{epoch}/{epochs}] | "
            f"Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"NMSE {nmse_db:.2f} dB{extra}"
        )

    model.load_state_dict(load_checkpoint(checkpoint_path), strict=True)
    fit_output_fir(model, x_train, y_train, args)
    final_metrics = evaluate_model(model, x_val, y_val, args)
    save_checkpoint(model, checkpoint_path)
    return final_metrics, history


def plot_history(all_history, output_prefix):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
    offset = 0
    for label, history in all_history:
        epochs = [offset + row[0] for row in history]
        nmse = [row[1] for row in history]
        ax1.plot(epochs, nmse, label=label)
        offset = epochs[-1]
    ax1.axhline(-35.0, color="black", linestyle="--", linewidth=1.0, label="-35 dB")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation NMSE (dB)")
    ax1.grid(True, linestyle=":", alpha=0.65)
    ax1.legend(loc="best")
    fig.tight_layout()
    path = FIGURE_DIR / f"{output_prefix}_convergence.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def build_model(config, g_real, g_imag):
    return DiagramDPDNet(g_real, g_imag, config=config).to(DEVICE)


def build_teacher_targets(teacher, x_train, x_val):
    d_train = as_sample_iq(run_sequence(teacher, x_train))
    d_val = as_sample_iq(run_sequence(teacher, x_val))
    return d_train, d_val


def prepare_teacher(x_train, y_train, x_val, y_val, g_real, g_imag, args):
    if args.teacher_mode == "none":
        print(">>> Teacher mode: none. Distillation target is the dataset y.")
        return None, y_train.copy(), y_val.copy(), []

    teacher_path = Path(args.teacher_checkpoint)
    teacher = build_model(TEACHER_CONFIG, g_real, g_imag)
    teacher_mults = real_multiplier_count(TEACHER_CONFIG)
    history = []

    if args.teacher_mode == "load":
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"teacher checkpoint not found: {teacher_path}. "
                "Use --teacher-mode train or pass --teacher-mode none."
            )
        teacher.load_state_dict(load_checkpoint(teacher_path), strict=True)
        print(f">>> Loaded teacher: {teacher_path} | real multipliers={teacher_mults}")
    elif args.teacher_mode == "train":
        print(f">>> Training teacher from scratch | real multipliers={teacher_mults}")
        d_train = y_train.copy()
        teacher_metrics, teacher_history = train_model(
            teacher,
            x_train,
            y_train,
            d_train,
            x_val,
            y_val,
            args,
            epochs=args.teacher_epochs,
            lr=args.teacher_lr,
            checkpoint_path=teacher_path,
            stage_name="teacher",
            teacher=True,
        )
        history.append(("teacher", teacher_history))
        print(
            f">>> Teacher trained | NMSE {teacher_metrics['nmse_db']:.2f} dB | "
            f"ACLR {teacher_metrics['pred_worst']:.2f} dBc | {teacher_path}"
        )
    else:
        raise ValueError(f"unsupported teacher mode: {args.teacher_mode}")

    d_train, d_val = build_teacher_targets(teacher, x_train, x_val)
    teacher_nmse, teacher_db = nmse_metrics(d_val, y_val)
    teacher_aclr = compute_aclr_metrics(d_val, y_val, args)
    print(
        f">>> Teacher target | NMSE {teacher_nmse:.6e} ({teacher_db:.2f} dB) | "
        f"ACLR {teacher_aclr['pred_worst']:.2f} dBc"
    )
    return teacher, d_train, d_val, history


def parse_args():
    parser = argparse.ArgumentParser(description="Final lightweight DPD script for <=999 real multipliers.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--teacher-mode", choices=["load", "train", "none"], default="load")
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER_CHECKPOINT))
    parser.add_argument("--teacher-epochs", type=int, default=500)
    parser.add_argument("--teacher-lr", type=float, default=3e-3)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--teacher-frame-length", type=int, default=1024)
    parser.add_argument("--teacher-stride-length", type=int, default=32)

    parser.add_argument("--student-preset", choices=sorted(STUDENT_CONFIGS), default=DEFAULT_STUDENT_PRESET)
    parser.add_argument("--max-student-multipliers", type=int, default=999)
    parser.add_argument("--stage1-epochs", type=int, default=120)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--stage3-epochs", type=int, default=80)
    parser.add_argument("--stage1-lr", type=float, default=5e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=64)
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=15)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--truth-loss-weight", type=float, default=1.0)
    parser.add_argument("--distill-loss-weight", type=float, default=0.06)
    parser.add_argument("--diff-loss-weight", type=float, default=0.005)
    parser.add_argument("--distill-diff-weight", type=float, default=0.03)
    parser.add_argument("--spectral-truth-weight", type=float, default=5e-5)
    parser.add_argument("--spectral-main-weight", type=float, default=1.0)
    parser.add_argument("--spectral-adjacent-weight", type=float, default=4.0)
    parser.add_argument("--spectral-floor-weight", type=float, default=0.05)
    parser.add_argument("--refit-every", type=int, default=10)
    parser.add_argument("--aclr-every", type=int, default=10)

    parser.add_argument("--fir-fit-domain", choices=["time", "freq"], default="freq")
    parser.add_argument("--fir-main-weight", type=float, default=1.0)
    parser.add_argument("--fir-adjacent-weight", type=float, default=1.2)
    parser.add_argument("--fir-floor-weight", type=float, default=0.02)
    parser.add_argument("--fir-ridge", type=float, default=1e-7)

    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(-100.0, 100.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-245.0, -180.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(180.0, 245.0))
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0)
    parser.add_argument("--nmse-target-db", type=float, default=-35.0)

    parser.add_argument("--final-checkpoint", default=str(DEFAULT_FINAL_CHECKPOINT))
    parser.add_argument("--output-prefix", default="final_lightweight_992_rich9_H10_L4_FIR45")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate --final-checkpoint and regenerate PSD/ACLR figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    final_checkpoint = Path(args.final_checkpoint)
    x_train, y_train, x_val, y_val, g_real, g_imag = split_data(data_path)
    student_config = dict(STUDENT_CONFIGS[args.student_preset])

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Train/val samples: {len(x_train)} / {len(x_val)}")
    print(f">>> Student preset: {args.student_preset}")
    print(f">>> Student config: {student_config}")
    print(f">>> Student real multipliers: {real_multiplier_count(student_config)}")
    print(f">>> Teacher mode: {args.teacher_mode}")

    student = build_model(student_config, g_real, g_imag)
    student_mults = real_multiplier_count(student_config)
    if student_mults > args.max_student_multipliers:
        raise RuntimeError(
            f"student multiplier budget exceeded: "
            f"{student_mults} > {args.max_student_multipliers}"
        )

    if args.evaluate_only:
        if not final_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {final_checkpoint}")
        student.load_state_dict(load_checkpoint(final_checkpoint), strict=True)
        final_metrics = evaluate_model(student, x_val, y_val, args)
        psd_path, aclr_path = save_eval_figures(final_metrics, args.output_prefix, args)
        print("\n>>> LIGHTWEIGHT EVALUATION RESULT")
        print(f"Checkpoint: {final_checkpoint}")
        print(f"PSD figure: {psd_path}")
        print(f"ACLR figure: {aclr_path}")
        print(f"Real multipliers: {student_mults}")
        print(f"Linear NMSE: {final_metrics['nmse_linear']:.6e}")
        print(f"dB NMSE: {final_metrics['nmse_db']:.2f} dB")
        print(
            f"ACLR worst: target y {final_metrics['target_worst']:.2f} dBc | "
            f"pred y {final_metrics['pred_worst']:.2f} dBc | "
            f"margin {args.aclr_target_dbc - final_metrics['pred_worst']:.2f} dB"
        )
        return

    _, d_train, _, teacher_history = prepare_teacher(x_train, y_train, x_val, y_val, g_real, g_imag, args)

    all_history = list(teacher_history)
    stage1_path = final_checkpoint.with_name(final_checkpoint.stem + "_stage1.pth")
    stage2_path = final_checkpoint.with_name(final_checkpoint.stem + "_stage2.pth")

    metrics1, history1 = train_model(
        student,
        x_train,
        y_train,
        d_train,
        x_val,
        y_val,
        args,
        epochs=args.stage1_epochs,
        lr=args.stage1_lr,
        checkpoint_path=stage1_path,
        stage_name="student-stage1",
    )
    all_history.append(("stage1", history1))
    print(f">>> Stage 1 done | NMSE {metrics1['nmse_db']:.2f} dB | ACLR {metrics1['pred_worst']:.2f} dBc")

    metrics2, history2 = train_model(
        student,
        x_train,
        y_train,
        d_train,
        x_val,
        y_val,
        args,
        epochs=args.stage2_epochs,
        lr=args.stage2_lr,
        checkpoint_path=stage2_path,
        stage_name="student-stage2",
    )
    all_history.append(("stage2", history2))
    print(f">>> Stage 2 done | NMSE {metrics2['nmse_db']:.2f} dB | ACLR {metrics2['pred_worst']:.2f} dBc")

    final_metrics, history3 = train_model(
        student,
        x_train,
        y_train,
        d_train,
        x_val,
        y_val,
        args,
        epochs=args.stage3_epochs,
        lr=args.stage3_lr,
        checkpoint_path=final_checkpoint,
        stage_name="student-stage3",
    )
    all_history.append(("stage3", history3))

    save_checkpoint(student, final_checkpoint)
    convergence_path = plot_history(all_history, args.output_prefix)
    psd_path, aclr_path = save_eval_figures(final_metrics, args.output_prefix, args)

    print("\n>>> FINAL LIGHTWEIGHT REPORT RESULT")
    print(f"Checkpoint: {final_checkpoint}")
    print(f"Convergence figure: {convergence_path}")
    print(f"PSD figure: {psd_path}")
    print(f"ACLR figure: {aclr_path}")
    print(f"Real multipliers: {student_mults}")
    print(f"Linear NMSE: {final_metrics['nmse_linear']:.6e}")
    print(f"dB NMSE: {final_metrics['nmse_db']:.2f} dB")
    print(
        f"ACLR worst: target y {final_metrics['target_worst']:.2f} dBc | "
        f"pred y {final_metrics['pred_worst']:.2f} dBc | "
        f"margin {args.aclr_target_dbc - final_metrics['pred_worst']:.2f} dB"
    )
    print(
        f"Target check: NMSE <= {args.nmse_target_db:.2f} dB, "
        f"ACLR <= {args.aclr_target_dbc:.2f} dBc"
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
