import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from scipy.signal import welch
from torch.utils.data import DataLoader, Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINTS = [
    "checkpoints/AI-DPD/final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth",
    "checkpoints/AI-DPD/final_minimal_H15_L6_rich9_FIR39_2268_NMSE40p03_ACLR45p04.pth",
    "checkpoints/AI-DPD/final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth",
    "checkpoints/AI-DPD/step800_scratch_h10_l4_base6_rank3_fir41_786.pth",
    "checkpoints/AI-DPD/final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth",
]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NARROW_MAIN_BAND_MHZ = (31.64, 36.64)
NARROW_LOWER_BAND_MHZ = (-129.84, -124.84)
NARROW_UPPER_BAND_MHZ = (121.08, 126.08)


def resolve_project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return project_candidate


@dataclass
class EvalResult:
    checkpoint: str
    real_multipliers: int
    nmse_linear: float
    nmse_db: float
    dataset_aclr_lower_db: float
    dataset_aclr_upper_db: float
    dataset_aclr_worst_db: float
    residual_aclr_lower_db: float
    residual_aclr_upper_db: float
    residual_aclr_worst_db: float
    residual_dataset_aclr_gap_db: float
    dataset_min_point_aclr_lower_db: float
    dataset_min_point_aclr_upper_db: float
    dataset_min_point_aclr_worst_db: float
    residual_min_point_aclr_lower_db: float
    residual_min_point_aclr_upper_db: float
    residual_min_point_aclr_worst_db: float
    dataset_narrow_aclr_lower_db: float
    dataset_narrow_aclr_upper_db: float
    dataset_narrow_aclr_worst_db: float
    residual_narrow_aclr_lower_db: float
    residual_narrow_aclr_upper_db: float
    residual_narrow_aclr_worst_db: float
    psd_path: str


class FramedDataset(Dataset):
    def __init__(self, features, targets, frame_length, stride_length):
        self.features = torch.as_tensor(features, dtype=torch.float64)
        self.targets = torch.as_tensor(targets, dtype=torch.float64)
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.num_frames = (self.features.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        end = start + self.frame_length
        return self.features[start:end, :].T, self.targets[start:end, :].T


class SpectrumAttention(nn.Module):
    def __init__(self, channels, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.env_conv = nn.Conv1d(
            input_channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
            dtype=torch.float64,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp, amp3, amp5, amp7):
        if self.input_channels == 3:
            env_features = torch.cat([amp, amp3, amp7], dim=1)
        elif self.input_channels == 4:
            env_features = torch.cat([amp, amp3, amp5, amp7], dim=1)
        else:
            raise ValueError(f"Unsupported attention input channels: {self.input_channels}")
        return x * self.sigmoid(self.env_conv(env_features))


class AutoDiagramDPDNet(nn.Module):

    def __init__(self, state_dict):
        super().__init__()
        if "in_conv.weight" not in state_dict or "out_conv.weight" not in state_dict:
            raise ValueError("Only DiagramDPDNet-style checkpoints are supported by this evaluator.")

        self.hidden = state_dict["in_conv.weight"].shape[0]
        self.input_channels = state_dict["in_conv.weight"].shape[1]
        self.attn_channels = state_dict["attn.env_conv.weight"].shape[1]
        self.has_g3 = "g3_real" in state_dict
        self.has_g5 = "g5_real" in state_dict
        self.extra_orders = sorted(
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("extra_poly_real.")
        )
        self.has_fir = "fir_real" in state_dict and "fir_imag" in state_dict

        block_ids = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("tcn_blocks.") and key.endswith(".0.weight")
            }
        )
        self.dilations = [1, 2, 4, 8, 16, 32][: len(block_ids)]

        self.g_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if self.has_g3:
            self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if self.has_g5:
            self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.extra_poly_real = nn.ParameterDict()
        self.extra_poly_imag = nn.ParameterDict()
        for order in self.extra_orders:
            self.extra_poly_real[str(order)] = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.extra_poly_imag[str(order)] = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.in_conv = nn.Conv1d(
            self.input_channels,
            self.hidden,
            kernel_size=1,
            bias=False,
            dtype=torch.float64,
        )
        self.act = nn.SiLU()
        blocks = []
        for block_id, dilation in zip(block_ids, self.dilations):
            layers = [
                nn.Conv1d(
                    self.hidden,
                    self.hidden,
                    kernel_size=5,
                    padding=2 * dilation,
                    dilation=dilation,
                    groups=self.hidden,
                    bias=False,
                    dtype=torch.float64,
                )
            ]
            if f"tcn_blocks.{block_id}.2.weight" in state_dict:
                rank = state_dict[f"tcn_blocks.{block_id}.1.weight"].shape[0]
                layers.extend(
                    [
                        nn.Conv1d(self.hidden, rank, kernel_size=1, bias=False, dtype=torch.float64),
                        nn.Conv1d(rank, self.hidden, kernel_size=1, bias=False, dtype=torch.float64),
                    ]
                )
            else:
                layers.append(
                    nn.Conv1d(
                        self.hidden,
                        self.hidden,
                        kernel_size=1,
                        bias=False,
                        dtype=torch.float64,
                    )
                )
            layers.append(nn.SiLU())
            blocks.append(nn.Sequential(*layers))
        self.tcn_blocks = nn.ModuleList(blocks)
        self.attn = SpectrumAttention(self.hidden, self.attn_channels)
        self.out_conv = nn.Conv1d(
            self.hidden,
            2,
            kernel_size=1,
            bias=False,
            dtype=torch.float64,
        )
        if self.has_fir:
            self.fir_real = nn.Parameter(torch.zeros_like(state_dict["fir_real"], dtype=torch.float64))
            self.fir_imag = nn.Parameter(torch.zeros_like(state_dict["fir_imag"], dtype=torch.float64))
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]
        amp = torch.sqrt(i_x**2 + q_x**2 + 1e-12)
        amp2 = amp**2
        amp3 = amp**3
        amp4 = amp2**2
        amp5 = amp**5
        amp7 = amp**7

        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        if self.has_g3:
            x3_i, x3_q = i_x * amp2, q_x * amp2
            lin_i = lin_i + x3_i * self.g3_real - x3_q * self.g3_imag
            lin_q = lin_q + x3_i * self.g3_imag + x3_q * self.g3_real

        if self.has_g5:
            x5_i, x5_q = i_x * amp4, q_x * amp4
            lin_i = lin_i + x5_i * self.g5_real - x5_q * self.g5_imag
            lin_q = lin_q + x5_i * self.g5_imag + x5_q * self.g5_real

        for order in self.extra_orders:
            amp_power = amp ** (order - 1)
            xo_i, xo_q = i_x * amp_power, q_x * amp_power
            gr = self.extra_poly_real[str(order)]
            gi = self.extra_poly_imag[str(order)]
            lin_i = lin_i + xo_i * gr - xo_q * gi
            lin_q = lin_q + xo_i * gi + xo_q * gr

        linear_out = torch.cat([lin_i, lin_q], dim=1)
        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)

        if self.input_channels == 6:
            features = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)
        elif self.input_channels == 9:
            i_past = torch.roll(i_x, shifts=1, dims=2)
            q_past = torch.roll(q_x, shifts=1, dims=2)
            features = torch.cat(
                [i_x, q_x, i_future, q_future, i_past, q_past, amp, amp3, amp5],
                dim=1,
            )
        elif self.input_channels == 11:
            i_past = torch.roll(i_x, shifts=1, dims=2)
            q_past = torch.roll(q_x, shifts=1, dims=2)
            i_delta = i_x - i_past
            q_delta = q_x - q_past
            features = torch.cat(
                [i_x, q_x, i_future, q_future, i_past, q_past, i_delta, q_delta, amp, amp3, amp5],
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported model input channels: {self.input_channels}")

        feat = self.act(self.in_conv(features))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        feat = self.attn(feat, amp, amp3, amp5, amp7)
        y = linear_out + self.out_conv(feat)
        if not self.has_fir:
            return y
        pad = len(self.fir_real) // 2
        yi, yq = y[:, 0:1, :], y[:, 1:2, :]
        kr = self.fir_real.view(1, 1, -1)
        ki = self.fir_imag.view(1, 1, -1)
        out_i = nn.functional.conv1d(yi, kr, padding=pad) - nn.functional.conv1d(yq, ki, padding=pad)
        out_q = nn.functional.conv1d(yi, ki, padding=pad) + nn.functional.conv1d(yq, kr, padding=pad)
        return torch.cat([out_i, out_q], dim=1)


def load_state_dict(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload: {checkpoint_path}")
    return state


def real_multiplier_count(state_dict):
    conv_macs = sum(
        tensor.numel()
        for key, tensor in state_dict.items()
        if hasattr(tensor, "numel")
        and (
            key == "in_conv.weight"
            or key == "out_conv.weight"
            or key.startswith("tcn_blocks.")
            or key == "attn.env_conv.weight"
        )
    )
    polynomial_orders = 1
    if "g3_real" in state_dict:
        polynomial_orders += 1
    if "g5_real" in state_dict:
        polynomial_orders += 1
    polynomial_orders += sum(1 for key in state_dict if key.startswith("extra_poly_real."))
    fir_multipliers = 0
    if "fir_real" in state_dict:
        fir_multipliers = 4 * int(state_dict["fir_real"].numel())
    return int(conv_macs + 4 * polynomial_orders + fir_multipliers)


def load_iq_arrays(data_path, normalization):
    data = scipy.io.loadmat(data_path)
    raw_x = data["x"].flatten()
    raw_y = data["y"].flatten()
    if "error" not in data:
        raise KeyError(f"MAT file {data_path} does not contain required 'error' data.")
    raw_error = data["error"].flatten()
    rms_x = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    rms_y = np.sqrt(np.mean(np.abs(raw_y) ** 2))
    max_x = np.max(np.abs(raw_x))
    max_y = np.max(np.abs(raw_y))

    if normalization == "rms":
        scale_x = rms_x
        scale_y = rms_y
    elif normalization == "max":
        scale_x = max_x
        scale_y = max_y
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    x = raw_x / scale_x
    y = raw_y / scale_y
    error = raw_error / scale_y
    min_len = min(len(x), len(y), len(error))
    x = x[:min_len]
    y = y[:min_len]
    error = error[:min_len]
    features = np.stack([np.real(x), np.imag(x)], axis=1)
    targets = np.stack([np.real(y), np.imag(y)], axis=1)
    dataset_errors = np.stack([np.real(error), np.imag(error)], axis=1)
    return features, targets, dataset_errors, rms_x, rms_y, max_x, max_y


def tensor_to_complex(iq):
    iq = np.squeeze(iq)
    if iq.shape[0] != 2 and iq.shape[-1] == 2:
        iq = iq.T
    return iq[0] + 1j * iq[1]


def psd_db(signal, fs_hz, nperseg):
    frequencies, density = welch(
        signal,
        fs=fs_hz,
        nperseg=min(nperseg, len(signal)),
        return_onesided=False,
    )
    frequencies = np.fft.fftshift(frequencies)
    density = np.fft.fftshift(density)
    return frequencies, 10 * np.log10(np.maximum(density, 1e-300))


def band_average(freq_hz, values_db, band_mhz):
    lo_hz = band_mhz[0] * 1e6
    hi_hz = band_mhz[1] * 1e6
    mask = (freq_hz >= lo_hz) & (freq_hz <= hi_hz)
    if not np.any(mask):
        raise ValueError(f"Band {band_mhz} MHz has no PSD bins.")
    return float(np.mean(values_db[mask]))


def aclr_from_psd(
    freq_hz,
    error_values_db,
    original_values_db,
    main_band_mhz,
    lower_band_mhz,
    upper_band_mhz,
):
    """Return error-adjacent minus original-main ACLR in dBc."""
    main_avg = band_average(freq_hz, original_values_db, main_band_mhz)
    lower_avg = band_average(freq_hz, error_values_db, lower_band_mhz)
    upper_avg = band_average(freq_hz, error_values_db, upper_band_mhz)
    lower_aclr = lower_avg - main_avg
    upper_aclr = upper_avg - main_avg
    return lower_aclr, upper_aclr, max(lower_aclr, upper_aclr)


def band_minimum(freq_hz, values_db, band_mhz):
    lo_hz, hi_hz = sorted((band_mhz[0] * 1e6, band_mhz[1] * 1e6))
    mask = (freq_hz >= lo_hz) & (freq_hz <= hi_hz)
    if not np.any(mask):
        raise ValueError(f"Band {band_mhz} MHz has no PSD bins.")
    return float(np.min(values_db[mask]))


def aclr_from_psd_minimum(freq_hz, error_values_db, original_values_db, main_band_mhz, lower_band_mhz, upper_band_mhz):
    """Return the requested lowest-point ACLR diagnostic in dBc."""
    main_min = band_minimum(freq_hz, original_values_db, main_band_mhz)
    lower_aclr = band_minimum(freq_hz, error_values_db, lower_band_mhz) - main_min
    upper_aclr = band_minimum(freq_hz, error_values_db, upper_band_mhz) - main_min
    return lower_aclr, upper_aclr, max(lower_aclr, upper_aclr)


def evaluate_nmse_frames(model, x_val, y_val, frame_length, stride_length, batch_size):
    dataset = FramedDataset(x_val, y_val, frame_length, stride_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_error = 0.0
    total_target = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            pred = model(batch_x)
            total_error += torch.sum((pred - batch_y) ** 2).item()
            total_target += torch.sum(batch_y**2).item()
    nmse = total_error / total_target
    return nmse, 10 * np.log10(nmse)


def evaluate_full_sequence(model, x_val, y_val):
    x_tensor = torch.as_tensor(x_val.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        pred = model(x_tensor).squeeze(0).cpu().numpy()
    target = y_val.T
    nmse = np.sum((pred - target) ** 2) / np.sum(target**2)
    return pred, target, nmse, 10 * np.log10(nmse)


def save_direct_psd_plot(
    checkpoint_name,
    target_iq,
    pred_iq,
    dataset_error_iq,
    out_dir,
    fs_hz,
    nperseg,
    main_band_mhz,
    lower_band_mhz,
    upper_band_mhz,
    aclr_target_dbc,
):
    target_complex = tensor_to_complex(target_iq)
    pred_complex = tensor_to_complex(pred_iq)
    residual_complex = target_complex - pred_complex
    dataset_error_complex = tensor_to_complex(dataset_error_iq)
    freq_hz, target_psd_db = psd_db(target_complex, fs_hz, nperseg)
    _, residual_psd_db = psd_db(residual_complex, fs_hz, nperseg)
    _, dataset_error_psd_db = psd_db(dataset_error_complex, fs_hz, nperseg)

    dataset_aclr_l, dataset_aclr_u, dataset_aclr_worst = aclr_from_psd(
        freq_hz,
        dataset_error_psd_db,
        target_psd_db,
        main_band_mhz,
        lower_band_mhz,
        upper_band_mhz,
    )
    residual_aclr_l, residual_aclr_u, residual_aclr_worst = aclr_from_psd(
        freq_hz,
        residual_psd_db,
        target_psd_db,
        main_band_mhz,
        lower_band_mhz,
        upper_band_mhz,
    )
    dataset_min_point = aclr_from_psd_minimum(
        freq_hz, dataset_error_psd_db, target_psd_db, main_band_mhz, lower_band_mhz, upper_band_mhz
    )
    residual_min_point = aclr_from_psd_minimum(
        freq_hz, residual_psd_db, target_psd_db, main_band_mhz, lower_band_mhz, upper_band_mhz
    )
    dataset_narrow = aclr_from_psd(
        freq_hz, dataset_error_psd_db, target_psd_db,
        NARROW_MAIN_BAND_MHZ, NARROW_LOWER_BAND_MHZ, NARROW_UPPER_BAND_MHZ,
    )
    residual_narrow = aclr_from_psd(
        freq_hz, residual_psd_db, target_psd_db,
        NARROW_MAIN_BAND_MHZ, NARROW_LOWER_BAND_MHZ, NARROW_UPPER_BAND_MHZ,
    )

    target_peak_db = np.max(target_psd_db)
    target_norm = target_psd_db - target_peak_db
    residual_norm = residual_psd_db - target_peak_db
    dataset_error_norm = dataset_error_psd_db - target_peak_db
    residual_dataset_gap = residual_aclr_worst - dataset_aclr_worst

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        freq_hz / 1e6,
        target_norm,
        color="tab:red",
        linestyle="--",
        linewidth=1.1,
        label="Target",
    )
    ax.plot(
        freq_hz / 1e6,
        dataset_error_norm,
        color="tab:gray",
        linestyle=":",
        linewidth=1.0,
        label="Dataset error",
    )
    ax.plot(
        freq_hz / 1e6,
        residual_norm,
        color="tab:blue",
        linewidth=1.1,
        label="Model residual (target - pred)",
    )

    for band, color, label in [
        (main_band_mhz, "tab:green", "Main avg band"),
        (lower_band_mhz, "tab:orange", "Lower leakage band"),
        (upper_band_mhz, "tab:purple", "Upper leakage band"),
    ]:
        ax.axvspan(band[0], band[1], color=color, alpha=0.12, label=label)
    main_norm_avg = band_average(freq_hz, target_norm, main_band_mhz)
    ax.axhline(
        main_norm_avg + aclr_target_dbc,
        color="black",
        linestyle=":",
        linewidth=1.1,
        label=f"{aclr_target_dbc:.0f} dBc",
    )

    ax.set_title("Error-adjacent / original-main ACLR")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Normalized PSD (dB)")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="best")
    fig.tight_layout()

    safe_name = Path(checkpoint_name).stem.replace(" ", "_")
    out_path = out_dir / f"{safe_name}_psd.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return (
        out_path,
        dataset_aclr_l,
        dataset_aclr_u,
        dataset_aclr_worst,
        residual_aclr_l,
        residual_aclr_u,
        residual_aclr_worst,
        residual_dataset_gap,
        *dataset_min_point,
        *residual_min_point,
        *dataset_narrow,
        *residual_narrow,
    )


def evaluate_checkpoint(args, checkpoint_path, x_val, y_val, dataset_error_val):
    state_dict = load_state_dict(checkpoint_path)
    model = AutoDiagramDPDNet(state_dict).to(DEVICE).eval()
    multipliers = real_multiplier_count(state_dict)

    pred_iq, target_iq, nmse_linear, nmse_db = evaluate_full_sequence(model, x_val, y_val)

    (
        psd_path,
        dataset_aclr_l,
        dataset_aclr_u,
        dataset_aclr_worst,
        residual_aclr_l,
        residual_aclr_u,
        residual_aclr_worst,
        residual_dataset_gap,
        dataset_min_point_l,
        dataset_min_point_u,
        dataset_min_point_worst,
        residual_min_point_l,
        residual_min_point_u,
        residual_min_point_worst,
        dataset_narrow_l,
        dataset_narrow_u,
        dataset_narrow_worst,
        residual_narrow_l,
        residual_narrow_u,
        residual_narrow_worst,
    ) = save_direct_psd_plot(
        checkpoint_path.name,
        target_iq,
        pred_iq,
        dataset_error_val.T,
        args.out_dir,
        args.fs_mhz * 1e6,
        args.nperseg,
        args.main_band,
        args.lower_band,
        args.upper_band,
        args.aclr_target_dbc,
    )

    return EvalResult(
        checkpoint=checkpoint_path.name,
        real_multipliers=multipliers,
        nmse_linear=nmse_linear,
        nmse_db=nmse_db,
        dataset_aclr_lower_db=dataset_aclr_l,
        dataset_aclr_upper_db=dataset_aclr_u,
        dataset_aclr_worst_db=dataset_aclr_worst,
        residual_aclr_lower_db=residual_aclr_l,
        residual_aclr_upper_db=residual_aclr_u,
        residual_aclr_worst_db=residual_aclr_worst,
        residual_dataset_aclr_gap_db=residual_dataset_gap,
        dataset_min_point_aclr_lower_db=dataset_min_point_l,
        dataset_min_point_aclr_upper_db=dataset_min_point_u,
        dataset_min_point_aclr_worst_db=dataset_min_point_worst,
        residual_min_point_aclr_lower_db=residual_min_point_l,
        residual_min_point_aclr_upper_db=residual_min_point_u,
        residual_min_point_aclr_worst_db=residual_min_point_worst,
        dataset_narrow_aclr_lower_db=dataset_narrow_l,
        dataset_narrow_aclr_upper_db=dataset_narrow_u,
        dataset_narrow_aclr_worst_db=dataset_narrow_worst,
        residual_narrow_aclr_lower_db=residual_narrow_l,
        residual_narrow_aclr_upper_db=residual_narrow_u,
        residual_narrow_aclr_worst_db=residual_narrow_worst,
        psd_path=str(psd_path),
    )


def write_summary(results, out_path):
    fieldnames = [
        "checkpoint",
        "real_multipliers",
        "nmse_linear",
        "nmse_db",
        "dataset_aclr_lower_db",
        "dataset_aclr_upper_db",
        "dataset_aclr_worst_db",
        "residual_aclr_lower_db",
        "residual_aclr_upper_db",
        "residual_aclr_worst_db",
        "residual_dataset_aclr_gap_db",
        "dataset_min_point_aclr_lower_db",
        "dataset_min_point_aclr_upper_db",
        "dataset_min_point_aclr_worst_db",
        "residual_min_point_aclr_lower_db",
        "residual_min_point_aclr_upper_db",
        "residual_min_point_aclr_worst_db",
        "dataset_narrow_aclr_lower_db",
        "dataset_narrow_aclr_upper_db",
        "dataset_narrow_aclr_worst_db",
        "residual_narrow_aclr_lower_db",
        "residual_narrow_aclr_upper_db",
        "residual_narrow_aclr_worst_db",
        "psd_path",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def print_table(results):
    headers = [
        "checkpoint",
        "MACs",
        "NMSE",
        "NMSE(dB)",
        "Data_mean/min/narrow",
        "Residual_mean/min/narrow",
        "PSD",
    ]
    rows = []
    for r in results:
        rows.append(
            [
                r.checkpoint,
                str(r.real_multipliers),
                f"{r.nmse_linear:.6e}",
                f"{r.nmse_db:.2f}",
                f"{r.dataset_aclr_worst_db:.2f}/{r.dataset_min_point_aclr_worst_db:.2f}/{r.dataset_narrow_aclr_worst_db:.2f}",
                f"{r.residual_aclr_worst_db:.2f}/{r.residual_min_point_aclr_worst_db:.2f}/{r.residual_narrow_aclr_worst_db:.2f}",
                r.psd_path,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row):
        return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print(fmt(headers))
    print(fmt(["-" * width for width in widths]))
    for row in rows:
        print(fmt(row))


def parse_args():
    parser = argparse.ArgumentParser(description="Unified evaluator for final Huawei AI-DPD checkpoints.")
    parser.add_argument(
        "--data",
        default="data/DPD_data.mat",
        help="Path to the MAT file containing x, y, and dataset reference error.",
    )
    parser.add_argument(
        "--checkpoint",
        nargs="*",
        default=None,
        help="One or more checkpoint paths. Defaults to the current final report candidates.",
    )
    parser.add_argument("--out-dir", default="eval_outputs_direct/unified_eval", help="Directory for CSV and PSD plots.")
    parser.add_argument("--normalization", choices=["rms", "max"], default="rms")
    parser.add_argument("--split", type=float, default=0.8, help="Train/validation split point.")
    parser.add_argument("--frame-length", type=int, default=800)
    parser.add_argument("--frame-stride", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument(
        "--main-band",
        type=float,
        nargs=2,
        default=(20.0, 80.0),
        metavar=("LO_MHZ", "HI_MHZ"),
        help="Original target main-channel band used by error-adjacent/original-main ACLR.",
    )
    parser.add_argument(
        "--lower-band",
        type=float,
        nargs=2,
        default=(-180.0, -120.0),
        metavar=("LO_MHZ", "HI_MHZ"),
    )
    parser.add_argument(
        "--upper-band",
        type=float,
        nargs=2,
        default=(120.0, 180.0),
        metavar=("LO_MHZ", "HI_MHZ"),
    )
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0, help="Target worst-band ACLR in dBc.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data = resolve_project_path(args.data)
    args.out_dir = resolve_project_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = args.checkpoint if args.checkpoint is not None else DEFAULT_CHECKPOINTS
    checkpoint_paths = [resolve_project_path(path) for path in checkpoints]
    checkpoint_paths = [path for path in checkpoint_paths if path.exists()]
    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoint files were found.")

    features, targets, dataset_errors, rms_x, rms_y, max_x, max_y = load_iq_arrays(
        args.data, args.normalization
    )
    split_idx = int(min(len(features), len(targets), len(dataset_errors)) * args.split)
    x_val = features[split_idx:]
    y_val = targets[split_idx:]
    dataset_error_val = dataset_errors[split_idx:]

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Data: {args.data} | validation samples: {len(x_val)}")
    print(
        "ACLR rules (reported together): mean_db=mean of PSD dB values over the full bands; "
        "min_point=lowest error PSD point minus lowest target PSD point; "
        f"narrow=mean_db over fixed 5 MHz bands {NARROW_MAIN_BAND_MHZ}/"
        f"{NARROW_LOWER_BAND_MHZ}/{NARROW_UPPER_BAND_MHZ}. "
        "Dataset reference error comes from DPD_data.mat; model error is target_y-pred_y. "
        f"main={tuple(args.main_band)} MHz, lower={tuple(args.lower_band)} MHz, "
        f"upper={tuple(args.upper_band)} MHz, target={args.aclr_target_dbc:.2f} dBc"
    )

    results = []
    for checkpoint_path in checkpoint_paths:
        try:
            results.append(
                evaluate_checkpoint(
                    args,
                    checkpoint_path,
                    x_val,
                    y_val,
                    dataset_error_val,
                )
            )
        except Exception as exc:
            print(f"[skip] {checkpoint_path}: {exc}")

    if not results:
        raise RuntimeError("No checkpoints could be evaluated.")

    summary_path = args.out_dir / "summary.csv"
    write_summary(results, summary_path)
    print_table(results)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
