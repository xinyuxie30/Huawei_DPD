import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
from scipy.signal import welch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 16
FRAME_LENGTH = 800
STRIDE_LENGTH = 32
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
EXPERIMENT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
BEST_CHECKPOINT = EXPERIMENT_DIR / "AI-DPD_best.pth"
REFERENCE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "archive" / "best_swa_model_A.pth"
DATA_PATH = PROJECT_ROOT / "data" / "DPD_data.mat"


class OptimalSpectrumAttention(nn.Module):
    def __init__(self, channels, input_channels=3, mode="amp37"):
        super(OptimalSpectrumAttention, self).__init__()
        self.input_channels = input_channels
        self.mode = mode
        self.env_conv = nn.Conv1d(input_channels, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp, amp3, amp7, amp5=None, thermal=None):
        if self.mode == "thermal16":
            if self.input_channels != 3:
                raise ValueError("thermal16 attention expects exactly 3 input channels")
            if thermal is None:
                raise ValueError("thermal feature is required for thermal16 attention")
            env_features = torch.cat([amp, amp3, thermal], dim=1)
        elif self.mode == "amp37_thermal16":
            if self.input_channels != 4:
                raise ValueError("amp37_thermal16 attention expects exactly 4 input channels")
            if thermal is None:
                raise ValueError("thermal feature is required for amp37_thermal16 attention")
            env_features = torch.cat([amp, amp3, amp7, thermal], dim=1)
        elif self.input_channels == 3:
            env_features = torch.cat([amp, amp3, amp7], dim=1)
        elif self.input_channels == 4:
            if amp5 is None:
                raise ValueError("amp5 is required for 4-channel attention")
            env_features = torch.cat([amp, amp3, amp5, amp7], dim=1)
        else:
            raise ValueError(f"Unsupported attention input channels: {self.input_channels}")
        weights = self.sigmoid(self.env_conv(env_features))
        return x * weights


def preset_config(name):
    presets = {
        "rich9_h15_l6_fir43_2284": {
            "hidden": 15,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8, 16, 32],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 43,
            "scratch_default": True,
        },
        "rich9_h15_l6_fir39_2268": {
            "hidden": 15,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8, 16, 32],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 39,
            "scratch_default": True,
        },
        "memory4_fir19_998": {
            "hidden": 10,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8],
            "attention_channels": 4,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 39,
        },
        "rich9_h15_l6_fir95_2492": {
            "hidden": 15,
            "input_mode": "rich9",
            "dilations": [1, 2, 4, 8, 16, 32],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 95,
            "scratch_default": True,
        },
        "h10_l3_base6_fir41_796": {
            "hidden": 10,
            "input_mode": "base6",
            "dilations": [1, 2, 4],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "output_fir_taps": 41,
        },
        "h10_l4_base6_rank3_fir41_786": {
            "hidden": 10,
            "input_mode": "base6",
            "dilations": [1, 2, 4, 8],
            "attention_channels": 3,
            "poly_orders": [1, 3, 5],
            "pointwise_rank": 3,
            "output_fir_taps": 41,
        },
    }
    if name not in presets:
        raise ValueError(f"Unknown preset: {name}")
    config = dict(presets[name])
    config["preset"] = name
    return config


def input_channels_for_mode(input_mode):
    if input_mode == "base6":
        return 6
    if input_mode == "rich9":
        return 9
    if input_mode == "rich11":
        return 11
    raise ValueError(f"Unsupported input mode: {input_mode}")


def real_multiplier_count_from_config(config):
    hidden = int(config["hidden"])
    input_channels = input_channels_for_mode(config["input_mode"])
    attention_channels = int(config["attention_channels"])
    blocks = len(config["dilations"])
    polynomial_orders = len(config.get("poly_orders", [1, 3, 5]))
    pointwise_rank = config.get("pointwise_rank")
    if pointwise_rank is None:
        block_multipliers = hidden * 5 + hidden * hidden
    else:
        rank = int(pointwise_rank)
        block_multipliers = hidden * 5 + 2 * hidden * rank
    fir_taps = int(config.get("output_fir_taps", 1))
    fir_multipliers = 4 * fir_taps if fir_taps > 1 else 0
    return int(
        4 * polynomial_orders
        + input_channels * hidden
        + blocks * block_multipliers
        + 2 * hidden
        + hidden * attention_channels * 3
        + fir_multipliers
    )


def build_features(i_x, q_x, amp, amp3, amp5, input_mode):
    i_future = torch.roll(i_x, shifts=-1, dims=2)
    q_future = torch.roll(q_x, shifts=-1, dims=2)
    if input_mode == "base6":
        return torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)

    i_past = torch.roll(i_x, shifts=1, dims=2)
    q_past = torch.roll(q_x, shifts=1, dims=2)
    if input_mode == "rich9":
        return torch.cat([i_x, q_x, i_future, q_future, i_past, q_past, amp, amp3, amp5], dim=1)

    if input_mode == "rich11":
        i_delta = i_x - i_past
        q_delta = q_x - q_past
        return torch.cat(
            [i_x, q_x, i_future, q_future, i_past, q_past, i_delta, q_delta, amp, amp3, amp5],
            dim=1,
        )

    raise ValueError(f"Unsupported input mode: {input_mode}")


# ==========================================
# EA-TCN  3 * 4 + 6 * 10 + (100+50) * 4 + 2 * 10 + 10 * 3 * 3 = 782

class DiagramDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag, config=None):
        super(DiagramDPDNet, self).__init__()
        self.config = preset_config("rich9_h15_l6_fir43_2284") if config is None else dict(config)
        self.hidden = int(self.config["hidden"])
        self.input_mode = self.config["input_mode"]
        self.input_channels = input_channels_for_mode(self.input_mode)
        self.attention_channels = int(self.config["attention_channels"])
        self.attention_mode = self.config.get("attention_mode", "amp37")
        self.thermal_window = int(self.config.get("thermal_window", 16))
        self.dilations = list(self.config["dilations"])
        self.poly_orders = list(self.config.get("poly_orders", [1, 3, 5]))
        self.pointwise_rank = self.config.get("pointwise_rank")

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))

        # 初始化为0，让网络在训练中自己去寻找最优的三阶压缩点
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.extra_poly_real = nn.ParameterDict()
        self.extra_poly_imag = nn.ParameterDict()
        for order in self.poly_orders:
            if order > 5:
                self.extra_poly_real[str(order)] = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
                self.extra_poly_imag[str(order)] = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.in_conv = nn.Conv1d(self.input_channels, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList()

        for dilation in self.dilations:
            if self.pointwise_rank is None:
                pointwise_layers = [
                    nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                ]
            else:
                rank = int(self.pointwise_rank)
                pointwise_layers = [
                    nn.Conv1d(self.hidden, rank, 1, bias=False, dtype=torch.float64),
                    nn.Conv1d(rank, self.hidden, 1, bias=False, dtype=torch.float64),
                ]
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(self.hidden, self.hidden, 5, padding=2 * dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                *pointwise_layers,
                nn.SiLU()
            ))

        self.attn = OptimalSpectrumAttention(self.hidden, self.attention_channels, mode=self.attention_mode)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)
        nn.init.constant_(self.out_conv.weight, 0.0)
        self.output_fir_taps = int(self.config.get("output_fir_taps", 1))
        if self.output_fir_taps > 1:
            center = self.output_fir_taps // 2
            fir_real = torch.zeros(self.output_fir_taps, dtype=torch.float64)
            fir_imag = torch.zeros(self.output_fir_taps, dtype=torch.float64)
            fir_real[center] = 1.0
            self.fir_real = nn.Parameter(fir_real)
            self.fir_imag = nn.Parameter(fir_imag)

    def forward_base(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        # 1. 基础一阶线性旁路
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        # 计算所有基础物理特征
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp4 = amp2 ** 2
        amp5 = amp ** 5
        amp7 = amp ** 7

        x3_i = i_x * amp2
        x3_q = q_x * amp2

        # 应用三阶复数权重吸收静态 AM-AM/AM-PM
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

        # 将一阶、三阶和五阶旁路合并
        lin_out = torch.cat([lin_i + lin3_i + lin5_i, lin_q + lin3_q + lin5_q], dim=1)

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
        feat = self.attn(feat, amp, amp3, amp7, amp5, thermal=thermal)
        residual = self.out_conv(feat)

        return lin_out + residual

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


class FramedDataset(Dataset):
    def __init__(self, features, targets, frame_length, stride_length):
        self.features = torch.as_tensor(features, dtype=torch.float64)
        self.targets = torch.as_tensor(targets, dtype=torch.float64)
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.num_frames = (self.features.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self): return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        return self.features[start:start + self.frame_length, :].T, self.targets[start:start + self.frame_length, :].T


class AutoDiagramDPDNet(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.hidden = state_dict["in_conv.weight"].shape[0]
        self.input_channels = state_dict["in_conv.weight"].shape[1]
        self.attn_channels = state_dict["attn.env_conv.weight"].shape[1]
        self.has_g3 = "g3_real" in state_dict
        self.has_g5 = "g5_real" in state_dict
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

        self.in_conv = nn.Conv1d(
            self.input_channels,
            self.hidden,
            1,
            bias=False,
            dtype=torch.float64,
        )
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.hidden,
                        self.hidden,
                        5,
                        padding=2 * dilation,
                        dilation=dilation,
                        groups=self.hidden,
                        bias=False,
                        dtype=torch.float64,
                    ),
                    nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                    nn.SiLU(),
                )
                for dilation in self.dilations
            ]
        )
        self.attn = OptimalSpectrumAttention(self.hidden, self.attn_channels)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp4 = amp2 ** 2
        amp5 = amp ** 5
        amp7 = amp ** 7

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

        if self.input_channels == 6:
            features = build_features(i_x, q_x, amp, amp3, amp5, "base6")
        elif self.input_channels == 9:
            features = build_features(i_x, q_x, amp, amp3, amp5, "rich9")
        elif self.input_channels == 11:
            features = build_features(i_x, q_x, amp, amp3, amp5, "rich11")
        else:
            raise ValueError(f"Unsupported model input channels: {self.input_channels}")

        feat = self.act(self.in_conv(features))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        feat = self.attn(feat, amp, amp3, amp7, amp5)
        return torch.cat([lin_i, lin_q], dim=1) + self.out_conv(feat)


def resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


def load_state_dict(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def is_subpath(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def default_checkpoint_for_preset(preset):
    return EXPERIMENT_DIR / f"AI-DPD_{preset}_best.pth"


def output_prefix_for_checkpoint(checkpoint_path, preset):
    return f"{Path(checkpoint_path).stem}_{preset}"


def load_partial_state(model, checkpoint_path):
    source = load_state_dict(checkpoint_path)
    target = model.state_dict()
    loaded = []
    partial = []
    skipped = []
    touched = set()

    for key, value in source.items():
        if key not in target:
            skipped.append(key)
            continue
        dst = target[key]
        if key == "in_conv.weight" and value.ndim == 3 and value.shape[1] == 6 and dst.shape[1] in (9, 11):
            merged = dst.clone()
            h = min(value.shape[0], dst.shape[0])
            merged[:h, :, :] = 0.0
            merged[:h, 0:4, :] = value[:h, 0:4, :].to(dtype=dst.dtype)
            amp_idx = 6 if dst.shape[1] == 9 else 8
            merged[:h, amp_idx, :] = value[:h, 4, :].to(dtype=dst.dtype)
            merged[:h, amp_idx + 1, :] = value[:h, 5, :].to(dtype=dst.dtype)
            target[key] = merged
            partial.append(key)
            touched.add(key)
            continue
        if key == "attn.env_conv.weight" and value.ndim == 3 and value.shape[1] == 3 and dst.shape[1] == 4:
            merged = dst.clone()
            h = min(value.shape[0], dst.shape[0])
            merged[:h, :, :] = 0.0
            merged[:h, 0, :] = value[:h, 0, :].to(dtype=dst.dtype)
            merged[:h, 1, :] = value[:h, 1, :].to(dtype=dst.dtype)
            if getattr(model, "config", {}).get("attention_mode") == "amp37_thermal16":
                merged[:h, 2, :] = value[:h, 2, :].to(dtype=dst.dtype)
            else:
                merged[:h, 3, :] = value[:h, 2, :].to(dtype=dst.dtype)
            target[key] = merged
            partial.append(key)
            touched.add(key)
            continue
        if (
            key.startswith("tcn_blocks.")
            and key.endswith(".1.weight")
            and f"{key[:-len('.1.weight')]}.2.weight" in target
            and value.ndim == 3
            and value.shape[-1] == 1
            and target[key].ndim == 3
            and target[f"{key[:-len('.1.weight')]}.2.weight"].ndim == 3
        ):
            expand_key = f"{key[:-len('.1.weight')]}.2.weight"
            compress = torch.zeros_like(target[key])
            expand = torch.zeros_like(target[expand_key])
            dense = value[:, :, 0].to(dtype=torch.float64)
            rank = min(compress.shape[0], dense.shape[0], dense.shape[1])
            try:
                u, s, vh = torch.linalg.svd(dense, full_matrices=False)
                root_s = torch.sqrt(s[:rank])
                compress[:rank, :dense.shape[1], 0] = (root_s[:, None] * vh[:rank, :]).to(dtype=compress.dtype)
                expand[:dense.shape[0], :rank, 0] = (u[:, :rank] * root_s[None, :]).to(dtype=expand.dtype)
            except RuntimeError:
                h0 = min(dense.shape[0], expand.shape[0], rank)
                h1 = min(dense.shape[1], compress.shape[1], rank)
                compress[:h1, :h1, 0] = torch.eye(h1, dtype=compress.dtype)
                expand[:h0, :h1, 0] = dense[:h0, :h1].to(dtype=expand.dtype)
            target[key] = compress
            target[expand_key] = expand
            partial.extend([key, expand_key])
            touched.add(key)
            touched.add(expand_key)
            continue
        if (
            key.startswith("tcn_blocks.")
            and key.endswith(".1.weight")
            and value.ndim == 3
            and value.shape[0] < dst.shape[0]
            and value.shape[1] < dst.shape[1]
        ):
            merged = dst.clone()
            h0 = min(value.shape[0], dst.shape[0])
            h1 = min(value.shape[1], dst.shape[1])
            merged[:h0, :, :] = 0.0
            merged[:h0, :h1, :] = value[:h0, :h1, :].to(dtype=dst.dtype)
            target[key] = merged
            partial.append(key)
            touched.add(key)
            continue
        if key == "out_conv.weight" and value.ndim == 3 and value.shape[1] < dst.shape[1]:
            merged = torch.zeros_like(dst)
            h = min(value.shape[1], dst.shape[1])
            merged[:, :h, :] = value[:, :h, :].to(dtype=dst.dtype)
            target[key] = merged
            partial.append(key)
            touched.add(key)
            continue
        if key in ("fir_real", "fir_imag") and value.ndim == 1 and dst.ndim == 1:
            src_center = value.shape[0] // 2
            dst_center = dst.shape[0] // 2
            half = min(src_center, dst_center, value.shape[0] - src_center - 1, dst.shape[0] - dst_center - 1)
            merged = dst.clone()
            merged[dst_center - half:dst_center + half + 1] = value[
                src_center - half:src_center + half + 1
            ].to(dtype=dst.dtype)
            target[key] = merged
            partial.append(key)
            touched.add(key)
            continue
        if value.shape == dst.shape:
            target[key] = value.to(dtype=dst.dtype)
            loaded.append(key)
            touched.add(key)
            continue
        if value.ndim == dst.ndim:
            merged = torch.zeros_like(dst)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, dst.shape))
            merged[slices] = value[slices].to(dtype=dst.dtype)
            target[key] = merged
            partial.append(key)
            touched.add(key)
        else:
            skipped.append(key)

    for key in list(target):
        if key not in touched and key.startswith("tcn_blocks.") and key.endswith(".1.weight"):
            target[key] = torch.zeros_like(target[key])

    model.load_state_dict(target, strict=True)
    return loaded, partial, skipped


def evaluate_nmse(model, x_val, y_val):
    pred = run_sequence(model, x_val)
    target = y_val.T
    nmse = np.sum((pred - target) ** 2) / np.sum(target ** 2)
    return nmse, 10 * np.log10(nmse)


def run_base_sequence(model, x_np):
    x_tensor = torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model.forward_base(x_tensor).squeeze(0).cpu().numpy()


def fit_output_fir_from_train(model, x_train, y_train):
    if getattr(model, "output_fir_taps", 1) <= 1:
        raise ValueError("Current preset does not enable output FIR.")
    taps = model.output_fir_taps
    radius = taps // 2
    base = run_base_sequence(model, x_train).T
    z = base[:, 0] + 1j * base[:, 1]
    target = y_train[:, 0] + 1j * y_train[:, 1]
    cols = []
    for offset in range(-radius, radius + 1):
        cols.append(z[radius + offset: len(z) - radius + offset])
    design = np.stack(cols, axis=1)
    rhs = target[radius: len(target) - radius]
    coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=torch.float64, device=DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=torch.float64, device=DEVICE))
    return coeffs


def fit_output_fir_freq_from_train(
    model,
    x_train,
    y_train,
    *,
    fs_mhz,
    main_band,
    lower_band,
    upper_band,
    main_weight,
    adjacent_weight,
    floor_weight,
    ridge,
):
    if getattr(model, "output_fir_taps", 1) <= 1:
        raise ValueError("Current preset does not enable output FIR.")
    taps = model.output_fir_taps
    radius = taps // 2
    base = run_base_sequence(model, x_train).T
    z = base[:, 0] + 1j * base[:, 1]
    target = y_train[:, 0] + 1j * y_train[:, 1]

    n = len(z)
    z_fft = np.fft.fft(z)
    target_fft = np.fft.fft(target)
    freq = np.fft.fftfreq(n, d=1.0 / fs_mhz)
    weights = np.full(n, float(floor_weight), dtype=np.float64)
    main_mask = (freq >= float(main_band[0])) & (freq <= float(main_band[1]))
    lower_mask = (freq >= float(lower_band[0])) & (freq <= float(lower_band[1]))
    upper_mask = (freq >= float(upper_band[0])) & (freq <= float(upper_band[1]))
    weights[main_mask] = float(main_weight)
    weights[lower_mask | upper_mask] = float(adjacent_weight)

    omega = 2 * np.pi * np.arange(n) / n
    offsets = np.arange(-radius, radius + 1)
    phase = np.exp(1j * omega[:, None] * offsets[None, :])
    design = z_fft[:, None] * phase
    weighted = design * np.sqrt(weights)[:, None]
    rhs = target_fft * np.sqrt(weights)
    lhs = weighted.conj().T @ weighted + float(ridge) * np.eye(taps)
    coeffs = np.linalg.solve(lhs, weighted.conj().T @ rhs)

    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=torch.float64, device=DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=torch.float64, device=DEVICE))
    return coeffs


def load_normalized_data(data_path):
    data = scipy.io.loadmat(data_path)
    raw_x, raw_y = data['x'].flatten(), data['y'].flatten()

    RMS_X = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    RMS_Y = np.sqrt(np.mean(np.abs(raw_y) ** 2))

    x_c = raw_x / RMS_X
    y_c = raw_y / RMS_Y
    return raw_x, raw_y, x_c, y_c


def load_reference_error(data_path, target_scale=None):
    """Load the dataset reference residual using the target signal's scale.

    ``DPD_data.mat`` defines ``error`` in the same physical units as ``y``.
    It must therefore never be normalized by its own RMS when it is compared
    with the main-band power of ``y``.
    """
    data = scipy.io.loadmat(data_path)
    raw_error = data["error"].flatten()
    if target_scale is None:
        raw_y = data["y"].flatten()
        target_scale = np.sqrt(np.mean(np.abs(raw_y) ** 2))
    return raw_error, raw_error / float(target_scale)


def iq_array(complex_signal):
    return np.stack([np.real(complex_signal), np.imag(complex_signal)], axis=1)


def tensor_to_complex(iq):
    iq = np.squeeze(iq)
    if iq.shape[0] == 2:
        return iq[0] + 1j * iq[1]
    return iq[:, 0] + 1j * iq[:, 1]


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
    low_hz, high_hz = band_mhz[0] * 1e6, band_mhz[1] * 1e6
    mask = (freq_hz >= low_hz) & (freq_hz <= high_hz)
    if not np.any(mask):
        raise ValueError(f"No PSD bins in band {band_mhz} MHz")
    return float(np.mean(values_db[mask]))


def aclr_from_psd(
    freq_hz,
    error_values_db,
    main_band,
    lower_band,
    upper_band,
    original_values_db=None,
):
    """Return residual ACLR as error adjacent-band PSD minus original main PSD.

    ``original_values_db`` is optional only for compatibility with older callers.
    New evaluation code must pass the target/original PSD explicitly; otherwise
    this falls back to the former single-spectrum calculation.
    """
    if original_values_db is None:
        original_values_db = error_values_db
    main_avg = band_average(freq_hz, original_values_db, main_band)
    lower_avg = band_average(freq_hz, error_values_db, lower_band)
    upper_avg = band_average(freq_hz, error_values_db, upper_band)
    lower_aclr = lower_avg - main_avg
    upper_aclr = upper_avg - main_avg
    return lower_aclr, upper_aclr, max(lower_aclr, upper_aclr)


def aclr_from_psd_minimum(freq_hz, error_values_db, original_values_db, main_band, lower_band, upper_band):
    def minimum(values, band):
        low_hz, high_hz = sorted((band[0] * 1e6, band[1] * 1e6))
        mask = (freq_hz >= low_hz) & (freq_hz <= high_hz)
        if not np.any(mask):
            raise ValueError(f"No PSD bins in band {band} MHz")
        return float(np.min(values[mask]))

    main_min = minimum(original_values_db, main_band)
    lower_aclr = minimum(error_values_db, lower_band) - main_min
    upper_aclr = minimum(error_values_db, upper_band) - main_min
    return lower_aclr, upper_aclr, max(lower_aclr, upper_aclr)


def run_sequence(model, x_np):
    x_tensor = torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model(x_tensor).squeeze(0).cpu().numpy()


def save_psd_and_aclr_figures(
    target_iq,
    pred_iq,
    output_prefix,
    main_band=(20.0, 80.0),
    lower_band=(-180.0, -120.0),
    upper_band=(120.0, 180.0),
    fs_hz=491.52e6,
    nperseg=4096,
    target_dbc=-45.0,
    reference_error_iq=None,
):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target_complex = tensor_to_complex(target_iq)
    pred_complex = tensor_to_complex(pred_iq)
    residual_complex = target_complex - pred_complex
    freq_hz, target_psd = psd_db(target_complex, fs_hz=fs_hz, nperseg=nperseg)
    _, pred_psd = psd_db(pred_complex, fs_hz=fs_hz, nperseg=nperseg)
    _, residual_psd = psd_db(residual_complex, fs_hz=fs_hz, nperseg=nperseg)

    residual_aclr = aclr_from_psd(
        freq_hz,
        residual_psd,
        main_band,
        lower_band,
        upper_band,
        original_values_db=target_psd,
    )
    residual_min_point_aclr = aclr_from_psd_minimum(
        freq_hz, residual_psd, target_psd, main_band, lower_band, upper_band
    )
    residual_narrow_aclr = aclr_from_psd(
        freq_hz, residual_psd, (31.64, 36.64), (-129.84, -124.84), (121.08, 126.08),
        original_values_db=target_psd,
    )
    reference_psd = None
    reference_aclr = None
    reference_min_point_aclr = None
    reference_narrow_aclr = None
    if reference_error_iq is not None:
        reference_error_complex = tensor_to_complex(reference_error_iq)
        if len(reference_error_complex) != len(target_complex):
            raise ValueError("reference_error_iq must have the same length as target_iq")
        _, reference_psd = psd_db(reference_error_complex, fs_hz=fs_hz, nperseg=nperseg)
        reference_aclr = aclr_from_psd(
            freq_hz,
            reference_psd,
            main_band,
            lower_band,
            upper_band,
            original_values_db=target_psd,
        )
        reference_min_point_aclr = aclr_from_psd_minimum(
            freq_hz, reference_psd, target_psd, main_band, lower_band, upper_band
        )
        reference_narrow_aclr = aclr_from_psd(
            freq_hz, reference_psd, (31.64, 36.64), (-129.84, -124.84), (121.08, 126.08),
            original_values_db=target_psd,
        )

    gap_to_reference = (
        residual_aclr[2] - reference_aclr[2] if reference_aclr is not None else float("nan")
    )
    target_margin = target_dbc - residual_aclr[2]

    psd_path = FIGURE_DIR / f"{output_prefix}_psd_aclr.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    target_peak = np.max(target_psd)
    target_norm = target_psd - target_peak
    pred_norm = pred_psd - target_peak
    residual_norm = residual_psd - target_peak
    ax.plot(freq_hz / 1e6, target_norm, "--", linewidth=1.1, label="Target")
    ax.plot(freq_hz / 1e6, pred_norm, linewidth=1.1, label="EA-TCN")
    ax.plot(freq_hz / 1e6, residual_norm, linewidth=1.0, label="Residual (target - pred)")
    if reference_psd is not None:
        ax.plot(
            freq_hz / 1e6,
            reference_psd - target_peak,
            ":",
            linewidth=1.0,
            label="Dataset reference error",
        )
    for band, label, color in [
        (main_band, "Main band", "tab:green"),
        (lower_band, "Lower adjacent", "tab:orange"),
        (upper_band, "Upper adjacent", "tab:purple"),
    ]:
        ax.axvspan(band[0], band[1], color=color, alpha=0.12, label=label)
    main_norm_avg = band_average(freq_hz, target_norm, main_band)
    ax.axhline(
        main_norm_avg + target_dbc,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"{target_dbc:.0f} dBc residual limit",
    )
    ax.set_title("Target, prediction, and residual spectra")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Normalized PSD (dB)")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="best")
    fig.tight_layout()
    try:
        fig.savefig(psd_path, dpi=300)
    except PermissionError:
        psd_path = FIGURE_DIR / f"{output_prefix}_psd_aclr_{int(time.time())}.png"
        fig.savefig(psd_path, dpi=300)
    plt.close(fig)

    aclr_path = FIGURE_DIR / f"{output_prefix}_aclr_bar.png"
    labels = ["Lower", "Upper", "Worst"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    if reference_aclr is not None:
        ax.bar(
            x - width / 2,
            reference_aclr,
            width,
            label="Dataset reference error",
            color="tab:red",
            alpha=0.82,
        )
        residual_x = x + width / 2
    else:
        residual_x = x
    ax.bar(
        residual_x,
        residual_aclr,
        width,
        label="Model residual",
        color="tab:blue",
        alpha=0.82,
    )
    ax.axhline(
        target_dbc,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"{target_dbc:.0f} dBc residual limit",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Residual ACLR (dBc)")
    ax.set_title("Residual ACLR: error adjacent band - target main band")
    ax.grid(True, axis="y", linestyle=":", alpha=0.65)
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(aclr_path, dpi=300)
    except PermissionError:
        aclr_path = FIGURE_DIR / f"{output_prefix}_aclr_bar_{int(time.time())}.png"
        fig.savefig(aclr_path, dpi=300)
    plt.close(fig)

    return {
        "reference_lower": reference_aclr[0] if reference_aclr is not None else float("nan"),
        "reference_upper": reference_aclr[1] if reference_aclr is not None else float("nan"),
        "reference_worst": reference_aclr[2] if reference_aclr is not None else float("nan"),
        "residual_lower": residual_aclr[0],
        "residual_upper": residual_aclr[1],
        "residual_worst": residual_aclr[2],
        "residual_min_point_lower": residual_min_point_aclr[0],
        "residual_min_point_upper": residual_min_point_aclr[1],
        "residual_min_point_worst": residual_min_point_aclr[2],
        "residual_narrow_lower": residual_narrow_aclr[0],
        "residual_narrow_upper": residual_narrow_aclr[1],
        "residual_narrow_worst": residual_narrow_aclr[2],
        "reference_min_point_worst": reference_min_point_aclr[2] if reference_min_point_aclr is not None else float("nan"),
        "reference_narrow_worst": reference_narrow_aclr[2] if reference_narrow_aclr is not None else float("nan"),
        "gap_to_reference": gap_to_reference,
        # Compatibility aliases for callers that still consume the old field names.
        "target_lower": reference_aclr[0] if reference_aclr is not None else float("nan"),
        "target_upper": reference_aclr[1] if reference_aclr is not None else float("nan"),
        "target_worst": reference_aclr[2] if reference_aclr is not None else float("nan"),
        "pred_lower": residual_aclr[0],
        "pred_upper": residual_aclr[1],
        "pred_worst": residual_aclr[2],
        "gap_to_target_y": gap_to_reference,
        "target_dbc": target_dbc,
        "target_margin": target_margin,
        "psd_path": psd_path,
        "aclr_path": aclr_path,
    }


def evaluate_aclr_visualization(
    model,
    x_val,
    y_val,
    output_prefix="step40_1_6_res_attention_direct",
    fs_mhz=491.52,
    main_band=(20.0, 80.0),
    lower_band=(-180.0, -120.0),
    upper_band=(120.0, 180.0),
    target_dbc=-45.0,
    reference_error_iq=None,
):
    model.eval()
    with torch.no_grad():
        pred_iq = run_sequence(model, x_val)

    fit_nmse = np.sum((pred_iq - y_val.T) ** 2) / np.sum(y_val.T ** 2)
    metrics = save_psd_and_aclr_figures(
        y_val.T,
        pred_iq,
        output_prefix,
        main_band=main_band,
        lower_band=lower_band,
        upper_band=upper_band,
        fs_hz=fs_mhz * 1e6,
        target_dbc=target_dbc,
        reference_error_iq=reference_error_iq,
    )
    print("\n>>> ACLR/PSD evaluation")
    print(f"Fit NMSE pred_y vs target_y: {fit_nmse:.6e} ({10 * np.log10(fit_nmse):.2f} dB)")
    print(
        "Residual ACLR worst (error adjacent - target main): "
        f"mean_db {metrics['residual_worst']:.2f}, "
        f"min_point {metrics['residual_min_point_worst']:.2f}, "
        f"narrow_5mhz {metrics['residual_narrow_worst']:.2f} "
        "dBc | "
        f"{target_dbc:.0f} dBc margin {metrics['target_margin']:.2f} dB"
    )
    if reference_error_iq is not None:
        print(
            "Dataset reference: "
            f"mean_db {metrics['reference_worst']:.2f}, "
            f"min_point {metrics['reference_min_point_worst']:.2f}, "
            f"narrow_5mhz {metrics['reference_narrow_worst']:.2f} "
            "dBc | "
            f"model-reference gap {metrics['gap_to_reference']:.2f} dB"
        )
    print(
        "Target requirement: "
        f"model residual worst ACLR <= {target_dbc:.2f} dBc"
    )
    print(f"PSD figure: {metrics['psd_path']}")
    print(f"ACLR figure: {metrics['aclr_path']}")
    return metrics


def weighted_spectral_nmse(
    pred,
    target,
    *,
    fs_mhz,
    main_band,
    lower_band,
    upper_band,
    main_weight,
    adjacent_weight,
    floor_weight,
):
    n = pred.shape[-1]
    pred_complex = torch.complex(pred[:, 0, :], pred[:, 1, :])
    target_complex = torch.complex(target[:, 0, :], target[:, 1, :])
    err_fft = torch.fft.fftshift(torch.fft.fft(pred_complex - target_complex, dim=-1, norm="ortho"), dim=-1)
    target_fft = torch.fft.fftshift(torch.fft.fft(target_complex, dim=-1, norm="ortho"), dim=-1)

    freq = torch.fft.fftshift(torch.fft.fftfreq(n, d=1.0 / fs_mhz, device=pred.device))
    weights = torch.full((n,), float(floor_weight), dtype=pred.dtype, device=pred.device)
    main_mask = (freq >= float(main_band[0])) & (freq <= float(main_band[1]))
    lower_mask = (freq >= float(lower_band[0])) & (freq <= float(lower_band[1]))
    upper_mask = (freq >= float(upper_band[0])) & (freq <= float(upper_band[1]))
    weights = torch.where(main_mask, torch.as_tensor(float(main_weight), dtype=pred.dtype, device=pred.device), weights)
    adjacent_value = torch.as_tensor(float(adjacent_weight), dtype=pred.dtype, device=pred.device)
    weights = torch.where(lower_mask | upper_mask, adjacent_value, weights)
    weights = weights.unsqueeze(0)

    err_power = (err_fft.real * err_fft.real + err_fft.imag * err_fft.imag) * weights
    target_power = (target_fft.real * target_fft.real + target_fft.imag * target_fft.imag) * weights
    return err_power.mean() / (target_power.mean() + torch.as_tensor(1e-24, dtype=pred.dtype, device=pred.device))


def loss_with_optional_terms(
    pred,
    target,
    *,
    crop_margin,
    diff_loss_weight,
    spectral_loss_weight,
    fs_mhz,
    main_band,
    lower_band,
    upper_band,
    spectral_main_weight,
    spectral_adjacent_weight,
    spectral_floor_weight,
):
    if crop_margin > 0 and pred.shape[-1] > 2 * crop_margin:
        region = slice(crop_margin, -crop_margin)
        pred = pred[:, :, region]
        target = target[:, :, region]

    loss = nn.functional.mse_loss(pred, target)
    if diff_loss_weight > 0:
        pred_d = pred[:, :, 1:] - pred[:, :, :-1]
        target_d = target[:, :, 1:] - target[:, :, :-1]
        loss = loss + diff_loss_weight * nn.functional.mse_loss(pred_d, target_d)
    if spectral_loss_weight > 0:
        loss = loss + spectral_loss_weight * weighted_spectral_nmse(
            pred,
            target,
            fs_mhz=fs_mhz,
            main_band=main_band,
            lower_band=lower_band,
            upper_band=upper_band,
            main_weight=spectral_main_weight,
            adjacent_weight=spectral_adjacent_weight,
            floor_weight=spectral_floor_weight,
        )
    return loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate Huawei AI-DPD models with unified NMSE and ACLR reporting.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to DPD_data.mat.")
    parser.add_argument(
        "--preset",
        default="rich9_h15_l6_fir43_2284",
        choices=[
            "rich9_h15_l6_fir43_2284",
            "rich9_h15_l6_fir39_2268",
            "memory4_fir19_998",
            "rich9_h15_l6_fir95_2492",
            "h10_l3_base6_fir41_796",
            "h10_l4_base6_rank3_fir41_786",
        ],
    )
    parser.add_argument("--checkpoint", default=None, help="Experiment checkpoint. Defaults to checkpoints/AI-DPD.")
    parser.add_argument("--init-checkpoint", default=str(REFERENCE_CHECKPOINT), help="Checkpoint for partial initialization.")
    parser.add_argument("--resume-checkpoint", default=None, help="Strictly load this checkpoint before continued training.")
    parser.add_argument("--force-init", action="store_true", help="Load --init-checkpoint even for scratch-default presets.")
    parser.add_argument("--no-init", action="store_true", help="Train from fresh initialization.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only generate ACLR/PSD figures.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--frame-length", type=int, default=FRAME_LENGTH)
    parser.add_argument("--stride-length", type=int, default=STRIDE_LENGTH)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scratch-lr", type=float, default=5e-3, help="Learning rate used when --no-init is set.")
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-nmse-db", type=float, default=None, help="Stop training once validation NMSE reaches this dB value.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--select-metric", choices=["frame", "sequence"], default="sequence")
    parser.add_argument("--train-extra-only", action="store_true", help="Freeze inherited weights and train new extra params only.")
    parser.add_argument("--train-fir-only", action="store_true", help="Freeze inherited weights and train output FIR params only.")
    parser.add_argument("--freeze-fir", action="store_true", help="Keep output FIR fixed while training other parameters.")
    parser.add_argument("--fit-output-fir", action="store_true", help="Fit enabled output FIR by LS on the training split before training.")
    parser.add_argument("--fit-output-fir-domain", choices=["time", "freq"], default="time")
    parser.add_argument("--fir-main-weight", type=float, default=1.0)
    parser.add_argument("--fir-adjacent-weight", type=float, default=1.0)
    parser.add_argument("--fir-floor-weight", type=float, default=0.05)
    parser.add_argument("--fir-ridge", type=float, default=1e-8)
    parser.add_argument("--crop-margin", type=int, default=0, help="Ignore frame edge samples in training loss.")
    parser.add_argument("--diff-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-main-weight", type=float, default=1.0)
    parser.add_argument("--spectral-adjacent-weight", type=float, default=1.0)
    parser.add_argument("--spectral-floor-weight", type=float, default=0.05)
    parser.add_argument("--fs-mhz", type=float, default=491.52, help="Dataset sample rate in MHz.")
    parser.add_argument("--main-band", type=float, nargs=2, default=(20.0, 80.0), metavar=("LO", "HI"))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-180.0, -120.0), metavar=("LO", "HI"))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(120.0, 180.0), metavar=("LO", "HI"))
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0, help="Target worst-band ACLR in dBc.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(args.checkpoint)
    raw_x, raw_y, x_c, y_c = load_normalized_data(data_path)
    rms_x = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    rms_y = np.sqrt(np.mean(np.abs(raw_y) ** 2))
    _, reference_error_c = load_reference_error(data_path, target_scale=rms_y)
    max_x = np.max(np.abs(raw_x))
    max_y = np.max(np.abs(raw_y))

    G_opt = np.vdot(x_c, y_c) / np.vdot(x_c, x_c)
    g_real_init = np.real(G_opt)
    g_imag_init = np.imag(G_opt)

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Target_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)
    Reference_error_np = iq_array(reference_error_c)

    min_len = min(len(X_np), len(Target_np), len(Reference_error_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]
    Reference_error_val = Reference_error_np[split_idx:min_len]

    model = DiagramDPDNet(g_real_init, g_imag_init).to(DEVICE)
    if args.evaluate_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
        evaluate_aclr_visualization(
            model,
            X_val,
            Y_val,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            target_dbc=args.aclr_target_dbc,
            reference_error_iq=Reference_error_val.T,
        )
        return

    train_loader = DataLoader(FramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(FramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float('inf')
    history_epochs = []
    history_nmse = []

    print(f"\n>>> 开始训练 EA-TCN (1/3/5阶旁路 + 动态 LR) (782乘法器)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        total_error_power, total_target_power = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
                preds = model(batch_x)
                total_error_power += torch.sum((preds - batch_y) ** 2).item()
                total_target_power += torch.sum(batch_y ** 2).item()

        linear_nmse = total_error_power / total_target_power
        val_nmse = 10 * np.log10(linear_nmse)

        # 动态监控
        scheduler.step(val_nmse)
        current_lr = optimizer.param_groups[0]['lr']

        history_epochs.append(epoch + 1)
        history_nmse.append(val_nmse)

        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Time: {elapsed_mins}m {elapsed_secs}s | LR: {current_lr:.2e} | NMSE: {val_nmse:.2f} dB ")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), checkpoint_path)

    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_nmse, label='Validation NMSE', color='b', linewidth=1.5)
    plt.axhline(y=min(history_nmse), color='r', linestyle='--', label=f'Best: {min(history_nmse):.2f} dB')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title(f"Legacy baseline validation NMSE (batch size={BATCH_SIZE})", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    convergence_path = FIGURE_DIR / "legacy_baseline_validation_nmse.png"
    plt.savefig(convergence_path, dpi=300)
    print(f"\n>>> 训练完成！收敛曲线已保存: {convergence_path}")

    model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
    evaluate_aclr_visualization(
        model,
        X_val,
        Y_val,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        target_dbc=args.aclr_target_dbc,
        reference_error_iq=Reference_error_val.T,
    )


def main_experiment():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = resolve_path(args.data)
    config = preset_config(args.preset)
    multipliers = real_multiplier_count_from_config(config)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else default_checkpoint_for_preset(args.preset)
    if not args.evaluate_only and not is_subpath(checkpoint_path, EXPERIMENT_DIR):
        raise ValueError(f"Training checkpoint must stay under {EXPERIMENT_DIR}")

    raw_x, raw_y, x_c, y_c = load_normalized_data(data_path)
    rms_y = np.sqrt(np.mean(np.abs(raw_y) ** 2))
    _, reference_error_c = load_reference_error(data_path, target_scale=rms_y)
    x_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    y_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)
    reference_error_np = iq_array(reference_error_c)
    min_len = min(len(x_np), len(y_np), len(reference_error_np))
    split_idx = int(min_len * 0.8)
    x_train, y_train = x_np[:split_idx], y_np[:split_idx]
    x_val, y_val = x_np[split_idx:min_len], y_np[split_idx:min_len]
    reference_error_val = reference_error_np[split_idx:min_len]

    x_train_c = x_c[:split_idx]
    y_train_c = y_c[:split_idx]
    g_opt = np.vdot(x_train_c, y_train_c) / np.vdot(x_train_c, x_train_c)
    g_real_init = np.real(g_opt)
    g_imag_init = np.imag(g_opt)

    model = DiagramDPDNet(g_real_init, g_imag_init, config=config).to(DEVICE)
    output_prefix = output_prefix_for_checkpoint(checkpoint_path, args.preset)

    if args.evaluate_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
        print(f">>> Evaluate preset={args.preset} | multipliers={multipliers} | checkpoint={checkpoint_path}")
        evaluate_aclr_visualization(
            model,
            x_val,
            y_val,
            output_prefix=output_prefix,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            target_dbc=args.aclr_target_dbc,
            reference_error_iq=reference_error_val.T,
        )
        return

    resume_path = resolve_path(args.resume_checkpoint) if args.resume_checkpoint else None
    scratch_mode = args.no_init or (bool(config.get("scratch_default", False)) and not args.force_init)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model.load_state_dict(load_state_dict(resume_path), strict=True)
        print(f">>> Resumed strict checkpoint: {resume_path}")
    elif config.get("scratch_default", False) and not args.no_init and not args.force_init:
        print(f">>> Preset {args.preset} is scratch-default. Existing checkpoints will not be loaded.")

    if resume_path is None and not scratch_mode:
        init_path = resolve_path(args.init_checkpoint)
        if init_path.exists():
            loaded, partial, skipped = load_partial_state(model, init_path)
            print(f">>> Partial init from: {init_path}")
            print(f">>> Loaded tensors: exact={len(loaded)}, partial={len(partial)}, skipped={len(skipped)}")
        else:
            print(f">>> Init checkpoint not found, training fresh: {init_path}")

    if args.fit_output_fir:
        if args.fit_output_fir_domain == "freq":
            coeffs = fit_output_fir_freq_from_train(
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
            print(
                ">>> Fitted output FIR by frequency-domain LS on training split | "
                f"taps={len(coeffs)} | adjacent_weight={args.fir_adjacent_weight:g}"
            )
        else:
            coeffs = fit_output_fir_from_train(model, x_train, y_train)
            print(f">>> Fitted output FIR by time-domain LS on training split | taps={len(coeffs)}")

    if args.train_extra_only or args.train_fir_only:
        if args.train_fir_only and args.freeze_fir:
            raise ValueError("--train-fir-only and --freeze-fir are mutually exclusive.")
        for _, param in model.named_parameters():
            param.requires_grad = False
        train_prefixes = []
        if args.train_extra_only:
            train_prefixes.append("extra_poly_")
        if args.train_fir_only:
            if getattr(model, "output_fir_taps", 1) <= 1:
                raise ValueError("--train-fir-only requires a preset with output_fir_taps > 1.")
            train_prefixes.append("fir_")
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in train_prefixes):
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f">>> Frozen-base training enabled ({', '.join(train_prefixes)}). Trainable parameters: {trainable}")

    if args.freeze_fir:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith("fir_"):
                param.requires_grad = False
                frozen += param.numel()
        if frozen == 0:
            raise ValueError("--freeze-fir requires a preset with output FIR parameters.")
        print(f">>> Output FIR frozen. Frozen parameters: {frozen}")

    train_loader = DataLoader(
        FramedDataset(x_train, y_train, args.frame_length, args.stride_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        FramedDataset(x_val, y_val, args.frame_length, args.frame_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    lr = args.lr if resume_path is not None else (args.scratch_lr if scratch_mode else args.lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr
    )

    initial_nmse, initial_nmse_db = evaluate_nmse(model, x_val, y_val)
    best_score = initial_nmse if args.select_metric == "sequence" else float("inf")
    best_nmse = initial_nmse
    torch.save(model.state_dict(), checkpoint_path)
    history_epochs = [0]
    history_nmse = [initial_nmse_db]
    start_time = time.time()

    print(f"\n>>> Start AI-DPD training | preset={args.preset} | multipliers={multipliers}")
    print(f">>> checkpoint={checkpoint_path}")
    print(f">>> lr={lr:.2e} | epochs={args.epochs} | select_metric={args.select_metric}")
    print(f">>> Initial NMSE: {initial_nmse_db:.2f} dB")
    if args.target_nmse_db is not None:
        print(f">>> Early-stop target: NMSE <= {args.target_nmse_db:.2f} dB")
        if initial_nmse_db <= args.target_nmse_db:
            print(">>> Initial model already meets target. Skipping training.")
            args.epochs = 0

    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_with_optional_terms(
                pred,
                batch_y,
                crop_margin=args.crop_margin,
                diff_loss_weight=args.diff_loss_weight,
                spectral_loss_weight=args.spectral_loss_weight,
                fs_mhz=args.fs_mhz,
                main_band=tuple(args.main_band),
                lower_band=tuple(args.lower_band),
                upper_band=tuple(args.upper_band),
                spectral_main_weight=args.spectral_main_weight,
                spectral_adjacent_weight=args.spectral_adjacent_weight,
                spectral_floor_weight=args.spectral_floor_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        total_error_power = 0.0
        total_target_power = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y = batch_y.to(DEVICE, non_blocking=True)
                pred = model(batch_x)
                total_error_power += torch.sum((pred - batch_y) ** 2).item()
                total_target_power += torch.sum(batch_y ** 2).item()

        frame_nmse = total_error_power / total_target_power
        frame_db = 10 * np.log10(frame_nmse)
        nmse, nmse_db = evaluate_nmse(model, x_val, y_val)
        score = nmse if args.select_metric == "sequence" else frame_nmse
        scheduler.step(nmse_db if args.select_metric == "sequence" else frame_db)

        history_epochs.append(epoch + 1)
        history_nmse.append(nmse_db)
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | Time: {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | Frame NMSE: {frame_db:.2f} dB | NMSE: {nmse_db:.2f} dB"
        )

        if score < best_score:
            best_score = score
            best_nmse = nmse
            torch.save(model.state_dict(), checkpoint_path)

        if args.target_nmse_db is not None and nmse_db <= args.target_nmse_db:
            print(f">>> Target reached at epoch {epoch + 1}: NMSE {nmse_db:.2f} dB")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_nmse, label="NMSE", color="b", linewidth=1.5)
    plt.axhline(y=min(history_nmse), color="r", linestyle="--", label=f"Best: {min(history_nmse):.2f} dB")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("NMSE (dB)", fontsize=12)
    plt.title(f"Validation NMSE: {args.preset} ({multipliers} real multipliers)", fontsize=14)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    convergence_path = FIGURE_DIR / f"{output_prefix}_convergence.png"
    plt.savefig(convergence_path, dpi=300)
    print(f"\n>>> Training complete. Convergence curve: {convergence_path}")
    print(f">>> Best NMSE: {10 * np.log10(best_nmse):.2f} dB")

    model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
    evaluate_aclr_visualization(
        model,
        x_val,
        y_val,
        output_prefix=output_prefix,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        target_dbc=args.aclr_target_dbc,
        reference_error_iq=reference_error_val.T,
    )


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main_experiment()
