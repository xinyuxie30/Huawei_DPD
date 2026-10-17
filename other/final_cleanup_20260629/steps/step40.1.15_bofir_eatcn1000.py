import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_INIT = CHECKPOINT_DIR / "EA-TCN_memory_fir1000_best.pth"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "BO-FIR_P2M5_FIR37_EA-TCN1000_best.pth"
DEFAULT_VARIANT = "p2m5_h10_fir37_1000"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


MODEL_CONFIG = {
    "preset": "bo_fir_eatcn1000",
    "hidden": 10,
    "bo_fir_taps": 5,
    "bo_fir_units": 2,
    "dilations": [1, 2, 4, 8],
    "kernel_size": 5,
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
    "output_fir_taps": 37,
}

VARIANT_CONFIGS = {
    DEFAULT_VARIANT: MODEL_CONFIG,
    "p2m5_h11_fir47_1164": {
        "preset": "bo_fir_p2m5_h11_fir47_1164",
        "hidden": 11,
        "bo_fir_taps": 5,
        "bo_fir_units": 2,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    },
    "p3m13_h10_fir47_1156": {
        "preset": "bo_fir_p3m13_h10_fir47_1156",
        "hidden": 10,
        "bo_fir_taps": 13,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    },
    "p3m5_h11_fir47_1184": {
        "preset": "bo_fir_p3m5_h11_fir47_1184",
        "hidden": 11,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    },
    "p3m5_h13_fir47_1456": {
        "preset": "bo_fir_p3m5_h13_fir47_1456",
        "hidden": 13,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    },
    "p3m5_h13_fir57_1496": {
        "preset": "bo_fir_p3m5_h13_fir57_1496",
        "hidden": 13,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h13_fir57_thermal16_1496": {
        "preset": "bo_fir_p3m5_h13_fir57_thermal16_1496",
        "hidden": 13,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "attention_mode": "thermal16",
        "thermal_window": 16,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h13_fir47_amp37thermal16_1495": {
        "preset": "bo_fir_p3m5_h13_fir47_amp37thermal16_1495",
        "hidden": 13,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 4,
        "attention_mode": "amp37_thermal16",
        "thermal_window": 16,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    },
    "p3m5_h16_l4_fir57_1964": {
        "preset": "bo_fir_p3m5_h16_l4_fir57_1964",
        "hidden": 16,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h18_l4_fir57_2316": {
        "preset": "bo_fir_p3m5_h18_l4_fir57_2316",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h18_l4_fir57_amp7_2334": {
        "preset": "bo_fir_p3m5_h18_l4_fir57_amp7_2334",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "input_extra": ["amp7"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h18_l4_fir57_amp7thermal16_2352": {
        "preset": "bo_fir_p3m5_h18_l4_fir57_amp7thermal16_2352",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "input_extra": ["amp7", "thermal16"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h18_l4_fir57_amp7amp9thermal16_2370": {
        "preset": "bo_fir_p3m5_h18_l4_fir57_amp7amp9thermal16_2370",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "input_extra": ["amp7", "amp9", "thermal16"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h18_l4_fir57_amp7amp9thermal16thermal64_2388": {
        "preset": "bo_fir_p3m5_h18_l4_fir57_amp7amp9thermal16thermal64_2388",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 3,
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h18_l4_fir71_attn4thermal_amp7amp9thermal16thermal64_2498": {
        "preset": "bo_fir_p3m5_h18_l4_fir71_attn4thermal_amp7amp9thermal16thermal64_2498",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 4,
        "attention_mode": "amp37_thermal16",
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 71,
    },
    "p3m5_h18_l4_fir67_attn4thermal_amp7amp9thermal16thermal64thermal128_2500": {
        "preset": "bo_fir_p3m5_h18_l4_fir67_attn4thermal_amp7amp9thermal16thermal64thermal128_2500",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 4,
        "attention_mode": "amp37_thermal16",
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64", "thermal128"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 67,
    },
    "p3m5_h18_l4_fir85_amp7amp9thermal16thermal64_2500": {
        "preset": "bo_fir_p3m5_h18_l4_fir85_amp7amp9thermal16thermal64_2500",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 3,
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 85,
    },
    "p3m5_h18_l4_fir57_attn4thermal_lag4amp9_2496": {
        "preset": "bo_fir_p3m5_h18_l4_fir57_attn4thermal_lag4amp9_2496",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 4,
        "attention_mode": "amp37_thermal16",
        "input_extra": [
            "amp7",
            "amp9",
            "thermal16",
            "thermal64",
            "thermal128",
            "lag4_amp9_i",
            "lag4_amp9_q",
        ],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h18_l4_fir49_attn4thermal_lag4lead2amp9_2500": {
        "preset": "bo_fir_p3m5_h18_l4_fir49_attn4thermal_lag4lead2amp9_2500",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 4,
        "attention_mode": "amp37_thermal16",
        "input_extra": [
            "amp7",
            "amp9",
            "thermal16",
            "thermal64",
            "thermal128",
            "lag4_amp9_i",
            "lag4_amp9_q",
            "lead2_amp9_i",
            "lead2_amp9_q",
        ],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 49,
    },
    "p3m5_h22_l4_g2_fir101_attn4thermal_thermal128_2500": {
        "preset": "bo_fir_p3m5_h22_l4_g2_fir101_attn4thermal_thermal128_2500",
        "hidden": 22,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "pointwise_groups": 2,
        "attention_channels": 4,
        "attention_mode": "amp37_thermal16",
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64", "thermal128"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 101,
    },
    "p3m5_h19_l4_fir37_amp7amp9thermal16thermal64_2500": {
        "preset": "bo_fir_p3m5_h19_l4_fir37_amp7amp9thermal16thermal64_2500",
        "hidden": 19,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 5,
        "attention_channels": 3,
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 37,
    },
    "p3m5_h18_l4_k7_fir21_amp7amp9thermal16thermal64_2388": {
        "preset": "bo_fir_p3m5_h18_l4_k7_fir21_amp7amp9thermal16thermal64_2388",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 7,
        "attention_channels": 3,
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 21,
    },
    "p3m5_h17_l4_k7_fir71_amp7amp9thermal16thermal64_2396": {
        "preset": "bo_fir_p3m5_h17_l4_k7_fir71_amp7amp9thermal16thermal64_2396",
        "hidden": 17,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "kernel_size": 7,
        "attention_channels": 3,
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 71,
    },
    "p3m5_h18_l4_fir43_attn4thermal_amp7amp9thermal16thermal64_2386": {
        "preset": "bo_fir_p3m5_h18_l4_fir43_attn4thermal_amp7amp9thermal16thermal64_2386",
        "hidden": 18,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 4,
        "attention_mode": "amp37_thermal16",
        "input_extra": ["amp7", "amp9", "thermal16", "thermal64"],
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 43,
    },
    "p3m5_h29_l4_r4_fir57_2388": {
        "preset": "bo_fir_p3m5_h29_l4_r4_fir57_2388",
        "hidden": 29,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "pointwise_rank": 4,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h20_l4_r8_fir57_2380": {
        "preset": "bo_fir_p3m5_h20_l4_r8_fir57_2380",
        "hidden": 20,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "pointwise_rank": 8,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h21_l4_r7_fir57_2316": {
        "preset": "bo_fir_p3m5_h21_l4_r7_fir57_2316",
        "hidden": 21,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "pointwise_rank": 7,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h32_l4_r3_fir57_2348": {
        "preset": "bo_fir_p3m5_h32_l4_r3_fir57_2348",
        "hidden": 32,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "pointwise_rank": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h16_l5_fir57_2300": {
        "preset": "bo_fir_p3m5_h16_l5_fir57_2300",
        "hidden": 16,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8, 16],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h15_l6_fir57_2400": {
        "preset": "bo_fir_p3m5_h15_l6_fir57_2400",
        "hidden": 15,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8, 16, 32],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h19_l4_fir31_2400": {
        "preset": "bo_fir_p3m5_h19_l4_fir31_2400",
        "hidden": 19,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 31,
    },
    "p3m5_h14_l5_fir57_1910": {
        "preset": "bo_fir_p3m5_h14_l5_fir57_1910",
        "hidden": 14,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8, 16],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p3m5_h15_l5_fir31_1996": {
        "preset": "bo_fir_p3m5_h15_l5_fir31_1996",
        "hidden": 15,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8, 16],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 31,
    },
    "p3m5_h13_l6_fir57_1964": {
        "preset": "bo_fir_p3m5_h13_l6_fir57_1964",
        "hidden": 13,
        "bo_fir_taps": 5,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8, 16, 32],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 57,
    },
    "p2m13_h13_fir47_1500": {
        "preset": "bo_fir_p2m13_h13_fir47_1500",
        "hidden": 13,
        "bo_fir_taps": 13,
        "bo_fir_units": 2,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    },
    "p3m13_h12_fir47_1412": {
        "preset": "bo_fir_p3m13_h12_fir47_1412",
        "hidden": 12,
        "bo_fir_taps": 13,
        "bo_fir_units": 3,
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    },
}

DEFAULT_CHECKPOINTS = {
    DEFAULT_VARIANT: DEFAULT_CHECKPOINT,
    "p2m5_h11_fir47_1164": CHECKPOINT_DIR / "BO-FIR_P2M5_H11_FIR47_1164_best.pth",
    "p3m13_h10_fir47_1156": CHECKPOINT_DIR / "BO-FIR_P3M13_H10_FIR47_1156_best.pth",
    "p3m5_h11_fir47_1184": CHECKPOINT_DIR / "BO-FIR_P3M5_H11_FIR47_1184_best.pth",
    "p3m5_h13_fir47_1456": CHECKPOINT_DIR / "BO-FIR_P3M5_H13_FIR47_1456_best.pth",
    "p3m5_h13_fir57_1496": CHECKPOINT_DIR / "BO-FIR_P3M5_H13_FIR57_1496_best.pth",
    "p3m5_h13_fir57_thermal16_1496": CHECKPOINT_DIR / "BO-FIR_P3M5_H13_FIR57_THERMAL16_1496_best.pth",
    "p3m5_h13_fir47_amp37thermal16_1495": CHECKPOINT_DIR / "BO-FIR_P3M5_H13_FIR47_AMP37THERMAL16_1495_best.pth",
    "p3m5_h16_l4_fir57_1964": CHECKPOINT_DIR / "BO-FIR_P3M5_H16_L4_FIR57_1964_best.pth",
    "p3m5_h18_l4_fir57_2316": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR57_2316_best.pth",
    "p3m5_h18_l4_fir57_amp7_2334": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR57_AMP7_2334_best.pth",
    "p3m5_h18_l4_fir57_amp7thermal16_2352": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR57_AMP7THERMAL16_2352_best.pth",
    "p3m5_h18_l4_fir57_amp7amp9thermal16_2370": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR57_AMP7AMP9THERMAL16_2370_best.pth",
    "p3m5_h18_l4_fir57_amp7amp9thermal16thermal64_2388": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR57_AMP7AMP9THERMAL16THERMAL64_2388_best.pth",
    "p3m5_h18_l4_fir71_attn4thermal_amp7amp9thermal16thermal64_2498": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR71_ATTN4THERMAL_AMP7AMP9THERMAL16THERMAL64_2498_best.pth",
    "p3m5_h18_l4_fir67_attn4thermal_amp7amp9thermal16thermal64thermal128_2500": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR67_ATTN4THERMAL_AMP7AMP9THERMAL16THERMAL64THERMAL128_2500_best.pth",
    "p3m5_h18_l4_fir85_amp7amp9thermal16thermal64_2500": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR85_AMP7AMP9THERMAL16THERMAL64_2500_best.pth",
    "p3m5_h18_l4_fir57_attn4thermal_lag4amp9_2496": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR57_ATTN4THERMAL_LAG4AMP9_2496_best.pth",
    "p3m5_h18_l4_fir49_attn4thermal_lag4lead2amp9_2500": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR49_ATTN4THERMAL_LAG4LEAD2AMP9_2500_best.pth",
    "p3m5_h22_l4_g2_fir101_attn4thermal_thermal128_2500": CHECKPOINT_DIR / "BO-FIR_P3M5_H22_L4_G2_FIR101_ATTN4THERMAL_THERMAL128_2500_best.pth",
    "p3m5_h19_l4_fir37_amp7amp9thermal16thermal64_2500": CHECKPOINT_DIR / "BO-FIR_P3M5_H19_L4_FIR37_AMP7AMP9THERMAL16THERMAL64_2500_best.pth",
    "p3m5_h18_l4_k7_fir21_amp7amp9thermal16thermal64_2388": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_K7_FIR21_AMP7AMP9THERMAL16THERMAL64_2388_best.pth",
    "p3m5_h17_l4_k7_fir71_amp7amp9thermal16thermal64_2396": CHECKPOINT_DIR / "BO-FIR_P3M5_H17_L4_K7_FIR71_AMP7AMP9THERMAL16THERMAL64_2396_best.pth",
    "p3m5_h18_l4_fir43_attn4thermal_amp7amp9thermal16thermal64_2386": CHECKPOINT_DIR / "BO-FIR_P3M5_H18_L4_FIR43_ATTN4THERMAL_AMP7AMP9THERMAL16THERMAL64_2386_best.pth",
    "p3m5_h29_l4_r4_fir57_2388": CHECKPOINT_DIR / "BO-FIR_P3M5_H29_L4_R4_FIR57_2388_best.pth",
    "p3m5_h20_l4_r8_fir57_2380": CHECKPOINT_DIR / "BO-FIR_P3M5_H20_L4_R8_FIR57_2380_best.pth",
    "p3m5_h21_l4_r7_fir57_2316": CHECKPOINT_DIR / "BO-FIR_P3M5_H21_L4_R7_FIR57_2316_best.pth",
    "p3m5_h32_l4_r3_fir57_2348": CHECKPOINT_DIR / "BO-FIR_P3M5_H32_L4_R3_FIR57_2348_best.pth",
    "p3m5_h16_l5_fir57_2300": CHECKPOINT_DIR / "BO-FIR_P3M5_H16_L5_FIR57_2300_best.pth",
    "p3m5_h15_l6_fir57_2400": CHECKPOINT_DIR / "BO-FIR_P3M5_H15_L6_FIR57_2400_best.pth",
    "p3m5_h19_l4_fir31_2400": CHECKPOINT_DIR / "BO-FIR_P3M5_H19_L4_FIR31_2400_best.pth",
    "p3m5_h14_l5_fir57_1910": CHECKPOINT_DIR / "BO-FIR_P3M5_H14_L5_FIR57_1910_best.pth",
    "p3m5_h15_l5_fir31_1996": CHECKPOINT_DIR / "BO-FIR_P3M5_H15_L5_FIR31_1996_best.pth",
    "p3m5_h13_l6_fir57_1964": CHECKPOINT_DIR / "BO-FIR_P3M5_H13_L6_FIR57_1964_best.pth",
    "p2m13_h13_fir47_1500": CHECKPOINT_DIR / "BO-FIR_P2M13_H13_FIR47_1500_best.pth",
    "p3m13_h12_fir47_1412": CHECKPOINT_DIR / "BO-FIR_P3M13_H12_FIR47_1412_best.pth",
}


def real_multiplier_count(config=MODEL_CONFIG):
    hidden = int(config["hidden"])
    bo_fir_taps = int(config["bo_fir_taps"])
    bo_fir_units = int(config["bo_fir_units"])
    blocks = len(config["dilations"])
    kernel_size = int(config.get("kernel_size", 5))
    attention_channels = int(config["attention_channels"])
    input_channels = 9 + len(config.get("input_extra", []))
    output_fir_taps = int(config["output_fir_taps"])
    physical = 12
    input_bo_fir = 4 * bo_fir_taps * bo_fir_units
    input_projection = input_channels * hidden
    tcn = blocks * (hidden * 5 + hidden * hidden)
    output_projection = 2 * hidden
    attention = hidden * attention_channels * 3
    pointwise_rank = config.get("pointwise_rank")
    pointwise_groups = int(config.get("pointwise_groups", 1))
    if pointwise_rank is None:
        tcn = blocks * (hidden * kernel_size + hidden * hidden // pointwise_groups)
    else:
        tcn = blocks * (hidden * kernel_size + 2 * hidden * int(pointwise_rank))
    output_fir = 4 * output_fir_taps
    return physical + input_bo_fir + input_projection + tcn + output_projection + attention + output_fir


class BOFIREATCN(nn.Module):
    def __init__(self, init_g_real, init_g_imag, config=None):
        super().__init__()
        self.config = dict(MODEL_CONFIG if config is None else config)
        self.hidden = int(self.config["hidden"])
        self.bo_fir_taps = int(self.config["bo_fir_taps"])
        self.bo_fir_units = int(self.config["bo_fir_units"])
        self.dilations = list(self.config["dilations"])
        self.kernel_size = int(self.config.get("kernel_size", 5))
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for centered TCN padding.")
        self.attention_channels = int(self.config.get("attention_channels", 3))
        self.attention_mode = self.config.get("attention_mode", "amp37")
        self.thermal_window = int(self.config.get("thermal_window", 16))
        self.input_extra = tuple(self.config.get("input_extra", []))
        self.input_channels = 9 + len(self.input_extra)
        self.pointwise_rank = self.config.get("pointwise_rank")
        self.pointwise_groups = int(self.config.get("pointwise_groups", 1))
        if self.pointwise_rank is None and self.hidden % self.pointwise_groups != 0:
            raise ValueError("hidden must be divisible by pointwise_groups.")
        self.output_fir_taps = int(self.config["output_fir_taps"])

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.bo_fir_i = nn.Parameter(torch.zeros(self.bo_fir_units, 1, self.bo_fir_taps, dtype=torch.float64))
        self.bo_fir_q = nn.Parameter(torch.zeros(self.bo_fir_units, 1, self.bo_fir_taps, dtype=torch.float64))
        self.init_bo_firs_as_rich9_delays()

        self.in_conv = nn.Conv1d(self.input_channels, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList()
        for dilation in self.dilations:
            depthwise = nn.Conv1d(
                self.hidden,
                self.hidden,
                self.kernel_size,
                padding=(self.kernel_size // 2) * dilation,
                dilation=dilation,
                groups=self.hidden,
                bias=False,
                dtype=torch.float64,
            )
            if self.pointwise_rank is None:
                self.tcn_blocks.append(
                    nn.Sequential(
                        depthwise,
                        nn.Conv1d(
                            self.hidden,
                            self.hidden,
                            1,
                            groups=self.pointwise_groups,
                            bias=False,
                            dtype=torch.float64,
                        ),
                        nn.SiLU(),
                    )
                )
            else:
                rank = int(self.pointwise_rank)
                self.tcn_blocks.append(
                    nn.Sequential(
                        depthwise,
                        nn.Conv1d(self.hidden, rank, 1, bias=False, dtype=torch.float64),
                        nn.Conv1d(rank, self.hidden, 1, bias=False, dtype=torch.float64),
                        nn.SiLU(),
                    )
                )
        self.attn = aidpd.OptimalSpectrumAttention(
            self.hidden,
            self.attention_channels,
            mode=self.attention_mode,
        )
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)
        nn.init.constant_(self.out_conv.weight, 0.0)

        center = self.output_fir_taps // 2
        fir_real = torch.zeros(self.output_fir_taps, dtype=torch.float64)
        fir_imag = torch.zeros(self.output_fir_taps, dtype=torch.float64)
        fir_real[center] = 1.0
        self.fir_real = nn.Parameter(fir_real)
        self.fir_imag = nn.Parameter(fir_imag)

    def init_bo_firs_as_rich9_delays(self):
        center = self.bo_fir_taps // 2
        with torch.no_grad():
            self.bo_fir_i.zero_()
            self.bo_fir_q.zero_()
            # Conv1d uses cross-correlation. With centered padding:
            # unit 0 = current sample, unit 1 = x[n+1]. Optional unit 2 = x[n-1].
            self.bo_fir_i[0, 0, center] = 1.0
            if self.bo_fir_units > 1:
                self.bo_fir_i[1, 0, min(center + 1, self.bo_fir_taps - 1)] = 1.0
            if self.bo_fir_units > 2:
                self.bo_fir_i[2, 0, max(center - 1, 0)] = 1.0

    def physical_branch(self, i_x, q_x):
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real
        amp2 = i_x * i_x + q_x * q_x
        amp4 = amp2 * amp2
        x3_i = i_x * amp2
        x3_q = q_x * amp2
        lin_i = lin_i + x3_i * self.g3_real - x3_q * self.g3_imag
        lin_q = lin_q + x3_i * self.g3_imag + x3_q * self.g3_real
        x5_i = i_x * amp4
        x5_q = q_x * amp4
        lin_i = lin_i + x5_i * self.g5_real - x5_q * self.g5_imag
        lin_q = lin_q + x5_i * self.g5_imag + x5_q * self.g5_real
        return torch.cat([lin_i, lin_q], dim=1)

    def bo_fir_features(self, i_x, q_x):
        def shifted(signal, samples):
            if samples > 0:
                return F.pad(signal[:, :, :-samples], (samples, 0), mode="replicate")
            if samples < 0:
                lead = -samples
                return F.pad(signal[:, :, lead:], (0, lead), mode="replicate")
            return signal

        pad = self.bo_fir_taps // 2
        i_f = F.conv1d(i_x, self.bo_fir_i, padding=pad) - F.conv1d(q_x, self.bo_fir_q, padding=pad)
        q_f = F.conv1d(i_x, self.bo_fir_q, padding=pad) + F.conv1d(q_x, self.bo_fir_i, padding=pad)
        i0, q0 = i_f[:, 0:1, :], q_f[:, 0:1, :]
        if self.bo_fir_units >= 3:
            i2, q2 = i_f[:, 2:3, :], q_f[:, 2:3, :]
        else:
            i2, q2 = torch.roll(i_x, shifts=1, dims=2), torch.roll(q_x, shifts=1, dims=2)
        amp = torch.sqrt(i0 * i0 + q0 * q0 + 1e-12)
        amp3 = amp ** 3
        amp5 = amp ** 5
        features = [i_f[:, 0:1], q_f[:, 0:1], i_f[:, 1:2], q_f[:, 1:2], i2, q2, amp, amp3, amp5]
        if "amp7" in self.input_extra:
            features.append(amp5 * amp * amp)
        if "amp9" in self.input_extra:
            amp7 = amp5 * amp * amp
            features.append(amp7 * amp * amp)
        if "thermal16" in self.input_extra:
            amp2 = amp * amp
            features.append(
                F.avg_pool1d(
                    F.pad(amp2, (self.thermal_window - 1, 0), mode="replicate"),
                    kernel_size=self.thermal_window,
                    stride=1,
                )
            )
        if "thermal64" in self.input_extra:
            amp2 = amp * amp
            features.append(
                F.avg_pool1d(
                    F.pad(amp2, (63, 0), mode="replicate"),
                    kernel_size=64,
                    stride=1,
                )
            )
        if "thermal128" in self.input_extra:
            amp2 = amp * amp
            features.append(
                F.avg_pool1d(
                    F.pad(amp2, (127, 0), mode="replicate"),
                    kernel_size=128,
                    stride=1,
                )
            )
        if "lag4_amp9_i" in self.input_extra or "lag4_amp9_q" in self.input_extra:
            lag_i = shifted(i_x, 4)
            lag_q = shifted(q_x, 4)
            lag_amp = torch.sqrt(lag_i * lag_i + lag_q * lag_q + 1e-12)
            lag_amp8 = lag_amp ** 8
            if "lag4_amp9_i" in self.input_extra:
                features.append(lag_i * lag_amp8)
            if "lag4_amp9_q" in self.input_extra:
                features.append(lag_q * lag_amp8)
        if "lead2_amp9_i" in self.input_extra or "lead2_amp9_q" in self.input_extra:
            lead_i = shifted(i_x, -2)
            lead_q = shifted(q_x, -2)
            lead_amp = torch.sqrt(lead_i * lead_i + lead_q * lead_q + 1e-12)
            lead_amp8 = lead_amp ** 8
            if "lead2_amp9_i" in self.input_extra:
                features.append(lead_i * lead_amp8)
            if "lead2_amp9_q" in self.input_extra:
                features.append(lead_q * lead_amp8)
        return torch.cat(features, dim=1), amp, amp3

    def forward_base(self, x_iq):
        i_x = x_iq[:, 0:1, :]
        q_x = x_iq[:, 1:2, :]
        lin_out = self.physical_branch(i_x, q_x)
        features, amp, amp3 = self.bo_fir_features(i_x, q_x)
        amp7 = amp ** 7
        amp5 = amp ** 5
        feat = self.act(self.in_conv(features))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        thermal = None
        if self.attention_mode in ("thermal16", "amp37_thermal16"):
            amp2 = amp * amp
            thermal = F.avg_pool1d(
                F.pad(amp2, (self.thermal_window - 1, 0), mode="replicate"),
                kernel_size=self.thermal_window,
                stride=1,
            )
        residual = self.out_conv(self.attn(feat, amp, amp3, amp7, amp5, thermal=thermal))
        return lin_out + residual

    def apply_output_fir(self, y):
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


def parse_args():
    parser = argparse.ArgumentParser(description="BO input-FIR + EA-TCN, 1000 real multipliers.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--variant", choices=sorted(VARIANT_CONFIGS), default=DEFAULT_VARIANT)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs-base", type=int, default=260)
    parser.add_argument("--epochs-long", type=int, default=160)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--long-lr", type=float, default=2e-6)
    parser.add_argument("--scheduler-patience", type=int, default=12)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--long-batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--long-frame-length", type=int, default=4096)
    parser.add_argument("--long-stride-length", type=int, default=256)
    parser.add_argument("--crop-margin", type=int, default=0)
    parser.add_argument("--diff-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-main-weight", type=float, default=0.25)
    parser.add_argument("--spectral-adjacent-weight", type=float, default=1.0)
    parser.add_argument("--spectral-floor-weight", type=float, default=0.05)
    parser.add_argument("--freeze-fir-base", action="store_true")
    parser.add_argument("--checkpoint-policy", choices=("nmse", "aclr_nmse"), default="aclr_nmse")
    parser.add_argument("--max-multipliers", type=int, default=1200)
    parser.add_argument("--target-fullseq-db", type=float, default=-40.0)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(-100.0, 100.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-245.0, -180.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(180.0, 245.0))
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0)
    return parser.parse_args()


def resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


def load_data(data_path):
    _, _, x_c, y_c = aidpd.load_normalized_data(data_path)
    x_np = aidpd.iq_array(x_c)
    y_np = aidpd.iq_array(y_c)
    split_idx = int(min(len(x_np), len(y_np)) * 0.8)
    g_opt = np.vdot(x_c[:split_idx], y_c[:split_idx]) / np.vdot(x_c[:split_idx], x_c[:split_idx])
    return x_np[:split_idx], y_np[:split_idx], x_np[split_idx:], y_np[split_idx:], float(g_opt.real), float(g_opt.imag)


def load_from_eatcn(model, checkpoint_path):
    source = aidpd.load_state_dict(checkpoint_path)
    target = model.state_dict()
    loaded = []
    cropped = []
    skipped = []
    source_hidden = source["in_conv.weight"].shape[0] if "in_conv.weight" in source else None
    source_block_ids = {
        int(key.split(".")[1])
        for key in source
        if key.startswith("tcn_blocks.") and key.endswith(".0.weight")
    }
    for key, value in source.items():
        if key not in target:
            skipped.append(key)
            continue
        dst = target[key]
        if value.shape == dst.shape:
            target[key] = value.to(dtype=dst.dtype)
            loaded.append(key)
        elif key in ("fir_real", "fir_imag") and value.ndim == 1 and dst.ndim == 1:
            src_center = value.shape[0] // 2
            dst_center = dst.shape[0] // 2
            half = min(src_center, dst_center, value.shape[0] - src_center - 1, dst.shape[0] - dst_center - 1)
            merged = dst.clone()
            merged[dst_center - half:dst_center + half + 1] = value[src_center - half:src_center + half + 1].to(dtype=dst.dtype)
            target[key] = merged
            cropped.append(key)
        elif key.startswith("tcn_blocks.") and key.endswith(".0.weight") and value.ndim == 3 and dst.ndim == 3:
            src_center = value.shape[-1] // 2
            dst_center = dst.shape[-1] // 2
            half = min(src_center, dst_center, value.shape[-1] - src_center - 1, dst.shape[-1] - dst_center - 1)
            merged = torch.zeros_like(dst)
            out_ch = min(value.shape[0], dst.shape[0])
            in_ch = min(value.shape[1], dst.shape[1])
            merged[
                :out_ch,
                :in_ch,
                dst_center - half:dst_center + half + 1,
            ] = value[
                :out_ch,
                :in_ch,
                src_center - half:src_center + half + 1,
            ].to(dtype=dst.dtype)
            target[key] = merged
            cropped.append(key)
        elif (
            key.startswith("tcn_blocks.")
            and key.endswith(".1.weight")
            and getattr(model, "pointwise_groups", 1) > 1
            and value.ndim == 3
            and dst.ndim == 3
        ):
            groups = int(model.pointwise_groups)
            group_size = dst.shape[0] // groups
            merged = torch.zeros_like(dst)
            for group_idx in range(groups):
                out_start = group_idx * group_size
                out_stop = min((group_idx + 1) * group_size, value.shape[0], dst.shape[0])
                in_start = group_idx * group_size
                in_stop = min((group_idx + 1) * group_size, value.shape[1])
                rows = max(0, out_stop - out_start)
                cols = max(0, in_stop - in_start)
                if rows > 0 and cols > 0:
                    merged[out_start:out_start + rows, :cols, :] = value[
                        out_start:out_start + rows,
                        in_start:in_start + cols,
                        :,
                    ].to(dtype=dst.dtype)
            target[key] = merged
            cropped.append(key)
        elif value.ndim == dst.ndim:
            merged = torch.zeros_like(dst)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, dst.shape))
            merged[slices] = value[slices].to(dtype=dst.dtype)
            target[key] = merged
            cropped.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(target, strict=True)
    if getattr(model, "pointwise_rank", None) is not None and source_hidden is not None:
        rank = int(model.pointwise_rank)
        with torch.no_grad():
            for block_idx in source_block_ids:
                if block_idx >= len(model.tcn_blocks):
                    continue
                src_key = f"tcn_blocks.{block_idx}.1.weight"
                if src_key not in source:
                    continue
                block = model.tcn_blocks[block_idx]
                if len(block) < 4 or not isinstance(block[1], nn.Conv1d) or not isinstance(block[2], nn.Conv1d):
                    continue
                full = torch.zeros(model.hidden, model.hidden, dtype=block[1].weight.dtype, device=block[1].weight.device)
                src = source[src_key].to(dtype=full.dtype, device=full.device).squeeze(-1)
                rows = min(src.shape[0], full.shape[0])
                cols = min(src.shape[1], full.shape[1])
                full[:rows, :cols] = src[:rows, :cols]
                u, s, vh = torch.linalg.svd(full, full_matrices=False)
                keep = min(rank, s.numel())
                sqrt_s = torch.sqrt(torch.clamp(s[:keep], min=0.0))
                left = torch.zeros(model.hidden, rank, dtype=full.dtype, device=full.device)
                right = torch.zeros(rank, model.hidden, dtype=full.dtype, device=full.device)
                left[:, :keep] = u[:, :keep] * sqrt_s.view(1, keep)
                right[:keep, :] = sqrt_s.view(keep, 1) * vh[:keep, :]
                block[1].weight.copy_(right.view(rank, model.hidden, 1))
                block[2].weight.copy_(left.view(model.hidden, rank, 1))
                cropped.append(f"tcn_blocks.{block_idx}.pointwise_svd_rank{rank}")
    if source_hidden is not None and hasattr(model, "hidden") and model.hidden > source_hidden:
        with torch.no_grad():
            nn.init.kaiming_uniform_(model.in_conv.weight[source_hidden:], a=float(np.sqrt(5)))
            nn.init.kaiming_uniform_(model.attn.env_conv.weight[source_hidden:], a=float(np.sqrt(5)))
            for block_idx in source_block_ids:
                if block_idx >= len(model.tcn_blocks):
                    continue
                block = model.tcn_blocks[block_idx]
                nn.init.kaiming_uniform_(block[0].weight[source_hidden:], a=float(np.sqrt(5)))
                if getattr(model, "pointwise_rank", None) is None:
                    nn.init.kaiming_uniform_(block[1].weight[source_hidden:], a=float(np.sqrt(5)))
            cropped.append(f"hidden.{source_hidden}_to_{model.hidden}.random_init")
    if source_block_ids and hasattr(model, "tcn_blocks"):
        max_source_block = max(source_block_ids)
        with torch.no_grad():
            for block_idx, block in enumerate(model.tcn_blocks):
                if block_idx <= max_source_block:
                    continue
                if isinstance(block[0], nn.Conv1d):
                    block[0].reset_parameters()
                if getattr(model, "pointwise_rank", None) is not None and len(block) >= 4:
                    if isinstance(block[1], nn.Conv1d):
                        block[1].reset_parameters()
                    if isinstance(block[2], nn.Conv1d):
                        block[2].weight.zero_()
                        if block[2].bias is not None:
                            block[2].bias.zero_()
                elif isinstance(block[1], nn.Conv1d):
                    block[1].weight.zero_()
                    if block[1].bias is not None:
                        block[1].bias.zero_()
                cropped.append(f"tcn_blocks.{block_idx}.residual_zero_init")
    return loaded, cropped, skipped


def make_loader(x_np, y_np, frame_length, stride_length, batch_size):
    return DataLoader(
        aidpd.FramedDataset(x_np, y_np, frame_length, stride_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def fullseq_nmse(model, x_val, y_val):
    nmse, db = aidpd.evaluate_full_sequence_nmse(model, x_val, y_val)
    return float(nmse), float(db)


def fullseq_aclr_worst(model, x_val, fs_mhz, main_band, lower_band, upper_band):
    pred_iq = aidpd.run_sequence(model, x_val)
    pred_complex = aidpd.tensor_to_complex(pred_iq)
    freq_hz, pred_psd = aidpd.psd_db(pred_complex, fs_hz=fs_mhz * 1e6)
    return float(aidpd.aclr_from_psd(freq_hz, pred_psd, main_band, lower_band, upper_band)[2])


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
        sl = slice(crop_margin, -crop_margin)
        pred = pred[:, :, sl]
        target = target[:, :, sl]
    loss = nn.functional.mse_loss(pred, target)
    if diff_loss_weight > 0:
        pred_d = pred[:, :, 1:] - pred[:, :, :-1]
        target_d = target[:, :, 1:] - target[:, :, :-1]
        loss = loss + diff_loss_weight * nn.functional.mse_loss(pred_d, target_d)
    if spectral_loss_weight > 0:
        spectral_loss = weighted_spectral_nmse(
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
        loss = loss + spectral_loss_weight * spectral_loss
    return loss


def set_fir_trainable(model, trainable):
    for name, param in model.named_parameters():
        if name.startswith("fir_"):
            param.requires_grad = trainable


def train_stage(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    stage_name,
    epochs,
    lr,
    scheduler_patience,
    scheduler_min_lr,
    frame_length,
    stride_length,
    batch_size,
    crop_margin,
    diff_loss_weight,
    spectral_loss_weight,
    spectral_main_weight,
    spectral_adjacent_weight,
    spectral_floor_weight,
    checkpoint_path,
    target_fullseq_db,
    checkpoint_policy,
    fs_mhz,
    main_band,
    lower_band,
    upper_band,
    aclr_target_dbc,
):
    if epochs <= 0:
        nmse, db = fullseq_nmse(model, x_val, y_val)
        torch.save(model.state_dict(), checkpoint_path)
        return [(0, db)], nmse, db
    loader = make_loader(x_train, y_train, frame_length, stride_length, batch_size)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )
    best_nmse, best_db = fullseq_nmse(model, x_val, y_val)
    saved_nmse, saved_db = best_nmse, best_db
    best_pass_nmse = float("inf")
    if checkpoint_policy == "aclr_nmse":
        initial_aclr = fullseq_aclr_worst(model, x_val, fs_mhz, main_band, lower_band, upper_band)
        print(f">>> Initial ACLR worst: {initial_aclr:.2f} dBc")
        if initial_aclr <= aclr_target_dbc:
            best_pass_nmse = best_nmse
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]
    start = time.time()
    print(f"\n>>> Stage {stage_name}: epochs={epochs}, lr={lr:.2e}, frame={frame_length}, batch={batch_size}, crop_margin={crop_margin}")
    print(f">>> Initial full-sequence NMSE: {best_nmse:.6e} ({best_db:.2f} dB)")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = loss_with_optional_terms(
                pred,
                batch_y,
                crop_margin=crop_margin,
                diff_loss_weight=diff_loss_weight,
                spectral_loss_weight=spectral_loss_weight,
                fs_mhz=fs_mhz,
                main_band=main_band,
                lower_band=lower_band,
                upper_band=upper_band,
                spectral_main_weight=spectral_main_weight,
                spectral_adjacent_weight=spectral_adjacent_weight,
                spectral_floor_weight=spectral_floor_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        nmse, db = fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        history.append((epoch, db))
        elapsed = time.time() - start
        print(f"{stage_name} Epoch [{epoch}/{epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | LR {optimizer.param_groups[0]['lr']:.2e} | Full NMSE {db:.3f} dB")
        if nmse < best_nmse:
            best_nmse = nmse
            best_db = db
            save_candidate = checkpoint_policy == "nmse"
            if checkpoint_policy == "aclr_nmse":
                candidate_aclr = fullseq_aclr_worst(model, x_val, fs_mhz, main_band, lower_band, upper_band)
                print(f">>> Candidate ACLR worst: {candidate_aclr:.2f} dBc")
                if candidate_aclr <= aclr_target_dbc and nmse < best_pass_nmse:
                    best_pass_nmse = nmse
                    save_candidate = True
                elif best_pass_nmse == float("inf"):
                    save_candidate = True
                    print(">>> No ACLR-passing checkpoint yet; saving best NMSE fallback.")
                else:
                    print(">>> Candidate improves NMSE but fails ACLR target; checkpoint kept unchanged.")
            if save_candidate:
                torch.save(model.state_dict(), checkpoint_path)
                saved_nmse, saved_db = nmse, db
            if saved_db <= target_fullseq_db:
                print(f">>> Target reached: {best_db:.2f} dB")
                break
    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    return history, saved_nmse, saved_db


def plot_history(histories, out_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    offset = 0
    for stage, history in histories:
        xs = [offset + item[0] for item in history]
        ys = [item[1] for item in history]
        plt.plot(xs, ys, label=stage)
        offset = xs[-1]
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.0, label="-37 dB")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("Full-sequence NMSE (dB)")
    plt.title("BO input-FIR + EA-TCN, 1000 real multipliers")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(args.data)
    config = dict(VARIANT_CONFIGS[args.variant])
    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINTS[args.variant]
    init_checkpoint = resolve_path(args.init_checkpoint)
    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)
    multipliers = real_multiplier_count(config)
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Variant: {args.variant}")
    print(f">>> Model config: {config}")
    print(f">>> Real multipliers: {multipliers}")
    if multipliers > args.max_multipliers:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers} > {args.max_multipliers}")

    model = BOFIREATCN(g_real, g_imag, config=config).to(DEVICE)
    if args.evaluate_only:
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        nmse, db = fullseq_nmse(model, x_val, y_val)
        print(f">>> Full-sequence NMSE: {nmse:.6e} ({db:.2f} dB)")
        aidpd.evaluate_aclr_visualization(model, x_val, y_val, output_prefix=checkpoint_path.stem, fs_mhz=args.fs_mhz, main_band=tuple(args.main_band), lower_band=tuple(args.lower_band), upper_band=tuple(args.upper_band), target_dbc=args.aclr_target_dbc)
        return

    if not args.scratch and init_checkpoint.exists():
        loaded, cropped, skipped = load_from_eatcn(model, init_checkpoint)
        print(f">>> Initialized from EA-TCN checkpoint: {init_checkpoint}")
        print(f">>> Loaded={len(loaded)}, cropped={cropped}, skipped={len(skipped)}")
    elif args.scratch:
        print(">>> Scratch mode.")
    else:
        print(f">>> Init checkpoint not found: {init_checkpoint}")

    histories = []
    initial_nmse, initial_db = fullseq_nmse(model, x_val, y_val)
    print(f">>> Initial full-sequence NMSE: {initial_nmse:.6e} ({initial_db:.2f} dB)")
    if args.freeze_fir_base:
        set_fir_trainable(model, False)
        print(">>> Output FIR frozen during short-frame training.")

    h1, _, db1 = train_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        stage_name="short-frame-freeze-fir" if args.freeze_fir_base else "short-frame",
        epochs=args.epochs_base,
        lr=args.base_lr,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
        frame_length=args.frame_length,
        stride_length=args.stride_length,
        batch_size=args.batch_size,
        crop_margin=args.crop_margin,
        diff_loss_weight=args.diff_loss_weight,
        spectral_loss_weight=args.spectral_loss_weight,
        spectral_main_weight=args.spectral_main_weight,
        spectral_adjacent_weight=args.spectral_adjacent_weight,
        spectral_floor_weight=args.spectral_floor_weight,
        checkpoint_path=checkpoint_path,
        target_fullseq_db=args.target_fullseq_db,
        checkpoint_policy=args.checkpoint_policy,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        aclr_target_dbc=args.aclr_target_dbc,
    )
    histories.append(("short-frame", h1))

    if db1 > args.target_fullseq_db and args.epochs_long > 0:
        set_fir_trainable(model, False)
        h2, _, _ = train_stage(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            stage_name="long-frame-freeze-fir",
            epochs=args.epochs_long,
            lr=args.long_lr,
            scheduler_patience=args.scheduler_patience,
            scheduler_min_lr=args.scheduler_min_lr,
            frame_length=args.long_frame_length,
            stride_length=args.long_stride_length,
            batch_size=args.long_batch_size,
            crop_margin=args.crop_margin,
            diff_loss_weight=0.0,
            spectral_loss_weight=args.spectral_loss_weight,
            spectral_main_weight=args.spectral_main_weight,
            spectral_adjacent_weight=args.spectral_adjacent_weight,
            spectral_floor_weight=args.spectral_floor_weight,
            checkpoint_path=checkpoint_path,
            target_fullseq_db=args.target_fullseq_db,
            checkpoint_policy=args.checkpoint_policy,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            aclr_target_dbc=args.aclr_target_dbc,
        )
        histories.append(("long-frame-freeze-fir", h2))

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    final_nmse, final_db = fullseq_nmse(model, x_val, y_val)
    print(f"\n>>> Final full-sequence NMSE: {final_nmse:.6e} ({final_db:.2f} dB)")
    print(f">>> Saved checkpoint: {checkpoint_path}")
    convergence_path = FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png"
    plot_history(histories, convergence_path)
    print(f">>> Convergence figure: {convergence_path}")
    aidpd.evaluate_aclr_visualization(model, x_val, y_val, output_prefix=checkpoint_path.stem, fs_mhz=args.fs_mhz, main_band=tuple(args.main_band), lower_band=tuple(args.lower_band), upper_band=tuple(args.upper_band), target_dbc=args.aclr_target_dbc)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
