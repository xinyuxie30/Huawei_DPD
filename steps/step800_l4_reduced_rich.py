"""
Standalone launcher for <=800-real-multiplier L4 Input-family DPD tests.

This file intentionally does not modify the final report scripts. It registers
new lightweight student variants on top of final_train_lightweight_999_distill:

1. h9_l4_semirich7_amp5_fir29_794
   L4 TCN, Input-7 = Input-6 + r[n]^5, FIR29, 794 MAC.

2. h9_l4_semirich8_pastiq_fir27_795
   L4 TCN, Input-8 = Input-6 + past I/Q, FIR27, 795 MAC.

3. h9_l4_base6_fir31_793
   L4 TCN, Input-6, FIR31, 793 MAC. This isolates depth gain.

Default run:
    python -B steps/step800_l4_reduced_rich.py

Pick another variant:
    python -B steps/step800_l4_reduced_rich.py --variant semirich8_pastiq
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "final_train_lightweight_999_distill.py"


VARIANTS = {
    "semirich7_amp5": {
        "preset": "h9_l4_semirich7_amp5_fir29_794",
        "checkpoint": "step800_l4_semirich7_amp5_fir29_794.pth",
        "prefix": "step800_l4_semirich7_amp5_fir29_794",
    },
    "semirich8_pastiq": {
        "preset": "h9_l4_semirich8_pastiq_fir27_795",
        "checkpoint": "step800_l4_semirich8_pastiq_fir27_795.pth",
        "prefix": "step800_l4_semirich8_pastiq_fir27_795",
    },
    "base6_l4": {
        "preset": "h9_l4_base6_fir31_793",
        "checkpoint": "step800_l4_base6_fir31_793.pth",
        "prefix": "step800_l4_base6_fir31_793",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("lightweight_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lightweight_base"] = module
    spec.loader.exec_module(module)
    return module


def register_reduced_rich_variants(base):
    original_input_channels_for_mode = base.input_channels_for_mode
    original_build_features = base.build_features

    def input_channels_for_mode(input_mode):
        if input_mode == "semirich7_amp5":
            return 7
        if input_mode == "semirich8_pastiq":
            return 8
        return original_input_channels_for_mode(input_mode)

    def build_features(i_x, q_x, amp, amp3, amp5, input_mode):
        i_future = base.torch.roll(i_x, shifts=-1, dims=2)
        q_future = base.torch.roll(q_x, shifts=-1, dims=2)
        if input_mode == "semirich7_amp5":
            return base.torch.cat([i_x, q_x, i_future, q_future, amp, amp3, amp5], dim=1)
        if input_mode == "semirich8_pastiq":
            i_past = base.torch.roll(i_x, shifts=1, dims=2)
            q_past = base.torch.roll(q_x, shifts=1, dims=2)
            return base.torch.cat([i_x, q_x, i_future, q_future, i_past, q_past, amp, amp3], dim=1)
        return original_build_features(i_x, q_x, amp, amp3, amp5, input_mode)

    base.input_channels_for_mode = input_channels_for_mode
    base.build_features = build_features

    base.STUDENT_CONFIGS.update(
        {
            "h9_l4_semirich7_amp5_fir29_794": {
                "name": "H9_L4_semirich7_amp5_FIR29_794",
                "hidden": 9,
                "input_mode": "semirich7_amp5",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 29,
            },
            "h9_l4_semirich8_pastiq_fir27_795": {
                "name": "H9_L4_semirich8_pastIQ_FIR27_795",
                "hidden": 9,
                "input_mode": "semirich8_pastiq",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 27,
            },
            "h9_l4_base6_fir31_793": {
                "name": "H9_L4_base6_FIR31_793",
                "hidden": 9,
                "input_mode": "base6",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 31,
            },
        }
    )


def pop_variant(argv):
    variant = "semirich7_amp5"
    cleaned = [argv[0]]
    idx = 1
    while idx < len(argv):
        item = argv[idx]
        if item == "--variant":
            if idx + 1 >= len(argv):
                raise ValueError("--variant requires one of: " + ", ".join(VARIANTS))
            variant = argv[idx + 1]
            idx += 2
            continue
        if item.startswith("--variant="):
            variant = item.split("=", 1)[1]
            idx += 1
            continue
        cleaned.append(item)
        idx += 1
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}. Choices: {', '.join(VARIANTS)}")
    return variant, cleaned


def has_arg(argv, name):
    return name in argv or any(item.startswith(name + "=") for item in argv)


def append_default(argv, name, value):
    if not has_arg(argv, name):
        argv.extend([name, str(value)])


def main():
    variant, argv = pop_variant(sys.argv)
    base = load_base_module()
    register_reduced_rich_variants(base)

    meta = VARIANTS[variant]
    checkpoint = PROJECT_ROOT / "checkpoints" / "AI-DPD" / meta["checkpoint"]

    append_default(argv, "--student-preset", meta["preset"])
    append_default(argv, "--max-student-multipliers", 800)
    append_default(argv, "--final-checkpoint", checkpoint)
    append_default(argv, "--output-prefix", meta["prefix"])

    print(f">>> step800 L4 Input-family variant: {variant}")
    print(f">>> preset: {meta['preset']}")
    print(f">>> checkpoint: {checkpoint}")
    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()
