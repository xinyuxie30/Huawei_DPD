"""
Scratch-only <=800-real-multiplier innovative DPD search.

This launcher intentionally does not modify the final report scripts. It
registers experimental sub-800 candidates on top of
final_train_lightweight_999_distill.py, but forces teacher-mode none and zero
distillation loss so the student is trained from random initialization against
the dataset target_y.

Main idea:
- Keep the expensive output FIR whenever possible, because ACLR is sensitive to
  residual group delay and PSD shape.
- Add TCN depth by replacing full pointwise 1x1 mixing with low-rank pointwise
  bottlenecks.
- Test Input-7/Input-8 only when the saved MAC budget does not cut FIR too
  aggressively.

Recommended first runs:
    python -B steps/step800_scratch_innovative.py --variant h10_l4_base6_rank3_fir41
    python -B steps/step800_scratch_innovative.py --variant h11_l4_base6_rank2_fir47
    python -B steps/step800_scratch_innovative.py --variant h10_l4_semirich8_rank3_fir37
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "final_train_lightweight_999_distill.py"


VARIANTS = {
    "h10_l4_base6_rank3_fir41": {
        "preset": "h10_l4_base6_rank3_fir41_786",
        "checkpoint": "step800_scratch_h10_l4_base6_rank3_fir41_786.pth",
        "prefix": "step800_scratch_h10_l4_base6_rank3_fir41_786",
    },
    "h10_l4_semirich7_rank3_fir39": {
        "preset": "h10_l4_semirich7_rank3_fir39_788",
        "checkpoint": "step800_scratch_h10_l4_semirich7_rank3_fir39_788.pth",
        "prefix": "step800_scratch_h10_l4_semirich7_rank3_fir39_788",
    },
    "h10_l4_semirich8_rank3_fir37": {
        "preset": "h10_l4_semirich8_rank3_fir37_790",
        "checkpoint": "step800_scratch_h10_l4_semirich8_rank3_fir37_790.pth",
        "prefix": "step800_scratch_h10_l4_semirich8_rank3_fir37_790",
    },
    "h11_l4_base6_rank2_fir47": {
        "preset": "h11_l4_base6_rank2_fir47_783",
        "checkpoint": "step800_scratch_h11_l4_base6_rank2_fir47_783.pth",
        "prefix": "step800_scratch_h11_l4_base6_rank2_fir47_783",
    },
    "h12_l5_base6_rank1_fir39": {
        "preset": "h12_l5_base6_rank1_fir39_792",
        "checkpoint": "step800_scratch_h12_l5_base6_rank1_fir39_792.pth",
        "prefix": "step800_scratch_h12_l5_base6_rank1_fir39_792",
    },
    "h13_l5_base6_rank1_fir27": {
        "preset": "h13_l5_base6_rank1_fir27_796",
        "checkpoint": "step800_scratch_h13_l5_base6_rank1_fir27_796.pth",
        "prefix": "step800_scratch_h13_l5_base6_rank1_fir27_796",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("lightweight_base_scratch800", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lightweight_base_scratch800"] = module
    spec.loader.exec_module(module)
    return module


def register_reduced_rich_modes(base):
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


def register_variants(base):
    register_reduced_rich_modes(base)
    base.STUDENT_CONFIGS.update(
        {
            "h10_l4_base6_rank3_fir41_786": {
                "name": "H10_L4_base6_rank3_FIR41_786",
                "hidden": 10,
                "input_mode": "base6",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "pointwise_rank": 3,
                "output_fir_taps": 41,
            },
            "h10_l4_semirich7_rank3_fir39_788": {
                "name": "H10_L4_semirich7_amp5_rank3_FIR39_788",
                "hidden": 10,
                "input_mode": "semirich7_amp5",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "pointwise_rank": 3,
                "output_fir_taps": 39,
            },
            "h10_l4_semirich8_rank3_fir37_790": {
                "name": "H10_L4_semirich8_pastIQ_rank3_FIR37_790",
                "hidden": 10,
                "input_mode": "semirich8_pastiq",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "pointwise_rank": 3,
                "output_fir_taps": 37,
            },
            "h11_l4_base6_rank2_fir47_783": {
                "name": "H11_L4_base6_rank2_FIR47_783",
                "hidden": 11,
                "input_mode": "base6",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "pointwise_rank": 2,
                "output_fir_taps": 47,
            },
            "h12_l5_base6_rank1_fir39_792": {
                "name": "H12_L5_base6_rank1_FIR39_792",
                "hidden": 12,
                "input_mode": "base6",
                "dilations": [1, 2, 4, 8, 16],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "pointwise_rank": 1,
                "output_fir_taps": 39,
            },
            "h13_l5_base6_rank1_fir27_796": {
                "name": "H13_L5_base6_rank1_FIR27_796",
                "hidden": 13,
                "input_mode": "base6",
                "dilations": [1, 2, 4, 8, 16],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "pointwise_rank": 1,
                "output_fir_taps": 27,
            },
        }
    )


def pop_variant(argv):
    variant = "h10_l4_base6_rank3_fir41"
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
        if item == "--list-variants":
            print("Available variants:")
            for name, meta in VARIANTS.items():
                print(f"  {name}: {meta['preset']}")
            raise SystemExit(0)
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
    register_variants(base)

    meta = VARIANTS[variant]
    checkpoint = PROJECT_ROOT / "checkpoints" / "AI-DPD" / meta["checkpoint"]

    scratch_defaults = {
        "--student-preset": meta["preset"],
        "--max-student-multipliers": 799,
        "--teacher-mode": "none",
        "--distill-loss-weight": 0.0,
        "--distill-diff-weight": 0.0,
        "--truth-loss-weight": 1.0,
        "--diff-loss-weight": 0.01,
        "--spectral-truth-weight": 2e-5,
        "--spectral-main-weight": 1.0,
        "--spectral-adjacent-weight": 4.0,
        "--spectral-floor-weight": 0.04,
        "--fir-fit-domain": "freq",
        "--fir-main-weight": 1.0,
        "--fir-adjacent-weight": 1.1,
        "--fir-floor-weight": 0.03,
        "--fir-ridge": 1e-7,
        "--stage1-epochs": 220,
        "--stage2-epochs": 120,
        "--stage3-epochs": 120,
        "--stage1-lr": 1e-3,
        "--stage2-lr": 2e-4,
        "--stage3-lr": 5e-5,
        "--batch-size": 16,
        "--frame-length": 1024,
        "--stride-length": 64,
        "--crop-margin": 64,
        "--refit-every": 10,
        "--aclr-every": 10,
        "--nmse-target-db": -40.0,
        "--final-checkpoint": checkpoint,
        "--output-prefix": meta["prefix"],
    }
    for key, value in scratch_defaults.items():
        append_default(argv, key, value)

    print(f">>> scratch800 innovative variant: {variant}")
    print(f">>> preset: {meta['preset']}")
    print(f">>> checkpoint: {checkpoint}")
    print(">>> training mode: scratch-only, teacher-mode none, no distillation")
    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()
