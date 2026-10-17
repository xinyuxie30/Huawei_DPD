"""
<=800-real-multiplier NMSE search launcher.

This keeps all new attempt artifacts under steps/step800_nmse_search and
reuses the existing training/evaluation math from final_train_lightweight_999_distill.py.
The unified NMSE/ACLR settings are not changed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "final_train_lightweight_999_distill.py"
ATTEMPT_ROOT = SCRIPT_DIR / "step800_nmse_search"
CHECKPOINT_DIR = ATTEMPT_ROOT / "checkpoints"
FIGURE_DIR = ATTEMPT_ROOT / "figures"


VARIANTS = {
    "h10_l3_rich9_fir33": {
        "preset": "h10_l3_rich9_fir33_794",
        "checkpoint": "h10_l3_rich9_fir33_794_distill.pth",
        "prefix": "h10_l3_rich9_fir33_794_distill",
    },
    "h10_l3_rich9_attn4_fir25": {
        "preset": "h10_l3_rich9_attn4_fir25_792",
        "checkpoint": "h10_l3_rich9_attn4_fir25_792_distill.pth",
        "prefix": "h10_l3_rich9_attn4_fir25_792_distill",
    },
    "h9_l3_rich9_fir57": {
        "preset": "h9_l3_rich9_fir57_798",
        "checkpoint": "h9_l3_rich9_fir57_798_distill.pth",
        "prefix": "h9_l3_rich9_fir57_798_distill",
    },
    "h9_l3_rich11_fir53": {
        "preset": "h9_l3_rich11_fir53_800",
        "checkpoint": "h9_l3_rich11_fir53_800_distill.pth",
        "prefix": "h9_l3_rich11_fir53_800_distill",
    },
    "h8_l4_rich11_fir49": {
        "preset": "h8_l4_rich11_fir49_800",
        "checkpoint": "h8_l4_rich11_fir49_800_distill.pth",
        "prefix": "h8_l4_rich11_fir49_800_distill",
    },
    "h10_l3_base6_poly7_fir39": {
        "preset": "h10_l3_base6_poly7_fir39_792",
        "checkpoint": "h10_l3_base6_poly7_fir39_792_distill.pth",
        "prefix": "h10_l3_base6_poly7_fir39_792_distill",
    },
    "h10_l4_rich9_rank3_fir33": {
        "preset": "h10_l4_rich9_rank3_fir33_784",
        "checkpoint": "h10_l4_rich9_rank3_fir33_784_distill.pth",
        "prefix": "h10_l4_rich9_rank3_fir33_784_distill",
    },
    "h9_l4_rich9_fir25": {
        "preset": "h9_l4_rich9_fir25_796",
        "checkpoint": "h9_l4_rich9_fir25_796_distill.pth",
        "prefix": "h9_l4_rich9_fir25_796_distill",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("lightweight_base_step800_nmse", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lightweight_base_step800_nmse"] = module
    spec.loader.exec_module(module)
    return module


def load_aidpd_module():
    script = SCRIPT_DIR / "AI-DPD.py"
    spec = importlib.util.spec_from_file_location("aidpd_step800_partial", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aidpd_step800_partial"] = module
    spec.loader.exec_module(module)
    return module


def register_variants(base):
    base.STUDENT_CONFIGS.update(
        {
            "h10_l3_rich9_fir33_794": {
                "name": "H10_L3_rich9_FIR33_794",
                "hidden": 10,
                "input_mode": "rich9",
                "dilations": [1, 2, 4],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 33,
            },
            "h10_l3_rich9_attn4_fir25_792": {
                "name": "H10_L3_rich9_attn4_FIR25_792",
                "hidden": 10,
                "input_mode": "rich9",
                "dilations": [1, 2, 4],
                "attention_channels": 4,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 25,
            },
            "h9_l3_rich9_fir57_798": {
                "name": "H9_L3_rich9_FIR57_798",
                "hidden": 9,
                "input_mode": "rich9",
                "dilations": [1, 2, 4],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 57,
            },
            "h9_l3_rich11_fir53_800": {
                "name": "H9_L3_rich11_FIR53_800",
                "hidden": 9,
                "input_mode": "rich11",
                "dilations": [1, 2, 4],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 53,
            },
            "h8_l4_rich11_fir49_800": {
                "name": "H8_L4_rich11_FIR49_800",
                "hidden": 8,
                "input_mode": "rich11",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 49,
            },
            "h10_l3_base6_poly7_fir39_792": {
                "name": "H10_L3_base6_poly7_FIR39_792",
                "hidden": 10,
                "input_mode": "base6",
                "dilations": [1, 2, 4],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5, 7],
                "output_fir_taps": 39,
            },
            "h10_l4_rich9_rank3_fir33_784": {
                "name": "H10_L4_rich9_rank3_FIR33_784",
                "hidden": 10,
                "input_mode": "rich9",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "pointwise_rank": 3,
                "output_fir_taps": 33,
            },
            "h9_l4_rich9_fir25_796": {
                "name": "H9_L4_rich9_FIR25_796",
                "hidden": 9,
                "input_mode": "rich9",
                "dilations": [1, 2, 4, 8],
                "attention_channels": 3,
                "attention_mode": "amp37",
                "poly_orders": [1, 3, 5],
                "output_fir_taps": 25,
            },
        }
    )


def pop_variant(argv):
    variant = "h10_l3_rich9_fir33"
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


def pop_partial_init(argv):
    init_checkpoint = None
    cleaned = [argv[0]]
    idx = 1
    while idx < len(argv):
        item = argv[idx]
        if item == "--partial-init-checkpoint":
            if idx + 1 >= len(argv):
                raise ValueError("--partial-init-checkpoint requires a checkpoint path")
            init_checkpoint = argv[idx + 1]
            idx += 2
            continue
        if item.startswith("--partial-init-checkpoint="):
            init_checkpoint = item.split("=", 1)[1]
            idx += 1
            continue
        cleaned.append(item)
        idx += 1
    return init_checkpoint, cleaned


def has_arg(argv, name):
    return name in argv or any(item.startswith(name + "=") for item in argv)


def append_default(argv, name, value):
    if not has_arg(argv, name):
        argv.extend([name, str(value)])


def main():
    variant, argv = pop_variant(sys.argv)
    partial_init_checkpoint, argv = pop_partial_init(argv)
    base = load_base_module()
    register_variants(base)

    ATTEMPT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base.CHECKPOINT_DIR = CHECKPOINT_DIR
    base.FIGURE_DIR = FIGURE_DIR

    if partial_init_checkpoint:
        aidpd = load_aidpd_module()
        init_path = Path(partial_init_checkpoint)
        if not init_path.is_absolute():
            init_path = PROJECT_ROOT / init_path
        original_build_model = base.build_model

        def build_model_with_partial_init(config, g_real, g_imag):
            model = original_build_model(config, g_real, g_imag)
            if config.get("name") != base.TEACHER_CONFIG["name"]:
                loaded, partial, skipped = aidpd.load_partial_state(model, init_path)
                print(f">>> Partial init from: {init_path}")
                print(
                    f">>> Partial init tensors: exact={len(loaded)}, "
                    f"partial={len(partial)}, skipped={len(skipped)}"
                )
            return model

        base.build_model = build_model_with_partial_init

    meta = VARIANTS[variant]
    suffix = "_partial" if partial_init_checkpoint else ""
    checkpoint = CHECKPOINT_DIR / meta["checkpoint"].replace(".pth", f"{suffix}.pth")
    output_prefix = meta["prefix"] + suffix

    defaults = {
        "--student-preset": meta["preset"],
        "--max-student-multipliers": 800,
        "--teacher-mode": "load",
        "--truth-loss-weight": 1.0,
        "--distill-loss-weight": 0.04,
        "--distill-diff-weight": 0.02,
        "--diff-loss-weight": 0.005,
        "--spectral-truth-weight": 2e-5,
        "--spectral-main-weight": 1.0,
        "--spectral-adjacent-weight": 4.0,
        "--spectral-floor-weight": 0.04,
        "--fir-fit-domain": "freq",
        "--fir-main-weight": 1.0,
        "--fir-adjacent-weight": 1.1,
        "--fir-floor-weight": 0.03,
        "--fir-ridge": 1e-7,
        "--stage1-epochs": 80,
        "--stage2-epochs": 40,
        "--stage3-epochs": 40,
        "--stage1-lr": 8e-4,
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
        "--output-prefix": output_prefix,
    }
    for key, value in defaults.items():
        append_default(argv, key, value)

    print(f">>> step800 NMSE-search variant: {variant}")
    print(f">>> preset: {meta['preset']}")
    print(f">>> checkpoint: {checkpoint}")
    print(f">>> artifacts: {ATTEMPT_ROOT}")
    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()
