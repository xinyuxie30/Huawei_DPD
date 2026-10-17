"""
Final 786-real-multiplier scratch-only EA-TCN training/evaluation launcher.

This wrapper keeps the released low-rank <=800 configuration reproducible
without requiring users to remember the long argument list. It delegates model
definition, training, FIR refit, and figure generation to
final_train_lightweight_999_distill.py.

Default model:
    H10/L4/Input-6/pointwise-rank3/FIR41
    12 + 60 + 4 * (50 + 60) + 20 + 90 + 164 = 786 real multipliers
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "final_train_lightweight_999_distill.py"

CHECKPOINT = PROJECT_ROOT / "checkpoints" / "AI-DPD" / "step800_scratch_h10_l4_base6_rank3_fir41_786.pth"
OUTPUT_PREFIX = "step800_scratch_h10_l4_base6_rank3_fir41_786"
PRESET = "h10_l4_base6_rank3_fir41_786"


def load_base_module():
    spec = importlib.util.spec_from_file_location("released_lightweight_786_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["released_lightweight_786_base"] = module
    spec.loader.exec_module(module)
    return module


def has_arg(argv, name):
    return name in argv or any(item.startswith(name + "=") for item in argv)


def append_default(argv, name, value):
    if not has_arg(argv, name):
        argv.extend([name, str(value)])


def main():
    base = load_base_module()
    argv = [sys.argv[0], *sys.argv[1:]]

    defaults = {
        "--student-preset": PRESET,
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
        "--final-checkpoint": CHECKPOINT,
        "--output-prefix": OUTPUT_PREFIX,
    }
    for key, value in defaults.items():
        append_default(argv, key, value)

    print(">>> final 786-MAC scratch-only EA-TCN")
    print(f">>> preset: {PRESET}")
    print(f">>> checkpoint: {CHECKPOINT}")
    print(">>> training mode: scratch-only, teacher-mode none, no distillation")
    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()
