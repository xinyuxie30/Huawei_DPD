"""Archived launcher for the ACLR50 checkpoint.

This file preserves the exact entry point needed to evaluate:
checkpoints/AI-DPD/SAVED_ACLR50_H18_FIR43_ATTN4THERMAL_2386_NMSE39p45_ACLR50p35.pth

It delegates to step40.1.15_bofir_eatcn1000.py, which contains the model
definition and shared evaluation code. Keep this file next to AI-DPD.py in the
steps directory so relative imports remain valid.
"""

import runpy
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "step40.1.15_bofir_eatcn1000.py"
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "AI-DPD"
    / "SAVED_ACLR50_H18_FIR43_ATTN4THERMAL_2386_NMSE39p45_ACLR50p35.pth"
)
VARIANT = "p3m5_h18_l4_fir43_attn4thermal_amp7amp9thermal16thermal64_2386"


def main():
    if len(sys.argv) == 1:
        sys.argv = [
            str(BASE_SCRIPT),
            "--variant",
            VARIANT,
            "--max-multipliers",
            "2400",
            "--checkpoint",
            str(CHECKPOINT),
            "--evaluate-only",
        ]
    else:
        sys.argv[0] = str(BASE_SCRIPT)
    runpy.run_path(str(BASE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
