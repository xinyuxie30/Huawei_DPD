import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
DISTILL_SCRIPT = SCRIPT_DIR / "TCN_800_distill.py"

DEFAULT_TEACHER = (
    CHECKPOINT_DIR
    / "BEST_PASS_RECOMMENDED_rich9_h15_l6_fir43_2284_NMSE40p15_ACLR48p77.pth"
)


def run_stage(stage_name, extra_args):
    cmd = [sys.executable, "-B", str(DISTILL_SCRIPT), *extra_args]
    print("\n" + "=" * 88)
    print(f">>> Running {stage_name}")
    print(">>> " + " ".join(f'"{item}"' if " " in item else item for item in cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"{stage_name} failed with exit code {completed.returncode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end reproducible <=800-MAC student-from-scratch distillation recipe."
    )
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER))
    parser.add_argument("--variant", default="h10_base6_l3_fir41")
    parser.add_argument("--stage1-epochs", type=int, default=120)
    parser.add_argument("--stage2-epochs", type=int, default=80)
    parser.add_argument("--stage3-epochs", type=int, default=80)
    parser.add_argument("--stage1-lr", type=float, default=5e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)
    parser.add_argument("--distill-loss-weight", type=float, default=0.06)
    parser.add_argument("--distill-diff-weight", type=float, default=0.03)
    parser.add_argument("--final-checkpoint", default=str(CHECKPOINT_DIR / "TCN_800_FINAL_H10_L3_FIR41_796_scratch_distill.pth"))
    return parser.parse_args()


def common_args(args):
    return [
        "--variant",
        args.variant,
        "--teacher-checkpoint",
        args.teacher_checkpoint,
        "--distill-loss-weight",
        str(args.distill_loss_weight),
        "--distill-diff-weight",
        str(args.distill_diff_weight),
        "--truth-loss-weight",
        "1.0",
        "--fir-fit-domain",
        "freq",
        "--fir-adjacent-weight",
        "1.2",
    ]


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    stage1_ckpt = CHECKPOINT_DIR / "TCN_800_final_stage1_scratch_e120.pth"
    stage2_ckpt = CHECKPOINT_DIR / "TCN_800_final_stage2_scratch_e200.pth"
    final_ckpt = Path(args.final_checkpoint)
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)

    run_stage(
        "stage 1: random student init",
        [
            *common_args(args),
            "--student-init-mode",
            "none",
            "--student-init-checkpoint",
            "none",
            "--epochs",
            str(args.stage1_epochs),
            "--lr",
            str(args.stage1_lr),
            "--checkpoint",
            str(stage1_ckpt),
        ],
    )
    run_stage(
        "stage 2: low-LR refinement",
        [
            *common_args(args),
            "--student-init-mode",
            "full",
            "--student-init-checkpoint",
            str(stage1_ckpt),
            "--epochs",
            str(args.stage2_epochs),
            "--lr",
            str(args.stage2_lr),
            "--checkpoint",
            str(stage2_ckpt),
        ],
    )
    run_stage(
        "stage 3: final low-LR refinement",
        [
            *common_args(args),
            "--student-init-mode",
            "full",
            "--student-init-checkpoint",
            str(stage2_ckpt),
            "--epochs",
            str(args.stage3_epochs),
            "--lr",
            str(args.stage3_lr),
            "--checkpoint",
            str(final_ckpt),
        ],
    )
    print("\n>>> Finished 800-MAC final recipe.")
    print(f">>> Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
