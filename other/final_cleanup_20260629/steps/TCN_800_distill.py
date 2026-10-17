import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
TCN800_PATH = SCRIPT_DIR / "TCN_800_aclr.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"

DEFAULT_TEACHER = (
    CHECKPOINT_DIR
    / "BEST_PASS_RECOMMENDED_rich9_h15_l6_fir43_2284_NMSE40p15_ACLR48p77.pth"
)
DEFAULT_STUDENT_INIT = (
    CHECKPOINT_DIR
    / "TCN_800_h10_base6_l3_fir41_freq_continue200_adj1p2.pth"
)
DEFAULT_OUT = CHECKPOINT_DIR / "TCN_800_distill_h10_l3_fir41_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_module("aidpd", AI_DPD_PATH)
t800 = load_module("t800", TCN800_PATH)


class DistillDataset(Dataset):
    def __init__(self, x_np, y_np, d_np, frame_length, stride_length):
        self.x = torch.as_tensor(x_np, dtype=torch.float64)
        self.y = torch.as_tensor(y_np, dtype=torch.float64)
        self.d = torch.as_tensor(d_np, dtype=torch.float64)
        self.frame_length = int(frame_length)
        self.starts = list(range(0, len(self.x) - self.frame_length + 1, int(stride_length)))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.frame_length
        return self.x[start:end].T, self.y[start:end].T, self.d[start:end].T


def resolve_path(path_like):
    if path_like is None or str(path_like).lower() == "none":
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x_train, y_train, d_train, args):
    return DataLoader(
        DistillDataset(x_train, y_train, d_train, args.frame_length, args.stride_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def as_sample_iq(iq):
    arr = np.asarray(iq)
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        return arr.T
    return arr


def crop_three(pred, truth, distill, margin):
    if margin > 0 and pred.shape[-1] > 2 * margin:
        sl = slice(margin, -margin)
        return pred[:, :, sl], truth[:, :, sl], distill[:, :, sl]
    return pred, truth, distill


def mse_loss(pred, target, margin):
    if margin > 0 and pred.shape[-1] > 2 * margin:
        pred = pred[:, :, margin:-margin]
        target = target[:, :, margin:-margin]
    return nn.functional.mse_loss(pred, target)


def loss_fn(pred, truth, distill, args):
    truth_loss = aidpd.loss_with_optional_terms(
        pred,
        truth,
        crop_margin=args.crop_margin,
        diff_loss_weight=args.diff_loss_weight,
        spectral_loss_weight=args.spectral_truth_weight,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        spectral_main_weight=args.spectral_main_weight,
        spectral_adjacent_weight=args.spectral_adjacent_weight,
        spectral_floor_weight=args.spectral_floor_weight,
    )
    distill_loss = mse_loss(pred, distill, args.crop_margin)
    if args.distill_diff_weight > 0:
        pred_c, _, distill_c = crop_three(pred, truth, distill, args.crop_margin)
        distill_loss = distill_loss + args.distill_diff_weight * nn.functional.mse_loss(
            pred_c[:, :, 1:] - pred_c[:, :, :-1],
            distill_c[:, :, 1:] - distill_c[:, :, :-1],
        )
    return args.truth_loss_weight * truth_loss + args.distill_loss_weight * distill_loss


def load_teacher(g_real, g_imag, args):
    teacher_path = resolve_path(args.teacher_checkpoint)
    if teacher_path is None:
        return None
    config = aidpd.preset_config(args.teacher_preset)
    teacher = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(DEVICE)
    teacher.load_state_dict(aidpd.load_state_dict(teacher_path), strict=True)
    teacher.eval()
    return teacher


def make_student(g_real, g_imag, args):
    variant = t800.VARIANTS[args.variant]
    config = dict(variant["config"])
    model, multipliers = t800.build_model(config, g_real, g_imag)
    init_path = resolve_path(args.student_init_checkpoint)
    t800.initialize_model(model, args.student_init_mode, init_path)
    return model, multipliers


def fit_fir(model, x_train, y_train, args):
    if args.fir_fit_domain == "freq":
        return aidpd.fit_output_fir_freq_from_train(
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
    return aidpd.fit_output_fir_from_train(model, x_train, y_train)


def score(metrics, aclr_target):
    return metrics["nmse_db"] + 0.35 * max(0.0, metrics["aclr_worst"] - aclr_target)


def plot_history(history, output_path):
    plt.figure(figsize=(9, 5.5))
    plt.plot([e for e, _, _ in history], [db for _, db, _ in history], label="NMSE")
    aclr_values = [aclr for _, _, aclr in history if np.isfinite(aclr)]
    if aclr_values:
        ax = plt.gca().twinx()
        ax.plot(
            [e for e, _, aclr in history if np.isfinite(aclr)],
            aclr_values,
            color="tab:red",
            alpha=0.65,
            label="ACLR",
        )
        ax.axhline(-45.0, color="tab:red", linestyle=":", linewidth=1.0)
        ax.set_ylabel("ACLR worst (dBc)")
    plt.gca().axhline(-35.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Teacher-assisted <=800-MAC TCN DPD distillation.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--variant", choices=sorted(t800.VARIANTS), default="h10_base6_l3_fir41")
    parser.add_argument("--teacher-preset", default="rich9_h15_l6_fir43_2284")
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER))
    parser.add_argument("--student-init-mode", choices=["none", "partial", "prune994", "full"], default="full")
    parser.add_argument("--student-init-checkpoint", default=str(DEFAULT_STUDENT_INIT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=64)
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=15)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--truth-loss-weight", type=float, default=1.0)
    parser.add_argument("--distill-loss-weight", type=float, default=0.08)
    parser.add_argument("--diff-loss-weight", type=float, default=0.005)
    parser.add_argument("--distill-diff-weight", type=float, default=0.05)
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
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, g_real, g_imag = t800.load_data(resolve_path(args.data))
    teacher = load_teacher(g_real, g_imag, args)
    if teacher is None:
        d_train = y_train.copy()
        d_val = y_val.copy()
    else:
        print(">>> Building teacher targets.")
        d_train = as_sample_iq(aidpd.run_sequence(teacher, x_train))
        d_val = as_sample_iq(aidpd.run_sequence(teacher, x_val))
        teacher_nmse = np.sum((d_val - y_val) ** 2) / np.sum(y_val ** 2)
        print(f">>> Teacher target NMSE: {teacher_nmse:.6e} ({10 * np.log10(teacher_nmse):.2f} dB)")

    model, multipliers = make_student(g_real, g_imag, args)
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Student variant: {args.variant} | multipliers={multipliers}")
    print(f">>> Checkpoint: {checkpoint_path}")

    fit_fir(model, x_train, y_train, args)
    init_metrics = t800.compute_aclr_metrics(
        model,
        x_val,
        y_val,
        args.fs_mhz,
        tuple(args.main_band),
        tuple(args.lower_band),
        tuple(args.upper_band),
    )
    best_score = score(init_metrics, args.aclr_target_dbc)
    best_metrics = init_metrics
    torch.save(model.state_dict(), checkpoint_path)
    print(f">>> Initial student | NMSE {init_metrics['nmse_db']:.2f} dB | ACLR {init_metrics['aclr_worst']:.2f} dBc")

    loader = make_loader(x_train, y_train, d_train, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    history = [(0, init_metrics["nmse_db"], init_metrics["aclr_worst"])]
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y, batch_d in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            batch_d = batch_d.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y, batch_d, args)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        nmse, nmse_db = aidpd.evaluate_full_sequence_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        aclr_text = ""
        aclr_value = float("nan")

        if epoch % args.refit_every == 0:
            fit_fir(model, x_train, y_train, args)

        if epoch % args.aclr_every == 0 or epoch == args.epochs:
            metrics = t800.compute_aclr_metrics(
                model,
                x_val,
                y_val,
                args.fs_mhz,
                tuple(args.main_band),
                tuple(args.lower_band),
                tuple(args.upper_band),
            )
            aclr_value = metrics["aclr_worst"]
            current_score = score(metrics, args.aclr_target_dbc)
            if current_score < best_score:
                best_score = current_score
                best_metrics = metrics
                torch.save(model.state_dict(), checkpoint_path)
            aclr_text = f" | ACLR {aclr_value:.2f} dBc | score {current_score:.2f}"

        history.append((epoch, float(nmse_db), aclr_value))
        elapsed = time.time() - start
        print(
            f"Epoch [{epoch}/{args.epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | Full NMSE {nmse_db:.2f} dB{aclr_text}"
        )

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    fit_fir(model, x_train, y_train, args)
    torch.save(model.state_dict(), checkpoint_path)
    final_metrics = t800.compute_aclr_metrics(
        model,
        x_val,
        y_val,
        args.fs_mhz,
        tuple(args.main_band),
        tuple(args.lower_band),
        tuple(args.upper_band),
    )

    plot_history(history, FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png")
    aidpd.evaluate_aclr_visualization(
        model,
        x_val,
        y_val,
        output_prefix=checkpoint_path.stem,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        target_dbc=args.aclr_target_dbc,
    )
    print(
        f">>> FINAL distilled {args.variant}: multipliers={multipliers} | "
        f"NMSE {final_metrics['nmse_db']:.2f} dB | ACLR {final_metrics['aclr_worst']:.2f} dBc"
    )
    print(f">>> Best tracked: NMSE {best_metrics['nmse_db']:.2f} dB | ACLR {best_metrics['aclr_worst']:.2f} dBc")
    print(f">>> Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
