import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_STUDENT = CHECKPOINT_DIR / "EA-TCN_memory_fir1000_best.pth"
DEFAULT_TEACHER = CHECKPOINT_DIR / "teacher32_6layer_scratch_best.pth"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "EA-TCN_blend_distilled1000_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


STUDENT_CONFIG = {
    "preset": "blend_distilled1000",
    "hidden": 10,
    "input_mode": "rich9",
    "dilations": [1, 2, 4, 8],
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
    "output_fir_taps": 47,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Distill teacher-student FIR blend into the 1000-MAC EA-TCN student.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--student-checkpoint", default=str(DEFAULT_STUDENT))
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--blend-fir-taps", type=int, default=63)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--truth-loss-weight", type=float, default=0.35)
    parser.add_argument("--distill-loss-weight", type=float, default=1.0)
    parser.add_argument("--diff-loss-weight", type=float, default=0.0)
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--fit-output-fir-every", type=int, default=50)
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
    return (
        x_np[:split_idx],
        y_np[:split_idx],
        x_np[split_idx:],
        y_np[split_idx:],
        float(g_opt.real),
        float(g_opt.imag),
    )


def run_model(model, x_np):
    model.eval()
    x_tensor = torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model(x_tensor).squeeze(0).cpu().numpy().T


def iq_to_complex(iq):
    return iq[:, 0] + 1j * iq[:, 1]


def complex_to_iq(signal):
    return np.stack([signal.real, signal.imag], axis=1)


def shifted_columns(signal, taps):
    if taps % 2 == 0:
        raise ValueError("Centered FIR calibration requires odd taps.")
    pad = taps // 2
    columns = []
    for tap in range(taps):
        shift = tap - pad
        if shift < 0:
            col = np.r_[signal[-shift:], np.repeat(signal[-1], -shift)]
        elif shift > 0:
            col = np.r_[np.repeat(signal[0], shift), signal[:-shift]]
        else:
            col = signal
        columns.append(col)
    return np.stack(columns, axis=1)


def fit_complex_fir(pred_train, target_train, taps):
    return np.linalg.lstsq(shifted_columns(pred_train, taps), target_train, rcond=None)[0]


def apply_complex_fir(pred, coeffs):
    return shifted_columns(pred, len(coeffs)) @ coeffs


def build_blend_targets(student_ref, teacher, x_train, y_train, x_val, alpha, fir_taps):
    student_train = run_model(student_ref, x_train)
    teacher_train = run_model(teacher, x_train)
    blend_train = (1.0 - alpha) * student_train + alpha * teacher_train
    coeffs = fit_complex_fir(iq_to_complex(blend_train), iq_to_complex(y_train), fir_taps)
    distill_train = complex_to_iq(apply_complex_fir(iq_to_complex(blend_train), coeffs))

    student_val = run_model(student_ref, x_val)
    teacher_val = run_model(teacher, x_val)
    blend_val = (1.0 - alpha) * student_val + alpha * teacher_val
    distill_val = complex_to_iq(apply_complex_fir(iq_to_complex(blend_val), coeffs))
    return distill_train, distill_val, coeffs


class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, x_np, y_np, d_np, frame_length, stride_length):
        self.x = torch.as_tensor(x_np, dtype=torch.float64)
        self.y = torch.as_tensor(y_np, dtype=torch.float64)
        self.d = torch.as_tensor(d_np, dtype=torch.float64)
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.num_frames = (self.x.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        end = start + self.frame_length
        return self.x[start:end].T, self.y[start:end].T, self.d[start:end].T


def make_loader(x_train, y_train, d_train, frame_length, stride_length, batch_size):
    return DataLoader(
        DistillDataset(x_train, y_train, d_train, frame_length, stride_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def fullseq_nmse(model, x_np, y_np):
    nmse, db = aidpd.evaluate_full_sequence_nmse(model, x_np, y_np)
    return float(nmse), float(db)


def loss_fn(pred, truth, distill, args):
    if args.crop_margin > 0 and pred.shape[-1] > 2 * args.crop_margin:
        sl = slice(args.crop_margin, -args.crop_margin)
        pred = pred[:, :, sl]
        truth = truth[:, :, sl]
        distill = distill[:, :, sl]
    loss = args.truth_loss_weight * nn.functional.mse_loss(pred, truth)
    loss = loss + args.distill_loss_weight * nn.functional.mse_loss(pred, distill)
    if args.diff_loss_weight > 0:
        pred_d = pred[:, :, 1:] - pred[:, :, :-1]
        distill_d = distill[:, :, 1:] - distill[:, :, :-1]
        loss = loss + args.diff_loss_weight * nn.functional.mse_loss(pred_d, distill_d)
    return loss


def plot_history(history, path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot([x for x, _ in history], [y for _, y in history], label="Validation NMSE")
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.0, label="-37 dB")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.title("Blend-target distillation into 1000-MAC EA-TCN")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data_path = resolve_path(args.data)
    student_path = resolve_path(args.student_checkpoint)
    teacher_path = resolve_path(args.teacher_checkpoint)
    checkpoint_path = resolve_path(args.checkpoint)
    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)

    student_macs = aidpd.real_multiplier_count_from_config(STUDENT_CONFIG)
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Student MACs: {student_macs}")
    print(f">>> Student init: {student_path}")
    print(f">>> Teacher: {teacher_path}")
    if student_macs > 1000:
        raise RuntimeError(f"Student multiplier budget exceeded: {student_macs}")

    student_ref = aidpd.DiagramDPDNet(g_real, g_imag, config=STUDENT_CONFIG).to(DEVICE)
    student_ref.load_state_dict(aidpd.load_state_dict(student_path), strict=True)
    teacher = aidpd.DiagramDPDNet(g_real, g_imag, config=aidpd.preset_config("teacher32_6layer")).to(DEVICE)
    teacher.load_state_dict(aidpd.load_state_dict(teacher_path), strict=True)

    print("\n>>> Building train-only calibrated blend targets.")
    d_train, d_val, coeffs = build_blend_targets(
        student_ref, teacher, x_train, y_train, x_val, args.alpha, args.blend_fir_taps
    )
    d_nmse = np.sum((d_val - y_val) ** 2) / np.sum(y_val ** 2)
    print(f">>> Blend target validation NMSE: {d_nmse:.6e} ({10 * np.log10(d_nmse):.2f} dB)")
    print(f">>> Blend alpha={args.alpha:.3f}, FIR taps={args.blend_fir_taps}, coeffs={len(coeffs)}")

    model = aidpd.DiagramDPDNet(g_real, g_imag, config=STUDENT_CONFIG).to(DEVICE)
    model.load_state_dict(aidpd.load_state_dict(student_path), strict=True)
    best_nmse, best_db = fullseq_nmse(model, x_val, y_val)
    initial_db = best_db
    history = [(0, best_db)]
    saved = False
    print(f">>> Initial student validation NMSE: {best_nmse:.6e} ({best_db:.2f} dB)")

    loader = make_loader(x_train, y_train, d_train, args.frame_length, args.stride_length, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-7
    )

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

        if args.fit_output_fir_every > 0 and epoch % args.fit_output_fir_every == 0:
            aidpd.fit_output_fir_from_train(model, x_train, y_train)

        nmse, db = fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        history.append((epoch, db))
        elapsed = time.time() - start
        print(
            f"Epoch [{epoch}/{args.epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | Val NMSE {db:.3f} dB"
        )

        if nmse < best_nmse:
            best_nmse = nmse
            best_db = db
            torch.save(model.state_dict(), checkpoint_path)
            saved = True
            if best_db <= args.target_fullseq_db:
                print(f">>> Target reached: {best_db:.2f} dB")
                break

    print(f"\n>>> Initial student NMSE: {initial_db:.3f} dB")
    print(f">>> Best distilled student NMSE: {best_nmse:.6e} ({best_db:.3f} dB)")
    if saved:
        print(f">>> Saved improved 1000-MAC checkpoint: {checkpoint_path}")
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    else:
        print(">>> No improved checkpoint saved.")

    convergence_path = FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png"
    plot_history(history, convergence_path)
    print(f">>> Convergence figure: {convergence_path}")

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


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
