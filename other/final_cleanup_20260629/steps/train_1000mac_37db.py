import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
STUDENT_CHECKPOINT = CHECKPOINT_DIR / "student_tcn1000_teacher37_best.pth"
TEACHER_CHECKPOINT = CHECKPOINT_DIR / "teacher32_6layer_scratch_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026

# This is exactly 1000 real multipliers under the project counting rule.
# It is a compressed version of other/step40.1.7.py:
# - keep base6 input, kernel-5 depthwise TCN, six dilations up to 32
# - reduce hidden from 32 to 16
# - replace full pointwise 16x16 by rank-1 pointwise factorization
# - add an 11-tap output complex FIR fitted only on the training split
#
# 3 complex polynomial branches = 12
# base6 input projection: 6 * 16 = 96
# 6 TCN blocks: 6 * (16 * 5 depthwise + 2 * 16 * 1 low-rank pointwise) = 672
# output projection: 2 * 16 = 32
# attention: 16 * 3 * 3 = 144
# output complex FIR: 4 * 11 = 44
# total = 12 + 96 + 672 + 32 + 144 + 44 = 1000
STUDENT_CONFIG = {
    "preset": "student_tcn1000_rank1_teacher37",
    "hidden": 16,
    "input_mode": "base6",
    "dilations": [1, 2, 4, 8, 16, 32],
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
    "pointwise_rank": 1,
    "output_fir_taps": 11,
}

TEACHER_CONFIG = {
    "preset": "teacher32_6layer",
    "hidden": 32,
    "input_mode": "base6",
    "dilations": [1, 2, 4, 8, 16, 32],
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
}


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


def load_data():
    data = scipy.io.loadmat(DATA_PATH)
    raw_x = data["x"].flatten()
    raw_y = data["y"].flatten()
    x = raw_x / np.sqrt(np.mean(np.abs(raw_x) ** 2))
    y = raw_y / np.sqrt(np.mean(np.abs(raw_y) ** 2))
    x_np = np.stack([x.real, x.imag], axis=1)
    y_np = np.stack([y.real, y.imag], axis=1)
    split_idx = int(min(len(x_np), len(y_np)) * 0.8)
    return x, y, x_np[:split_idx], y_np[:split_idx], x_np[split_idx:], y_np[split_idx:], split_idx


def train_gain_from_train_split(x_complex, y_complex, split_idx):
    x_train = x_complex[:split_idx]
    y_train = y_complex[:split_idx]
    gain = np.vdot(x_train, y_train) / np.vdot(x_train, x_train)
    return float(gain.real), float(gain.imag)


def fullseq_nmse_db(model, x_val, y_val):
    nmse, nmse_db = aidpd.evaluate_full_sequence_nmse(model, x_val, y_val)
    return nmse, nmse_db


def load_teacher(g_real, g_imag):
    if not TEACHER_CHECKPOINT.exists():
        print(f">>> Teacher not found, training without distillation: {TEACHER_CHECKPOINT}")
        return None
    teacher = aidpd.DiagramDPDNet(g_real, g_imag, config=TEACHER_CONFIG).to(DEVICE)
    teacher.load_state_dict(aidpd.load_state_dict(TEACHER_CHECKPOINT), strict=True)
    teacher.eval()
    print(f">>> Loaded teacher: {TEACHER_CHECKPOINT}")
    return teacher


def initialize_student_from_teacher(student):
    if not TEACHER_CHECKPOINT.exists():
        print(f">>> Teacher init skipped, checkpoint not found: {TEACHER_CHECKPOINT}")
        return

    source = aidpd.load_state_dict(TEACHER_CHECKPOINT)
    target = student.state_dict()
    copied = 0
    compressed = 0

    direct_prefix = (
        "g_real",
        "g_imag",
        "g3_real",
        "g3_imag",
        "g5_real",
        "g5_imag",
        "in_conv.weight",
        "attn.env_conv.weight",
        "out_conv.weight",
    )

    for key in direct_prefix:
        if key not in source or key not in target:
            continue
        src = source[key]
        dst = target[key]
        merged = dst.clone()
        if src.ndim == 0:
            merged.copy_(src.to(dtype=dst.dtype))
        else:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, dst.shape))
            merged[slices] = src[slices].to(dtype=dst.dtype)
        target[key] = merged
        copied += 1

    for block_idx in range(len(STUDENT_CONFIG["dilations"])):
        depth_key = f"tcn_blocks.{block_idx}.0.weight"
        teacher_pw_key = f"tcn_blocks.{block_idx}.1.weight"
        student_pw_in_key = f"tcn_blocks.{block_idx}.1.weight"
        student_pw_out_key = f"tcn_blocks.{block_idx}.2.weight"

        if depth_key in source and depth_key in target:
            src = source[depth_key]
            dst = target[depth_key]
            merged = dst.clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, dst.shape))
            merged[slices] = src[slices].to(dtype=dst.dtype)
            target[depth_key] = merged
            copied += 1

        if teacher_pw_key not in source or student_pw_in_key not in target or student_pw_out_key not in target:
            continue

        teacher_pw = source[teacher_pw_key].squeeze(-1).to(dtype=torch.float64)
        hidden = target[student_pw_out_key].shape[0]
        rank = target[student_pw_in_key].shape[0]
        teacher_pw = teacher_pw[:hidden, :hidden]
        u, s, vh = torch.linalg.svd(teacher_pw, full_matrices=False)
        s_root = torch.sqrt(torch.clamp(s[:rank], min=0.0))
        pw_in = (s_root[:, None] * vh[:rank, :]).unsqueeze(-1)
        pw_out = (u[:, :rank] * s_root[None, :]).unsqueeze(-1)
        target[student_pw_in_key] = pw_in.to(dtype=target[student_pw_in_key].dtype)
        target[student_pw_out_key] = pw_out.to(dtype=target[student_pw_out_key].dtype)
        compressed += 1

    student.load_state_dict(target, strict=True)
    print(f">>> Compressed teacher init: copied={copied}, svd_pointwise={compressed}")


def make_loader(x, y, frame_length, stride_length, batch_size, shuffle=True):
    dataset = aidpd.FramedDataset(x, y, frame_length, stride_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_stage(
    model,
    teacher,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    epochs,
    lr,
    frame_length,
    stride_length,
    batch_size,
    kd_weight,
    target_db,
    checkpoint_path,
    stage_name,
):
    loader = make_loader(x_train, y_train, frame_length, stride_length, batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_nmse, best_db = fullseq_nmse_db(model, x_val, y_val)
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]
    start = time.time()
    print(f"\n>>> Stage {stage_name} | epochs={epochs} lr={lr:.2e} frame={frame_length} batch={batch_size}")
    print(f">>> Initial full NMSE: {best_db:.2f} dB")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            if teacher is not None and kd_weight > 0:
                with torch.no_grad():
                    teacher_pred = teacher(batch_x)
                loss = loss + kd_weight * criterion(pred, teacher_pred)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        nmse, nmse_db = fullseq_nmse_db(model, x_val, y_val)
        scheduler.step(nmse_db)
        history.append((epoch, nmse_db))
        elapsed = time.time() - start
        print(
            f"{stage_name} Epoch [{epoch}/{epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | Full NMSE {nmse_db:.2f} dB"
        )
        if nmse < best_nmse:
            best_nmse = nmse
            best_db = nmse_db
            torch.save(model.state_dict(), checkpoint_path)
        if best_db <= target_db:
            print(f">>> Target reached in stage {stage_name}: {best_db:.2f} dB")
            break

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    return history, best_nmse, best_db


def plot_history(all_history, out_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    cursor = 0
    for stage_name, history in all_history:
        xs = [cursor + item[0] for item in history]
        ys = [item[1] for item in history]
        plt.plot(xs, ys, label=stage_name)
        cursor = xs[-1]
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.2, label="-37 dB target")
    plt.xlabel("Epoch")
    plt.ylabel("Full-sequence NMSE (dB)")
    plt.title("1000-multiplier student training")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    x_complex, y_complex, x_train, y_train, x_val, y_val, split_idx = load_data()
    g_real, g_imag = train_gain_from_train_split(x_complex, y_complex, split_idx)

    multipliers = aidpd.real_multiplier_count_from_config(STUDENT_CONFIG)
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Student config: {STUDENT_CONFIG['preset']}")
    print(f">>> Real multipliers: {multipliers}")
    if multipliers > 1000:
        raise RuntimeError(f"Student exceeds 1000 multipliers: {multipliers}")

    student = aidpd.DiagramDPDNet(g_real, g_imag, config=STUDENT_CONFIG).to(DEVICE)
    initialize_student_from_teacher(student)
    teacher = load_teacher(g_real, g_imag)

    all_history = []

    h1, _, db1 = train_stage(
        student,
        teacher,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=260,
        lr=2e-3,
        frame_length=800,
        stride_length=32,
        batch_size=64,
        kd_weight=0.05,
        target_db=-37.0,
        checkpoint_path=STUDENT_CHECKPOINT,
        stage_name="compressed-teacher-distill",
    )
    all_history.append(("compressed-teacher-distill", h1))

    if db1 > -37.0:
        print("\n>>> Fitting output FIR by LS before long-frame fine tune")
        aidpd.fit_output_fir_from_train(student, x_train, y_train)
        torch.save(student.state_dict(), STUDENT_CHECKPOINT)

        # Keep FIR fixed in the long-frame phase. This preserves the frequency correction while the base adapts.
        for name, param in student.named_parameters():
            if name.startswith("fir_"):
                param.requires_grad = False

        h2, _, db2 = train_stage(
            student,
            teacher,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=180,
            lr=1e-5,
            frame_length=4096,
            stride_length=256,
            batch_size=4,
            kd_weight=0.02,
            target_db=-37.0,
            checkpoint_path=STUDENT_CHECKPOINT,
            stage_name="long-frame",
        )
        all_history.append(("long-frame", h2))
    else:
        db2 = db1

    student.load_state_dict(aidpd.load_state_dict(STUDENT_CHECKPOINT), strict=True)
    print("\n>>> Final FIR-LS refit")
    aidpd.fit_output_fir_from_train(student, x_train, y_train)
    torch.save(student.state_dict(), STUDENT_CHECKPOINT)
    final_nmse, final_db = fullseq_nmse_db(student, x_val, y_val)

    plot_history(all_history, FIGURE_DIR / "student_tcn1000_teacher37_convergence.png")
    print(f"\n>>> Final full-sequence NMSE: {final_nmse:.6e} ({final_db:.2f} dB)")
    print(f">>> Saved checkpoint: {STUDENT_CHECKPOINT}")

    aidpd.evaluate_aclr_visualization(
        student,
        x_val,
        y_val,
        output_prefix="student_tcn1000_teacher37",
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
