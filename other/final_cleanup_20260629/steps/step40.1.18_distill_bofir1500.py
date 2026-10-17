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
BOFIR_PATH = SCRIPT_DIR / "step40.1.15_bofir_eatcn1000.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"

DEFAULT_VARIANT = "p3m5_h13_fir47_1456"
DEFAULT_STUDENT = CHECKPOINT_DIR / "probe_bofir_p3m5_h13_fir47_1456_train180_freezefir.pth"
DEFAULT_TEACHER = CHECKPOINT_DIR / "teacher32_6layer_scratch_best.pth"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "BO-FIR_P3M5_H13_FIR47_1456_distilled_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_module("aidpd", AI_DPD_PATH)
bofir = load_module("bofir", BOFIR_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description="Distill calibrated teacher blend into a <=1500-MAC BO-FIR EA-TCN.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--variant", choices=sorted(bofir.VARIANT_CONFIGS), default=DEFAULT_VARIANT)
    parser.add_argument("--student-checkpoint", default=str(DEFAULT_STUDENT))
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--max-multipliers", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--blend-fir-taps", type=int, default=63)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scheduler-patience", type=int, default=16)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--truth-loss-weight", type=float, default=0.75)
    parser.add_argument("--distill-loss-weight", type=float, default=0.25)
    parser.add_argument("--diff-loss-weight", type=float, default=0.0)
    parser.add_argument("--spectral-truth-weight", type=float, default=0.0)
    parser.add_argument("--spectral-distill-weight", type=float, default=0.0)
    parser.add_argument("--spectral-main-weight", type=float, default=0.25)
    parser.add_argument("--spectral-adjacent-weight", type=float, default=1.0)
    parser.add_argument("--spectral-floor-weight", type=float, default=0.05)
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--freeze-fir", action="store_true")
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
        raise ValueError("FIR taps must be odd.")
    radius = taps // 2
    cols = []
    for tap in range(taps):
        shift = tap - radius
        if shift < 0:
            cols.append(np.r_[signal[-shift:], np.repeat(signal[-1], -shift)])
        elif shift > 0:
            cols.append(np.r_[np.repeat(signal[0], shift), signal[:-shift]])
        else:
            cols.append(signal)
    return np.stack(cols, axis=1)


def fit_complex_fir(pred_train, target_train, taps):
    return np.linalg.lstsq(shifted_columns(pred_train, taps), target_train, rcond=None)[0]


def apply_complex_fir(pred, coeffs):
    return shifted_columns(pred, len(coeffs)) @ coeffs


def build_model(config, checkpoint_path, g_real, g_imag):
    model = bofir.BOFIREATCN(g_real, g_imag, config=config).to(DEVICE)
    try:
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    except RuntimeError:
        loaded, cropped, skipped = bofir.load_from_eatcn(model, checkpoint_path)
        print(f">>> Partial BO-FIR load: exact={len(loaded)}, cropped={cropped}, skipped={len(skipped)}")
    return model


def build_blend_targets(student_ref, teacher, x_train, y_train, x_val, y_val, alpha, fir_taps):
    student_train = run_model(student_ref, x_train)
    teacher_train = run_model(teacher, x_train)
    blend_train = (1.0 - alpha) * student_train + alpha * teacher_train
    coeffs = fit_complex_fir(iq_to_complex(blend_train), iq_to_complex(y_train), fir_taps)
    d_train = complex_to_iq(apply_complex_fir(iq_to_complex(blend_train), coeffs))

    student_val = run_model(student_ref, x_val)
    teacher_val = run_model(teacher, x_val)
    blend_val = (1.0 - alpha) * student_val + alpha * teacher_val
    d_val = complex_to_iq(apply_complex_fir(iq_to_complex(blend_val), coeffs))

    d_nmse = np.sum((d_val - y_val) ** 2) / np.sum(y_val ** 2)
    freq_hz, psd = aidpd.psd_db(iq_to_complex(d_val), fs_hz=491.52e6)
    d_aclr = aidpd.aclr_from_psd(freq_hz, psd, (-100.0, 100.0), (-245.0, -180.0), (180.0, 245.0))[2]
    return d_train, d_val, coeffs, float(d_nmse), float(10 * np.log10(d_nmse)), float(d_aclr)


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
        truth_d = truth[:, :, 1:] - truth[:, :, :-1]
        loss = loss + args.diff_loss_weight * nn.functional.mse_loss(pred_d, truth_d)
    if args.spectral_truth_weight > 0:
        loss = loss + args.spectral_truth_weight * bofir.weighted_spectral_nmse(
            pred,
            truth,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            main_weight=args.spectral_main_weight,
            adjacent_weight=args.spectral_adjacent_weight,
            floor_weight=args.spectral_floor_weight,
        )
    if args.spectral_distill_weight > 0:
        loss = loss + args.spectral_distill_weight * bofir.weighted_spectral_nmse(
            pred,
            distill,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            main_weight=args.spectral_main_weight,
            adjacent_weight=args.spectral_adjacent_weight,
            floor_weight=args.spectral_floor_weight,
        )
    return loss


def fullseq_nmse(model, x_np, y_np):
    nmse, db = aidpd.evaluate_full_sequence_nmse(model, x_np, y_np)
    return float(nmse), float(db)


def fullseq_aclr(model, x_np, args):
    pred = aidpd.run_sequence(model, x_np)
    freq_hz, psd = aidpd.psd_db(aidpd.tensor_to_complex(pred), fs_hz=args.fs_mhz * 1e6)
    return float(aidpd.aclr_from_psd(freq_hz, psd, tuple(args.main_band), tuple(args.lower_band), tuple(args.upper_band))[2])


def set_fir_trainable(model, trainable):
    for name, param in model.named_parameters():
        if name.startswith("fir_"):
            param.requires_grad = trainable


def plot_history(history, path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot([x for x, _ in history], [y for _, y in history], label="Validation NMSE")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("Full-sequence NMSE (dB)")
    plt.title("BO-FIR 1500-MAC distillation")
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
    config = dict(bofir.VARIANT_CONFIGS[args.variant])
    multipliers = bofir.real_multiplier_count(config)
    if multipliers > args.max_multipliers:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers} > {args.max_multipliers}")

    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)
    student_ref = build_model(config, student_path, g_real, g_imag)
    teacher = aidpd.DiagramDPDNet(g_real, g_imag, config=aidpd.preset_config("teacher32_6layer")).to(DEVICE)
    teacher.load_state_dict(aidpd.load_state_dict(teacher_path), strict=True)

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Variant: {args.variant} | multipliers={multipliers}")
    print(f">>> Student init: {student_path}")
    print(f">>> Teacher: {teacher_path}")

    d_train, d_val, coeffs, d_nmse, d_db, d_aclr = build_blend_targets(
        student_ref, teacher, x_train, y_train, x_val, y_val, args.alpha, args.blend_fir_taps
    )
    print(f">>> Distill target: NMSE {d_nmse:.6e} ({d_db:.2f} dB), ACLR {d_aclr:.2f} dBc")
    print(f">>> Blend alpha={args.alpha:.3f}, FIR taps={len(coeffs)}")

    model = build_model(config, student_path, g_real, g_imag)
    if args.freeze_fir:
        set_fir_trainable(model, False)
        print(">>> Output FIR frozen.")

    best_nmse, best_db = fullseq_nmse(model, x_val, y_val)
    best_aclr = fullseq_aclr(model, x_val, args)
    best_pass_nmse = best_nmse if best_aclr <= args.aclr_target_dbc else float("inf")
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]

    loader = make_loader(x_train, y_train, d_train, args.frame_length, args.stride_length, args.batch_size)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    start = time.time()
    print(f">>> Initial student: NMSE {best_db:.2f} dB, ACLR {best_aclr:.2f} dBc")

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

        nmse, db = fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        history.append((epoch, db))
        elapsed = time.time() - start
        print(
            f"Epoch [{epoch}/{args.epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | Full NMSE {db:.3f} dB"
        )
        if nmse < best_nmse:
            candidate_aclr = fullseq_aclr(model, x_val, args)
            print(f">>> Candidate ACLR worst: {candidate_aclr:.2f} dBc")
            best_nmse = nmse
            best_db = db
            if candidate_aclr <= args.aclr_target_dbc and nmse < best_pass_nmse:
                best_pass_nmse = nmse
                torch.save(model.state_dict(), checkpoint_path)
            elif best_pass_nmse == float("inf"):
                torch.save(model.state_dict(), checkpoint_path)
            else:
                print(">>> Candidate improves NMSE but fails ACLR target; checkpoint kept unchanged.")
            if best_db <= args.target_fullseq_db and candidate_aclr <= args.aclr_target_dbc:
                print(f">>> Target reached: {best_db:.2f} dB, ACLR {candidate_aclr:.2f} dBc")
                break

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    final_nmse, final_db = fullseq_nmse(model, x_val, y_val)
    final_aclr = fullseq_aclr(model, x_val, args)
    print(f"\n>>> Final saved: NMSE {final_nmse:.6e} ({final_db:.2f} dB), ACLR {final_aclr:.2f} dBc")
    print(f">>> Saved checkpoint: {checkpoint_path}")
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
