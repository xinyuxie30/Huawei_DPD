import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BOFIR_PATH = SCRIPT_DIR / "step40.1.15_bofir_eatcn1000.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_VARIANT = "p3m5_h18_l4_fir57_amp7amp9thermal16thermal64_2388"
DEFAULT_INIT = CHECKPOINT_DIR / "probe_distill_h18_thermal64_2388_crop256_frame4096_e160_dw015.pth"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "probe_bofir_h18_thermal64_2388_fullseq_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bofir = load_module("bofir", BOFIR_PATH)
aidpd = bofir.aidpd


def parse_args():
    parser = argparse.ArgumentParser(description="Full-sequence fine-tuning for <=2400-MAC BO-FIR EA-TCN.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--variant", choices=sorted(bofir.VARIANT_CONFIGS), default=DEFAULT_VARIANT)
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--scheduler-patience", type=int, default=16)
    parser.add_argument("--scheduler-min-lr", type=float, default=5e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--crop-margin", type=int, default=512)
    parser.add_argument("--diff-loss-weight", type=float, default=0.0)
    parser.add_argument("--freeze-fir", action="store_true")
    parser.add_argument("--freeze-physical", action="store_true")
    parser.add_argument("--max-multipliers", type=int, default=2400)
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


def tensor_sequence(x_np):
    return torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)


def cropped_loss(pred, target, crop_margin, diff_loss_weight):
    if crop_margin > 0 and pred.shape[-1] > 2 * crop_margin:
        pred = pred[:, :, crop_margin:-crop_margin]
        target = target[:, :, crop_margin:-crop_margin]
    loss = nn.functional.mse_loss(pred, target)
    if diff_loss_weight > 0:
        pred_d = pred[:, :, 1:] - pred[:, :, :-1]
        target_d = target[:, :, 1:] - target[:, :, :-1]
        loss = loss + diff_loss_weight * nn.functional.mse_loss(pred_d, target_d)
    return loss


def set_trainable(model, args):
    if args.freeze_fir:
        for name, param in model.named_parameters():
            if name.startswith("fir_"):
                param.requires_grad = False
    if args.freeze_physical:
        physical = ("g_real", "g_imag", "g3_real", "g3_imag", "g5_real", "g5_imag")
        for name, param in model.named_parameters():
            if name in physical:
                param.requires_grad = False


def plot_history(history, out_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    xs = [item[0] for item in history]
    ys = [item[1] for item in history]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, color="tab:blue", linewidth=1.5, label="Validation full-sequence NMSE")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.title("BO-FIR EA-TCN full-sequence fine-tune")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = dict(bofir.VARIANT_CONFIGS[args.variant])
    multipliers = bofir.real_multiplier_count(config)
    if multipliers > args.max_multipliers:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers} > {args.max_multipliers}")

    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(resolve_path(args.data))
    init_checkpoint = resolve_path(args.init_checkpoint)
    checkpoint_path = resolve_path(args.checkpoint)
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")

    model = bofir.BOFIREATCN(g_real, g_imag, config=config).to(DEVICE)
    try:
        model.load_state_dict(aidpd.load_state_dict(init_checkpoint), strict=True)
        print(f">>> Strict init from: {init_checkpoint}")
    except RuntimeError:
        loaded, cropped, skipped = bofir.load_from_eatcn(model, init_checkpoint)
        print(f">>> Partial init from: {init_checkpoint}")
        print(f">>> Loaded={len(loaded)}, cropped={len(cropped)}, skipped={len(skipped)}")

    set_trainable(model, args)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters.")

    x_train_t = tensor_sequence(x_train)
    y_train_t = tensor_sequence(y_train)
    best_nmse, best_db = bofir.fullseq_nmse(model, x_val, y_val)
    best_pass_nmse = float("inf")
    initial_aclr = bofir.fullseq_aclr_worst(model, x_val, args.fs_mhz, tuple(args.main_band), tuple(args.lower_band), tuple(args.upper_band))
    if initial_aclr <= args.aclr_target_dbc:
        best_pass_nmse = best_nmse
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Variant: {args.variant} | multipliers={multipliers}")
    print(f">>> Initial NMSE: {best_nmse:.6e} ({best_db:.3f} dB), ACLR {initial_aclr:.2f} dBc")
    print(f">>> Fullseq fine-tune: epochs={args.epochs}, lr={args.lr:.2e}, crop={args.crop_margin}")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    start = time.time()
    saved_nmse, saved_db = best_nmse, best_db
    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(x_train_t)
        loss = cropped_loss(pred, y_train_t, args.crop_margin, args.diff_loss_weight)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, max_norm=0.2)
        optimizer.step()

        nmse, db = bofir.fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        history.append((epoch, db))
        elapsed = time.time() - start
        print(
            f"Epoch [{epoch}/{args.epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | Full NMSE {db:.3f} dB"
        )

        if nmse < best_nmse:
            best_nmse, best_db = nmse, db
            candidate_aclr = bofir.fullseq_aclr_worst(
                model,
                x_val,
                args.fs_mhz,
                tuple(args.main_band),
                tuple(args.lower_band),
                tuple(args.upper_band),
            )
            print(f">>> Candidate ACLR worst: {candidate_aclr:.2f} dBc")
            if candidate_aclr <= args.aclr_target_dbc and nmse < best_pass_nmse:
                best_pass_nmse = nmse
                saved_nmse, saved_db = nmse, db
                torch.save(model.state_dict(), checkpoint_path)
            if saved_db <= args.target_fullseq_db:
                print(f">>> Target reached: {saved_db:.2f} dB")
                break

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    final_nmse, final_db = bofir.fullseq_nmse(model, x_val, y_val)
    convergence_path = FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png"
    plot_history(history, convergence_path)

    print(f"\n>>> Final saved NMSE: {final_nmse:.6e} ({final_db:.3f} dB)")
    print(f">>> Saved checkpoint: {checkpoint_path}")
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
    main()
