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
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_INIT = CHECKPOINT_DIR / "EA-TCN_memory_fir1000_best.pth"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "EA-TCN_fullseq_finetune1000_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


MODEL_CONFIG = {
    "preset": "fullseq_finetune1000",
    "hidden": 10,
    "input_mode": "rich9",
    "dilations": [1, 2, 4, 8],
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
    "output_fir_taps": 47,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-sequence fine-tuning for the 1000-MAC EA-TCN model."
    )
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--diff-loss-weight", type=float, default=0.0)
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--fit-fir-before", action="store_true")
    parser.add_argument("--freeze-fir", action="store_true")
    parser.add_argument("--freeze-physical", action="store_true")
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


def tensor_sequence(x_np):
    return torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)


def fullseq_nmse(model, x_np, y_np):
    nmse, nmse_db = aidpd.evaluate_full_sequence_nmse(model, x_np, y_np)
    return float(nmse), float(nmse_db)


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
            if name in physical or name.startswith("extra_poly_"):
                param.requires_grad = False


def plot_history(history, out_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    xs = [item[0] for item in history]
    ys = [item[1] for item in history]
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, color="tab:blue", linewidth=1.5, label="Validation full-sequence NMSE")
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.0, label="-37 dB")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.title("Full-sequence fine-tuning, 1000 real multipliers")
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

    data_path = resolve_path(args.data)
    init_checkpoint = resolve_path(args.init_checkpoint)
    checkpoint_path = resolve_path(args.checkpoint)
    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)

    multipliers = aidpd.real_multiplier_count_from_config(MODEL_CONFIG)
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Init checkpoint: {init_checkpoint}")
    print(f">>> Candidate checkpoint: {checkpoint_path}")
    print(f">>> Real multipliers: {multipliers}")
    if multipliers > 1000:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers}")

    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")

    model = aidpd.DiagramDPDNet(g_real, g_imag, config=MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(aidpd.load_state_dict(init_checkpoint), strict=True)
    if args.fit_fir_before:
        print("\n>>> Re-fitting 47-tap output FIR on train split before fine-tuning.")
        aidpd.fit_output_fir_from_train(model, x_train, y_train)

    set_trainable(model, args)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters left after freeze options.")

    x_train_t = tensor_sequence(x_train)
    y_train_t = tensor_sequence(y_train)
    best_nmse, best_db = fullseq_nmse(model, x_val, y_val)
    initial_db = best_db
    history = [(0, best_db)]
    saved = False
    print(f">>> Initial validation full-sequence NMSE: {best_nmse:.6e} ({best_db:.2f} dB)")
    print(
        f">>> Fine-tune: epochs={args.epochs}, lr={args.lr:.2e}, "
        f"crop={args.crop_margin}, diff={args.diff_loss_weight}"
    )

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-9
    )

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(x_train_t)
        loss = cropped_loss(pred, y_train_t, args.crop_margin, args.diff_loss_weight)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, max_norm=0.25)
        optimizer.step()

        nmse, nmse_db = fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse)
        history.append((epoch, nmse_db))
        elapsed = time.time() - start
        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"Val full NMSE {nmse_db:.3f} dB"
        )

        if nmse < best_nmse:
            best_nmse = nmse
            best_db = nmse_db
            torch.save(model.state_dict(), checkpoint_path)
            saved = True
            if best_db <= args.target_fullseq_db:
                print(f">>> Target reached: {best_db:.2f} dB")
                break

    print(f"\n>>> Initial NMSE: {initial_db:.3f} dB")
    print(f">>> Best NMSE: {best_nmse:.6e} ({best_db:.3f} dB)")
    if saved:
        print(f">>> Saved improved checkpoint: {checkpoint_path}")
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
    main()
