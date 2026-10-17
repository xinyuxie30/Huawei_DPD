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
DATA_PATH = PROJECT_ROOT / "data" / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "lightweight_reference_rich9_H10_L4_FIR47_1000.pth"
DEFAULT_INIT = PROJECT_ROOT / "checkpoints" / "archive" / "best_swa_model_A.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


# Lightweight 1000-real-multiplier reference under the project counting rule:
# polynomial 1/3/5 branches: 3 * 4 = 12
# Input-9 projection: 9 * 10 = 90
# four depthwise-separable TCN blocks: 4 * (10 * 5 + 10 * 10) = 600
# output projection: 2 * 10 = 20
# spectrum attention: 10 * 3 * 3 = 90
# 47-tap complex output FIR: 47 * 4 = 188
# total = 1000
MODEL_CONFIG = {
    "preset": "lightweight_reference_rich9_H10_L4_FIR47_1000",
    "hidden": 10,
    "input_mode": "rich9",
    "dilations": [1, 2, 4, 8],
    "attention_channels": 3,
    "poly_orders": [1, 3, 5],
    "output_fir_taps": 47,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight EA-TCN reference training under 1000 real multipliers."
    )
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--scratch", action="store_true", help="Do not initialize from the 782-MAC baseline.")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs-base", type=int, default=600)
    parser.add_argument("--epochs-long", type=int, default=240)
    parser.add_argument("--base-lr", type=float, default=5e-3)
    parser.add_argument("--long-lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--long-batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=800)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--long-frame-length", type=int, default=4096)
    parser.add_argument("--long-stride-length", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=MODEL_CONFIG["hidden"])
    parser.add_argument("--input-mode", choices=["base6", "rich9"], default=MODEL_CONFIG["input_mode"])
    parser.add_argument("--attention-channels", type=int, default=MODEL_CONFIG["attention_channels"])
    parser.add_argument("--output-fir-taps", type=int, default=MODEL_CONFIG["output_fir_taps"])
    parser.add_argument("--max-multipliers", type=int, default=1000)
    parser.add_argument("--fir-fit-domain", choices=["time", "freq"], default="time")
    parser.add_argument("--fir-main-weight", type=float, default=1.0)
    parser.add_argument("--fir-adjacent-weight", type=float, default=1.0)
    parser.add_argument("--fir-floor-weight", type=float, default=0.02)
    parser.add_argument("--fir-ridge", type=float, default=1e-6)
    parser.add_argument("--diff-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-nmse-db", type=float, default=-37.0)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(20.0, 80.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-180.0, -120.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(120.0, 180.0))
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
    raw_x, raw_y, x_c, y_c = aidpd.load_normalized_data(data_path)
    x_np = aidpd.iq_array(x_c)
    y_np = aidpd.iq_array(y_c)
    split_idx = int(min(len(x_np), len(y_np)) * 0.8)
    x_train = x_np[:split_idx]
    y_train = y_np[:split_idx]
    x_val = x_np[split_idx:]
    y_val = y_np[split_idx:]
    g_opt = np.vdot(x_c[:split_idx], y_c[:split_idx]) / np.vdot(x_c[:split_idx], x_c[:split_idx])
    return x_train, y_train, x_val, y_val, float(g_opt.real), float(g_opt.imag)


def make_loader(x_np, y_np, frame_length, stride_length, batch_size, shuffle=True):
    return DataLoader(
        aidpd.FramedDataset(x_np, y_np, frame_length, stride_length),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_nmse(model, x_val, y_val):
    nmse, nmse_db = aidpd.evaluate_nmse(model, x_val, y_val)
    return float(nmse), float(nmse_db)


def loss_with_optional_diff(pred, target, diff_loss_weight):
    loss = nn.functional.mse_loss(pred, target)
    if diff_loss_weight <= 0:
        return loss
    pred_d = pred[:, :, 1:] - pred[:, :, :-1]
    target_d = target[:, :, 1:] - target[:, :, :-1]
    return loss + diff_loss_weight * nn.functional.mse_loss(pred_d, target_d)


def set_fir_trainable(model, trainable):
    for name, param in model.named_parameters():
        if name.startswith("fir_"):
            param.requires_grad = trainable


def fit_output_fir(args, model, x_train, y_train):
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


def train_stage(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    stage_name,
    epochs,
    lr,
    frame_length,
    stride_length,
    batch_size,
    diff_loss_weight,
    checkpoint_path,
    target_nmse_db,
):
    if epochs <= 0:
        nmse, nmse_db = evaluate_nmse(model, x_val, y_val)
        return [(0, nmse_db)], nmse, nmse_db

    train_loader = make_loader(x_train, y_train, frame_length, stride_length, batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=5e-7
    )

    best_nmse, best_db = evaluate_nmse(model, x_val, y_val)
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]
    start_time = time.time()
    print(f"\n>>> Stage {stage_name}: epochs={epochs}, lr={lr:.2e}, frame={frame_length}, batch={batch_size}")
    print(f">>> Initial NMSE: {best_nmse:.6e} ({best_db:.2f} dB)")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = loss_with_optional_diff(pred, batch_y, diff_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        nmse, nmse_db = evaluate_nmse(model, x_val, y_val)
        scheduler.step(nmse_db)
        history.append((epoch, nmse_db))
        elapsed = time.time() - start_time
        print(
            f"{stage_name} Epoch [{epoch}/{epochs}] | "
            f"Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"NMSE {nmse_db:.2f} dB"
        )

        if nmse < best_nmse:
            best_nmse = nmse
            best_db = nmse_db
            torch.save(model.state_dict(), checkpoint_path)

        if best_db <= target_nmse_db:
            print(f">>> Target reached in {stage_name}: {best_db:.2f} dB")
            break

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    return history, best_nmse, best_db


def plot_history(histories, out_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    offset = 0
    for stage_name, history in histories:
        xs = [offset + item[0] for item in history]
        ys = [item[1] for item in history]
        plt.plot(xs, ys, label=stage_name)
        offset = xs[-1]
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.1, label="-37 dB")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.title("Lightweight EA-TCN reference: Input-9/H10/L4/FIR47, 1000 real multipliers")
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

    checkpoint_path = resolve_path(args.checkpoint)
    init_checkpoint = resolve_path(args.init_checkpoint)
    data_path = resolve_path(args.data)
    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)

    model_config = dict(MODEL_CONFIG)
    model_config["hidden"] = int(args.hidden)
    model_config["input_mode"] = args.input_mode
    model_config["attention_channels"] = int(args.attention_channels)
    model_config["output_fir_taps"] = int(args.output_fir_taps)
    multipliers = aidpd.real_multiplier_count_from_config(model_config)
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Validation samples: {len(x_val)}")
    print(f">>> Model config: {model_config}")
    print(f">>> Real multipliers: {multipliers}")
    if multipliers > args.max_multipliers:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers} > {args.max_multipliers}")

    model = aidpd.DiagramDPDNet(g_real, g_imag, config=model_config).to(DEVICE)
    if args.evaluate_only:
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        nmse, nmse_db = evaluate_nmse(model, x_val, y_val)
        print(f">>> NMSE: {nmse:.6e} ({nmse_db:.2f} dB)")
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
        return

    if not args.scratch and init_checkpoint.exists():
        loaded, partial, skipped = aidpd.load_partial_state(model, init_checkpoint)
        print(f">>> Partial init from: {init_checkpoint}")
        print(f">>> Loaded tensors: exact={len(loaded)}, partial={len(partial)}, skipped={len(skipped)}")
    elif args.scratch:
        print(">>> Scratch mode: no checkpoint initialization.")
    else:
        print(f">>> Init checkpoint not found, training fresh: {init_checkpoint}")

    histories = []
    h1, _, db1 = train_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        stage_name="short-frame",
        epochs=args.epochs_base,
        lr=args.base_lr,
        frame_length=args.frame_length,
        stride_length=args.stride_length,
        batch_size=args.batch_size,
        diff_loss_weight=args.diff_loss_weight,
        checkpoint_path=checkpoint_path,
        target_nmse_db=args.target_nmse_db,
    )
    histories.append(("short-frame", h1))

    print(
        f"\n>>> Fitting {model.output_fir_taps}-tap complex output FIR by "
        f"{args.fir_fit_domain}-domain least squares on training split only."
    )
    fit_output_fir(args, model, x_train, y_train)
    torch.save(model.state_dict(), checkpoint_path)
    fir_nmse, fir_db = evaluate_nmse(model, x_val, y_val)
    print(f">>> After LS FIR NMSE: {fir_nmse:.6e} ({fir_db:.2f} dB)")

    if fir_db > args.target_nmse_db:
        set_fir_trainable(model, False)
        h2, _, _ = train_stage(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            stage_name="long-frame-freeze-fir",
            epochs=args.epochs_long,
            lr=args.long_lr,
            frame_length=args.long_frame_length,
            stride_length=args.long_stride_length,
            batch_size=args.long_batch_size,
            diff_loss_weight=0.0,
            checkpoint_path=checkpoint_path,
            target_nmse_db=args.target_nmse_db,
        )
        histories.append(("long-frame-freeze-fir", h2))

        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        print("\n>>> Final train-only LS FIR refit.")
        fit_output_fir(args, model, x_train, y_train)
        torch.save(model.state_dict(), checkpoint_path)

    final_nmse, final_db = evaluate_nmse(model, x_val, y_val)
    print(f"\n>>> Final NMSE: {final_nmse:.6e} ({final_db:.2f} dB)")
    print(f">>> Saved checkpoint: {checkpoint_path}")

    convergence_path = FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png"
    plot_history(histories, convergence_path)
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
