import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "data" / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"


class FramedDataset(Dataset):
    def __init__(self, x_np, y_np, frame_length, stride_length):
        self.x = torch.as_tensor(x_np, dtype=torch.float64)
        self.y = torch.as_tensor(y_np, dtype=torch.float64)
        self.frame_length = int(frame_length)
        self.stride_length = int(stride_length)
        self.num_frames = (self.x.shape[0] - self.frame_length) // self.stride_length + 1
        if self.num_frames <= 0:
            raise ValueError("frame_length is longer than the training sequence.")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        end = start + self.frame_length
        return self.x[start:end, :].T, self.y[start:end, :].T


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd_final_module", AI_DPD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {AI_DPD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aidpd_final_module"] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_split(aidpd, data_path):
    _, _, x_c, y_c = aidpd.load_normalized_data(data_path)
    x_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    y_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)
    min_len = min(len(x_np), len(y_np))
    split_idx = int(min_len * 0.8)

    x_train = x_np[:split_idx]
    y_train = y_np[:split_idx]
    x_val = x_np[split_idx:min_len]
    y_val = y_np[split_idx:min_len]

    g_opt = np.vdot(x_c[:split_idx], y_c[:split_idx]) / np.vdot(x_c[:split_idx], x_c[:split_idx])
    return x_train, y_train, x_val, y_val, float(np.real(g_opt)), float(np.imag(g_opt))


def make_model(aidpd, config, g_real, g_imag):
    return aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(aidpd.DEVICE)


def aclr_worst(aidpd, model, x_val, y_val, args):
    pred_iq = aidpd.run_sequence(model, x_val)
    pred_complex = aidpd.tensor_to_complex(pred_iq)
    target_complex = aidpd.tensor_to_complex(y_val.T)
    error_complex = target_complex - pred_complex
    freq_hz, error_psd = aidpd.psd_db(error_complex, fs_hz=args.fs_mhz * 1e6)
    _, target_psd = aidpd.psd_db(target_complex, fs_hz=args.fs_mhz * 1e6)
    return float(
        aidpd.aclr_from_psd(
            freq_hz,
            error_psd,
            tuple(args.main_band),
            tuple(args.lower_band),
            tuple(args.upper_band),
            original_values_db=target_psd,
        )[2]
    )


def fit_final_fir(aidpd, model, x_train, y_train, args):
    if args.fit_domain == "freq":
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


def build_and_evaluate_final(aidpd, base_checkpoint, x_train, y_train, x_val, y_val, g_real, g_imag, args, save=True):
    final_config = aidpd.preset_config("rich9_h15_l6_fir43_2284")
    final_config["output_fir_taps"] = int(args.final_fir_taps)
    final_model = make_model(aidpd, final_config, g_real, g_imag)
    aidpd.load_partial_state(final_model, base_checkpoint)
    fit_final_fir(aidpd, final_model, x_train, y_train, args)

    nmse, nmse_db = aidpd.evaluate_nmse(final_model, x_val, y_val)
    aclr = aclr_worst(aidpd, final_model, x_val, y_val, args)
    multipliers = aidpd.real_multiplier_count_from_config(final_config)

    if save:
        torch.save(final_model.state_dict(), args.final_checkpoint)
        prefix = args.figure_prefix or args.final_checkpoint.stem
        aidpd.evaluate_aclr_visualization(
            final_model,
            x_val,
            y_val,
            output_prefix=prefix,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            target_dbc=args.aclr_target_dbc,
        )

    return final_model, float(nmse), float(nmse_db), float(aclr), int(multipliers)


def candidate_taps(args):
    taps = [int(args.final_fir_taps)]
    if not args.no_fallback_scan:
        taps.extend(int(item) for item in args.fallback_fir_taps)
    unique = []
    for item in taps:
        if item <= 0 or item % 2 == 0:
            raise ValueError(f"FIR taps must be positive odd integers, got {item}")
        if item not in unique:
            unique.append(item)
    return unique


def update_output_names_for_selected_taps(args, selected_taps, selected_multipliers):
    default_checkpoint_name = "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth"
    default_prefix = "final_main_H15_L6_rich9_FIR43_2284"
    if args.final_checkpoint.name == default_checkpoint_name and selected_taps != 43:
        args.final_checkpoint = args.final_checkpoint.with_name(
            f"final_main_H15_L6_rich9_FIR{selected_taps}_{selected_multipliers}_PASS.pth"
        )
    if args.figure_prefix == default_prefix and selected_taps != 43:
        args.figure_prefix = f"final_main_H15_L6_rich9_FIR{selected_taps}_{selected_multipliers}"


def export_best_final(aidpd, base_checkpoint, x_train, y_train, x_val, y_val, g_real, g_imag, args):
    original_taps = int(args.final_fir_taps)
    records = []
    best_pass = None

    print(">>> Final FIR candidate scan")
    for taps in candidate_taps(args):
        args.final_fir_taps = taps
        _, nmse, db, aclr, multipliers = build_and_evaluate_final(
            aidpd,
            base_checkpoint,
            x_train,
            y_train,
            x_val,
            y_val,
            g_real,
            g_imag,
            args,
            save=False,
        )
        records.append((taps, multipliers, nmse, db, aclr))
        print(
            f">>> Candidate FIR{taps}: multipliers={multipliers} | "
            f"NMSE={db:.4f} dB | ACLR={aclr:.4f} dBc"
        )
        if multipliers > args.max_final_multipliers:
            continue
        if db <= args.nmse_target_db and aclr <= args.aclr_target_dbc:
            candidate = (multipliers, taps, nmse, db, aclr)
            if best_pass is None or candidate[0] < best_pass[0] or (
                candidate[0] == best_pass[0] and candidate[3] < best_pass[3]
            ):
                best_pass = candidate

    if best_pass is not None:
        multipliers, taps, _, _, _ = best_pass
        args.final_fir_taps = taps
        update_output_names_for_selected_taps(args, taps, multipliers)
        _, final_nmse, final_db, final_aclr, final_multipliers = build_and_evaluate_final(
            aidpd,
            base_checkpoint,
            x_train,
            y_train,
            x_val,
            y_val,
            g_real,
            g_imag,
            args,
            save=True,
        )
        return final_nmse, final_db, final_aclr, final_multipliers

    args.final_fir_taps = original_taps
    _, final_nmse, final_db, final_aclr, final_multipliers = build_and_evaluate_final(
        aidpd,
        base_checkpoint,
        x_train,
        y_train,
        x_val,
        y_val,
        g_real,
        g_imag,
        args,
        save=True,
    )
    summary = "; ".join(
        f"FIR{taps}: {db:.3f} dB/{aclr:.2f} dBc/{multipliers} mult"
        for taps, multipliers, _, db, aclr in records
    )
    raise RuntimeError(
        "Final model did not meet targets after fallback scan: "
        f"NMSE {final_db:.2f} dB, ACLR {final_aclr:.2f} dBc. "
        f"Candidates: {summary}"
    )


def train_base_and_export_final(aidpd, x_train, y_train, x_val, y_val, g_real, g_imag, args):
    base_config = aidpd.preset_config("rich9_h15_l6_fir95_2492")
    base_model = make_model(aidpd, base_config, g_real, g_imag)
    base_multipliers = aidpd.real_multiplier_count_from_config(base_config)

    train_loader = DataLoader(
        FramedDataset(x_train, y_train, args.frame_length, args.stride_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    criterion = nn.MSELoss()

    best_nmse = float("inf")
    best_db = float("inf")
    start = time.time()

    print(f">>> Device: {aidpd.DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(">>> Base model: rich9_h15_l6_fir95_2492")
    print(f">>> Base real multipliers before FIR pruning: {base_multipliers}")
    print(f">>> Final FIR taps: {args.final_fir_taps}")
    print(f">>> Base checkpoint: {args.base_checkpoint}")
    print(f">>> Final checkpoint: {args.final_checkpoint}")
    print(f">>> Train split: {len(x_train)} samples | Val split: {len(x_val)} samples")

    for epoch in range(1, args.max_epochs + 1):
        base_model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(aidpd.DEVICE, non_blocking=True)
            batch_y = batch_y.to(aidpd.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = base_model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        nmse, nmse_db = aidpd.evaluate_nmse(base_model, x_val, y_val)
        scheduler.step(nmse)
        if nmse < best_nmse:
            best_nmse = float(nmse)
            best_db = float(nmse_db)
            torch.save(base_model.state_dict(), args.base_checkpoint)

        elapsed = time.time() - start
        print(
            f"Epoch [{epoch}/{args.max_epochs}] | "
            f"Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"Base full NMSE {nmse_db:.2f} dB | Best {best_db:.2f} dB"
        )

        if nmse_db <= args.base_target_db:
            final_nmse, final_db, final_aclr, final_multipliers = export_best_final(
                aidpd,
                args.base_checkpoint,
                x_train,
                y_train,
                x_val,
                y_val,
                g_real,
                g_imag,
                args,
            )
            print(
                ">>> Final check after FIR pruning/refit: "
                f"multipliers={final_multipliers} | "
                f"NMSE={final_nmse:.6e} ({final_db:.4f} dB) | "
                f"ACLR={final_aclr:.4f} dBc"
            )
            if final_db <= args.nmse_target_db and final_aclr <= args.aclr_target_dbc:
                print(">>> PASS: final model meets both NMSE and ACLR targets.")
                return final_nmse, final_db, final_aclr, final_multipliers

    print(">>> Max epochs reached. Exporting final model from the best base checkpoint.")
    final_nmse, final_db, final_aclr, final_multipliers = export_best_final(
        aidpd,
        args.base_checkpoint,
        x_train,
        y_train,
        x_val,
        y_val,
        g_real,
        g_imag,
        args,
    )
    return final_nmse, final_db, final_aclr, final_multipliers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final main DPD training: H15/L6/Input-9/FIR43, 2284 real multipliers, target NMSE <= -40 dB and ACLR <= -45 dBc."
    )
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--base-target-db", type=float, default=-40.0)
    parser.add_argument("--nmse-target-db", type=float, default=-40.0)
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0)
    parser.add_argument("--final-fir-taps", type=int, default=43)
    parser.add_argument("--fallback-fir-taps", type=int, nargs="*", default=[47, 51, 55, 63, 71, 79, 95])
    parser.add_argument("--no-fallback-scan", action="store_true")
    parser.add_argument("--max-final-multipliers", type=int, default=2500)
    parser.add_argument("--fit-domain", choices=["time", "freq"], default="time")
    parser.add_argument("--fir-main-weight", type=float, default=1.0)
    parser.add_argument("--fir-adjacent-weight", type=float, default=1.0)
    parser.add_argument("--fir-floor-weight", type=float, default=0.05)
    parser.add_argument("--fir-ridge", type=float, default=1e-8)

    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--main-band", type=float, nargs=2, default=(20.0, 80.0))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-180.0, -120.0))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(120.0, 180.0))

    parser.add_argument(
        "--base-checkpoint",
        default=str(CHECKPOINT_DIR / "final_main_base_H15_L6_rich9_FIR95_2492.pth"),
    )
    parser.add_argument(
        "--final-checkpoint",
        default=str(CHECKPOINT_DIR / "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth"),
    )
    parser.add_argument("--figure-prefix", default="final_main_H15_L6_rich9_FIR43_2284")
    parser.add_argument("--reuse-base", action="store_true", help="Skip base training and export from --base-checkpoint.")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate --final-checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    aidpd = load_aidpd_module()
    set_seed(args.seed)

    args.data = resolve_path(args.data)
    args.base_checkpoint = resolve_path(args.base_checkpoint)
    args.final_checkpoint = resolve_path(args.final_checkpoint)
    args.base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.final_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, g_real, g_imag = load_split(aidpd, args.data)

    if args.evaluate_only:
        final_config = aidpd.preset_config("rich9_h15_l6_fir43_2284")
        final_config["output_fir_taps"] = int(args.final_fir_taps)
        model = make_model(aidpd, final_config, g_real, g_imag)
        model.load_state_dict(aidpd.load_state_dict(args.final_checkpoint), strict=True)
        multipliers = aidpd.real_multiplier_count_from_config(final_config)
        print(f">>> Evaluate final main model | multipliers={multipliers} | checkpoint={args.final_checkpoint}")
        aidpd.evaluate_aclr_visualization(
            model,
            x_val,
            y_val,
            output_prefix=args.figure_prefix or args.final_checkpoint.stem,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            target_dbc=args.aclr_target_dbc,
        )
        return

    if args.reuse_base:
        if not args.base_checkpoint.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint}")
        final_nmse, final_db, final_aclr, final_multipliers = export_best_final(
            aidpd,
            args.base_checkpoint,
            x_train,
            y_train,
            x_val,
            y_val,
            g_real,
            g_imag,
            args,
        )
    else:
        final_nmse, final_db, final_aclr, final_multipliers = train_base_and_export_final(
            aidpd, x_train, y_train, x_val, y_val, g_real, g_imag, args
        )

    print("\n>>> Final main model summary")
    print(f"Final checkpoint: {args.final_checkpoint}")
    print(f"Real multipliers: {final_multipliers}")
    print(f"Linear NMSE: {final_nmse:.6e}")
    print(f"dB NMSE: {final_db:.4f} dB")
    print(f"ACLR worst: {final_aclr:.4f} dBc")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
