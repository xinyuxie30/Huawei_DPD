"""Train the 2492-MAC EA-TCN base, prune to FIR43, and refit by train-only LS.

This runner records the genuine full-sequence validation NMSE after every GPU
training epoch.  The deployable point is produced from the best FIR95 base by
copying the shared weights into the 2284-MAC FIR43 model and fitting only its
complex output FIR with least squares on the training split.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import random
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
DEFAULT_DATA = PROJECT_ROOT / "data" / "DPD_data.mat"
DEFAULT_OUTPUT = SCRIPT_DIR / "ea_tcn_2492_to_2284"


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd_2492_pipeline", AI_DPD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {AI_DPD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aidpd_2492_pipeline"] = module
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


class FramedDataset(Dataset):
    def __init__(self, x_np, y_np, frame_length, stride_length, dtype):
        self.x = torch.as_tensor(x_np, dtype=dtype)
        self.y = torch.as_tensor(y_np, dtype=dtype)
        self.frame_length = int(frame_length)
        self.stride_length = int(stride_length)
        self.num_frames = (len(self.x) - self.frame_length) // self.stride_length + 1
        if self.num_frames <= 0:
            raise ValueError("frame_length exceeds the training sequence")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        start = index * self.stride_length
        end = start + self.frame_length
        return self.x[start:end].T, self.y[start:end].T


def load_split(data_path):
    _, _, x_complex, y_complex = aidpd.load_normalized_data(data_path)
    x_np = np.stack([np.real(x_complex), np.imag(x_complex)], axis=1)
    y_np = np.stack([np.real(y_complex), np.imag(y_complex)], axis=1)
    length = min(len(x_np), len(y_np))
    split = int(length * 0.8)
    g_opt = np.vdot(x_complex[:split], y_complex[:split]) / np.vdot(
        x_complex[:split], x_complex[:split]
    )
    return (
        x_np[:split],
        y_np[:split],
        x_np[split:length],
        y_np[split:length],
        float(np.real(g_opt)),
        float(np.imag(g_opt)),
    )


def evaluate_nmse_db(model, x_tensor, y_tensor):
    model.eval()
    with torch.no_grad():
        prediction = model(x_tensor)
        nmse = torch.sum((prediction - y_tensor).square()) / torch.sum(y_tensor.square())
    return float(10.0 * torch.log10(nmse).item())


def copy_base_to_deploy(base_model, deploy_model):
    """Copy shared weights and center-crop FIR95 into FIR43 before LS."""
    source = base_model.state_dict()
    target = deploy_model.state_dict()
    for key, destination in target.items():
        if key not in source:
            continue
        value = source[key]
        if value.shape == destination.shape:
            target[key] = value.to(dtype=destination.dtype, device=destination.device)
            continue
        if key in ("fir_real", "fir_imag") and value.ndim == destination.ndim == 1:
            source_center = value.shape[0] // 2
            target_center = destination.shape[0] // 2
            radius = destination.shape[0] // 2
            target[key] = value[
                source_center - radius : source_center + radius + 1
            ].to(dtype=destination.dtype, device=destination.device)
    deploy_model.load_state_dict(target, strict=True)


def fit_output_fir_ls(model, x_train, y_train, device, dtype):
    """Fit the complex FIR from the training split only."""
    taps = int(model.output_fir_taps)
    radius = taps // 2
    x_tensor = torch.as_tensor(x_train.T, dtype=dtype, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        base = model.forward_base(x_tensor).squeeze(0).detach().cpu().numpy().T
    z = base[:, 0].astype(np.float64) + 1j * base[:, 1].astype(np.float64)
    target = y_train[:, 0].astype(np.float64) + 1j * y_train[:, 1].astype(np.float64)
    design = np.stack(
        [z[radius + offset : len(z) - radius + offset] for offset in range(-radius, radius + 1)],
        axis=1,
    )
    rhs = target[radius : len(target) - radius]
    coefficients, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coefficients.real, dtype=dtype, device=device))
        model.fir_imag.copy_(torch.as_tensor(coefficients.imag, dtype=dtype, device=device))
    return coefficients


def build_deploy_model(base_model, g_real, g_imag, x_train, y_train, device, dtype):
    config = aidpd.preset_config("rich9_h15_l6_fir43_2284")
    model = aidpd.DiagramDPDNet(g_real, g_imag, config=config).to(device=device, dtype=dtype)
    copy_base_to_deploy(base_model, model)
    fit_output_fir_ls(model, x_train, y_train, device, dtype)
    return model, config


def write_history(output_dir, history, loss_history, lr_history, deploy_epoch=None, deploy_nmse=None):
    path = output_dir / "pipeline_history.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "base_2492_validation_nmse_db",
                "train_mse",
                "learning_rate",
                "deploy_2284_validation_nmse_db",
            ]
        )
        for epoch, nmse_db in enumerate(history):
            deploy_value = deploy_nmse if deploy_epoch == epoch else ""
            writer.writerow(
                [
                    epoch,
                    f"{nmse_db:.8f}",
                    "" if not math.isfinite(loss_history[epoch]) else f"{loss_history[epoch]:.10e}",
                    f"{lr_history[epoch]:.10e}",
                    "" if deploy_value == "" else f"{deploy_value:.8f}",
                ]
            )
    return path


def save_state(path, epoch, model, optimizer, scheduler, history, loss_history, lr_history, best):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "loss_history": loss_history,
            "lr_history": lr_history,
            "best": best,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="GPU train FIR95/2492 then prune+LS to FIR43/2284")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-epochs", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-trigger-db", type=float, default=-39.80)
    parser.add_argument("--deploy-target-db", type=float, default=-40.0)
    parser.add_argument("--check-deploy-every", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-from",
        default="",
        help="Optional FIR95 checkpoint for continued training; empty means scratch.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "runner_state.pth"
    best_path = checkpoint_dir / "ea_tcn_fir95_2492_best.pth"
    final_path = checkpoint_dir / "ea_tcn_fir43_2284_pruned_ls.pth"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for this requested run")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    x_train, y_train, x_val, y_val, g_real, g_imag = load_split(Path(args.data).resolve())
    dataset = FramedDataset(x_train, y_train, args.frame_length, args.stride_length, dtype)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    x_val_tensor = torch.as_tensor(x_val.T, dtype=dtype, device=device).unsqueeze(0)
    y_val_tensor = torch.as_tensor(y_val.T, dtype=dtype, device=device).unsqueeze(0)

    base_config = aidpd.preset_config("rich9_h15_l6_fir95_2492")
    base_model = aidpd.DiagramDPDNet(g_real, g_imag, config=base_config).to(
        device=device, dtype=dtype
    )
    optimizer = torch.optim.AdamW(
        base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )

    start_epoch = 0
    elapsed_before = 0.0
    if args.resume:
        saved = torch.load(state_path, map_location=device, weights_only=False)
        base_model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = list(saved["history"])
        loss_history = list(saved["loss_history"])
        lr_history = list(saved["lr_history"])
        best = dict(saved["best"])
        start_epoch = int(saved["epoch"])
        elapsed_before = float(best.get("elapsed_seconds", 0.0))
    else:
        if args.initialize_from:
            source = torch.load(Path(args.initialize_from).resolve(), map_location=device, weights_only=True)
            if isinstance(source, dict) and "state_dict" in source:
                source = source["state_dict"]
            base_model.load_state_dict(source, strict=True)
        initial_nmse = evaluate_nmse_db(base_model, x_val_tensor, y_val_tensor)
        history = [initial_nmse]
        loss_history = [math.nan]
        lr_history = [args.lr]
        best = {"epoch": 0, "nmse_db": initial_nmse, "elapsed_seconds": 0.0}
        torch.save(base_model.state_dict(), best_path)
        write_history(output_dir, history, loss_history, lr_history)

    print(f">>> Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f">>> PyTorch/CUDA: {torch.__version__} / {torch.version.cuda}", flush=True)
    print(f">>> Dtype: {args.dtype} | frames={len(dataset)} | batch={args.batch_size}", flush=True)
    print(
        f">>> Base multipliers={aidpd.real_multiplier_count_from_config(base_config)} | "
        "deployment multipliers=2284",
        flush=True,
    )
    print(f">>> Initial/best NMSE: {history[0]:.4f}/{best['nmse_db']:.4f} dB", flush=True)

    criterion = nn.MSELoss()
    run_start = time.time()
    final_nmse = math.inf
    deploy_epoch = None
    deployed = None
    for epoch in range(start_epoch + 1, args.max_epochs + 1):
        epoch_start = time.time()
        base_model.train()
        loss_sum = 0.0
        batches = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = base_model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1

        nmse_db = evaluate_nmse_db(base_model, x_val_tensor, y_val_tensor)
        scheduler.step(nmse_db)
        history.append(nmse_db)
        loss_history.append(loss_sum / max(batches, 1))
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        if nmse_db < float(best["nmse_db"]):
            best = {
                "epoch": epoch,
                "nmse_db": nmse_db,
                "elapsed_seconds": elapsed_before + time.time() - run_start,
            }
            torch.save(base_model.state_dict(), best_path)

        should_check = (
            float(best["nmse_db"]) <= args.base_trigger_db
            and (epoch % args.check_deploy_every == 0 or nmse_db == float(best["nmse_db"]))
        )
        if should_check:
            current_state = {
                key: value.detach().clone() for key, value in base_model.state_dict().items()
            }
            base_model.load_state_dict(
                torch.load(best_path, map_location=device, weights_only=True), strict=True
            )
            deployed, deploy_config = build_deploy_model(
                base_model, g_real, g_imag, x_train, y_train, device, dtype
            )
            final_nmse = evaluate_nmse_db(deployed, x_val_tensor, y_val_tensor)
            deploy_epoch = epoch
            print(
                f">>> FIR43 prune + train-only LS check: {final_nmse:.4f} dB",
                flush=True,
            )
            if final_nmse < args.deploy_target_db:
                torch.save(deployed.state_dict(), final_path)
                write_history(
                    output_dir,
                    history,
                    loss_history,
                    lr_history,
                    deploy_epoch=deploy_epoch,
                    deploy_nmse=final_nmse,
                )
                save_state(
                    state_path,
                    epoch,
                    base_model,
                    optimizer,
                    scheduler,
                    history,
                    loss_history,
                    lr_history,
                    best,
                )
                print(
                    f">>> PASS: deployable 2284 model is below {args.deploy_target_db:.2f} dB",
                    flush=True,
                )
                break
            base_model.load_state_dict(current_state, strict=True)

        write_history(output_dir, history, loss_history, lr_history)
        save_state(
            state_path,
            epoch,
            base_model,
            optimizer,
            scheduler,
            history,
            loss_history,
            lr_history,
            best,
        )
        print(
            f"Epoch [{epoch}/{args.max_epochs}] | {time.time() - epoch_start:.2f}s | "
            f"NMSE={nmse_db:.4f} dB | best={float(best['nmse_db']):.4f} dB | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
    else:
        base_model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True), strict=True
        )
        deployed, deploy_config = build_deploy_model(
            base_model, g_real, g_imag, x_train, y_train, device, dtype
        )
        final_nmse = evaluate_nmse_db(deployed, x_val_tensor, y_val_tensor)
        deploy_epoch = args.max_epochs
        torch.save(deployed.state_dict(), final_path)
        write_history(
            output_dir,
            history,
            loss_history,
            lr_history,
            deploy_epoch=deploy_epoch,
            deploy_nmse=final_nmse,
        )

    if deployed is None:
        base_model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True), strict=True
        )
        deployed, deploy_config = build_deploy_model(
            base_model, g_real, g_imag, x_train, y_train, device, dtype
        )
        final_nmse = evaluate_nmse_db(deployed, x_val_tensor, y_val_tensor)
        deploy_epoch = len(history) - 1
        torch.save(deployed.state_dict(), final_path)
        write_history(
            output_dir,
            history,
            loss_history,
            lr_history,
            deploy_epoch=deploy_epoch,
            deploy_nmse=final_nmse,
        )

    metadata = {
        "created_unix": time.time(),
        "data": str(Path(args.data).resolve()),
        "split": "first 80% train / last 20% validation",
        "train_samples": len(x_train),
        "validation_samples": len(x_val),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "dtype": args.dtype,
        "seed": args.seed,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "frame_length": args.frame_length,
        "stride_length": args.stride_length,
        "base_preset": "rich9_h15_l6_fir95_2492",
        "base_real_multipliers": aidpd.real_multiplier_count_from_config(base_config),
        "base_best_epoch": int(best["epoch"]),
        "base_best_validation_nmse_db": float(best["nmse_db"]),
        "deployment_preset": "rich9_h15_l6_fir43_2284",
        "deployment_real_multipliers": aidpd.real_multiplier_count_from_config(deploy_config),
        "deployment_step": "FIR95 center prune to FIR43, then train-only complex least squares",
        "deployment_epoch": deploy_epoch,
        "deployment_validation_nmse_db": final_nmse,
        "elapsed_training_seconds": float(best.get("elapsed_seconds", time.time() - run_start)),
        "base_checkpoint": str(best_path),
        "deployment_checkpoint": str(final_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f">>> Result: base best={float(best['nmse_db']):.4f} dB @ epoch {int(best['epoch'])}; "
        f"2284 deploy={final_nmse:.4f} dB",
        flush=True,
    )
    print(f">>> History: {output_dir / 'pipeline_history.csv'}", flush=True)
    print(f">>> Metadata: {output_dir / 'metadata.json'}", flush=True)
    if final_nmse >= args.deploy_target_db:
        raise RuntimeError(
            f"Deployment NMSE {final_nmse:.4f} dB did not beat {args.deploy_target_db:.4f} dB"
        )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
