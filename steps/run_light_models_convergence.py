"""From-scratch base EA-TCN convergence runner for the 998/786/796 presets.

Mirrors steps/run_2284_baseline_convergence.py's protocol (AdamW lr 3e-3,
ReduceLROnPlateau patience 12, MSE loss, 1024-frame / 32-stride / batch 64,
seed 2026) but trains a single EA-TCN per preset and logs validation NMSE
every epoch, so the report gets a smooth convergence curve for each light
model.  A closed-form output-FIR least squares step (the "FIR pruning + LS"
star) is applied after training.  Saves base weights, nmse_history.csv and a
convergence PNG under eval_outputs_direct/base_convergence/<label>/.

The 2284 curve already exists in steps/baseline_convergence_2284/ and is not
recomputed here.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent.parent
AI_DPD_PATH = ROOT / "steps" / "AI-DPD.py"
OUT_ROOT = ROOT / "eval_outputs_direct" / "base_convergence"

PRESETS = {
    "998": "memory4_fir19_998",
    "786": "h10_l4_base6_rank3_fir41_786",
    "796": "h10_l3_base6_fir41_796",
    "2284": "rich9_h15_l6_fir43_2284",
}


def load_aidpd():
    spec = importlib.util.spec_from_file_location("aidpd_conv_module", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aidpd_conv_module"] = module
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd()


class Frames(Dataset):
    def __init__(self, x_np, y_np, frame, stride, dtype):
        self.x = torch.as_tensor(x_np, dtype=dtype)
        self.y = torch.as_tensor(y_np, dtype=dtype)
        self.frame = int(frame)
        self.stride = int(stride)
        self.n = (len(self.x) - self.frame) // self.stride + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.stride
        return self.x[s:s + self.frame].T, self.y[s:s + self.frame].T


def evaluate_nmse_db(model, xt, yt):
    model.eval()
    with torch.no_grad():
        pred = model(xt)
        nmse = torch.sum((pred - yt).square()) / torch.sum(yt.square())
    return float(10.0 * torch.log10(nmse).item())


def fit_output_fir_ls(model, x_train, y_train, device, dtype):
    taps = int(getattr(model, "output_fir_taps", 1))
    if taps <= 1 or not hasattr(model, "forward_base"):
        return math.nan
    radius = taps // 2
    xt = torch.as_tensor(x_train.T, dtype=dtype, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        base = model.forward_base(xt).squeeze(0).detach().cpu().numpy().T
    z = base[:, 0].astype(np.float64) + 1j * base[:, 1].astype(np.float64)
    tgt = y_train[:, 0].astype(np.float64) + 1j * y_train[:, 1].astype(np.float64)
    design = np.stack([z[radius + o:len(z) - radius + o] for o in range(-radius, radius + 1)], axis=1)
    rhs = tgt[radius:len(tgt) - radius]
    coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=dtype, device=device))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=dtype, device=device))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=PRESETS)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--frame-length", type=int, default=1024)
    p.add_argument("--stride-length", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--scheduler-patience", type=int, default=12)
    p.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--target-db", type=float, default=None)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    targets = {"2284": -40.0, "998": -37.0, "786": -35.0, "796": -35.0}
    target_db = args.target_db if args.target_db is not None else targets[args.model]

    raw_x, raw_y, x_c, y_c = aidpd.load_normalized_data(ROOT / "data" / "DPD_data.mat")
    x_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    y_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)
    split = int(len(x_np) * 0.8)
    x_train, y_train = x_np[:split], y_np[:split]
    x_val, y_val = x_np[split:], y_np[split:]
    xc_tr, yc_tr = x_c[:split], y_c[:split]
    g_opt = np.vdot(xc_tr, yc_tr) / np.vdot(xc_tr, xc_tr)

    config = aidpd.preset_config(PRESETS[args.model])
    macs = aidpd.real_multiplier_count_from_config(config)
    model = aidpd.DiagramDPDNet(float(np.real(g_opt)), float(np.imag(g_opt)), config=config)
    model = model.to(device=device, dtype=dtype)

    loader = DataLoader(Frames(x_train, y_train, args.frame_length, args.stride_length, dtype),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    xv = torch.as_tensor(x_val.T, dtype=dtype, device=device).unsqueeze(0)
    yv = torch.as_tensor(y_val.T, dtype=dtype, device=device).unsqueeze(0)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5,
                                                       patience=args.scheduler_patience,
                                                       min_lr=args.scheduler_min_lr)
    criterion = nn.MSELoss()
    out_dir = OUT_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    history = [evaluate_nmse_db(model, xv, yv)]
    best_db, best_epoch = history[0], 0
    best_path = out_dir / "base_best.pth"
    torch.save(model.state_dict(), best_path)
    print(f">>> {args.model} preset={config['preset']} MAC={macs} init NMSE {history[0]:.3f} dB "
          f"| device {device} dtype {args.dtype}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = n = 0
        for bx, by in loader:
            bx = bx.to(device); by = by.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            loss_sum += float(loss.item()); n += 1
        db = evaluate_nmse_db(model, xv, yv)
        sched.step(db)
        history.append(db)
        if db < best_db:
            best_db, best_epoch = db, epoch
            torch.save(model.state_dict(), best_path)
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"    ep {epoch:03d} val NMSE {db:.3f} dB | best {best_db:.3f}@{best_epoch} "
                  f"| lr {optim.param_groups[0]['lr']:.2e} | loss {loss_sum/max(n,1):.3e}")

    # closed-form output-FIR least squares on the best checkpoint (the report's LS star)
    model.load_state_dict(torch.load(best_path, map_location=device))
    fit_output_fir_ls(model, x_train, y_train, device, dtype)
    post_ls_db = evaluate_nmse_db(model, xv, yv)
    torch.save(model.state_dict(), out_dir / "base_fir_ls.pth")
    print(f">>> {args.model} best {best_db:.3f} dB @ep{best_epoch} | post-FIR-LS {post_ls_db:.3f} dB "
          f"| {time.time()-t0:.0f}s")

    with (out_dir / "nmse_history.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "ea_tcn"])
        for e, v in enumerate(history):
            w.writerow([e, f"{v:.8f}"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ep = np.arange(len(history))
    ax.plot(ep, history, color="#0067B1", lw=1.6, label=f"EA-TCN-{args.model} ({macs} MAC)")
    ax.scatter([len(history) + 2], [post_ls_db], marker="*", s=130, color="#0067B1",
               edgecolors="white", linewidths=0.8, zorder=6)
    ax.annotate(f"FIR-LS {post_ls_db:.2f}", xy=(len(history) + 2, post_ls_db),
                xytext=(6, -10), textcoords="offset points", fontsize=8, color="#0067B1")
    ax.axhline(target_db, color="#C62828", ls=":", lw=1.4, label=f"Target {target_db:.0f} dB")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation NMSE (dB)")
    ax.set_title(f"EA-TCN-{args.model} from-scratch convergence")
    ax.grid(True, ls=":", lw=0.7, alpha=0.55)
    ax.legend(loc="best", fontsize=9); ax.set_xlim(0, len(history) + 10)
    fig.tight_layout()
    fig.savefig(out_dir / "nmse_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">>> saved: {out_dir}")


if __name__ == "__main__":
    main()
