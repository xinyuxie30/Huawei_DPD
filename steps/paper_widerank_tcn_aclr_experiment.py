"""Standalone wider low-rank TCN-DPD experiment under the 2284 budget."""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = load_module("paper_ttrans", ROOT / "steps" / "paper_ttrans_tcn_aclr_experiment.py")
evaluator = load_module("final_evaluate_unified", ROOT / "steps" / "final_evaluate_unified.py")
metric_exp = load_module("aclr_three_metric_experiment", ROOT / "steps" / "aclr_three_metric_experiment.py")


HIDDEN = 17
RANK = 6
LAYERS = 6
FIR_TAPS = 43
MULTIPLIERS = 12 + 9 * HIDDEN + LAYERS * (5 * HIDDEN + 2 * HIDDEN * RANK) + 2 * HIDDEN + 9 * HIDDEN + 4 * FIR_TAPS
BASE_CHECKPOINT = ROOT / "checkpoints" / "AI-DPD" / "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth"
TEACHER_CHECKPOINT = ROOT / "eval_outputs_direct" / "aclr_three_metric_experiments" / "aclr_teacher32" / "teacher32" / "best.pth"
OUT_ROOT = ROOT / "eval_outputs_direct" / "paper_widerank_tcn_experiments"


class LowRankDSBlock(nn.Module):
    def __init__(self, channels, rank, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation, dilation=dilation,
            groups=channels, bias=False, dtype=torch.float64,
        )
        self.down = nn.Conv1d(channels, rank, 1, bias=False, dtype=torch.float64)
        self.up = nn.Conv1d(rank, channels, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.up(self.down(self.depthwise(x))))


class WideRankTCN(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("g1_real", "g1_imag", "g3_real", "g3_imag", "g5_real", "g5_imag"):
            setattr(self, name, nn.Parameter(torch.tensor(0.0, dtype=torch.float64)))
        self.input_projection = nn.Conv1d(9, HIDDEN, 1, bias=False, dtype=torch.float64)
        self.input_activation = nn.SiLU()
        self.blocks = nn.ModuleList([LowRankDSBlock(HIDDEN, RANK, 2**i) for i in range(LAYERS)])
        self.local_attention = nn.Conv1d(3, HIDDEN, 3, padding=1, bias=False, dtype=torch.float64)
        self.output_projection = nn.Conv1d(HIDDEN, 2, 1, bias=False, dtype=torch.float64)
        self.fir_real = nn.Parameter(torch.zeros(FIR_TAPS, dtype=torch.float64))
        self.fir_imag = nn.Parameter(torch.zeros(FIR_TAPS, dtype=torch.float64))
        self.fir_real.data[FIR_TAPS // 2] = 1.0

    def base_forward(self, x):
        i, q = x[:, 0:1], x[:, 1:2]
        amp = torch.sqrt(i.square() + q.square() + 1e-12)
        amp2, amp3, amp4, amp5, amp7 = amp.square(), amp.pow(3), amp.pow(4), amp.pow(5), amp.pow(7)
        features = torch.cat([
            i, q, torch.roll(i, -1, 2), torch.roll(q, -1, 2),
            torch.roll(i, 1, 2), torch.roll(q, 1, 2), amp, amp3, amp5,
        ], dim=1)
        lin_i = i * self.g1_real - q * self.g1_imag
        lin_q = i * self.g1_imag + q * self.g1_real
        for power_i, power_q, gr, gi in [
            (i * amp2, q * amp2, self.g3_real, self.g3_imag),
            (i * amp4, q * amp4, self.g5_real, self.g5_imag),
        ]:
            lin_i = lin_i + power_i * gr - power_q * gi
            lin_q = lin_q + power_i * gi + power_q * gr
        hidden = self.input_activation(self.input_projection(features))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden * torch.sigmoid(self.local_attention(torch.cat([amp, amp3, amp7], dim=1)))
        return torch.cat([lin_i, lin_q], dim=1) + self.output_projection(hidden)

    def forward(self, x):
        base = self.base_forward(x)
        pad = FIR_TAPS // 2
        real, imag = self.fir_real.view(1, 1, -1), self.fir_imag.view(1, 1, -1)
        i, q = base[:, 0:1], base[:, 1:2]
        return torch.cat([
            F.conv1d(i, real, padding=pad) - F.conv1d(q, imag, padding=pad),
            F.conv1d(i, imag, padding=pad) + F.conv1d(q, real, padding=pad),
        ], dim=1)


def initialize_svd(model, state):
    old_h = state["in_conv.weight"].shape[0]
    with torch.no_grad():
        for new, old in [("g1_real", "g_real"), ("g1_imag", "g_imag"), ("g3_real", "g3_real"), ("g3_imag", "g3_imag"), ("g5_real", "g5_real"), ("g5_imag", "g5_imag")]:
            getattr(model, new).copy_(state[old])
        model.input_projection.weight.zero_()
        model.input_projection.weight[:old_h].copy_(state["in_conv.weight"])
        for index, block in enumerate(model.blocks):
            block.depthwise.weight.zero_()
            block.depthwise.weight[:old_h].copy_(state[f"tcn_blocks.{index}.0.weight"])
            matrix = state[f"tcn_blocks.{index}.1.weight"].squeeze(-1)
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            rank = min(RANK, len(s))
            root_s = torch.sqrt(s[:rank])
            block.down.weight.zero_(); block.up.weight.zero_()
            block.down.weight[:rank, :old_h, 0].copy_(root_s[:, None] * vh[:rank])
            block.up.weight[:old_h, :rank, 0].copy_(u[:, :rank] * root_s[None, :])
        model.local_attention.weight.zero_()
        model.local_attention.weight[:old_h].copy_(state["attn.env_conv.weight"])
        model.output_projection.weight.zero_()
        model.output_projection.weight[:, :old_h].copy_(state["out_conv.weight"])
        model.fir_real.copy_(state["fir_real"]); model.fir_imag.copy_(state["fir_imag"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--spectral-weight", type=float, default=1e-3)
    args = parser.parse_args()
    if MULTIPLIERS > 2284:
        raise ValueError(f"Budget exceeded: {MULTIPLIERS}")
    device = evaluator.DEVICE
    x, y, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split = int(len(x) * 0.8); x_train, y_train, x_val, y_val = x[:split], y[:split], x[split:], y[split:]
    teacher = evaluator.AutoDiagramDPDNet(evaluator.load_state_dict(TEACHER_CHECKPOINT)).to(device).eval()
    teacher_train, _, _, _ = evaluator.evaluate_full_sequence(teacher, x_train, y_train)
    model = WideRankTCN().to(device)
    initialize_svd(model, evaluator.load_state_dict(BASE_CHECKPOINT))
    penalty = metric_exp.SpectralPenalty(y_train, 1024, 256, 491.52, device)
    loader = DataLoader(paper.SignalFrames(x_train, y_train, teacher_train.T, 4096, 2048), batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)
    out = OUT_ROOT / args.tag; out.mkdir(parents=True, exist_ok=True)
    best = paper.evaluate_metrics(evaluator, model, x_val, y_val, device); best_path = out / "best.pth"; torch.save(model.state_dict(), best_path)
    print(f"multipliers={MULTIPLIERS} initial={best}")
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0
        for bx, by, bt in loader:
            bx, by, bt = bx.to(device), by.to(device), bt.to(device); optimizer.zero_grad(); pred = model(bx)
            core = slice(256, -256); p, t, teacher_y = pred[..., core], by[..., core], bt[..., core]
            denom = torch.sum(t.square()) + 1e-24
            truth = torch.sum((p-t).square()) / denom; distill = torch.sum((p-teacher_y).square()) / denom
            spec_loss, _ = penalty(pred, by, -45.5); guard = torch.relu(10*torch.log10(truth+1e-24)+40).square()
            loss = 0.5*truth + distill + args.spectral_weight*spec_loss + 5e-3*guard
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); loss_sum += float(loss.detach())
        scheduler.step()
        if epoch % args.eval_every and epoch != args.epochs: continue
        metrics = paper.evaluate_metrics(evaluator, model, x_val, y_val, device); rows.append({"epoch":epoch,"loss":loss_sum/len(loader),**metrics})
        print(f"epoch={epoch:03d} {metrics}")
        if paper.candidate_key(metrics, -40.0) < paper.candidate_key(best, -40.0): best=metrics; torch.save(model.state_dict(), best_path)
    with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    row={"multipliers":MULTIPLIERS,**best,"checkpoint":str(best_path)}
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    print("BEST",best,best_path)


if __name__ == "__main__":
    main()
