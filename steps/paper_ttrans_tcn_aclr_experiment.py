"""Standalone TCN/Transformer DPD experiment inspired by two 2025 papers.

No existing training or release source is modified. Results are written under
eval_outputs_direct/paper_ttrans_tcn_experiments.
"""

import argparse
import csv
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "paper_ttrans_tcn_experiments"
EVALUATOR_PATH = ROOT / "steps" / "final_evaluate_unified.py"
METRIC_PATH = ROOT / "steps" / "aclr_three_metric_experiment.py"
DEFAULT_TEACHER = ROOT / "eval_outputs_direct" / "aclr_three_metric_experiments" / "aclr_teacher32" / "teacher32" / "best.pth"
DEFAULT_INITS = {
    "ttrans_probe_2532": ROOT / "checkpoints" / "AI-DPD" / "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth",
    "ttrans_probe_2376": ROOT / "checkpoints" / "AI-DPD" / "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth",
    "ttrans_2284": ROOT / "checkpoints" / "AI-DPD" / "final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth",
    "ttrans_998": ROOT / "checkpoints" / "AI-DPD" / "final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth",
    "ttrans_786": ROOT / "checkpoints" / "AI-DPD" / "step800_scratch_h10_l4_base6_rank3_fir41_786.pth",
    "ttrans_796": ROOT / "checkpoints" / "AI-DPD" / "final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Variant:
    hidden: int
    layers: int
    attn_dim: int
    heads: int
    fir_taps: int
    input_channels: int
    local_attention_channels: int
    use_fifth_order: bool
    nmse_target_db: float
    max_multipliers: int


VARIANTS = {
    "ttrans_probe_2532": Variant(15, 6, 4, 1, 43, 9, 3, True, -40.0, 2532),
    "ttrans_probe_2376": Variant(15, 6, 2, 1, 43, 9, 3, True, -40.0, 2376),
    "ttrans_2284": Variant(15, 6, 2, 1, 21, 9, 3, False, -40.0, 2284),
    "ttrans_998": Variant(9, 4, 4, 1, 39, 9, 0, True, -37.0, 998),
    "ttrans_786": Variant(8, 4, 4, 1, 25, 6, 0, True, -35.0, 786),
    "ttrans_796": Variant(9, 3, 4, 1, 33, 6, 0, True, -35.0, 796),
}


def multiplier_count(v):
    return (
        (12 if v.use_fifth_order else 8)
        + v.input_channels * v.hidden
        + v.layers * (5 * v.hidden + v.hidden * v.hidden)
        + 2 * v.hidden * v.attn_dim
        + 8 * v.attn_dim * v.attn_dim
        + 2 * v.hidden
        + 3 * v.local_attention_channels * v.hidden
        + 4 * v.fir_taps
    )


class SignalFrames(Dataset):
    def __init__(self, x, y, teacher, length, stride):
        self.x = torch.as_tensor(x, dtype=torch.float64)
        self.y = torch.as_tensor(y, dtype=torch.float64)
        self.teacher = torch.as_tensor(teacher, dtype=torch.float64)
        self.starts = list(range(0, len(x) - length + 1, stride))
        self.length = length

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        stop = start + self.length
        return self.x[start:stop].T, self.y[start:stop].T, self.teacher[start:stop].T


class NoncausalDSBlock(nn.Module):
    """Noncausal dilated depthwise-separable block with a smooth SiLU."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.act = nn.SiLU()
        self.depthwise = nn.Conv1d(
            channels, channels, 5, padding=2 * dilation, dilation=dilation,
            groups=channels, bias=False, dtype=torch.float64,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False, dtype=torch.float64)

    def forward(self, x):
        return x + self.act(self.pointwise(self.depthwise(x)))


class PooledTransformerContext(nn.Module):
    """Bottleneck self-attention over pooled tokens for affordable global memory."""

    def __init__(self, channels, embed_dim, heads, pool_stride=16):
        super().__init__()
        self.pool_stride = pool_stride
        self.to_tokens = nn.Conv1d(channels, embed_dim, 1, bias=False, dtype=torch.float64)
        self.norm1 = nn.LayerNorm(embed_dim, dtype=torch.float64)
        self.attn = nn.MultiheadAttention(
            embed_dim, heads, batch_first=True, bias=False, dtype=torch.float64
        )
        self.norm2 = nn.LayerNorm(embed_dim, dtype=torch.float64)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim, bias=False, dtype=torch.float64),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim, bias=False, dtype=torch.float64),
        )
        self.to_channels = nn.Conv1d(embed_dim, channels, 1, bias=False, dtype=torch.float64)
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

    def forward(self, x):
        tokens = F.avg_pool1d(self.to_tokens(x), self.pool_stride, self.pool_stride)
        tokens = tokens.transpose(1, 2)
        normed = self.norm1(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        context = self.to_channels(tokens.transpose(1, 2))
        context = F.interpolate(context, size=x.shape[-1], mode="linear", align_corners=False)
        return x + torch.tanh(self.gate) * context


class PaperTTransDPD(nn.Module):
    def __init__(self, variant, init_gain):
        super().__init__()
        self.variant = variant
        self.g1_real = nn.Parameter(torch.tensor(init_gain.real, dtype=torch.float64))
        self.g1_imag = nn.Parameter(torch.tensor(init_gain.imag, dtype=torch.float64))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if variant.use_fifth_order:
            self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.input_projection = nn.Conv1d(
            variant.input_channels, variant.hidden, 1, bias=False, dtype=torch.float64
        )
        self.input_activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            [NoncausalDSBlock(variant.hidden, 2**i) for i in range(variant.layers)]
        )
        self.context = PooledTransformerContext(
            variant.hidden, variant.attn_dim, variant.heads
        )
        self.local_attention = None
        if variant.local_attention_channels:
            self.local_attention = nn.Conv1d(
                variant.local_attention_channels,
                variant.hidden,
                3,
                padding=1,
                bias=False,
                dtype=torch.float64,
            )
        self.output_projection = nn.Conv1d(
            variant.hidden, 2, 1, bias=False, dtype=torch.float64
        )
        nn.init.zeros_(self.output_projection.weight)
        fir_real = torch.zeros(variant.fir_taps, dtype=torch.float64)
        fir_imag = torch.zeros(variant.fir_taps, dtype=torch.float64)
        fir_real[variant.fir_taps // 2] = 1.0
        self.fir_real = nn.Parameter(fir_real)
        self.fir_imag = nn.Parameter(fir_imag)

    def base_forward(self, x):
        i, q = x[:, 0:1], x[:, 1:2]
        amp = torch.sqrt(i.square() + q.square() + 1e-12)
        amp2, amp3, amp4, amp5, amp7 = amp.square(), amp.pow(3), amp.pow(4), amp.pow(5), amp.pow(7)
        if self.variant.input_channels in (9, 11):
            i_future, q_future = torch.roll(i, -1, 2), torch.roll(q, -1, 2)
            i_past, q_past = torch.roll(i, 1, 2), torch.roll(q, 1, 2)
            rich9 = [i, q, i_future, q_future, i_past, q_past, amp, amp3, amp5]
            features = torch.cat(
                rich9 + ([i / amp, q / amp] if self.variant.input_channels == 11 else []), dim=1
            )
        else:
            features = torch.cat([i, q, amp, amp3, i / amp, q / amp], dim=1)
        lin_i = i * self.g1_real - q * self.g1_imag
        lin_q = i * self.g1_imag + q * self.g1_real
        x3_i, x3_q, x5_i, x5_q = i * amp2, q * amp2, i * amp4, q * amp4
        lin_i = lin_i + x3_i * self.g3_real - x3_q * self.g3_imag
        lin_q = lin_q + x3_i * self.g3_imag + x3_q * self.g3_real
        if self.variant.use_fifth_order:
            lin_i = lin_i + x5_i * self.g5_real - x5_q * self.g5_imag
            lin_q = lin_q + x5_i * self.g5_imag + x5_q * self.g5_real
        hidden = self.input_activation(self.input_projection(features))
        for block in self.blocks:
            hidden = block(hidden)
        if self.local_attention is not None:
            if self.variant.local_attention_channels == 3:
                environment = torch.cat([amp, amp3, amp7], dim=1)
            else:
                environment = torch.cat([amp, amp3, amp5, amp7], dim=1)
            hidden = hidden * torch.sigmoid(self.local_attention(environment))
        hidden = self.context(hidden)
        return torch.cat([lin_i, lin_q], dim=1) + self.output_projection(hidden)

    def forward(self, x):
        base = self.base_forward(x)
        pad = len(self.fir_real) // 2
        real, imag = self.fir_real.view(1, 1, -1), self.fir_imag.view(1, 1, -1)
        i, q = base[:, 0:1], base[:, 1:2]
        return torch.cat([
            F.conv1d(i, real, padding=pad) - F.conv1d(q, imag, padding=pad),
            F.conv1d(i, imag, padding=pad) + F.conv1d(q, real, padding=pad),
        ], dim=1)


def initialize_from_existing(model, state):
    """Shape-aware migration; leaves the new Transformer at an identity start."""
    h = model.variant.hidden
    with torch.no_grad():
        for new_name, old_name in [
            ("g1_real", "g_real"), ("g1_imag", "g_imag"),
            ("g3_real", "g3_real"), ("g3_imag", "g3_imag"),
            ("g5_real", "g5_real"), ("g5_imag", "g5_imag"),
        ]:
            if old_name in state and hasattr(model, new_name):
                getattr(model, new_name).copy_(state[old_name])
        model.input_projection.weight.zero_()
        old_in = state["in_conv.weight"]
        copied_inputs = min(old_in.shape[1], model.input_projection.weight.shape[1])
        model.input_projection.weight[:h, :copied_inputs].copy_(old_in[:h, :copied_inputs])
        for index, block in enumerate(model.blocks):
            depth_key = f"tcn_blocks.{index}.0.weight"
            point_key = f"tcn_blocks.{index}.1.weight"
            if depth_key in state:
                block.depthwise.weight.copy_(state[depth_key][:h])
            if point_key in state and state[point_key].shape[0] >= h and state[point_key].shape[1] >= h:
                block.pointwise.weight.copy_(state[point_key][:h, :h])
        model.output_projection.weight.copy_(state["out_conv.weight"][:, :h])
        if model.local_attention is not None and "attn.env_conv.weight" in state:
            old_attention = state["attn.env_conv.weight"]
            channels = min(old_attention.shape[1], model.local_attention.weight.shape[1])
            model.local_attention.weight[:, :channels].copy_(old_attention[:h, :channels])
        if len(model.fir_real) == len(state["fir_real"]):
            model.fir_real.copy_(state["fir_real"])
            model.fir_imag.copy_(state["fir_imag"])
        else:
            model.fir_real.zero_()
            model.fir_imag.zero_()
            old_center = len(state["fir_real"]) // 2
            new_center = len(model.fir_real) // 2
            radius = min(old_center, new_center)
            model.fir_real[new_center-radius:new_center+radius+1].copy_(
                state["fir_real"][old_center-radius:old_center+radius+1]
            )
            model.fir_imag[new_center-radius:new_center+radius+1].copy_(
                state["fir_imag"][old_center-radius:old_center+radius+1]
            )
        model.context.gate.zero_()


def run_full_sequence(model, x, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x.T, dtype=torch.float64, device=device).unsqueeze(0))
    pred = pred.squeeze(0).cpu().numpy()
    target = y.T
    nmse = np.sum((pred - target) ** 2) / np.sum(target**2)
    return pred, target, float(nmse), float(10 * np.log10(nmse))


def evaluate_metrics(evaluator, model, x, y, device):
    pred_iq, target_iq, nmse, nmse_db = run_full_sequence(model, x, y, device)
    target = evaluator.tensor_to_complex(target_iq)
    pred = evaluator.tensor_to_complex(pred_iq)
    freq, target_psd = evaluator.psd_db(target, 491.52e6, 4096)
    _, error_psd = evaluator.psd_db(target - pred, 491.52e6, 4096)
    full = evaluator.aclr_from_psd(
        freq, error_psd, target_psd, (20, 80), (-180, -120), (120, 180)
    )
    point = evaluator.aclr_from_psd_minimum(
        freq, error_psd, target_psd, (20, 80), (-180, -120), (120, 180)
    )
    narrow = evaluator.aclr_from_psd(
        freq, error_psd, target_psd,
        evaluator.NARROW_MAIN_BAND_MHZ,
        evaluator.NARROW_LOWER_BAND_MHZ,
        evaluator.NARROW_UPPER_BAND_MHZ,
    )
    return {
        "nmse_linear": nmse,
        "nmse_db": nmse_db,
        "full_worst": float(full[2]),
        "point_worst": float(point[2]),
        "narrow_worst": float(narrow[2]),
        "three_metric_worst": float(max(full[2], point[2], narrow[2])),
    }


def candidate_key(metrics, nmse_target):
    violation = max(0.0, metrics["nmse_db"] - nmse_target)
    return (
        violation > 0,
        violation,
        max(0.0, metrics["three_metric_worst"] + 45),
        metrics["three_metric_worst"],
        metrics["nmse_db"],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="ttrans_2284")
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=4096)
    parser.add_argument("--frame-stride", type=int, default=2048)
    parser.add_argument("--edge-trim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--truth-weight", type=float, default=0.5)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--spectral-weight", type=float, default=3e-4)
    parser.add_argument("--nmse-guard-weight", type=float, default=1e-3)
    parser.add_argument("--aclr-target-db", type=float, default=-45.5)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--train-context-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    evaluator = load_module("final_evaluate_unified", EVALUATOR_PATH)
    metric_exp = load_module("aclr_three_metric_experiment", METRIC_PATH)
    device = evaluator.DEVICE
    variant = VARIANTS[args.variant]
    multipliers = multiplier_count(variant)
    if multipliers > variant.max_multipliers:
        raise ValueError(f"Multiplier budget exceeded: {multipliers}>{variant.max_multipliers}")

    x, y, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split = int(len(x) * 0.8)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    x_complex = x_train[:, 0] + 1j * x_train[:, 1]
    y_complex = y_train[:, 0] + 1j * y_train[:, 1]
    gain = np.vdot(x_complex, y_complex) / np.vdot(x_complex, x_complex)

    teacher_state = evaluator.load_state_dict(args.teacher_checkpoint)
    teacher_model = evaluator.AutoDiagramDPDNet(teacher_state).to(device).eval()
    teacher_train, _, _, _ = evaluator.evaluate_full_sequence(teacher_model, x_train, y_train)
    teacher_train = teacher_train.T
    model = PaperTTransDPD(variant, gain).to(device)
    init_checkpoint = args.init_checkpoint or DEFAULT_INITS[args.variant]
    initialize_from_existing(model, evaluator.load_state_dict(init_checkpoint))
    print(f"Partial architecture migration: {init_checkpoint}")
    if args.train_context_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.context.parameters():
            parameter.requires_grad = True
        print("Training only the new pooled Transformer context")
    penalty = metric_exp.SpectralPenalty(y_train, 1024, args.edge_trim, 491.52, device)
    loader = DataLoader(
        SignalFrames(x_train, y_train, teacher_train, args.frame_length, args.frame_stride),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    out_dir = OUT_ROOT / args.tag / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path, best_path = out_dir / "history.csv", out_dir / "best.pth"
    best = evaluate_metrics(evaluator, model, x_val, y_val, device)
    torch.save(model.state_dict(), best_path)
    print(
        f"variant={args.variant} multipliers={multipliers} initial "
        f"NMSE={best['nmse_db']:.2f} ACLR={best['full_worst']:.2f}/"
        f"{best['point_worst']:.2f}/{best['narrow_worst']:.2f}"
    )
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y, batch_teacher in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_teacher = batch_teacher.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            core = slice(args.edge_trim, -args.edge_trim)
            core_pred, core_y, core_teacher = pred[..., core], batch_y[..., core], batch_teacher[..., core]
            denominator = torch.sum(core_y.square()) + 1e-24
            truth_nmse = torch.sum((core_pred - core_y).square()) / denominator
            teacher_nmse = torch.sum((core_pred - core_teacher).square()) / denominator
            spectral_loss, _ = penalty(pred, batch_y, args.aclr_target_db)
            nmse_db = 10 * torch.log10(truth_nmse + 1e-24)
            guard = torch.relu(nmse_db - variant.nmse_target_db).square()
            loss = (
                args.truth_weight * truth_nmse
                + args.teacher_weight * teacher_nmse
                + args.spectral_weight * spectral_loss
                + args.nmse_guard_weight * guard
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        scheduler.step()
        if epoch % args.eval_every and epoch != args.epochs:
            continue
        metrics = evaluate_metrics(evaluator, model, x_val, y_val, device)
        rows.append({"epoch": epoch, "loss": epoch_loss / len(loader), **metrics})
        print(
            f"epoch={epoch:03d} NMSE={metrics['nmse_db']:.3f} "
            f"ACLR={metrics['full_worst']:.2f}/{metrics['point_worst']:.2f}/"
            f"{metrics['narrow_worst']:.2f} lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if candidate_key(metrics, variant.nmse_target_db) < candidate_key(best, variant.nmse_target_db):
            best = metrics
            torch.save(model.state_dict(), best_path)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary_path = out_dir / "summary.csv"
    row = {"variant": args.variant, "multipliers": multipliers, **best, "checkpoint": str(best_path)}
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    print(f"BEST {best} | {best_path}")


if __name__ == "__main__":
    main()
