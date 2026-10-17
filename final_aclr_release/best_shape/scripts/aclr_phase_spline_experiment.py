"""Phase-normalized spline-LUT output correction for EA-TCN (architecture experiment).

Adds an APNKAN-inspired correction branch to a frozen/loaded EA-TCN release model:

    r(n)   = conj(x(n)) / |x(n)|                       (phase normalization)
    u_m(n) = x(n+m) * r(n),  m in --offsets            (rotated memory features)
    feats  = [|x(n)|] + [Re u_m, Im u_m for m != 0]    (n_feat real features)
    o_k(n) = sum_i  spline_ik(feat_i(n)),  k = I,Q     (first-order spline LUT edges)
    corr   = (o_I + j o_Q) * conj(r(n))                (phase restore)
    y_pre  = EA-TCN_pre_fir(x) + corr;  y = FIR(y_pre) (correction feeds the output FIR)

The branch cost is paid inside the MAC budget by shrinking the output FIR:
    KAN rule (APNKAN report): 2*n_feat + n_edges ; phase restore: 4 ;
    neighbor rotations (strict): 4 per non-zero offset.
Default --taps-reduction 14 frees 4*14 = 56 MACs which covers the default branch
(9 feats -> 2*9 + 18 + 4 + 16 = 56).  Spline tables start at zero, so the wrapped
model is exactly the truncated+refit base at epoch 0.  Training: progressive
sub-band (HPF error) loss + NMSE guard + periodic closed-form FIR refit.
Release checkpoints are never modified.
"""

import argparse
import csv
import importlib.util
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "eval_outputs_direct" / "aclr_ref_convention"


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhaseSplineCorrection(nn.Module):
    def __init__(self, offsets, grid, amp_max):
        super().__init__()
        self.offsets = [int(m) for m in offsets]
        if 0 not in self.offsets:
            raise ValueError("--offsets must include 0 (envelope feature)")
        self.grid = int(grid)
        self.n_feat = 1 + 2 * (len(self.offsets) - 1)
        gmin = [0.0] + [-amp_max] * (2 * (len(self.offsets) - 1))
        gmax = [amp_max] * self.n_feat
        self.register_buffer("gmin", torch.tensor(gmin, dtype=torch.float64))
        self.register_buffer("gmax", torch.tensor(gmax, dtype=torch.float64))
        self.tables = nn.Parameter(torch.zeros(self.n_feat, 2, self.grid + 1, dtype=torch.float64))

    @property
    def n_edges(self):
        return self.n_feat * 2

    def mac_cost(self, strict=True):
        kan = 2 * self.n_feat + self.n_edges
        restore = 4
        rotations = 4 * (len(self.offsets) - 1) if strict else 0
        return kan + restore + rotations

    def forward(self, x_iq):
        xc = torch.complex(x_iq[:, 0], x_iq[:, 1])
        amp = xc.abs()
        r = xc.conj() / (amp + 1e-12)
        feats = [amp]
        for m in self.offsets:
            if m == 0:
                continue
            um = torch.roll(xc, shifts=-m, dims=-1) * r
            feats.append(um.real)
            feats.append(um.imag)
        f = torch.stack(feats, dim=1)  # (B, n_feat, T)
        delta = (self.gmax - self.gmin) / self.grid
        v = (f - self.gmin[None, :, None]) / delta[None, :, None]
        v = v.clamp(0.0, float(self.grid))
        q = v.floor().clamp(0.0, float(self.grid - 1))
        t = (v - q).unsqueeze(2)  # (B, n_feat, 1, T)
        idx = q.long().unsqueeze(2).expand(-1, -1, 2, -1)  # (B, n_feat, 2, T)
        tab = self.tables.unsqueeze(0).expand(idx.shape[0], -1, -1, -1)
        c0 = torch.take_along_dim(tab, idx, dim=3)
        c1 = torch.take_along_dim(tab, idx + 1, dim=3)
        s = c0 + t * (c1 - c0)
        o = s.sum(dim=1)  # (B, 2, T)
        oc = torch.complex(o[:, 0], o[:, 1]) * r.conj()
        return torch.stack([oc.real, oc.imag], dim=1)


class PhaseSplineDPD(nn.Module):
    """EA-TCN base (pre-FIR) + phase-spline correction + reduced complex output FIR."""

    def __init__(self, evaluator, state, taps_reduction, offsets, grid, amp_max):
        super().__init__()
        self.base = evaluator.AutoDiagramDPDNet(state)
        if not self.base.has_fir:
            raise ValueError("base checkpoint has no output FIR")
        self.base.has_fir = False  # wrapper owns the FIR
        orig = state["fir_real"].numel()
        new_taps = orig - int(taps_reduction)
        if new_taps < 5 or new_taps % 2 == 0:
            raise ValueError(f"invalid reduced tap count {new_taps} (from {orig})")
        lo = (orig - new_taps) // 2
        self.fir_real = nn.Parameter(state["fir_real"][lo : lo + new_taps].clone().double())
        self.fir_imag = nn.Parameter(state["fir_imag"][lo : lo + new_taps].clone().double())
        self.corr = PhaseSplineCorrection(offsets, grid, amp_max)
        self.has_fir = True
        self.orig_taps = orig
        self.new_taps = new_taps

    def forward(self, x_iq):
        y = self.base(x_iq) + self.corr(x_iq)
        if not self.has_fir:
            return y
        pad = len(self.fir_real) // 2
        yi, yq = y[:, 0:1, :], y[:, 1:2, :]
        kr = self.fir_real.view(1, 1, -1)
        ki = self.fir_imag.view(1, 1, -1)
        out_i = nn.functional.conv1d(yi, kr, padding=pad) - nn.functional.conv1d(yq, ki, padding=pad)
        out_q = nn.functional.conv1d(yi, ki, padding=pad) + nn.functional.conv1d(yq, kr, padding=pad)
        return torch.cat([out_i, out_q], dim=1)

    def mac_report(self, evaluator, state, strict=True):
        base_total = evaluator.real_multiplier_count(state)
        base_no_fir = base_total - 4 * self.orig_taps
        fir = 4 * self.new_taps
        corr = self.corr.mac_cost(strict=strict)
        return {
            "base_no_fir": base_no_fir,
            "fir": fir,
            "corr": corr,
            "total": base_no_fir + fir + corr,
            "orig_total": base_total,
        }


def refit_fir(evaluator, sweep_mod, metrics_mod, model, x_train, target_train_c, adjacent_weight, floor_weight, ridge, fs_mhz):
    base_train = sweep_mod.run_base_sequence(evaluator, model, x_train)
    designer = sweep_mod.FirDesigner(base_train, target_train_c, len(model.fir_real), fs_mhz)
    w = designer.weights(metrics_mod, adjacent_weight, floor_weight)
    coeffs = designer.solve(w, ridge)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=torch.float64, device=evaluator.DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=torch.float64, device=evaluator.DEVICE))


def evaluate(evaluator, metrics_mod, model, x_val, y_val, nperseg):
    pred, target, _, _ = evaluator.evaluate_full_sequence(model.eval(), x_val, y_val)
    pred_c = evaluator.tensor_to_complex(pred)
    target_c = evaluator.tensor_to_complex(target)
    metrics = metrics_mod.ref_metrics(evaluator, pred_c, target_c, nperseg)
    freq, target_psd = evaluator.psd_db(target_c, metrics_mod.FS_HZ, nperseg)
    _, error_psd = evaluator.psd_db(target_c - pred_c, metrics_mod.FS_HZ, nperseg)
    lower, upper, worst = evaluator.aclr_from_psd_minimum(
        freq, error_psd, target_psd, (20.0, 80.0), (-180.0, -120.0), (120.0, 180.0)
    )
    metrics.update(
        {
            "point_lower_db": float(lower),
            "point_upper_db": float(upper),
            "point_worst_db": float(worst),
        }
    )
    return metrics


def selection_value(metrics, select_metric):
    return metrics["point_worst_db"] if select_metric == "point" else metrics["aclr_db"]


def scratch_init(model, x_train, y_train):
    """Re-initialize the wrapper for a from-scratch progressive trajectory.

    Mirrors the original recipe: order-1 complex gain from closed-form LS,
    higher poly orders zero, residual pointwise convs and out_conv zero
    (blocks start as identity), spline tables zero, FIR = unit impulse
    (immediately re-fitted in closed form by the fresh-start path).
    """
    base = model.base
    with torch.no_grad():
        xc = torch.as_tensor(x_train[:, 0] + 1j * x_train[:, 1], dtype=torch.complex128)
        yc = torch.as_tensor(y_train[:, 0] + 1j * y_train[:, 1], dtype=torch.complex128)
        g = (torch.sum(xc.conj() * yc) / torch.sum(xc.abs().square())).item()
        base.g_real.fill_(g.real)
        base.g_imag.fill_(g.imag)
        if base.has_g3:
            base.g3_real.zero_()
            base.g3_imag.zero_()
        if base.has_g5:
            base.g5_real.zero_()
            base.g5_imag.zero_()
        for key in base.extra_poly_real:
            base.extra_poly_real[key].zero_()
            base.extra_poly_imag[key].zero_()
        base.in_conv.reset_parameters()
        for block in base.tcn_blocks:
            convs = [m for m in block if isinstance(m, nn.Conv1d)]
            for conv in convs:
                conv.reset_parameters()
            convs[-1].weight.zero_()
        base.attn.env_conv.reset_parameters()
        base.out_conv.weight.zero_()
        model.corr.tables.zero_()
        model.fir_real.zero_()
        model.fir_imag.zero_()
        model.fir_real[len(model.fir_real) // 2] = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tag", default="phase_spline")
    parser.add_argument("--taps-reduction", type=int, default=14)
    parser.add_argument("--mac-limit", type=int, default=None,
                        help="override the MAC ceiling (default: original checkpoint MAC)")
    parser.add_argument("--offsets", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--train-scope", choices=["corr", "all"], default="corr")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=4096)
    parser.add_argument("--frame-stride", type=int, default=1024)
    parser.add_argument("--edge-crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="lr for spline tables")
    parser.add_argument("--base-lr", type=float, default=3e-5, help="lr for base when scope=all")
    parser.add_argument("--lr-gamma", type=float, default=0.995)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--r-max", type=float, default=0.995)
    parser.add_argument("--ramp-epochs", type=int, default=20)
    parser.add_argument("--hpf-taps", type=int, default=257)
    parser.add_argument("--hpf-cutoff-mhz", type=float, default=98.0)
    parser.add_argument("--nmse-guard", type=float, required=True)
    parser.add_argument("--nmse-select", type=float, default=None,
                        help="val NMSE bound for model selection (default: guard)")
    parser.add_argument("--select-metric", choices=["ref", "point"], default="ref",
                        help="select best checkpoint by integrated (ref) or lowest-point ACLR")
    parser.add_argument("--init", choices=["checkpoint", "scratch"], default="checkpoint",
                        help="scratch = random re-init (LS order-1 gain, zero residuals) for a "
                             "full progressive trajectory instead of fine-tuning")
    parser.add_argument("--guard-weight", type=float, default=3e-2)
    parser.add_argument("--init-adjacent-weight", type=float, default=1.0)
    parser.add_argument("--refit-every", type=int, default=10)
    parser.add_argument("--refit-adjacent-weight", type=float, default=10.0)
    parser.add_argument("--refit-floor-weight", type=float, default=1.0)
    parser.add_argument("--refit-ridge", type=float, default=1e-8)
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--fs-mhz", type=float, default=491.52)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--smoke", action="store_true", help="setup + sanity checks only")
    args = parser.parse_args()
    nmse_select = args.nmse_select if args.nmse_select is not None else args.nmse_guard

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    evaluator = load_module("final_evaluate_unified", "steps/final_evaluate_unified.py")
    metrics_mod = load_module("aclr_ref_metrics", "steps/aclr_ref_metrics.py")
    sweep_mod = load_module("aclr_ref_fir_sweep", "steps/aclr_ref_fir_sweep.py")
    ft_mod = load_module("aclr_ref_subband_finetune", "steps/aclr_ref_subband_finetune.py")

    features, targets, _, *_ = evaluator.load_iq_arrays(ROOT / "data" / "DPD_data.mat", "rms")
    split_idx = int(len(features) * args.split)
    x_train, y_train = features[:split_idx], targets[:split_idx]
    x_val, y_val = features[split_idx:], targets[split_idx:]
    target_train_c = y_train[:, 0] + 1j * y_train[:, 1]
    amp_max = 1.05 * float(np.max(np.hypot(features[:, 0], features[:, 1])))

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
    else:
        checkpoint_path = ROOT / metrics_mod.MODEL_SPECS[args.model][0]
    state = evaluator.load_state_dict(checkpoint_path)

    wrapper_resume = any(key.startswith("base.") for key in state)
    if wrapper_resume:
        base_state = {key[len("base."):]: value for key, value in state.items() if key.startswith("base.")}
        model = PhaseSplineDPD(evaluator, base_state, args.taps_reduction, args.offsets, args.grid, amp_max)
        model.load_state_dict(state, strict=True)
        mac_state = base_state
    else:
        model = PhaseSplineDPD(evaluator, state, args.taps_reduction, args.offsets, args.grid, amp_max)
        mac_state = state
    model = model.to(evaluator.DEVICE)
    if args.init == "scratch":
        if wrapper_resume:
            raise SystemExit("--init scratch requires a release-format --checkpoint (shapes only)")
        scratch_init(model, x_train, y_train)
    macs = model.mac_report(evaluator, mac_state, strict=True)
    mac_limit = args.mac_limit if args.mac_limit is not None else macs["orig_total"]
    print(
        f"[{args.model}] MAC: base(no FIR) {macs['base_no_fir']} + FIR {macs['fir']} "
        f"({model.orig_taps}->{model.new_taps} taps) + corr {macs['corr']} = {macs['total']} "
        f"(limit {mac_limit}, original {macs['orig_total']})"
    )
    if macs["total"] > mac_limit:
        raise SystemExit("MAC budget exceeded — increase --taps-reduction")

    # NMSE-optimal refit of the reduced FIR (fresh starts only), then sanity report.
    if not wrapper_resume:
        refit_fir(evaluator, sweep_mod, metrics_mod, model.eval(), x_train, target_train_c,
                  args.init_adjacent_weight, 1.0, args.refit_ridge, args.fs_mhz)
    start = evaluate(evaluator, metrics_mod, model, x_val, y_val, args.nperseg)
    print(
        f"[{args.model}] truncated+refit start: NMSE {start['nmse_db']:.3f} | "
        f"ACLR {start['aclr_db']:.2f} (L {start['aclr_lower_db']:.2f} / U {start['aclr_upper_db']:.2f}) | "
        f"point {start['point_worst_db']:.2f}"
    )
    if args.smoke:
        return

    for p in model.parameters():
        p.requires_grad = False
    model.corr.tables.requires_grad = True
    groups = [{"params": [model.corr.tables], "lr": args.lr}]
    if args.train_scope == "all":
        base_params = [p for p in model.base.parameters()]
        for p in base_params:
            p.requires_grad = True
        model.fir_real.requires_grad = True
        model.fir_imag.requires_grad = True
        groups.append({"params": base_params + [model.fir_real, model.fir_imag], "lr": args.base_lr})

    hpf = ft_mod.build_hpf(args.hpf_taps, args.hpf_cutoff_mhz, args.fs_mhz, evaluator.DEVICE)
    tau = args.ramp_epochs / (-math.log(1.0 - min(args.r_max, 1.0 - 1e-9)))
    loader = torch.utils.data.DataLoader(
        ft_mod.Frames(x_train, y_train, args.frame_length, args.frame_stride),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    crop = args.edge_crop

    output_dir = OUT_ROOT / args.tag / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    best = start if start["nmse_db"] <= nmse_select else None
    best_path = output_dir / "best.pth"
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, best_path)

    with (output_dir / "history.csv").open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "r", "loss", "nmse_db", "aclr_db", "aclr_lower_db", "aclr_upper_db",
                        "point_lower_db", "point_upper_db", "point_worst_db"],
        )
        writer.writeheader()
        writer.writerow({"epoch": 0, "r": 0.0, "loss": float("nan"), **start})
        handle.flush()
        for epoch in range(1, args.epochs + 1):
            r = min(args.r_max, 1.0 - math.exp(-epoch / tau))
            model.train()
            loss_sum = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(evaluator.DEVICE)
                batch_y = batch_y.to(evaluator.DEVICE)
                optimizer.zero_grad()
                pred = model(batch_x)
                err = (pred - batch_y)[:, :, crop:-crop]
                tgt = batch_y[:, :, crop:-crop]
                loss_fb = err.square().mean()
                err_hp = nn.functional.conv1d(err, hpf, groups=2)
                loss = (1.0 - r) * loss_fb + r * err_hp.square().mean()
                nmse_db = 10.0 * torch.log10(err.square().sum() / (tgt.square().sum() + 1e-24) + 1e-24)
                loss = loss + args.guard_weight * torch.relu(nmse_db - args.nmse_guard).square()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.clip_grad)
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
            scheduler.step()

            if args.refit_every > 0 and epoch % args.refit_every == 0:
                refit_fir(evaluator, sweep_mod, metrics_mod, model.eval(), x_train, target_train_c,
                          args.refit_adjacent_weight, args.refit_floor_weight, args.refit_ridge, args.fs_mhz)

            if epoch % args.eval_every != 0 and epoch != args.epochs:
                continue
            metrics = evaluate(evaluator, metrics_mod, model, x_val, y_val, args.nperseg)
            torch.save(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                output_dir / "last.pth",
            )
            writer.writerow({"epoch": epoch, "r": r, "loss": loss_sum / max(1, len(loader)), **metrics})
            handle.flush()
            print(
                f"[{args.model}] ep {epoch:03d} r={r:.4f} NMSE {metrics['nmse_db']:.3f} | "
                f"ACLR {metrics['aclr_db']:.2f} (L {metrics['aclr_lower_db']:.2f} / "
                f"U {metrics['aclr_upper_db']:.2f}) | point {metrics['point_worst_db']:.2f}"
            )
            if metrics["nmse_db"] <= nmse_select and (
                best is None
                or selection_value(metrics, args.select_metric) < selection_value(best, args.select_metric)
            ):
                best = metrics
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, best_path)

    if best is None:
        print(f"[{args.model}] no candidate satisfied NMSE <= {nmse_select}")
    else:
        print(
            f"[{args.model}] best (NMSE <= {nmse_select}, by {args.select_metric}): "
            f"NMSE {best['nmse_db']:.3f} | ACLR {best['aclr_db']:.2f} | "
            f"point {best['point_worst_db']:.2f} -> {best_path}"
        )


if __name__ == "__main__":
    main()
