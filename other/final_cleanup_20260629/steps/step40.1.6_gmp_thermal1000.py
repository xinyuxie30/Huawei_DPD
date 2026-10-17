import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "EA-TCN_gmp_thermal1000_best.pth"
DEFAULT_INIT = CHECKPOINT_DIR / "EA-TCN_memory_fir1000_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


def safe_shift(x, shift):
    if shift == 0:
        return x
    if shift > 0:
        return torch.cat([x[:, :, :1].expand(-1, -1, shift), x[:, :, :-shift]], dim=2)
    shift = -shift
    return torch.cat([x[:, :, shift:], x[:, :, -1:].expand(-1, -1, shift)], dim=2)


def causal_average(x, taps):
    if taps <= 1:
        return x
    kernel = torch.ones(1, 1, taps, dtype=x.dtype, device=x.device) / taps
    padded = F.pad(x, (taps - 1, 0), mode="replicate")
    return F.conv1d(padded, kernel)


class SpectrumAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.env_conv = nn.Conv1d(3, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp, amp3, amp7):
        env = torch.cat([amp, amp3, amp7], dim=1)
        return x * self.sigmoid(self.env_conv(env))


class GMPThermalDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super().__init__()
        self.hidden = 10
        self.output_fir_taps = 43

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        # Four residual branches selected from residual-feature analysis:
        # 1) x[n-4] * |x[n-4]|^6
        # 2) x[n-4] * |x[n-4]|^8
        # 3) x[n-2] * |x[n-4]|^6
        # 4) x[n-4] * MA_8(|x|^6), a short thermal/envelope memory proxy
        self.extra_real = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.extra_imag = nn.Parameter(torch.zeros(4, dtype=torch.float64))

        self.in_conv = nn.Conv1d(9, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList()
        for dilation in [1, 2, 4, 8]:
            self.tcn_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.hidden,
                        self.hidden,
                        5,
                        padding=2 * dilation,
                        dilation=dilation,
                        groups=self.hidden,
                        bias=False,
                        dtype=torch.float64,
                    ),
                    nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                    nn.SiLU(),
                )
            )

        self.attn = SpectrumAttention(self.hidden)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)
        nn.init.constant_(self.out_conv.weight, 0.0)

        center = self.output_fir_taps // 2
        fir_real = torch.zeros(self.output_fir_taps, dtype=torch.float64)
        fir_imag = torch.zeros(self.output_fir_taps, dtype=torch.float64)
        fir_real[center] = 1.0
        self.fir_real = nn.Parameter(fir_real)
        self.fir_imag = nn.Parameter(fir_imag)

    def complex_branch(self, x_i, x_q, envelope, idx):
        gr = self.extra_real[idx]
        gi = self.extra_imag[idx]
        zi = x_i * envelope
        zq = x_q * envelope
        return zi * gr - zq * gi, zi * gi + zq * gr

    def forward_base(self, x_iq):
        i_x = x_iq[:, 0:1, :]
        q_x = x_iq[:, 1:2, :]

        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp4 = amp2 ** 2
        amp6 = amp2 ** 3
        amp7 = amp ** 7
        amp8 = amp4 ** 2

        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        x3_i = i_x * amp2
        x3_q = q_x * amp2
        lin_i = lin_i + x3_i * self.g3_real - x3_q * self.g3_imag
        lin_q = lin_q + x3_i * self.g3_imag + x3_q * self.g3_real

        x5_i = i_x * amp4
        x5_q = q_x * amp4
        lin_i = lin_i + x5_i * self.g5_real - x5_q * self.g5_imag
        lin_q = lin_q + x5_i * self.g5_imag + x5_q * self.g5_real

        i_lag2 = safe_shift(i_x, 2)
        q_lag2 = safe_shift(q_x, 2)
        i_lag4 = safe_shift(i_x, 4)
        q_lag4 = safe_shift(q_x, 4)
        amp_lag4 = safe_shift(amp, 4)
        amp_lag4_6 = amp_lag4 ** 6
        amp_lag4_8 = amp_lag4 ** 8
        thermal_w8_p6 = causal_average(amp6, 8)

        for idx, (bi, bq, env) in enumerate(
            [
                (i_lag4, q_lag4, amp_lag4_6),
                (i_lag4, q_lag4, amp_lag4_8),
                (i_lag2, q_lag2, amp_lag4_6),
                (i_lag4, q_lag4, thermal_w8_p6),
            ]
        ):
            ei, eq = self.complex_branch(bi, bq, env, idx)
            lin_i = lin_i + ei
            lin_q = lin_q + eq

        i_future = safe_shift(i_x, -1)
        q_future = safe_shift(q_x, -1)
        i_past = safe_shift(i_x, 1)
        q_past = safe_shift(q_x, 1)
        features = torch.cat([i_x, q_x, i_future, q_future, i_past, q_past, amp, amp3, amp ** 5], dim=1)

        feat = self.act(self.in_conv(features))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        residual = self.out_conv(self.attn(feat, amp, amp3, amp7))
        return torch.cat([lin_i, lin_q], dim=1) + residual

    def apply_output_fir(self, y):
        pad = self.output_fir_taps // 2
        real = self.fir_real.view(1, 1, -1)
        imag = self.fir_imag.view(1, 1, -1)
        y_i = y[:, 0:1, :]
        y_q = y[:, 1:2, :]
        out_i = F.conv1d(y_i, real, padding=pad) - F.conv1d(y_q, imag, padding=pad)
        out_q = F.conv1d(y_i, imag, padding=pad) + F.conv1d(y_q, real, padding=pad)
        return torch.cat([out_i, out_q], dim=1)

    def forward(self, x_iq):
        return self.apply_output_fir(self.forward_base(x_iq))


def real_multiplier_count():
    return 12 + 16 + 90 + 4 * (10 * 5 + 10 * 10) + 20 + 90 + 4 * 43


def parse_args():
    parser = argparse.ArgumentParser(description="EA-TCN + explicit GMP/thermal branches, 1000 real multipliers.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--epochs-extra", type=int, default=80)
    parser.add_argument("--epochs-base", type=int, default=240)
    parser.add_argument("--epochs-long", type=int, default=360)
    parser.add_argument("--epochs-full", type=int, default=80)
    parser.add_argument("--extra-lr", type=float, default=1e-4)
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--long-lr", type=float, default=1e-6)
    parser.add_argument("--full-lr", type=float, default=5e-7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--long-batch-size", type=int, default=2)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--long-frame-length", type=int, default=8192)
    parser.add_argument("--long-stride-length", type=int, default=512)
    parser.add_argument("--crop-margin", type=int, default=80)
    parser.add_argument("--long-crop-margin", type=int, default=160)
    parser.add_argument("--full-crop-margin", type=int, default=256)
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
    raw_x, raw_y, x_c, y_c = aidpd.load_normalized_data(data_path)
    x_np = aidpd.iq_array(x_c)
    y_np = aidpd.iq_array(y_c)
    split_idx = int(min(len(x_np), len(y_np)) * 0.8)
    g_opt = np.vdot(x_c[:split_idx], y_c[:split_idx]) / np.vdot(x_c[:split_idx], x_c[:split_idx])
    return x_np[:split_idx], y_np[:split_idx], x_np[split_idx:], y_np[split_idx:], float(g_opt.real), float(g_opt.imag)


def make_loader(x_np, y_np, frame_length, stride_length, batch_size):
    dataset = aidpd.FramedDataset(x_np, y_np, frame_length, stride_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())


def run_sequence(model, x_np, base=False):
    x_tensor = torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = model.forward_base(x_tensor) if base else model(x_tensor)
    return out.squeeze(0).cpu().numpy()


def fullseq_nmse(model, x_val, y_val):
    pred = run_sequence(model, x_val)
    target = y_val.T
    nmse = np.sum((pred - target) ** 2) / np.sum(target ** 2)
    return float(nmse), float(10 * np.log10(nmse))


def cropped_mse(pred, target, crop):
    if crop <= 0 or pred.shape[-1] <= 2 * crop:
        return F.mse_loss(pred, target)
    return F.mse_loss(pred[:, :, crop:-crop], target[:, :, crop:-crop])


def set_fir_trainable(model, trainable):
    for name, param in model.named_parameters():
        if name.startswith("fir_"):
            param.requires_grad = trainable


def set_trainable_mode(model, mode):
    for name, param in model.named_parameters():
        if mode == "extra":
            param.requires_grad = name.startswith("extra_")
        elif mode == "base":
            param.requires_grad = not name.startswith("fir_")
        elif mode == "all":
            param.requires_grad = True
        else:
            raise ValueError(f"Unsupported trainable mode: {mode}")


def load_partial_without_fir(model, checkpoint_path):
    source = aidpd.load_state_dict(checkpoint_path)
    target = model.state_dict()
    loaded = 0
    partial = 0
    skipped = 0
    for key, value in source.items():
        if key.startswith("fir_"):
            skipped += 1
            continue
        if key not in target:
            skipped += 1
            continue
        dst = target[key]
        merged = dst.clone()
        if value.shape == dst.shape:
            merged.copy_(value.to(dtype=dst.dtype))
            loaded += 1
        elif value.ndim == dst.ndim:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, dst.shape))
            merged[slices] = value[slices].to(dtype=dst.dtype)
            partial += 1
        else:
            skipped += 1
            continue
        target[key] = merged
    model.load_state_dict(target, strict=True)
    return loaded, partial, skipped


def fit_output_fir_from_train(model, x_train, y_train):
    taps = model.output_fir_taps
    radius = taps // 2
    base_iq = run_sequence(model, x_train, base=True).T
    z = base_iq[:, 0] + 1j * base_iq[:, 1]
    target = y_train[:, 0] + 1j * y_train[:, 1]
    cols = [z[radius + offset : len(z) - radius + offset] for offset in range(-radius, radius + 1)]
    design = np.stack(cols, axis=1)
    rhs = target[radius : len(target) - radius]
    coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=torch.float64, device=DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=torch.float64, device=DEVICE))
    return coeffs


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
    crop_margin,
    checkpoint_path,
    target_fullseq_db,
):
    loader = make_loader(x_train, y_train, frame_length, stride_length, batch_size)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=12, min_lr=5e-7)
    best_nmse, best_db = fullseq_nmse(model, x_val, y_val)
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]
    start = time.time()
    print(f"\n>>> Stage {stage_name}: epochs={epochs}, lr={lr:.2e}, frame={frame_length}, crop={crop_margin}")
    print(f">>> Initial full-sequence NMSE: {best_nmse:.6e} ({best_db:.2f} dB)")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = cropped_mse(pred, batch_y, crop_margin)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        nmse, nmse_db = fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse_db)
        history.append((epoch, nmse_db))
        elapsed = time.time() - start
        print(
            f"{stage_name} Epoch [{epoch}/{epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | Full NMSE {nmse_db:.2f} dB"
        )
        if nmse < best_nmse:
            best_nmse = nmse
            best_db = nmse_db
            torch.save(model.state_dict(), checkpoint_path)
        if best_db <= target_fullseq_db:
            print(f">>> Target reached in {stage_name}: {best_db:.2f} dB")
            break

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    return history, best_nmse, best_db


def train_full_sequence_stage(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    epochs,
    lr,
    crop_margin,
    checkpoint_path,
    target_fullseq_db,
):
    if epochs <= 0:
        nmse, nmse_db = fullseq_nmse(model, x_val, y_val)
        return [(0, nmse_db)], nmse, nmse_db

    x_tensor = torch.as_tensor(x_train.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    y_tensor = torch.as_tensor(y_train.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-7)
    best_nmse, best_db = fullseq_nmse(model, x_val, y_val)
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]
    start = time.time()
    print(f"\n>>> Stage full-sequence: epochs={epochs}, lr={lr:.2e}, crop={crop_margin}")
    print(f">>> Initial full-sequence NMSE: {best_nmse:.6e} ({best_db:.2f} dB)")

    for epoch in range(1, epochs + 1):
        model.train()
        pred = model(x_tensor)
        loss = cropped_mse(pred, y_tensor, crop_margin)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        nmse, nmse_db = fullseq_nmse(model, x_val, y_val)
        scheduler.step(nmse_db)
        history.append((epoch, nmse_db))
        elapsed = time.time() - start
        print(
            f"full-sequence Epoch [{epoch}/{epochs}] | Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | Full NMSE {nmse_db:.2f} dB"
        )
        if nmse < best_nmse:
            best_nmse = nmse
            best_db = nmse_db
            torch.save(model.state_dict(), checkpoint_path)
        if best_db <= target_fullseq_db:
            print(f">>> Target reached in full-sequence: {best_db:.2f} dB")
            break

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    return history, best_nmse, best_db


def plot_history(histories, out_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    offset = 0
    for label, history in histories:
        xs = [offset + item[0] for item in history]
        ys = [item[1] for item in history]
        plt.plot(xs, ys, label=label)
        offset = xs[-1]
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.1, label="-37 dB")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("Full-sequence NMSE (dB)")
    plt.title("EA-TCN GMP/thermal cropped training, 1000 real multipliers")
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
    checkpoint_path = resolve_path(args.checkpoint)
    init_checkpoint = resolve_path(args.init_checkpoint)
    x_train, y_train, x_val, y_val, g_real, g_imag = load_data(data_path)
    model = GMPThermalDPDNet(g_real, g_imag).to(DEVICE)

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Validation samples: {len(x_val)}")
    print(f">>> Real multipliers: {real_multiplier_count()}")
    if real_multiplier_count() > 1000:
        raise RuntimeError(f"Multiplier budget exceeded: {real_multiplier_count()}")

    if args.evaluate_only:
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        nmse, nmse_db = fullseq_nmse(model, x_val, y_val)
        print(f">>> Full-sequence NMSE: {nmse:.6e} ({nmse_db:.2f} dB)")
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
        loaded, partial, skipped = load_partial_without_fir(model, init_checkpoint)
        print(f">>> Partial init without FIR from: {init_checkpoint}")
        print(f">>> Loaded tensors: exact={loaded}, partial={partial}, skipped={skipped}")
    elif args.scratch:
        print(">>> Scratch mode: no checkpoint initialization.")
    else:
        print(f">>> Init checkpoint not found, training fresh: {init_checkpoint}")

    print("\n>>> Initial train-only LS FIR fit.")
    fit_output_fir_from_train(model, x_train, y_train)
    init_nmse, init_db = fullseq_nmse(model, x_val, y_val)
    print(f">>> After initial LS FIR full-sequence NMSE: {init_nmse:.6e} ({init_db:.2f} dB)")

    histories = []
    set_trainable_mode(model, "extra")
    h0, _, _ = train_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        stage_name="extra-branches",
        epochs=args.epochs_extra,
        lr=args.extra_lr,
        frame_length=args.frame_length,
        stride_length=args.stride_length,
        batch_size=args.batch_size,
        crop_margin=args.crop_margin,
        checkpoint_path=checkpoint_path,
        target_fullseq_db=args.target_fullseq_db,
    )
    histories.append(("extra-branches", h0))

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    print("\n>>> Post-extra train-only LS FIR refit.")
    fit_output_fir_from_train(model, x_train, y_train)
    torch.save(model.state_dict(), checkpoint_path)
    extra_nmse, extra_db = fullseq_nmse(model, x_val, y_val)
    print(f">>> After post-extra LS FIR full-sequence NMSE: {extra_nmse:.6e} ({extra_db:.2f} dB)")

    set_trainable_mode(model, "base")
    h1, _, db1 = train_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        stage_name="cropped-short",
        epochs=args.epochs_base,
        lr=args.base_lr,
        frame_length=args.frame_length,
        stride_length=args.stride_length,
        batch_size=args.batch_size,
        crop_margin=args.crop_margin,
        checkpoint_path=checkpoint_path,
        target_fullseq_db=args.target_fullseq_db,
    )
    histories.append(("cropped-short", h1))

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    print("\n>>> Mid train-only LS FIR refit.")
    fit_output_fir_from_train(model, x_train, y_train)
    torch.save(model.state_dict(), checkpoint_path)
    mid_nmse, mid_db = fullseq_nmse(model, x_val, y_val)
    print(f">>> After mid LS FIR full-sequence NMSE: {mid_nmse:.6e} ({mid_db:.2f} dB)")

    if mid_db > args.target_fullseq_db:
        set_fir_trainable(model, False)
        h2, _, _ = train_stage(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            stage_name="cropped-long",
            epochs=args.epochs_long,
            lr=args.long_lr,
            frame_length=args.long_frame_length,
            stride_length=args.long_stride_length,
            batch_size=args.long_batch_size,
            crop_margin=args.long_crop_margin,
            checkpoint_path=checkpoint_path,
            target_fullseq_db=args.target_fullseq_db,
        )
        histories.append(("cropped-long", h2))

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    set_trainable_mode(model, "base")
    set_fir_trainable(model, False)
    h3, _, _ = train_full_sequence_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs_full,
        lr=args.full_lr,
        crop_margin=args.full_crop_margin,
        checkpoint_path=checkpoint_path,
        target_fullseq_db=args.target_fullseq_db,
    )
    histories.append(("full-sequence", h3))

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    print("\n>>> Final train-only LS FIR refit.")
    fit_output_fir_from_train(model, x_train, y_train)
    torch.save(model.state_dict(), checkpoint_path)
    final_nmse, final_db = fullseq_nmse(model, x_val, y_val)
    print(f"\n>>> Final full-sequence NMSE: {final_nmse:.6e} ({final_db:.2f} dB)")
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
