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
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "EA-TCN_tres_gru_fir1000_best.pth"
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


class TResGRUFIRDPDNet(nn.Module):
    """OpenDPDv2-style TRes-GRU direct DPD model.

    Equivalent real multiplier count under the project rule:
    - 1/3/5 complex physical branches: 3 * 4 = 12
    - GRU input/recurrent weights: 3 * 13 * (6 + 13) = 741
    - GRU output FC: 13 * 2 = 26
    - TCN residual skip: 2 * 3 * 3 + 3 * 2 * 1 = 24
    - 47-tap complex output FIR: 47 * 4 = 188
    Total = 991.
    """

    def __init__(self, init_g_real, init_g_imag, hidden_size=13, fir_taps=47, dtype=torch.float32):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.output_fir_taps = int(fir_taps)
        self.model_dtype = dtype

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=dtype))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=dtype))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=dtype))

        self.gru = nn.GRU(
            input_size=6,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bias=True,
            dtype=dtype,
        )
        self.fc_out = nn.Linear(self.hidden_size, 2, bias=False, dtype=dtype)

        self.tcn_skip = nn.Sequential(
            nn.Conv1d(2, 3, kernel_size=3, padding=16, dilation=16, bias=False, dtype=dtype),
            nn.Hardswish(),
            nn.Conv1d(3, 2, kernel_size=1, bias=False, dtype=dtype),
            nn.Hardswish(),
        )

        center = self.output_fir_taps // 2
        fir_real = torch.zeros(self.output_fir_taps, dtype=dtype)
        fir_imag = torch.zeros(self.output_fir_taps, dtype=dtype)
        fir_real[center] = 1.0
        self.fir_real = nn.Parameter(fir_real)
        self.fir_imag = nn.Parameter(fir_imag)
        self.reset_residual_parameters()

    def reset_residual_parameters(self):
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight_ih" in name:
                for gate in param.chunk(3, dim=0):
                    nn.init.xavier_uniform_(gate)
            elif "weight_hh" in name:
                for gate in param.chunk(3, dim=0):
                    nn.init.orthogonal_(gate)

        # Start from the physical/FIR solution. The residual paths then learn corrections.
        nn.init.constant_(self.fc_out.weight, 0.0)
        nn.init.xavier_uniform_(self.tcn_skip[0].weight)
        nn.init.constant_(self.tcn_skip[2].weight, 0.0)

    def forward_physical(self, i_x, q_x, amp, amp2, amp4):
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
        return torch.cat([lin_i, lin_q], dim=1)

    def forward_base(self, x_iq):
        x_iq = x_iq.to(dtype=self.g_real.dtype)
        i_x = x_iq[:, 0:1, :]
        q_x = x_iq[:, 1:2, :]
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp4 = amp2 ** 2

        physical = self.forward_physical(i_x, q_x, amp, amp2, amp4)

        i_future = safe_shift(i_x, -1)
        q_future = safe_shift(q_x, -1)
        features = torch.cat([i_x, q_x, amp, amp3, i_future, q_future], dim=1).transpose(1, 2).contiguous()
        # cuDNN GRU can reject float64 sequences on some Windows/CUDA builds.
        if features.dtype == torch.float64 or features.shape[1] > 65535:
            with torch.backends.cudnn.flags(enabled=False):
                gru_out, _ = self.gru(features)
        else:
            gru_out, _ = self.gru(features)
        rnn_residual = self.fc_out(gru_out).transpose(1, 2)
        tcn_residual = self.tcn_skip(x_iq)
        return physical + rnn_residual + tcn_residual

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


def real_multiplier_count(hidden_size=13, fir_taps=47):
    return int(12 + 3 * hidden_size * (6 + hidden_size) + 2 * hidden_size + 24 + 4 * fir_taps)


def parse_args():
    parser = argparse.ArgumentParser(description="TRes-GRU + TCN residual + FIR direct DPD, under 1000 real multipliers.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from --checkpoint if it exists.")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=13)
    parser.add_argument("--fir-taps", type=int, default=47)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--epochs-short", type=int, default=500)
    parser.add_argument("--epochs-long", type=int, default=360)
    parser.add_argument("--short-lr", type=float, default=1e-3)
    parser.add_argument("--long-lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--long-batch-size", type=int, default=2)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--stride-length", type=int, default=32)
    parser.add_argument("--long-frame-length", type=int, default=8192)
    parser.add_argument("--long-stride-length", type=int, default=512)
    parser.add_argument("--crop-margin", type=int, default=96)
    parser.add_argument("--long-crop-margin", type=int, default=256)
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


def dtype_from_arg(dtype_name):
    return torch.float32 if dtype_name == "float32" else torch.float64


def make_loader(x_np, y_np, frame_length, stride_length, batch_size):
    dataset = aidpd.FramedDataset(x_np, y_np, frame_length, stride_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_sequence(model, x_np, base=False):
    x_tensor = torch.as_tensor(x_np.T, dtype=model.g_real.dtype, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        y = model.forward_base(x_tensor) if base else model(x_tensor)
    return y.squeeze(0).cpu().numpy()


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


def load_scalar_init(model, checkpoint_path):
    source = aidpd.load_state_dict(checkpoint_path)
    copied = 0
    with torch.no_grad():
        for name in ["g_real", "g_imag", "g3_real", "g3_imag", "g5_real", "g5_imag"]:
            if name in source and hasattr(model, name):
                param = getattr(model, name)
                param.copy_(source[name].to(dtype=param.dtype, device=DEVICE))
                copied += 1
    return copied


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
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=model.fir_real.dtype, device=DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=model.fir_imag.dtype, device=DEVICE))
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
    if epochs <= 0:
        nmse, nmse_db = fullseq_nmse(model, x_val, y_val)
        return [(0, nmse_db)], nmse, nmse_db

    loader = make_loader(x_train, y_train, frame_length, stride_length, batch_size)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
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
            batch_x = batch_x.to(DEVICE, dtype=model.g_real.dtype, non_blocking=True)
            batch_y = batch_y.to(DEVICE, dtype=model.g_real.dtype, non_blocking=True)
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
    plt.title("TRes-GRU + TCN residual + train-only LS FIR")
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
    model_dtype = dtype_from_arg(args.dtype)
    model = TResGRUFIRDPDNet(
        g_real,
        g_imag,
        hidden_size=args.hidden_size,
        fir_taps=args.fir_taps,
        dtype=model_dtype,
    ).to(DEVICE)

    multipliers = real_multiplier_count(args.hidden_size, args.fir_taps)
    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Validation samples: {len(x_val)}")
    print(f">>> DType: {args.dtype}")
    print(f">>> Real multipliers: {multipliers}")
    if multipliers > 1000:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers}")

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

    if args.resume and checkpoint_path.exists():
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        print(f">>> Resumed exact model from: {checkpoint_path}")
    elif not args.scratch and init_checkpoint.exists():
        copied = load_scalar_init(model, init_checkpoint)
        print(f">>> Scalar physical-branch init from: {init_checkpoint} | copied={copied}")
    elif args.scratch:
        print(">>> Scratch mode: no checkpoint initialization.")
    else:
        print(f">>> Init checkpoint not found, using train-split linear gain only: {init_checkpoint}")

    print("\n>>> Initial train-only LS FIR fit.")
    fit_output_fir_from_train(model, x_train, y_train)
    init_nmse, init_db = fullseq_nmse(model, x_val, y_val)
    print(f">>> After initial LS FIR full-sequence NMSE: {init_nmse:.6e} ({init_db:.2f} dB)")
    torch.save(model.state_dict(), checkpoint_path)

    histories = []
    set_fir_trainable(model, False)
    h1, _, db1 = train_stage(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        stage_name="short-frame",
        epochs=args.epochs_short,
        lr=args.short_lr,
        frame_length=args.frame_length,
        stride_length=args.stride_length,
        batch_size=args.batch_size,
        crop_margin=args.crop_margin,
        checkpoint_path=checkpoint_path,
        target_fullseq_db=args.target_fullseq_db,
    )
    histories.append(("short-frame", h1))

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
            stage_name="long-frame",
            epochs=args.epochs_long,
            lr=args.long_lr,
            frame_length=args.long_frame_length,
            stride_length=args.long_stride_length,
            batch_size=args.long_batch_size,
            crop_margin=args.long_crop_margin,
            checkpoint_path=checkpoint_path,
            target_fullseq_db=args.target_fullseq_db,
        )
        histories.append(("long-frame", h2))

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
