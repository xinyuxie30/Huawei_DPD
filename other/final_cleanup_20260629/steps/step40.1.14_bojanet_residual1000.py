import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AI_DPD_PATH = SCRIPT_DIR / "AI-DPD.py"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AI-DPD"
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
DEFAULT_INIT = CHECKPOINT_DIR / "EA-TCN_memory_fir1000_best.pth"
DEFAULT_TEACHER = CHECKPOINT_DIR / "teacher32_6layer_scratch_best.pth"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "BO-JANET_residual1000_fixed_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 2026


def load_aidpd_module():
    spec = importlib.util.spec_from_file_location("aidpd", AI_DPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aidpd = load_aidpd_module()


MODEL_CONFIG = {
    "memory_depth": 16,
    "parallel_firs": 6,
    "hidden": 10,
    "output_fir_taps": 31,
}


def real_multiplier_count(config=MODEL_CONFIG):
    memory_depth = int(config["memory_depth"])
    parallel_firs = int(config["parallel_firs"])
    hidden = int(config["hidden"])
    output_fir_taps = int(config["output_fir_taps"])
    physical = 12
    input_complex_firs = 4 * memory_depth * parallel_firs
    janet = 4 * parallel_firs * hidden + 2 * hidden * hidden
    output_projection = 4 * hidden
    output_fir = 4 * output_fir_taps
    return physical + input_complex_firs + janet + output_projection + output_fir


class BOJANETResidualDPD(nn.Module):
    def __init__(self, init_g_real, init_g_imag, config=None):
        super().__init__()
        self.config = dict(MODEL_CONFIG if config is None else config)
        self.memory_depth = int(self.config["memory_depth"])
        self.parallel_firs = int(self.config["parallel_firs"])
        self.hidden = int(self.config["hidden"])
        self.output_fir_taps = int(self.config["output_fir_taps"])

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.fir_i = nn.Parameter(torch.empty(self.parallel_firs, 1, self.memory_depth, dtype=torch.float64))
        self.fir_q = nn.Parameter(torch.empty(self.parallel_firs, 1, self.memory_depth, dtype=torch.float64))
        self.W_fi = nn.Linear(2 * self.parallel_firs, self.hidden, bias=True, dtype=torch.float64)
        self.W_fh = nn.Linear(self.hidden, self.hidden, bias=False, dtype=torch.float64)
        self.W_gi = nn.Linear(2 * self.parallel_firs, self.hidden, bias=True, dtype=torch.float64)
        self.W_gh = nn.Linear(self.hidden, self.hidden, bias=False, dtype=torch.float64)
        self.out_i = nn.Linear(self.hidden, 1, bias=False, dtype=torch.float64)
        self.out_q = nn.Linear(self.hidden, 1, bias=False, dtype=torch.float64)

        center = self.output_fir_taps // 2
        fir_real = torch.zeros(self.output_fir_taps, dtype=torch.float64)
        fir_imag = torch.zeros(self.output_fir_taps, dtype=torch.float64)
        fir_real[center] = 1.0
        self.fir_real = nn.Parameter(fir_real)
        self.fir_imag = nn.Parameter(fir_imag)

        phase_index = torch.arange(self.hidden, dtype=torch.long) % self.parallel_firs
        self.register_buffer("phase_index", phase_index, persistent=False)
        self.reset_bojanet_parameters()

    def reset_bojanet_parameters(self):
        nn.init.xavier_uniform_(self.fir_i, gain=0.2)
        nn.init.xavier_uniform_(self.fir_q, gain=0.2)
        for module in [self.W_fi, self.W_gi]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            nn.init.constant_(module.bias, 0.0)
        for module in [self.W_fh, self.W_gh]:
            nn.init.orthogonal_(module.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_i.weight, gain=0.02)
        nn.init.xavier_uniform_(self.out_q.weight, gain=0.02)

    def physical_branch(self, i_x, q_x):
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        amp2 = i_x * i_x + q_x * q_x
        amp4 = amp2 * amp2

        x3_i = i_x * amp2
        x3_q = q_x * amp2
        lin_i = lin_i + x3_i * self.g3_real - x3_q * self.g3_imag
        lin_q = lin_q + x3_i * self.g3_imag + x3_q * self.g3_real

        x5_i = i_x * amp4
        x5_q = q_x * amp4
        lin_i = lin_i + x5_i * self.g5_real - x5_q * self.g5_imag
        lin_q = lin_q + x5_i * self.g5_imag + x5_q * self.g5_real
        return torch.cat([lin_i, lin_q], dim=1)

    def input_parallel_firs(self, i_x, q_x):
        pad = (self.memory_depth - 1, 0)
        i_pad = F.pad(i_x, pad, mode="replicate")
        q_pad = F.pad(q_x, pad, mode="replicate")
        i_fir = F.conv1d(i_pad, self.fir_i) - F.conv1d(q_pad, self.fir_q)
        q_fir = F.conv1d(i_pad, self.fir_q) + F.conv1d(q_pad, self.fir_i)
        return i_fir, q_fir

    def bojanet_residual(self, i_x, q_x):
        i_fir, q_fir = self.input_parallel_firs(i_x, q_x)
        magnitude = torch.sqrt(i_fir * i_fir + q_fir * q_fir + 1e-12)
        mag2 = magnitude * magnitude
        sin_theta = q_fir / (magnitude + 1e-12)
        cos_theta = i_fir / (magnitude + 1e-12)

        features = torch.cat([magnitude, mag2], dim=1).transpose(1, 2)
        cos_seq = cos_theta.transpose(1, 2)
        sin_seq = sin_theta.transpose(1, 2)
        batch, seq_len, _ = features.shape

        h = torch.zeros(batch, self.hidden, dtype=features.dtype, device=features.device)
        outputs = []
        for t in range(seq_len):
            l_t = features[:, t, :]
            f_t = torch.sigmoid(self.W_fi(l_t) + self.W_fh(h))
            g_t = torch.tanh(self.W_gi(l_t) + self.W_gh(h))
            h = f_t * h + (1.0 - f_t) * g_t
            outputs.append(h)

        h_seq = torch.stack(outputs, dim=1)
        cos_h = cos_seq[:, :, self.phase_index]
        sin_h = sin_seq[:, :, self.phase_index]
        i_rot = h_seq * cos_h
        q_rot = h_seq * sin_h
        out_i = self.out_i(i_rot) - self.out_q(q_rot)
        out_q = self.out_q(q_rot) + self.out_i(i_rot)
        return torch.cat([out_i, out_q], dim=2).transpose(1, 2)

    def forward_base(self, x_iq):
        i_x = x_iq[:, 0:1, :]
        q_x = x_iq[:, 1:2, :]
        return self.physical_branch(i_x, q_x) + self.bojanet_residual(i_x, q_x)

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


class DistillDataset(Dataset):
    def __init__(self, x_np, y_np, d_np, frame_length, stride_length):
        self.x = torch.as_tensor(x_np, dtype=torch.float64)
        self.y = torch.as_tensor(y_np, dtype=torch.float64)
        self.d = None if d_np is None else torch.as_tensor(d_np, dtype=torch.float64)
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.num_frames = (self.x.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        end = start + self.frame_length
        x = self.x[start:end].T
        y = self.y[start:end].T
        if self.d is None:
            return x, y, y
        return x, y, self.d[start:end].T


def parse_args():
    parser = argparse.ArgumentParser(description="BO-JANET residual DPD, adapted from OpenDPD baseline, 1000 real multipliers.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--init-checkpoint", default=str(DEFAULT_INIT))
    parser.add_argument("--teacher-checkpoint", default=str(DEFAULT_TEACHER))
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--use-blend-target", action="store_true")
    parser.add_argument("--blend-alpha", type=float, default=0.85)
    parser.add_argument("--blend-fir-taps", type=int, default=63)
    parser.add_argument("--truth-loss-weight", type=float, default=1.0)
    parser.add_argument("--distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--epochs-base", type=int, default=300)
    parser.add_argument("--epochs-long", type=int, default=80)
    parser.add_argument("--base-lr", type=float, default=2e-3)
    parser.add_argument("--long-lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--long-batch-size", type=int, default=4)
    parser.add_argument("--frame-length", type=int, default=512)
    parser.add_argument("--stride-length", type=int, default=256)
    parser.add_argument("--long-frame-length", type=int, default=4096)
    parser.add_argument("--long-stride-length", type=int, default=2048)
    parser.add_argument("--crop-margin", type=int, default=64)
    parser.add_argument("--long-crop-margin", type=int, default=160)
    parser.add_argument("--fit-output-fir-every", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-frame-length", type=int, default=4096)
    parser.add_argument("--eval-warmup", type=int, default=512)
    parser.add_argument("--ls-frame-length", type=int, default=4096)
    parser.add_argument("--ls-stride-length", type=int, default=2048)
    parser.add_argument("--ls-max-frames", type=int, default=16)
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
    return x_np[:split_idx], y_np[:split_idx], x_np[split_idx:], y_np[split_idx:], float(g_opt.real), float(g_opt.imag)


def load_physical_from_checkpoint(model, checkpoint_path):
    source = aidpd.load_state_dict(checkpoint_path)
    loaded = []
    with torch.no_grad():
        for key in ["g_real", "g_imag", "g3_real", "g3_imag", "g5_real", "g5_imag"]:
            if key in source and hasattr(model, key):
                getattr(model, key).copy_(source[key].to(dtype=torch.float64, device=DEVICE))
                loaded.append(key)
    return loaded


def run_model(model, x_np):
    model.eval()
    x_tensor = torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model(x_tensor).squeeze(0).cpu().numpy().T


def iq_to_complex(iq):
    return iq[:, 0] + 1j * iq[:, 1]


def complex_to_iq(signal):
    return np.stack([signal.real, signal.imag], axis=1)


def shifted_columns(signal, taps):
    pad = taps // 2
    cols = []
    for tap in range(taps):
        shift = tap - pad
        if shift < 0:
            col = np.r_[signal[-shift:], np.repeat(signal[-1], -shift)]
        elif shift > 0:
            col = np.r_[np.repeat(signal[0], shift), signal[:-shift]]
        else:
            col = signal
        cols.append(col)
    return np.stack(cols, axis=1)


def fit_complex_fir(pred_train, target_train, taps):
    return np.linalg.lstsq(shifted_columns(pred_train, taps), target_train, rcond=None)[0]


def apply_complex_fir(pred, coeffs):
    return shifted_columns(pred, len(coeffs)) @ coeffs


def collect_base_fir_rows(model, x_np, y_np, frame_length, stride_length, batch_size, crop_margin, max_frames):
    dataset = aidpd.FramedDataset(x_np, y_np, frame_length, stride_length)
    if max_frames > 0 and len(dataset) > max_frames:
        indices = np.linspace(0, len(dataset) - 1, max_frames, dtype=np.int64)
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    pred_rows = []
    target_rows = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            pred = model.forward_base(batch_x).cpu().numpy()
            target = batch_y.numpy()
            for pred_frame, target_frame in zip(pred, target):
                if crop_margin > 0 and pred_frame.shape[-1] > 2 * crop_margin:
                    pred_frame = pred_frame[:, crop_margin:-crop_margin]
                    target_frame = target_frame[:, crop_margin:-crop_margin]
                pred_c = pred_frame[0] + 1j * pred_frame[1]
                target_c = target_frame[0] + 1j * target_frame[1]
                pred_rows.append(shifted_columns(pred_c, model.output_fir_taps))
                target_rows.append(target_c)
    return np.concatenate(pred_rows, axis=0), np.concatenate(target_rows, axis=0)


def fit_output_fir_from_frames(model, x_train, y_train, args):
    matrix, target = collect_base_fir_rows(
        model,
        x_train,
        y_train,
        args.ls_frame_length,
        args.ls_stride_length,
        args.long_batch_size,
        args.long_crop_margin,
        args.ls_max_frames,
    )
    coeffs = np.linalg.lstsq(matrix, target, rcond=None)[0]
    with torch.no_grad():
        model.fir_real.copy_(torch.as_tensor(coeffs.real, dtype=torch.float64, device=DEVICE))
        model.fir_imag.copy_(torch.as_tensor(coeffs.imag, dtype=torch.float64, device=DEVICE))
    return coeffs


def maybe_build_blend_target(args, x_train, y_train, x_val, g_real, g_imag):
    if not args.use_blend_target:
        return None
    teacher_path = resolve_path(args.teacher_checkpoint)
    student_path = resolve_path(args.init_checkpoint)
    if not teacher_path.exists() or not student_path.exists():
        raise FileNotFoundError("Blend target requires existing teacher and student checkpoints.")

    student_config = {
        "hidden": 10,
        "input_mode": "rich9",
        "dilations": [1, 2, 4, 8],
        "attention_channels": 3,
        "poly_orders": [1, 3, 5],
        "output_fir_taps": 47,
    }
    student = aidpd.DiagramDPDNet(g_real, g_imag, config=student_config).to(DEVICE)
    student.load_state_dict(aidpd.load_state_dict(student_path), strict=True)
    teacher = aidpd.DiagramDPDNet(g_real, g_imag, config=aidpd.preset_config("teacher32_6layer")).to(DEVICE)
    teacher.load_state_dict(aidpd.load_state_dict(teacher_path), strict=True)

    student_train = run_model(student, x_train)
    teacher_train = run_model(teacher, x_train)
    blend_train = (1.0 - args.blend_alpha) * student_train + args.blend_alpha * teacher_train
    coeffs = fit_complex_fir(iq_to_complex(blend_train), iq_to_complex(y_train), args.blend_fir_taps)
    distill_train = complex_to_iq(apply_complex_fir(iq_to_complex(blend_train), coeffs))

    student_val = run_model(student, x_val)
    teacher_val = run_model(teacher, x_val)
    blend_val = (1.0 - args.blend_alpha) * student_val + args.blend_alpha * teacher_val
    distill_val = complex_to_iq(apply_complex_fir(iq_to_complex(blend_val), coeffs))
    return distill_train, distill_val


def make_loader(x_np, y_np, d_np, frame_length, stride_length, batch_size):
    return DataLoader(
        DistillDataset(x_np, y_np, d_np, frame_length, stride_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def frame_nmse(model, x_np, y_np, frame_length, batch_size, crop_margin):
    loader = DataLoader(
        aidpd.FramedDataset(x_np, y_np, frame_length, frame_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    err = 0.0
    power = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            if crop_margin > 0 and pred.shape[-1] > 2 * crop_margin:
                pred = pred[:, :, crop_margin:-crop_margin]
                batch_y = batch_y[:, :, crop_margin:-crop_margin]
            err += torch.sum((pred - batch_y) ** 2).item()
            power += torch.sum(batch_y ** 2).item()
    nmse = err / power
    return float(nmse), float(10 * np.log10(nmse))


def predict_with_warmup(model, x_np, frame_length, warmup):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_np), frame_length):
            prefix = max(0, start - warmup)
            end = min(len(x_np), start + frame_length)
            x_chunk = torch.as_tensor(x_np[prefix:end].T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
            pred = model(x_chunk).squeeze(0).cpu().numpy().T
            discard = start - prefix
            outputs.append(pred[discard:discard + (end - start)])
    return np.concatenate(outputs, axis=0)


def fullseq_nmse(model, x_val, y_val, args):
    pred = predict_with_warmup(model, x_val, args.eval_frame_length, args.eval_warmup)
    nmse = np.sum((pred - y_val) ** 2) / np.sum(y_val ** 2)
    return float(nmse), float(10 * np.log10(nmse))


def evaluate_aclr_chunked(model, x_val, y_val, checkpoint_stem, args):
    pred = predict_with_warmup(model, x_val, args.eval_frame_length, args.eval_warmup)
    fit_nmse = np.sum((pred - y_val) ** 2) / np.sum(y_val ** 2)
    metrics = aidpd.save_psd_and_aclr_figures(
        y_val.T,
        pred.T,
        output_prefix=checkpoint_stem,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        fs_hz=args.fs_mhz * 1e6,
        target_dbc=args.aclr_target_dbc,
    )
    print("\n>>> ACLR/PSD evaluation")
    print(f"Fit NMSE pred_y vs target_y: {fit_nmse:.6e} ({10 * np.log10(fit_nmse):.2f} dB)")
    print(
        "ACLR worst: "
        f"target y {metrics['target_worst']:.2f} dBc | "
        f"pred y {metrics['pred_worst']:.2f} dBc | "
        f"pred-target gap {metrics['gap_to_target_y']:.2f} dB | "
        f"-45 dBc margin {metrics['target_margin']:.2f} dB"
    )
    print(f"Target requirement: pred worst ACLR <= {args.aclr_target_dbc:.2f} dBc")
    print(f"PSD figure: {metrics['psd_path']}")
    print(f"ACLR figure: {metrics['aclr_path']}")
    return metrics


def cropped_mse(pred, target, margin):
    if margin > 0 and pred.shape[-1] > 2 * margin:
        pred = pred[:, :, margin:-margin]
        target = target[:, :, margin:-margin]
    return nn.functional.mse_loss(pred, target)


def train_stage(
    model,
    x_train,
    y_train,
    d_train,
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
    args,
):
    if epochs <= 0:
        nmse, db = fullseq_nmse(model, x_val, y_val, args)
        return [(0, db)], nmse, db

    loader = make_loader(x_train, y_train, d_train, frame_length, stride_length, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12, min_lr=1e-7
    )
    best_nmse, best_db = fullseq_nmse(model, x_val, y_val, args)
    torch.save(model.state_dict(), checkpoint_path)
    history = [(0, best_db)]
    start = time.time()
    print(f"\n>>> Stage {stage_name}: epochs={epochs}, lr={lr:.2e}, frame={frame_length}, batch={batch_size}")
    print(f">>> Initial full-sequence NMSE: {best_nmse:.6e} ({best_db:.2f} dB)")

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y, batch_d in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            batch_d = batch_d.to(DEVICE, non_blocking=True)
            pred = model(batch_x)
            loss = args.truth_loss_weight * cropped_mse(pred, batch_y, crop_margin)
            if args.distill_loss_weight > 0:
                loss = loss + args.distill_loss_weight * cropped_mse(pred, batch_d, crop_margin)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        if args.fit_output_fir_every > 0 and epoch % args.fit_output_fir_every == 0:
            fit_output_fir_from_frames(model, x_train, y_train, args)

        frame_value, frame_db = frame_nmse(model, x_val, y_val, frame_length, batch_size, crop_margin)
        scheduler.step(frame_value)
        db = frame_db
        nmse = frame_value
        did_full_eval = epoch % args.eval_every == 0 or epoch == epochs
        if did_full_eval:
            nmse, db = fullseq_nmse(model, x_val, y_val, args)
        history.append((epoch, db))
        elapsed = time.time() - start
        full_text = f"Full NMSE {db:.2f} dB" if did_full_eval else "Full NMSE pending"
        print(
            f"{stage_name} Epoch [{epoch}/{epochs}] | "
            f"Time {int(elapsed // 60)}m {int(elapsed % 60)}s | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"Frame NMSE {frame_db:.2f} dB | {full_text}"
        )
        if did_full_eval and nmse < best_nmse:
            best_nmse = nmse
            best_db = db
            torch.save(model.state_dict(), checkpoint_path)
            if best_db <= args.target_fullseq_db:
                print(f">>> Target reached in {stage_name}: {best_db:.2f} dB")
                break

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    return history, best_nmse, best_db


def plot_history(histories, out_path):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    offset = 0
    for stage, history in histories:
        xs = [offset + item[0] for item in history]
        ys = [item[1] for item in history]
        plt.plot(xs, ys, label=stage)
        offset = xs[-1]
    plt.axhline(-37.0, color="black", linestyle="--", linewidth=1.0, label="-37 dB")
    plt.axhline(-40.0, color="tab:red", linestyle=":", linewidth=1.1, label="-40 dB")
    plt.xlabel("Epoch")
    plt.ylabel("Full-sequence NMSE (dB)")
    plt.title("BO-JANET residual, 1000 real multipliers")
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
    multipliers = real_multiplier_count()

    print(f">>> Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f">>> CUDA: {torch.cuda.get_device_name(0)}")
    print(f">>> Data: {data_path}")
    print(f">>> Model config: {MODEL_CONFIG}")
    print(f">>> Real multipliers: {multipliers}")
    if multipliers > 1000:
        raise RuntimeError(f"Multiplier budget exceeded: {multipliers}")

    model = BOJANETResidualDPD(g_real, g_imag).to(DEVICE)
    if args.evaluate_only:
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        nmse, db = fullseq_nmse(model, x_val, y_val, args)
        print(f">>> Full-sequence NMSE: {nmse:.6e} ({db:.2f} dB)")
        evaluate_aclr_chunked(model, x_val, y_val, checkpoint_path.stem, args)
        return

    if not args.scratch and init_checkpoint.exists():
        loaded = load_physical_from_checkpoint(model, init_checkpoint)
        print(f">>> Loaded physical scalars from: {init_checkpoint}")
        print(f">>> Loaded: {loaded}")
    elif args.scratch:
        print(">>> Scratch mode: physical scalars initialized from train-only linear gain.")
    else:
        print(f">>> Init checkpoint not found: {init_checkpoint}")

    distill_train = None
    if args.use_blend_target:
        print("\n>>> Building blend teacher target for BO-JANET training.")
        distill_train, distill_val = maybe_build_blend_target(args, x_train, y_train, x_val, g_real, g_imag)
        blend_nmse = np.sum((distill_val - y_val) ** 2) / np.sum(y_val ** 2)
        print(f">>> Blend target validation NMSE: {blend_nmse:.6e} ({10 * np.log10(blend_nmse):.2f} dB)")

    print("\n>>> Initial train-only LS output FIR fit.")
    fit_output_fir_from_frames(model, x_train, y_train, args)
    init_nmse, init_db = fullseq_nmse(model, x_val, y_val, args)
    print(f">>> After initial LS FIR full-sequence NMSE: {init_nmse:.6e} ({init_db:.2f} dB)")

    histories = []
    h1, best_nmse, best_db = train_stage(
        model,
        x_train,
        y_train,
        distill_train,
        x_val,
        y_val,
        stage_name="short-frame",
        epochs=args.epochs_base,
        lr=args.base_lr,
        frame_length=args.frame_length,
        stride_length=args.stride_length,
        batch_size=args.batch_size,
        crop_margin=args.crop_margin,
        checkpoint_path=checkpoint_path,
        args=args,
    )
    histories.append(("short-frame", h1))

    print("\n>>> Mid train-only LS output FIR refit.")
    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    before_ls_nmse, before_ls_db = fullseq_nmse(model, x_val, y_val, args)
    fit_output_fir_from_frames(model, x_train, y_train, args)
    mid_nmse, mid_db = fullseq_nmse(model, x_val, y_val, args)
    print(f">>> After mid LS FIR full-sequence NMSE: {mid_nmse:.6e} ({mid_db:.2f} dB)")
    if mid_nmse < before_ls_nmse:
        torch.save(model.state_dict(), checkpoint_path)
        best_nmse, best_db = mid_nmse, mid_db
        print(">>> Mid LS FIR improved validation NMSE; checkpoint updated.")
    else:
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        best_nmse, best_db = before_ls_nmse, before_ls_db
        print(">>> Mid LS FIR did not improve validation NMSE; rolled back to stage best.")

    if best_db > args.target_fullseq_db and args.epochs_long > 0:
        h2, best_nmse, best_db = train_stage(
            model,
            x_train,
            y_train,
            distill_train,
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
            args=args,
        )
        histories.append(("long-frame", h2))

        print("\n>>> Final train-only LS output FIR refit.")
        model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
        before_ls_nmse, before_ls_db = fullseq_nmse(model, x_val, y_val, args)
        fit_output_fir_from_frames(model, x_train, y_train, args)
        final_ls_nmse, final_ls_db = fullseq_nmse(model, x_val, y_val, args)
        print(f">>> After final LS FIR full-sequence NMSE: {final_ls_nmse:.6e} ({final_ls_db:.2f} dB)")
        if final_ls_nmse < before_ls_nmse:
            torch.save(model.state_dict(), checkpoint_path)
            best_nmse, best_db = final_ls_nmse, final_ls_db
            print(">>> Final LS FIR improved validation NMSE; checkpoint updated.")
        else:
            model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
            best_nmse, best_db = before_ls_nmse, before_ls_db
            print(">>> Final LS FIR did not improve validation NMSE; rolled back to stage best.")

    model.load_state_dict(aidpd.load_state_dict(checkpoint_path), strict=True)
    final_nmse, final_db = fullseq_nmse(model, x_val, y_val, args)
    print(f"\n>>> Final full-sequence NMSE: {final_nmse:.6e} ({final_db:.2f} dB)")
    print(f">>> Saved checkpoint: {checkpoint_path}")

    convergence_path = FIGURE_DIR / f"{checkpoint_path.stem}_convergence.png"
    plot_history(histories, convergence_path)
    print(f">>> Convergence figure: {convergence_path}")

    evaluate_aclr_chunked(model, x_val, y_val, checkpoint_path.stem, args)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
