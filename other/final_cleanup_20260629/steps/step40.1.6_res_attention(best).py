import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
from scipy.signal import welch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 16
FRAME_LENGTH = 800
STRIDE_LENGTH = 32
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIGURE_DIR = PROJECT_ROOT / "eval_outputs_direct" / "figures"
BEST_CHECKPOINT = PROJECT_ROOT / "best_raw_model_A.pth"
DATA_PATH = PROJECT_ROOT / "DPD_data.mat"


class OptimalSpectrumAttention(nn.Module):
    def __init__(self, channels):
        super(OptimalSpectrumAttention, self).__init__()
        self.env_conv = nn.Conv1d(3, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp, amp3, amp7):
        env_features = torch.cat([amp, amp3, amp7], dim=1)
        weights = self.sigmoid(self.env_conv(env_features))
        return x * weights


# ==========================================
# EA-TCN  3 * 4 + 6 * 10 + (100+50) * 4 + 2 * 10 + 10 * 3 * 3 = 782

class DiagramDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(DiagramDPDNet, self).__init__()
        self.hidden = 10

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))

        # 初始化为0，让网络在训练中自己去寻找最优的三阶压缩点
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.in_conv = nn.Conv1d(6, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList()

        for dilation in [1, 2, 4, 8]:
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(self.hidden, self.hidden, 5, padding=2 * dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                nn.SiLU()
            ))

        self.attn = OptimalSpectrumAttention(self.hidden)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)
        nn.init.constant_(self.out_conv.weight, 0.0)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        # 1. 基础一阶线性旁路
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        # 计算所有基础物理特征
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp4 = amp2 ** 2
        amp7 = amp ** 7

        x3_i = i_x * amp2
        x3_q = q_x * amp2

        # 应用三阶复数权重吸收静态 AM-AM/AM-PM
        lin3_i = x3_i * self.g3_real - x3_q * self.g3_imag
        lin3_q = x3_i * self.g3_imag + x3_q * self.g3_real
        x5_i = i_x * amp4
        x5_q = q_x * amp4
        lin5_i = x5_i * self.g5_real - x5_q * self.g5_imag
        lin5_q = x5_i * self.g5_imag + x5_q * self.g5_real

        # 将一阶、三阶和五阶旁路合并
        lin_out = torch.cat([lin_i + lin3_i + lin5_i, lin_q + lin3_q + lin5_q], dim=1)

        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)

        x_6ch = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)

        feat = self.act(self.in_conv(x_6ch))


        for block in self.tcn_blocks:
            feat = feat + block(feat)

        feat = self.attn(feat, amp, amp3, amp7)
        residual = self.out_conv(feat)

        return lin_out + residual


class FramedDataset(Dataset):
    def __init__(self, features, targets, frame_length, stride_length):
        self.features = torch.as_tensor(features, dtype=torch.float64)
        self.targets = torch.as_tensor(targets, dtype=torch.float64)
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.num_frames = (self.features.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self): return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        return self.features[start:start + self.frame_length, :].T, self.targets[start:start + self.frame_length, :].T


class AutoDiagramDPDNet(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.hidden = state_dict["in_conv.weight"].shape[0]
        self.input_channels = state_dict["in_conv.weight"].shape[1]
        self.attn_channels = state_dict["attn.env_conv.weight"].shape[1]
        self.has_g3 = "g3_real" in state_dict
        self.has_g5 = "g5_real" in state_dict
        block_ids = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("tcn_blocks.") and key.endswith(".0.weight")
            }
        )
        self.dilations = [1, 2, 4, 8, 16, 32][: len(block_ids)]

        self.g_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if self.has_g3:
            self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        if self.has_g5:
            self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
            self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.in_conv = nn.Conv1d(
            self.input_channels,
            self.hidden,
            1,
            bias=False,
            dtype=torch.float64,
        )
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList(
            [
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
                for dilation in self.dilations
            ]
        )
        self.attn = OptimalSpectrumAttention(self.hidden)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp4 = amp2 ** 2
        amp5 = amp ** 5
        amp7 = amp ** 7

        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real
        if self.has_g3:
            x3_i, x3_q = i_x * amp2, q_x * amp2
            lin_i = lin_i + x3_i * self.g3_real - x3_q * self.g3_imag
            lin_q = lin_q + x3_i * self.g3_imag + x3_q * self.g3_real
        if self.has_g5:
            x5_i, x5_q = i_x * amp4, q_x * amp4
            lin_i = lin_i + x5_i * self.g5_real - x5_q * self.g5_imag
            lin_q = lin_q + x5_i * self.g5_imag + x5_q * self.g5_real

        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)
        if self.input_channels == 6:
            features = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)
        elif self.input_channels == 9:
            i_past = torch.roll(i_x, shifts=1, dims=2)
            q_past = torch.roll(q_x, shifts=1, dims=2)
            features = torch.cat(
                [i_x, q_x, i_future, q_future, i_past, q_past, amp, amp3, amp5],
                dim=1,
            )
        else:
            raise ValueError(f"Unsupported model input channels: {self.input_channels}")

        feat = self.act(self.in_conv(features))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        feat = self.attn(feat, amp, amp3, amp7)
        return torch.cat([lin_i, lin_q], dim=1) + self.out_conv(feat)


def resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


def load_state_dict(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def load_normalized_data(data_path):
    data = scipy.io.loadmat(data_path)
    raw_x, raw_y = data['x'].flatten(), data['y'].flatten()

    RMS_X = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    RMS_Y = np.sqrt(np.mean(np.abs(raw_y) ** 2))

    x_c = raw_x / RMS_X
    y_c = raw_y / RMS_Y
    return raw_x, raw_y, x_c, y_c


def iq_array(complex_signal):
    return np.stack([np.real(complex_signal), np.imag(complex_signal)], axis=1)


def tensor_to_complex(iq):
    iq = np.squeeze(iq)
    if iq.shape[0] == 2:
        return iq[0] + 1j * iq[1]
    return iq[:, 0] + 1j * iq[:, 1]


def psd_db(signal, fs_hz=491.52e6, nperseg=4096):
    freq, density = welch(
        signal,
        fs=fs_hz,
        nperseg=min(nperseg, len(signal)),
        return_onesided=False,
    )
    freq = np.fft.fftshift(freq)
    density = np.fft.fftshift(density)
    return freq, 10 * np.log10(np.maximum(density, 1e-300))


def band_average(freq_hz, values_db, band_mhz):
    low_hz, high_hz = band_mhz[0] * 1e6, band_mhz[1] * 1e6
    mask = (freq_hz >= low_hz) & (freq_hz <= high_hz)
    if not np.any(mask):
        raise ValueError(f"No PSD bins in band {band_mhz} MHz")
    return float(np.mean(values_db[mask]))


def aclr_from_psd(freq_hz, values_db, main_band, lower_band, upper_band):
    main_avg = band_average(freq_hz, values_db, main_band)
    lower_avg = band_average(freq_hz, values_db, lower_band)
    upper_avg = band_average(freq_hz, values_db, upper_band)
    lower_aclr = lower_avg - main_avg
    upper_aclr = upper_avg - main_avg
    return lower_aclr, upper_aclr, max(lower_aclr, upper_aclr)


def run_sequence(model, x_np):
    x_tensor = torch.as_tensor(x_np.T, dtype=torch.float64, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        return model(x_tensor).squeeze(0).cpu().numpy()


def save_psd_and_aclr_figures(
    target_iq,
    pred_iq,
    output_prefix,
    main_band=(-100.0, 100.0),
    lower_band=(-245.0, -180.0),
    upper_band=(180.0, 245.0),
    fs_hz=491.52e6,
    nperseg=4096,
    target_dbc=-45.0,
):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target_complex = tensor_to_complex(target_iq)
    pred_complex = tensor_to_complex(pred_iq)
    freq_hz, target_psd = psd_db(target_complex, fs_hz=fs_hz, nperseg=nperseg)
    _, pred_psd = psd_db(pred_complex, fs_hz=fs_hz, nperseg=nperseg)

    target_aclr = aclr_from_psd(freq_hz, target_psd, main_band, lower_band, upper_band)
    pred_aclr = aclr_from_psd(freq_hz, pred_psd, main_band, lower_band, upper_band)
    gap_to_target_y = pred_aclr[2] - target_aclr[2]
    target_margin = target_dbc - pred_aclr[2]

    psd_path = FIGURE_DIR / f"{output_prefix}_psd_aclr.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    target_norm = target_psd - np.max(target_psd)
    pred_norm = pred_psd - np.max(pred_psd)
    ax.plot(freq_hz / 1e6, target_norm, "--", linewidth=1.1, label="Target y")
    ax.plot(freq_hz / 1e6, pred_norm, linewidth=1.1, label="Predicted y = f(x)")
    for band, label, color in [
        (main_band, "Main band", "tab:green"),
        (lower_band, "Lower adjacent", "tab:orange"),
        (upper_band, "Upper adjacent", "tab:purple"),
    ]:
        ax.axvspan(band[0], band[1], color=color, alpha=0.12, label=label)
    main_norm_avg = band_average(freq_hz, pred_norm, main_band)
    ax.axhline(
        main_norm_avg + target_dbc,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="-45 dBc target from predicted main avg",
    )
    ax.set_title(f"Predicted PSD ACLR {pred_aclr[2]:.2f} dBc | target margin {target_margin:.2f} dB")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Normalized PSD (dB)")
    ax.grid(True, linestyle=":", alpha=0.65)
    ax.legend(loc="best")
    fig.tight_layout()
    try:
        fig.savefig(psd_path, dpi=300)
    except PermissionError:
        psd_path = FIGURE_DIR / f"{output_prefix}_psd_aclr_{int(time.time())}.png"
        fig.savefig(psd_path, dpi=300)
    plt.close(fig)

    aclr_path = FIGURE_DIR / f"{output_prefix}_aclr_bar.png"
    labels = ["Lower", "Upper", "Worst"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, target_aclr, width, label="Target y", color="tab:red", alpha=0.82)
    ax.bar(x + width / 2, pred_aclr, width, label="Predicted y", color="tab:blue", alpha=0.82)
    ax.axhline(target_dbc, color="black", linestyle="--", linewidth=1.2, label="-45 dBc target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ACLR leakage ratio (dBc, lower is better)")
    ax.set_title(f"ACLR comparison | pred-target gap {gap_to_target_y:.2f} dB | margin {target_margin:.2f} dB")
    ax.grid(True, axis="y", linestyle=":", alpha=0.65)
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(aclr_path, dpi=300)
    except PermissionError:
        aclr_path = FIGURE_DIR / f"{output_prefix}_aclr_bar_{int(time.time())}.png"
        fig.savefig(aclr_path, dpi=300)
    plt.close(fig)

    return {
        "target_lower": target_aclr[0],
        "target_upper": target_aclr[1],
        "target_worst": target_aclr[2],
        "pred_lower": pred_aclr[0],
        "pred_upper": pred_aclr[1],
        "pred_worst": pred_aclr[2],
        "gap_to_target_y": gap_to_target_y,
        "target_dbc": target_dbc,
        "target_margin": target_margin,
        "psd_path": psd_path,
        "aclr_path": aclr_path,
    }


def evaluate_aclr_visualization(
    model,
    x_val,
    y_val,
    output_prefix="step40_1_6_res_attention_direct",
    fs_mhz=491.52,
    main_band=(-100.0, 100.0),
    lower_band=(-245.0, -180.0),
    upper_band=(180.0, 245.0),
    target_dbc=-45.0,
):
    model.eval()
    with torch.no_grad():
        pred_iq = run_sequence(model, x_val)

    fit_nmse = np.sum((pred_iq - y_val.T) ** 2) / np.sum(y_val.T ** 2)
    metrics = save_psd_and_aclr_figures(
        y_val.T,
        pred_iq,
        output_prefix,
        main_band=main_band,
        lower_band=lower_band,
        upper_band=upper_band,
        fs_hz=fs_mhz * 1e6,
        target_dbc=target_dbc,
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
    print(
        "Target requirement: "
        f"pred worst ACLR <= {target_dbc:.2f} dBc"
    )
    print(f"PSD figure: {metrics['psd_path']}")
    print(f"ACLR figure: {metrics['aclr_path']}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate the 782-multiplier EA-TCN DPD model.")
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to DPD_data.mat.")
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT), help="Best checkpoint for evaluation.")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only generate ACLR/PSD figures.")
    parser.add_argument("--fs-mhz", type=float, default=491.52, help="Dataset sample rate in MHz.")
    parser.add_argument("--main-band", type=float, nargs=2, default=(-100.0, 100.0), metavar=("LO", "HI"))
    parser.add_argument("--lower-band", type=float, nargs=2, default=(-245.0, -180.0), metavar=("LO", "HI"))
    parser.add_argument("--upper-band", type=float, nargs=2, default=(180.0, 245.0), metavar=("LO", "HI"))
    parser.add_argument("--aclr-target-dbc", type=float, default=-45.0, help="Target ACLR leakage ratio in dBc.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = resolve_path(args.data)
    checkpoint_path = resolve_path(args.checkpoint)
    raw_x, raw_y, x_c, y_c = load_normalized_data(data_path)
    rms_x = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    rms_y = np.sqrt(np.mean(np.abs(raw_y) ** 2))
    max_x = np.max(np.abs(raw_x))
    max_y = np.max(np.abs(raw_y))

    G_opt = np.vdot(x_c, y_c) / np.vdot(x_c, x_c)
    g_real_init = np.real(G_opt)
    g_imag_init = np.imag(G_opt)

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Target_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]

    model = DiagramDPDNet(g_real_init, g_imag_init).to(DEVICE)
    if args.evaluate_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
        evaluate_aclr_visualization(
            model,
            X_val,
            Y_val,
            fs_mhz=args.fs_mhz,
            main_band=tuple(args.main_band),
            lower_band=tuple(args.lower_band),
            upper_band=tuple(args.upper_band),
            target_dbc=args.aclr_target_dbc,
        )
        return

    train_loader = DataLoader(FramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(FramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float('inf')
    history_epochs = []
    history_nmse = []

    print(f"\n>>> 开始训练 EA-TCN (1/3/5阶旁路 + 动态 LR) (782乘法器)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        total_error_power, total_target_power = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
                preds = model(batch_x)
                total_error_power += torch.sum((preds - batch_y) ** 2).item()
                total_target_power += torch.sum(batch_y ** 2).item()

        linear_nmse = total_error_power / total_target_power
        val_nmse = 10 * np.log10(linear_nmse)

        # 动态监控
        scheduler.step(val_nmse)
        current_lr = optimizer.param_groups[0]['lr']

        history_epochs.append(epoch + 1)
        history_nmse.append(val_nmse)

        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Time: {elapsed_mins}m {elapsed_secs}s | LR: {current_lr:.2e} | NMSE: {val_nmse:.2f} dB ")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), checkpoint_path)

    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_nmse, label='Validation NMSE', color='b', linewidth=1.5)
    plt.axhline(y=min(history_nmse), color='r', linestyle='--', label=f'Best: {min(history_nmse):.2f} dB')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title(f'EA-TCN 782 Multipliers (BS={BATCH_SIZE})', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    convergence_path = FIGURE_DIR / 'EA-TCN-res-attention-782.png'
    plt.savefig(convergence_path, dpi=300)
    print(f"\n>>> 训练完成！收敛曲线已保存: {convergence_path}")

    model.load_state_dict(load_state_dict(checkpoint_path), strict=True)
    evaluate_aclr_visualization(
        model,
        X_val,
        Y_val,
        fs_mhz=args.fs_mhz,
        main_band=tuple(args.main_band),
        lower_band=tuple(args.lower_band),
        upper_band=tuple(args.upper_band),
        target_dbc=args.aclr_target_dbc,
    )


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()
