import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import welch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 0. 补充缺失的模型类 (直接放在这里避免 import 报错)
# ==========================================
class HuaweiTCN_Sniper(nn.Module):
    def __init__(self, in_channels=18, hidden_channels=20):
        super(HuaweiTCN_Sniper, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = 2
        self.kernel_size = 5

        self.network = nn.Sequential(
            nn.Conv1d(self.in_channels, self.hidden_channels, 1, bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 1, dilation=1, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 2, dilation=2, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 4, dilation=4, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 8, dilation=8, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.out_channels, 1, bias=False, dtype=torch.float64),
        )

    def forward(self, x_18ch):
        residual = self.network(x_18ch)
        original_iq = x_18ch[:, 0:2, :]
        return original_iq + residual

    # ==========================================


# 1. 频域评估工具：ACLR 计算与 PSD 绘图
# ==========================================
def to_complex(iq_data):
    """将 (Seq_Len, 2) 或 (2, Seq_Len) 转换为 1D 复数数组"""
    if hasattr(iq_data, 'detach'):
        iq_data = iq_data.detach().cpu().numpy()
    iq_data = np.squeeze(iq_data)
    if iq_data.shape[0] != 2 and iq_data.shape[-1] == 2:
        iq_data = iq_data.T
    return iq_data[0] + 1j * iq_data[1]


def calculate_aclr(complex_signal, fs=800e6, main_bw=200e6, adj_offset=200e6, adj_bw=200e6):
    """通过对功率谱密度(PSD)进行数值积分来计算 ACLR。"""
    frequencies, psd = welch(complex_signal, fs=fs, nperseg=4096, return_onesided=False)

    frequencies = np.fft.fftshift(frequencies)
    psd = np.fft.fftshift(psd)

    main_mask = (frequencies >= -main_bw / 2) & (frequencies <= main_bw / 2)
    adj_lower_mask = (frequencies >= -adj_offset - adj_bw / 2) & (frequencies <= -adj_offset + adj_bw / 2)
    adj_upper_mask = (frequencies >= adj_offset - adj_bw / 2) & (frequencies <= adj_offset + adj_bw / 2)

    main_power = np.trapezoid(psd[main_mask], frequencies[main_mask])
    adj_lower_power = np.trapezoid(psd[adj_lower_mask], frequencies[adj_lower_mask])
    adj_upper_power = np.trapezoid(psd[adj_upper_mask], frequencies[adj_upper_mask])

    aclr_lower = 10 * np.log10(adj_lower_power / main_power)
    aclr_upper = 10 * np.log10(adj_upper_power / main_power)

    return aclr_lower, aclr_upper


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_huawei_psd(pred_complex, target_complex, fs=800e6, nperseg=4096, smoothing_window=10):
    """绘制归一化 PSD 频谱对比图"""
    frequencies_1, psd_1 = welch(pred_complex, fs=fs, nperseg=nperseg, return_onesided=False)
    frequencies_2, psd_2 = welch(target_complex, fs=fs, nperseg=nperseg, return_onesided=False)

    half_nfft = int(nperseg / 2)
    frequencies_1 = np.concatenate((frequencies_1[half_nfft:], frequencies_1[:half_nfft]))
    psd_1 = np.concatenate((psd_1[half_nfft:], psd_1[:half_nfft]))
    psd_2 = np.concatenate((psd_2[half_nfft:], psd_2[:half_nfft]))

    psd_1_normalized = 10 * np.log10(psd_1 / np.max(psd_1))
    psd_2_normalized = 10 * np.log10(psd_2 / np.max(psd_2))

    psd_1_smoothed = moving_average(psd_1_normalized, smoothing_window)
    psd_2_smoothed = moving_average(psd_2_normalized, smoothing_window)

    freq_adjusted = frequencies_1[int(smoothing_window / 2): -int(smoothing_window / 2) + 1]

    plt.figure(figsize=(10, 6))
    plt.plot(freq_adjusted / 1e6, psd_1_smoothed, label="Predicted DPD", color='blue', alpha=0.8)
    plt.plot(freq_adjusted / 1e6, psd_2_smoothed, label="Ground Truth", color='red', linestyle='--', alpha=0.8)

    plt.title('Huawei DPD: Normalized Power Spectral Density (PSD)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Normalized PSD (dB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================
# 2. 执行推理与评估
# ==========================================
def evaluate():
    print(">>> 1. Loading Data and Extracting Features...")
    data = scipy.io.loadmat('DPD_data.mat')
    raw_x = data['x'].flatten()
    raw_y = data['y'].flatten()

    x_norm = raw_x / np.max(np.abs(raw_x))
    y_norm = raw_y / np.max(np.abs(raw_y))

    u = x_norm
    real, imag, mag = np.real(u), np.imag(u), np.abs(u)

    mag2 = mag ** 2
    mag4 = mag2 ** 2
    mag6 = mag4 * mag2
    mag8 = mag4 ** 2
    mag10 = mag8 * mag2
    mag12 = mag6 ** 2

    u_delay = np.concatenate([[0], u[:-1]])
    diff_u = u - u_delay
    diff_r, diff_i = np.real(diff_u), np.imag(diff_u)

    channels = [
        real, imag,
        real * mag2, imag * mag2,
        real * mag4, imag * mag4,
        real * mag6, imag * mag6,
        real * mag8, imag * mag8,
        real * mag10, imag * mag10,
        real * mag12, imag * mag12,
        diff_r, diff_i,
        diff_r * mag2, diff_i * mag2
    ]

    X_np = np.stack(channels, axis=1)
    Target_np = np.stack([np.real(y_norm), np.imag(y_norm)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    X_np = X_np[:min_len]
    Target_np = Target_np[:min_len]

    split_idx = int(min_len * 0.8)
    X_val = torch.from_numpy(X_np[split_idx:]).double().to(DEVICE)
    Y_val = Target_np[split_idx:]

    print(">>> 2. Loading Best Model...")
    model = HuaweiTCN_Sniper(in_channels=18, hidden_channels=20).to(DEVICE)
    # 加载你的权重文件
    model.load_state_dict(torch.load('best_huawei_tcn.pth'))
    model.eval()

    print(">>> 3. Running Full Sequence Inference...")
    with torch.no_grad():
        X_val_input = X_val.transpose(0, 1).unsqueeze(0)
        preds = model(X_val_input)
        preds_np = preds.squeeze(0).transpose(0, 1).cpu().numpy()

    print(">>> 4. Calculating ACLR and Plotting...")
    pred_complex = to_complex(preds_np)
    target_complex = to_complex(Y_val)

    pred_aclr_l, pred_aclr_u = calculate_aclr(pred_complex)
    target_aclr_l, target_aclr_u = calculate_aclr(target_complex)

    print("\n" + "=" * 40)
    print(" 📊 ACLR Performance (200MHz Bandwidth)")
    print("=" * 40)
    print(f"Predicted DPD Lower ACLR : {pred_aclr_l:.2f} dBc")
    print(f"Predicted DPD Upper ACLR : {pred_aclr_u:.2f} dBc")
    print("-" * 40)
    print(f"Ground Truth  Lower ACLR : {target_aclr_l:.2f} dBc")
    print(f"Ground Truth  Upper ACLR : {target_aclr_u:.2f} dBc")
    print("=" * 40 + "\n")

    plot_huawei_psd(pred_complex, target_complex)


if __name__ == '__main__':
    evaluate()