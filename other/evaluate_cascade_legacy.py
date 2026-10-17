import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import welch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 模型架构定义 (复用 HuaweiTCN_Sniper)
# ==========================================
class HuaweiTCN_Sniper(nn.Module):
    def __init__(self, in_channels=18, hidden_channels=20):
        super(HuaweiTCN_Sniper, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1, bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, stride=1, padding=2, dilation=1, groups=hidden_channels,
                      bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, stride=1, padding=4, dilation=2, groups=hidden_channels,
                      bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, stride=1, padding=8, dilation=4, groups=hidden_channels,
                      bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, stride=1, padding=16, dilation=8, groups=hidden_channels,
                      bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(hidden_channels, 2, 1, bias=False, dtype=torch.float64),
        )

    def forward(self, x_18ch):
        return x_18ch[:, 0:2, :] + self.network(x_18ch)


# ==========================================
# 2. 纯 PyTorch 版本的 18 通道特征提取器
# ==========================================
def extract_18ch_features_pt(u_tensor):
    """
    在 GPU 上动态将 (Batch, 2, SeqLen) 转换为 (Batch, 18, SeqLen)
    """
    real = u_tensor[:, 0, :]
    imag = u_tensor[:, 1, :]

    mag2 = real ** 2 + imag ** 2
    mag = torch.sqrt(mag2 + 1e-12)
    mag4 = mag2 ** 2
    mag6 = mag4 * mag2
    mag8 = mag4 ** 2
    mag10 = mag8 * mag2
    mag12 = mag6 ** 2

    # 一阶延迟与差分
    u_delay = torch.cat([torch.zeros_like(u_tensor[:, :, :1]), u_tensor[:, :, :-1]], dim=-1)
    diff_u = u_tensor - u_delay
    diff_r = diff_u[:, 0, :]
    diff_i = diff_u[:, 1, :]

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
    return torch.stack(channels, dim=1)


# ==========================================
# 3. 频域工具
# ==========================================
def to_complex(iq_data):
    if hasattr(iq_data, 'detach'): iq_data = iq_data.detach().cpu().numpy()
    iq_data = np.squeeze(iq_data)
    if iq_data.shape[0] != 2 and iq_data.shape[-1] == 2: iq_data = iq_data.T
    return iq_data[0] + 1j * iq_data[1]


def calculate_aclr(complex_signal, fs=800e6, main_bw=200e6, adj_offset=200e6, adj_bw=200e6):
    frequencies, psd = welch(complex_signal, fs=fs, nperseg=4096, return_onesided=False)
    frequencies, psd = np.fft.fftshift(frequencies), np.fft.fftshift(psd)

    main_mask = (frequencies >= -main_bw / 2) & (frequencies <= main_bw / 2)
    adj_l_mask = (frequencies >= -adj_offset - adj_bw / 2) & (frequencies <= -adj_offset + adj_bw / 2)
    adj_u_mask = (frequencies >= adj_offset - adj_bw / 2) & (frequencies <= adj_offset + adj_bw / 2)

    main_pwr = np.trapezoid(psd[main_mask], frequencies[main_mask])
    adj_l_pwr = np.trapezoid(psd[adj_l_mask], frequencies[adj_l_mask])
    adj_u_pwr = np.trapezoid(psd[adj_u_mask], frequencies[adj_u_mask])

    return 10 * np.log10(adj_l_pwr / main_pwr), 10 * np.log10(adj_u_pwr / main_pwr)


# ==========================================
# 4. 级联评估主程序
# ==========================================
def evaluate_cascade():
    print(">>> 1. Loading Base Data...")
    data = scipy.io.loadmat('DPD_data.mat')
    x_norm = data['x'].flatten() / np.max(np.abs(data['x']))

    # 提取后20%作为验证集
    split_idx = int(len(x_norm) * 0.8)
    x_val_complex = x_norm[split_idx:]

    # 构造基础张量 (Batch=1, Channels=2, SeqLen)
    x_val_tensor = torch.tensor(np.stack([np.real(x_val_complex), np.imag(x_val_complex)], axis=0),
                                dtype=torch.float64).unsqueeze(0).to(DEVICE)

    print(">>> 2. Loading Models...")
    dpd_model = HuaweiTCN_Sniper().to(DEVICE)
    dpd_model.load_state_dict(torch.load('best_huawei_tcn.pth'))
    dpd_model.eval()

    pa_model = HuaweiTCN_Sniper().to(DEVICE)
    pa_model.load_state_dict(torch.load('best_pa_model.pth'))  # 第一步训练出的 PA 权重
    pa_model.eval()

    print(">>> 3. Running Cascade Inference...")
    with torch.no_grad():
        # 链路 A: 原始信号 -> 直接过 PA (无 DPD)
        x_val_18ch = extract_18ch_features_pt(x_val_tensor)
        distorted_out = pa_model(x_val_18ch)

        # 链路 B: 原始信号 -> DPD解药 -> 提取解药特征 -> 过 PA (有 DPD)
        pred_dpd_out = dpd_model(x_val_18ch)
        pred_dpd_18ch = extract_18ch_features_pt(pred_dpd_out)
        linearized_out = pa_model(pred_dpd_18ch)

    # 转换为复数用于画图
    distorted_cplx = to_complex(distorted_out)
    linearized_cplx = to_complex(linearized_out)

    print(">>> 4. Calculating ACLR...")
    dist_l, dist_u = calculate_aclr(distorted_cplx)
    lin_l, lin_u = calculate_aclr(linearized_cplx)

    print("\n" + "=" * 45)
    print(" 📊 Real PA Output ACLR Performance")
    print("=" * 45)
    print(f"[Without DPD] Lower ACLR : {dist_l:.2f} dBc | Upper : {dist_u:.2f} dBc")
    print(f"[With DPD]    Lower ACLR : {lin_l:.2f} dBc | Upper : {lin_u:.2f} dBc")
    print(f"✅ ACLR Improvement: ~{abs((dist_l + dist_u) / 2 - (lin_l + lin_u) / 2):.2f} dB")
    print("=" * 45 + "\n")

    # 绘制最终的射频输出 PSD 对比图
    frequencies_1, psd_dist = welch(distorted_cplx, fs=800e6, nperseg=4096, return_onesided=False)
    frequencies_2, psd_lin = welch(linearized_cplx, fs=800e6, nperseg=4096, return_onesided=False)

    half = int(4096 / 2)
    freq = np.concatenate((frequencies_1[half:], frequencies_1[:half]))
    psd_dist = 10 * np.log10(np.concatenate((psd_dist[half:], psd_dist[:half])))
    psd_lin = 10 * np.log10(np.concatenate((psd_lin[half:], psd_lin[:half])))

    plt.figure(figsize=(10, 6))
    plt.plot(freq / 1e6, psd_dist - np.max(psd_dist), label="PA Output (Without DPD)", color='red', alpha=0.7)
    plt.plot(freq / 1e6, psd_lin - np.max(psd_lin), label="PA Output (With TCN DPD)", color='blue', alpha=0.9)
    plt.title('True RF Spectrum at PA Output')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Normalized PSD (dB)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    evaluate_cascade()