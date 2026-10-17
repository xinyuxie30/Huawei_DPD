import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


# ==========================================
# 1. 华为赛题适配：数据截帧加载器
# ==========================================
class HuaweiFramedDataset(Dataset):
    """
    基于 OpenDPDv2 IQFramedDataset 思想适配的华为赛题数据集加载器
    利用滑动窗口（Sliding Window）将几十万点的连续长序列切分为重叠短帧，彻底解决梯度消失。
    """

    def __init__(self, features, targets, frame_length=1024, stride_length=512):
        """
        :param features: 已经提取好18通道物理特征的输入数据，维度 (N, 18)
        :param targets: 华为官方提供的理想解药信号，维度 (N, 2)
        :param frame_length: 每一帧的时间长度（例如 1024）
        :param stride_length: 滑动窗口的步长（例如 512，代表 50% 的帧间重叠）
        """
        # 将传入的 NumPy 数组或 Tensor 统一转为 PyTorch Tensor
        self.features = torch.as_tensor(features, dtype=torch.float64)
        self.targets = torch.as_tensor(targets, dtype=torch.float64)

        self.frame_length = frame_length
        self.stride_length = stride_length
        self.sequence_length = self.features.shape[0]

        # 精确复刻 OpenDPDv2 的帧数计算公式
        # num_frames = (sequence_length - frame_length) // stride_length + 1
        self.num_frames = (self.sequence_length - self.frame_length) // self.stride_length + 1

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # 动态切片，不预先占用巨量内存
        start_idx = idx * self.stride_length
        end_idx = start_idx + self.frame_length

        # 提取当前短帧的特征和标签
        # feature 维度: (frame_length, 18)
        # target 维度: (frame_length, 2)
        frame_feature = self.features[start_idx:end_idx, :]
        frame_target = self.targets[start_idx:end_idx, :]

        # 调整维度以适应 PyTorch 的 Conv1D (Batch, Channels, Length)
        # 将 (1024, 18) 转为 (18, 1024)
        frame_feature = frame_feature.transpose(0, 1)
        frame_target = frame_target.transpose(0, 1)

        return frame_feature, frame_target


# ==========================================
# 2. 华为赛题适配：满血版 TCN 算力引擎
# ==========================================
class HuaweiTCN_Sniper(nn.Module):
    """
    专为华为赛题重构的满血版 TCN 预失真网络。
    彻底剥离内置特征提取，直接接收 18 通道高阶物理特征。
    当 hidden_channels=20 时，严格消耗 800 个实数乘法器。
    """

    def __init__(self, in_channels=18, hidden_channels=20):
        super(HuaweiTCN_Sniper, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = 2
        self.kernel_size = 5

        # 核心 TCN 残差引擎 (保持 FP64 精度)
        self.network = nn.Sequential(
            # 1. 跨通道特征融合 (18 -> 20)
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=1, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),

            # 2. 深度可分离空洞卷积提取长时记忆 (感受野极速扩张)
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

            # 3. 降维输出 (20 -> 2)
            nn.Conv1d(self.hidden_channels, self.out_channels, kernel_size=1, bias=False, dtype=torch.float64),
        )

    def forward(self, x_18ch):
        """
        :param x_18ch: 外部预先计算好的 18 通道特征，维度 (Batch, 18, Seq_Len)
        :return: 预测的解药信号，维度 (Batch, 2, Seq_Len)
        """
        # 网络提取残差补偿
        residual = self.network(x_18ch)

        # 提取输入中的原始 I/Q 信号 (假设前两个通道是纯 I 和 Q)
        original_iq = x_18ch[:, 0:2, :]

        # 预失真输出 = 线性直通信号 + 非线性/记忆残差补偿
        out = original_iq + residual
        return out


# ==========================================
# 3. 华为赛题适配：评估与绘图工具
# ==========================================
def moving_average(x, w):
    """滑动平均滤波，用于平滑 PSD 曲线"""
    return np.convolve(x, np.ones(w), 'valid') / w


def to_complex(iq_data):
    """
    自动将 PyTorch Tensor 或 Numpy Array 的 I/Q 数据转换为 1D 复数 Numpy 数组
    支持形状: (2, Seq_Len) 或 (Seq_Len, 2)
    """
    # 如果是 PyTorch Tensor，先转到 CPU 并转为 numpy
    if hasattr(iq_data, 'detach'):
        iq_data = iq_data.detach().cpu().numpy()

    # 去除 batch 维度 (如果传入的是单个 batch 的数据，例如 shape 为 (1, 2, N))
    iq_data = np.squeeze(iq_data)

    # 确保 I 和 Q 分别在第一维和第二维
    if iq_data.shape[0] != 2 and iq_data.shape[-1] == 2:
        iq_data = iq_data.T  # 转置为 (2, Seq_Len)

    # 组合为复数信号: I + j*Q
    complex_signal = iq_data[0] + 1j * iq_data[1]
    return complex_signal


def plot_huawei_psd(pred_iq, target_iq, label_1="Prediction (DPD)", label_2="Ground Truth", fs=800e6, nperseg=2560,
                    smoothing_window=10):
    """
    适配华为赛题的 PSD (功率谱密度) 绘制函数。
    用来直观查看网络是否成功压低了带外“肩膀”(Out-of-band emissions)。
    """
    complex_signal_1 = to_complex(pred_iq)
    complex_signal_2 = to_complex(target_iq)

    # 截取计算长度
    complex_signal_1_subset = complex_signal_1[:nperseg]
    complex_signal_2_subset = complex_signal_2[:nperseg]

    # Welch 法计算 PSD
    frequencies_1, psd_1 = welch(complex_signal_1_subset, fs=fs, nperseg=nperseg, return_onesided=False)
    frequencies_2, psd_2 = welch(complex_signal_2_subset, fs=fs, nperseg=nperseg, return_onesided=False)

    # 频谱移位 (FFT Shift)，将 0 频移到中心
    half_nfft = int(nperseg / 2)
    frequencies_1 = np.concatenate((frequencies_1[half_nfft:], frequencies_1[:half_nfft]))
    frequencies_2 = np.concatenate((frequencies_2[half_nfft:], frequencies_2[:half_nfft]))
    psd_1 = np.concatenate((psd_1[half_nfft:], psd_1[:half_nfft]))
    psd_2 = np.concatenate((psd_2[half_nfft:], psd_2[:half_nfft]))

    # 对数归一化
    psd_1_normalized = 10 * np.log10(psd_1 / np.max(psd_1))
    psd_2_normalized = 10 * np.log10(psd_2 / np.max(psd_2))

    # 平滑处理
    psd_1_smoothed = moving_average(psd_1_normalized, smoothing_window)
    psd_2_smoothed = moving_average(psd_2_normalized, smoothing_window)

    # 调整频率轴以匹配卷积后的长度
    freq_adjusted = frequencies_1[int(smoothing_window / 2): -int(smoothing_window / 2) + 1]

    plt.figure(figsize=(10, 6))
    plt.plot(freq_adjusted / 1e6, psd_1_smoothed, label=label_1, color='blue', alpha=0.8)
    plt.plot(freq_adjusted / 1e6, psd_2_smoothed, label=label_2, color='red', linestyle='--', alpha=0.8)

    plt.title('Huawei DPD: Normalized Power Spectral Density (PSD)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Normalized PSD (dB)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_huawei_constellation(pred_iq, target_iq, nperseg=2560, n_subc=64, n_ch=10):
    """
    适配华为赛题的星座图绘制函数。
    注意：对 5G OFDM 信号，此方法仅能提供粗略的频域散点分布参考。
    """
    complex_signal_1 = to_complex(pred_iq)
    complex_signal_2 = to_complex(target_iq)

    # 截取并计算 FFT
    spectrum_prediction = np.fft.fftshift(np.fft.fft(complex_signal_1[:nperseg], n=nperseg))
    spectrum_ground_truth = np.fft.fftshift(np.fft.fft(complex_signal_2[:nperseg], n=nperseg))

    X_pred, y_pred, X_gt, y_gt = [], [], [], []

    # 提取子载波并归一化
    for i in range(n_ch):
        left_index = int(nperseg / 2) - n_subc * int(n_ch / 2)

        pred_i = spectrum_prediction[left_index + n_subc * i: left_index + n_subc * (i + 1)]
        pred_i /= np.max([np.max(np.abs(np.real(pred_i))), np.max(np.abs(np.imag(pred_i)))])
        X_pred.append(pred_i.real)
        y_pred.append(pred_i.imag)

        gt_i = spectrum_ground_truth[left_index + n_subc * i: left_index + n_subc * (i + 1)]
        gt_i /= np.max([np.max(np.abs(np.real(gt_i))), np.max(np.abs(np.imag(gt_i)))])
        X_gt.append(gt_i.real)
        y_gt.append(gt_i.imag)

    X_pred = np.concatenate(X_pred, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    X_gt = np.concatenate(X_gt, axis=0)
    y_gt = np.concatenate(y_gt, axis=0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_pred, y_pred, c='blue', label="Prediction (DPD)", alpha=0.3, edgecolors='none', s=10)
    ax.scatter(X_gt, y_gt, c='red', label="Ground Truth", alpha=0.3, edgecolors='none', s=10)

    ax.legend(loc='upper right')
    ax.grid(True)
    plt.title('Huawei DPD: FFT Scatter / Constellation Diagram')
    plt.xlabel('I (Real part)')
    plt.ylabel('Q (Imaginary part)')
    plt.tight_layout()
    plt.show()