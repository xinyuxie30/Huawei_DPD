import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==========================================
# LUT 参数配置
# ==========================================
NUM_BINS = 1024


def generate_lut(x_train, y_train, num_bins):
    """
    使用无偏最小二乘法 (Least Squares) 生成高稳定复数 LUT
    """
    amp_train = np.abs(x_train)
    max_amp = np.max(amp_train)

    # 使用 Y*conj(X) 和 |X|^2 进行分箱累计，彻底避免除零爆炸
    cross_corr = np.zeros(num_bins, dtype=np.complex128)
    auto_corr = np.zeros(num_bins, dtype=np.float64)

    bin_indices = np.clip((amp_train / max_amp * (num_bins - 1)).astype(int), 0, num_bins - 1)

    for i in range(len(x_train)):
        idx = bin_indices[i]
        cross_corr[idx] += y_train[i] * np.conj(x_train[i])
        auto_corr[idx] += np.abs(x_train[i]) ** 2

    lut = np.zeros(num_bins, dtype=np.complex128)
    valid_indices = []

    for i in range(num_bins):
        if auto_corr[i] > 1e-12:  # 只有该区间有足够的能量时才计算增益
            lut[i] = cross_corr[i] / auto_corr[i]
            valid_indices.append(i)

    valid_indices = np.array(valid_indices)

    # 线性插值填补空区间
    if len(valid_indices) > 0 and len(valid_indices) < num_bins:
        f = interp1d(valid_indices, lut[valid_indices], kind='linear', bounds_error=False,
                     fill_value=(lut[valid_indices[0]], lut[valid_indices[-1]]))
        lut = f(np.arange(num_bins))

    return lut, max_amp


def apply_lut(x, lut, max_amp):
    num_bins = len(lut)
    amp_x = np.abs(x)
    bin_indices = np.clip((amp_x / max_amp * (num_bins - 1)).astype(int), 0, num_bins - 1)
    gains = lut[bin_indices]
    return x * gains


def main():
    print(">>> 正在加载数据并进行 RMS 归一化...")
    data = scipy.io.loadmat('DPD_data.mat')
    raw_x, raw_y = data['x'].flatten(), data['y'].flatten()

    RMS_X = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    RMS_Y = np.sqrt(np.mean(np.abs(raw_y) ** 2))
    x_c = raw_x / RMS_X
    y_c = raw_y / RMS_Y

    split_idx = int(len(x_c) * 0.8)
    x_train, y_train = x_c[:split_idx], y_c[:split_idx]
    x_val, y_val = x_c[split_idx:], y_c[split_idx:]

    print(f">>> 正在生成 1D 复数 LUT (深度: {NUM_BINS})...")
    lut, max_amp = generate_lut(x_train, y_train, NUM_BINS)

    print(">>> 正在评估验证集...")
    y_val_lut = apply_lut(x_val, lut, max_amp)
    residual_val = y_val - y_val_lut

    error_power = np.sum(np.abs(residual_val) ** 2)
    target_power = np.sum(np.abs(y_val) ** 2)
    nmse_lut = 10 * np.log10(error_power / target_power)

    print("-" * 50)
    print(f"[ 0 MACs 性能基线 ]")
    print(f"验证集 LUT NMSE: {nmse_lut:.2f} dB")
    print("-" * 50)

    # ==========================================
    # 可视化分析 (已加入抗异常值画图逻辑)
    # ==========================================
    plot_indices = np.random.choice(len(x_val), 5000, replace=False)
    x_plot = x_val[plot_indices]
    y_plot = y_val[plot_indices]
    y_lut_plot = y_val_lut[plot_indices]
    res_plot = residual_val[plot_indices]

    amp_x = np.abs(x_plot)

    # 计算相位偏差并限制在 -180 到 180 度之间
    phase_diff_y = np.angle(y_plot / (x_plot + 1e-12)) * 180 / np.pi
    phase_diff_lut = np.angle(y_lut_plot / (x_plot + 1e-12)) * 180 / np.pi

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 图1: AM-AM
    axs[0].scatter(amp_x, np.abs(y_plot), s=1, c='blue', alpha=0.3, label='Original PA (w/o DPD)')
    axs[0].scatter(amp_x, np.abs(y_lut_plot), s=1, c='red', alpha=1.0, label='LUT Output')
    # 动态限制Y轴，切除极个别离谱的噪声点
    axs[0].set_ylim(0, np.percentile(np.abs(y_plot), 99.9) * 1.1)
    axs[0].set_title('AM-AM Characteristic')
    axs[0].set_xlabel('Input Amplitude |x|')
    axs[0].set_ylabel('Output Amplitude |y|')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # 图2: AM-PM
    axs[1].scatter(amp_x, phase_diff_y, s=1, c='blue', alpha=0.3, label='Original PA (w/o DPD)')
    axs[1].scatter(amp_x, phase_diff_lut, s=1, c='red', alpha=1.0, label='LUT Output')
    # 限制相位显示范围在合理区间内
    axs[1].set_ylim(-100, 100)
    axs[1].set_title('AM-PM Characteristic')
    axs[1].set_xlabel('Input Amplitude |x|')
    axs[1].set_ylabel('Phase Deviation (Degrees)')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.7)

    # 图3: 残差
    axs[2].scatter(amp_x, np.abs(res_plot), s=1, c='green', alpha=0.3)
    axs[2].axhline(0, color='r', linestyle='--')
    axs[2].set_ylim(0, np.percentile(np.abs(res_plot), 99.9) * 1.5)
    axs[2].set_title('Extracted Memory Residuals |Y_true - Y_lut|')
    axs[2].set_xlabel('Input Amplitude |x|')
    axs[2].set_ylabel('Residual Amplitude')
    axs[2].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('LUT_baseline_analysis.png', dpi=300)
    print(">>> 高清物理分析图已重新保存为 LUT_baseline_analysis.png")

    # 导出纯残差数据
    y_lut_full = apply_lut(x_c, lut, max_amp)
    residual_full = y_c - y_lut_full

    out_dict = {'x': raw_x, 'y_res': residual_full * RMS_Y, 'y_lut': y_lut_full * RMS_Y}
    scipy.io.savemat('DPD_residual_data.mat', out_dict)
    print(">>> 纯记忆效应残差数据集已更新并保存为 DPD_residual_data.mat")


if __name__ == '__main__':
    main()