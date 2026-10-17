import numpy as np
import scipy.io
import scipy.signal as signal
import matplotlib.pyplot as plt


def analyze_dataset():
    print(">>> 开始解析 DPD_data.mat 数据集物理特性...")

    # 1. 数据加载与基础量级分析
    data = scipy.io.loadmat('DPD_data.mat')
    x = data['x'].flatten()
    y = data['y'].flatten()

    # 转换为复数
    X_complex = x
    Y_complex = y

    # RMS 功率计算
    P_in_rms = np.sqrt(np.mean(np.abs(X_complex) ** 2))
    P_out_rms = np.sqrt(np.mean(np.abs(Y_complex) ** 2))
    print(f"[1] 基础功率特性:")
    print(f"    - 输入 RMS: {P_in_rms:.4f}")
    print(f"    - 输出 RMS: {P_out_rms:.4f}")
    print(f"    - 宏观线性增益: {P_out_rms / P_in_rms:.4f}")

    # 2. 互相关分析 (寻找时序错位)
    print("\n[2] 时序对齐分析:")
    # 截取前 10000 个点进行互相关计算以加速
    corr = signal.correlate(np.abs(Y_complex[:10000]), np.abs(X_complex[:10000]), mode='full')
    lags = signal.correlation_lags(10000, 10000, mode='full')
    best_lag = lags[np.argmax(corr)]
    print(f"    - 整数级延迟 (Integer Delay): {best_lag} 个采样点")
    if best_lag != 0:
        print("    ! 警告: 数据集存在整数级错位，必须在预处理时强制对齐。")
    else:
        print("    - 数据已在整数级对齐。")

    # 3. 线性极限测试 (Wiener Filter / 最小二乘 FIR)
    print("\n[3] 线性记忆极限评估:")
    # 我们测试一个标量 G 和一个 11-Tap 的 FIR 滤波器能把误差压到多少
    # 标量 G
    G_opt = np.vdot(X_complex, Y_complex) / np.vdot(X_complex, X_complex)
    Y_linear_scalar = X_complex * G_opt
    nmse_scalar = 10 * np.log10(np.sum(np.abs(Y_complex - Y_linear_scalar) ** 2) / np.sum(np.abs(Y_complex) ** 2))

    # 11-Tap FIR
    pad_X = np.pad(X_complex, (5, 5), mode='constant')
    X_matrix = np.array([pad_X[i:i + 11] for i in range(len(X_complex))])
    # LS 求解 w = (X^H X)^-1 X^H Y
    w_opt = np.linalg.lstsq(X_matrix, Y_complex, rcond=None)[0]
    Y_linear_fir = X_matrix @ w_opt
    nmse_fir = 10 * np.log10(np.sum(np.abs(Y_complex - Y_linear_fir) ** 2) / np.sum(np.abs(Y_complex) ** 2))

    print(f"    - 纯标量增益 (G_opt) 极限 NMSE: {nmse_scalar:.2f} dB")
    print(f"    - 11-Tap FIR 滤波器 极限 NMSE: {nmse_fir:.2f} dB")
    if abs(nmse_scalar - nmse_fir) > 2.0:
        print("    ! 结论: 线性群延迟极其严重，必须在预处理/网络前置接入多抽头 FIR。")

    # 4. AM-AM 与 AM-PM 提取与绘图
    print("\n>>> 正在绘制物理特征图...")
    # 降采样绘图，避免点数过多导致卡顿
    plot_step = 10
    X_plot = X_complex[::plot_step]
    Y_plot = Y_complex[::plot_step]

    # 对齐增益与相位以便观察失真
    Y_aligned = Y_plot / G_opt

    mag_X = np.abs(X_plot)
    mag_Y = np.abs(Y_aligned)
    phase_diff = np.angle(Y_aligned) - np.angle(X_plot)
    # 将相位差限制在 -pi 到 pi
    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # AM-AM
    axs[0].scatter(mag_X, mag_Y, s=1, alpha=0.3, c='blue')
    axs[0].plot([0, np.max(mag_X)], [0, np.max(mag_X)], 'r--', label='Ideal Linear')
    axs[0].set_title('AM-AM Characteristic')
    axs[0].set_xlabel('Input Magnitude |x|')
    axs[0].set_ylabel('Normalized Output Magnitude |y|')
    axs[0].grid(True, linestyle=':')
    axs[0].legend()

    # AM-PM
    axs[1].scatter(mag_X, np.degrees(phase_diff), s=1, alpha=0.3, c='green')
    axs[1].set_title('AM-PM Distortion')
    axs[1].set_xlabel('Input Magnitude |x|')
    axs[1].set_ylabel('Phase Deviation (Degrees)')
    axs[1].grid(True, linestyle=':')

    # 误差频域功率谱密度 (PSD)
    axs[2].psd(Y_complex, Fs=491.52e6, NFFT=4096, color='black', label='Output Y')
    axs[2].psd(Y_complex - Y_linear_scalar, Fs=491.52e6, NFFT=4096, color='red', label='Residual (Scalar G)')
    axs[2].set_title('Power Spectral Density')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('dataset_physics_analysis.png', dpi=300)
    print(">>> 分析完成，图表已保存至 'dataset_physics_analysis.png'")


if __name__ == '__main__':
    analyze_dataset()