import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 设置字体，防止中文乱码 (如果你系统没这个字体，可能会显示方框，不影响分析)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
FILE_NAME = 'DPD_data.mat'

# ================= 1. 数据读取 =================
print(f"1. 正在读取 {FILE_NAME} ...")
try:
    data = scipy.io.loadmat(FILE_NAME)
    # 提取并展平数据
    x = data['x'].flatten()  # 输入
    y = data['y'].flatten()  # 输出
    print(f"   -> 读取成功！共 {len(x)} 个采样点。")

    # 计算幅度
    x_amp = np.abs(x)
    y_amp = np.abs(y)

    # 准备一个“安全面具” (Mask)
    # 只分析输入幅度大于 0.01 的点，避免除以 0 产生的噪音干扰
    mask = x_amp > 0.01
    x_safe = x[mask]
    y_safe = y[mask]
    x_amp_safe = x_amp[mask]

except Exception as e:
    print(f"出错啦：{e}")
    exit()

# ================= 2. 第一张图：AM-AM (幅度线性度) =================
print("\n[图1/4] 正在绘制 AM-AM 图...")
plt.figure(figsize=(10, 6))
# s=0.01 表示点非常小，c='blue' 蓝色，alpha=0.3 透明度
plt.scatter(x_amp, y_amp, s=0.01, c='blue', alpha=0.3)
plt.title("1. AM-AM 特性 (输入幅度 vs 输出幅度)")
plt.xlabel("输入幅度 |x|")
plt.ylabel("输出幅度 |y|")
plt.grid(True)
print("   -> 请观察：线条是否变粗(记忆效应)？顶部是否变弯(压缩)？")
plt.show()  # 【注意】必须关掉这个窗口，代码才会继续！

# ================= 3. 第二张图：Gain (增益压缩) =================
print("\n[图2/4] 正在绘制 增益(Gain) 图...")
# 计算增益 = 输出幅度 / 输入幅度
gain = np.abs(y_safe) / np.abs(x_safe)

plt.figure(figsize=(10, 6))
plt.scatter(x_amp_safe, gain, s=0.01, c='green', alpha=0.3)
plt.title("2. 增益曲线 (Gain vs Input)")
plt.xlabel("输入幅度 |x|")
plt.ylabel("增益 (Mag Gain)")
plt.grid(True)
print("   -> 请观察：最右侧是否掉下来了？(掉下来代表功放推不动了)")
plt.show()

# ================= 4. 第三张图：AM-PM (相位失真) =================
print("\n[图3/4] 正在绘制 AM-PM 相位失真图...")
# 计算相位差：angle(输出/输入)
# 结果是弧度，要乘以 180/pi 转成角度
phase_diff = np.angle(y_safe / x_safe) * (180 / np.pi)

# 有时候相位会跳变，我们把中心对齐到 0 度附近方便观察
phase_diff = phase_diff - np.mean(phase_diff)

plt.figure(figsize=(10, 6))
plt.scatter(x_amp_safe, phase_diff, s=0.01, c='purple', alpha=0.3)
plt.title("3. AM-PM 特性 (相位失真)")
plt.xlabel("输入幅度 |x|")
plt.ylabel("相位偏差 (度)")
plt.grid(True)
print("   -> 请观察：线条是平的吗？如果右边翘起来或散开，说明有相位失真！")
plt.show()

# ================= 5. 第四张图：频谱图 (PSD) =================
print("\n[图4/4] 正在绘制 频谱图 (PSD)...")
plt.figure(figsize=(10, 6))
plt.psd(x, NFFT=1024, Fs=491.52, label='输入 (理想)', color='blue')
plt.psd(y, NFFT=1024, Fs=491.52, label='输出 (失真)', color='red', alpha=0.7)
plt.title("4. 频谱对比 (这是最终考核标准)")
plt.legend()
print("   -> 请观察：红色的“肩膀”比蓝色高多少？这就是我们要消除的噪音。")
plt.show()

print("\n✅ 所有可视化结束！")