import torch
import torch.nn as nn
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.fft

# ================= 1. 基础配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 必须保持双精度，这是精度的底线
torch.set_default_dtype(torch.float64)
print(f"🚀 [Step 30: 极致算力 ILA 狙击版 (逻辑修正版)] 启动！")
print(f"   战术：严格遵循赛题！输入 x (干净信号) -> 预测 y (理想解药)")
print(f"   审计：全系统预估消耗 663 个乘法器 (上限 800)！")

# ================= 2. 数据处理与 13 阶特征构建 =================
data = scipy.io.loadmat('DPD_data.mat')
raw_x = data['x'].flatten()
raw_y = data['y'].flatten()

x_norm = raw_x / np.max(np.abs(raw_x))
y_norm = raw_y / np.max(np.abs(raw_y))

memory_depth = 8  # 滑窗大小 9 (0到8)

# ★ 修正 1：输入必须是干净信号 x ★
u = x_norm
real = np.real(u)
imag = np.imag(u)
mag = np.abs(u)

# ★ 消耗极小算力的 13 阶特征链 ★
mag2 = mag ** 2
mag4 = mag2 ** 2
mag6 = mag4 * mag2
mag8 = mag4 ** 2
mag10 = mag8 * mag2
mag12 = mag6 ** 2

# 差分特征 (捕捉高频毛刺)
u_delay = np.concatenate([[0], u[:-1]])
diff_u = u - u_delay
diff_r = np.real(diff_u)
diff_i = np.imag(diff_u)
diff_mag2_r = diff_r * mag2
diff_mag2_i = diff_i * mag2

# 18 通道全火力配置
channels = [
    real, imag,
    real * mag2, imag * mag2,
    real * mag4, imag * mag4,
    real * mag6, imag * mag6,
    real * mag8, imag * mag8,
    real * mag10, imag * mag10,
    real * mag12, imag * mag12,
    diff_r, diff_i,
    diff_mag2_r, diff_mag2_i
]

pad_zeros = np.zeros(memory_depth)
padded_channels = [np.concatenate([pad_zeros, c]) for c in channels]

X_list = []
for c in padded_channels:
    stack = []
    for i in range(memory_depth + 1):
        if i == 0:
            stack.append(c[memory_depth:])
        else:
            stack.append(c[memory_depth - i: -i])
    X_list.append(np.stack(stack, axis=1))

X_np = np.stack(X_list, axis=1)  # [N, 18, 9]

# ★ 修正 2：目标必须是带补偿的理想解药 y ★
Target_np = np.stack([np.real(y_norm), np.imag(y_norm)], axis=1)

min_len = min(len(X_np), len(Target_np))
X_np = X_np[:min_len]
Target_np = Target_np[:min_len]

# 80% 训练，20% 验证
split_idx = int(min_len * 0.8)
X_train = torch.from_numpy(X_np[:split_idx]).double().to(DEVICE)
Y_train = torch.from_numpy(Target_np[:split_idx]).double().to(DEVICE)

X_val = torch.from_numpy(X_np[split_idx:]).double().to(DEVICE)
y_val_dirty_complex = raw_y[split_idx:split_idx + len(X_val)]

print(f"✅ 18 通道高阶数据就绪。")


# ================= 3. 核心：663 乘法器狙击模型 =================
class SniperDPD(nn.Module):
    def __init__(self):
        super(SniperDPD, self).__init__()
        # 层1：记忆深度卷积 (18 * 9 = 162 乘法器)
        self.depth_conv = nn.Conv1d(18, 18, kernel_size=memory_depth + 1, groups=18, bias=False)
        # 层2：特征压缩组合 (18 * 24 = 432 乘法器)
        self.point_conv1 = nn.Conv1d(18, 24, kernel_size=1)
        # ★ 换装 PReLU：消除死区神经元，让微小的高阶互调梯度完美回传
        self.act1 = nn.PReLU(num_parameters=24)

        # 层3：输出 I 和 Q (24 * 2 = 48 乘法器)
        self.point_conv2 = nn.Conv1d(24, 2, kernel_size=1)

        self.skip_scale = nn.Parameter(torch.ones(1, dtype=torch.float64))

    def forward(self, x):
        skip = x[:, 0:2, 0] * self.skip_scale
        out = self.depth_conv(x)
        out = self.act1(self.point_conv1(out))
        out = self.point_conv2(out)
        return skip + out.squeeze(-1)


model = SniperDPD().double().to(DEVICE)


# ================= 4. 频域 Loss (升级为对数谱惩罚) =================
def freq_loss(pred, target):
    pred_c = torch.view_as_complex(pred)
    target_c = torch.view_as_complex(target)

    pred_fft = torch.fft.fft(pred_c) / len(pred_c)
    target_fft = torch.fft.fft(target_c) / len(target_c)

    # ★ 核心改动：转到对数域 (类似 dB) ★
    # 强制网络关注那些绝对数值极小，但对数差异巨大的高频远端区域
    log_pred = torch.log10(torch.abs(pred_fft) + 1e-12)
    log_target = torch.log10(torch.abs(target_fft) + 1e-12)

    return torch.mean((log_pred - log_target) ** 2)


# ================= 5. 高速训练循环 =================
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=300, factor=0.5)

print("\n🚀 开始高速训练 (5000轮)...")
start_time = time.time()

TOTAL_EPOCHS = 5000
WARMUP_EPOCHS = 500  # ★ 缩短 Warmup，提早 500 轮开始压迫肩膀

for epoch in range(TOTAL_EPOCHS + 1):
    model.train()
    pred = model(X_train)

    loss_time = nn.MSELoss()(pred, Y_train)

    # 动态加入一点频域惩罚，让预测出来的肩膀更贴合官方解药的肩膀
    if epoch < WARMUP_EPOCHS:
        alpha = 0.0
    else:
        alpha = 0.005  # ★ 对数 Loss 的 alpha 调大，给足压力

    loss_f = freq_loss(pred, Y_train)
    loss = loss_time + (alpha * loss_f)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss.item())

    if epoch % 500 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).cpu().numpy()
            v_complex = val_pred[:, 0] + 1j * val_pred[:, 1]

            # ★ 修正 3：验证阶段拿官方解药 y 作为 Target 来算误差 ★
            r_complex = y_val_dirty_complex / np.max(np.abs(raw_y))
            err = np.sum(np.abs(r_complex - v_complex) ** 2)
            sig = np.sum(np.abs(r_complex) ** 2)
            v_nmse = 10 * np.log10(err / sig)
        print(f"Epoch {epoch}: Train Loss = {loss.item():.7f} | Val y-NMSE = {v_nmse:.2f} dB")

print(f"训练耗时 {time.time() - start_time:.1f}s")

# ================= 6. 最终验证与战果 =================
model.eval()
with torch.no_grad():
    z_pred = model(X_val).cpu().numpy()

z_complex = z_pred[:, 0] + 1j * z_pred[:, 1]

# ★ 修正 4：计算最终预测解药和官方解药的拟合度 ★
y_ref_normalized = y_val_dirty_complex / np.max(np.abs(raw_y))

err = np.sum(np.abs(y_ref_normalized - z_complex) ** 2)
sig = np.sum(np.abs(y_ref_normalized) ** 2)
nmse = 10 * np.log10(err / sig)

print(f"\n🏆 [Step 30: 极致狙击战果 (逻辑修正)]")
print(f"   📊 预估算力: 663 实数乘法器 (符合 <800 限制)")
print(f"   ✅ 解药拟合度 NMSE (E(f)): {nmse:.2f} dB")
print(f"   注：由于是直接拟合解药 y，本地不再进行伪 ACLR 测算。")

# 只对比目标 y 和 预测的 y
plt.figure(figsize=(12, 7))
plt.psd(y_ref_normalized, NFFT=1024, Fs=491.52, color='red', alpha=0.5, label='官方理想解药 (y)')
plt.psd(z_complex, NFFT=1024, Fs=491.52, color='green', linestyle='--', linewidth=2, label='Sniper 预测解药 (pr_y)')
plt.title(f"Sniper DPD | 解药拟合度 NMSE = {nmse:.2f} dB")
plt.legend()
plt.ylim([-75, 5])
plt.grid(True)
plt.show()