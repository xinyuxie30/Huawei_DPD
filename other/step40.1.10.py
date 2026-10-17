import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 通用组件：物理注意力与基础残差块
# ==========================================
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
# 1. 代理 PA 模型 (Surrogate PA) - 无算力限制
# 负责以绝对精度拟合真实物理 PA 的正向特征 (X -> Y)
# ==========================================
class SurrogatePAModel(nn.Module):
    def __init__(self, g_real_init, g_imag_init):
        super(SurrogatePAModel, self).__init__()
        self.hidden = 32  # 满血容量，确保 PA 代理精度极高

        self.g_real = nn.Parameter(torch.tensor(g_real_init, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(g_imag_init, dtype=torch.float64))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.in_conv = nn.Conv1d(6, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()

        self.tcn_blocks = nn.ModuleList()
        for dilation in [1, 2, 4, 8, 16, 32]:
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
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp7 = amp ** 7

        x3_i, x3_q = i_x * amp2, q_x * amp2
        lin3_i = x3_i * self.g3_real - x3_q * self.g3_imag
        lin3_q = x3_i * self.g3_imag + x3_q * self.g3_real

        lin_out = torch.cat([lin_i + lin3_i, lin_q + lin3_q], dim=1)
        i_future, q_future = torch.roll(i_x, shifts=-1, dims=2), torch.roll(q_x, shifts=-1, dims=2)
        x_6ch = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)

        feat = self.act(self.in_conv(x_6ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        feat = self.attn(feat, amp, amp3, amp7)

        return lin_out + self.out_conv(feat)


# ==========================================
# 2. 轻量级 DPD 模型 - 严格遵守 <800 MACs
# 负责在算力红线内寻找最优的预失真反函数
# ==========================================
class LightweightDPDNet(nn.Module):
    def __init__(self, g_real_inv, g_imag_inv):
        super(LightweightDPDNet, self).__init__()
        self.hidden = 10  # 严格控制在 776 MACs

        # DPD 的线性骨架应当初始化为物理 PA 增益的逆函数
        self.g_real = nn.Parameter(torch.tensor(g_real_inv, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(g_imag_inv, dtype=torch.float64))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

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
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2, amp3, amp7 = amp ** 2, amp ** 3, amp ** 7

        x3_i, x3_q = i_x * amp2, q_x * amp2
        lin3_i = x3_i * self.g3_real - x3_q * self.g3_imag
        lin3_q = x3_i * self.g3_imag + x3_q * self.g3_real

        lin_out = torch.cat([lin_i + lin3_i, lin_q + lin3_q], dim=1)
        i_future, q_future = torch.roll(i_x, shifts=-1, dims=2), torch.roll(q_x, shifts=-1, dims=2)
        x_6ch = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)

        feat = self.act(self.in_conv(x_6ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat)
        feat = self.attn(feat, amp, amp3, amp7)

        return lin_out + self.out_conv(feat)


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


def get_ideal_linear_target(batch_x, g_real, g_imag):
    """
    计算 DLA 中的端到端系统目标输出
    理想情况下，系统应为一个仅包含主增益 G_opt 的完美线性放大器
    """
    i_x, q_x = batch_x[:, 0:1, :], batch_x[:, 1:2, :]
    lin_i = i_x * g_real - q_x * g_imag
    lin_q = i_x * g_imag + q_x * g_real
    return torch.cat([lin_i, lin_q], dim=1)


def main():
    data = scipy.io.loadmat('DPD_data.mat')
    raw_x, raw_y = data['x'].flatten(), data['y'].flatten()

    RMS_X = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    RMS_Y = np.sqrt(np.mean(np.abs(raw_y) ** 2))
    x_c = raw_x / RMS_X
    y_c = raw_y / RMS_Y

    # 计算 PA 物理正向增益 G_opt
    G_opt = np.vdot(x_c, y_c) / np.vdot(x_c, x_c)
    pa_g_real = np.real(G_opt)
    pa_g_imag = np.imag(G_opt)

    # 计算 DPD 物理逆向增益 G_inv
    G_inv = 1.0 / G_opt
    dpd_g_real = np.real(G_inv)
    dpd_g_imag = np.imag(G_inv)

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Target_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]

    train_loader = DataLoader(FramedDataset(X_train, Y_train, 800, 32),
                              batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(FramedDataset(X_val, Y_val, 800, 800),
                            batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    pa_surrogate = SurrogatePAModel(pa_g_real, pa_g_imag).to(DEVICE)
    dpd_model = LightweightDPDNet(dpd_g_real, dpd_g_imag).to(DEVICE)
    criterion = nn.MSELoss()

    # ====================================================================
    # 阶段 1：训练/加载 代理 PA 模型 (必须精确捕获物理前向响应)
    # ====================================================================
    surrogate_path = 'best_surrogate_pa.pth'
    if os.path.exists(surrogate_path):
        print(f"\n>>> [Stage 1] 检测到已存在的 PA 代理模型权重，直接加载: {surrogate_path}")
        pa_surrogate.load_state_dict(torch.load(surrogate_path, map_location=DEVICE))
    else:
        print(f"\n>>> [Stage 1] 未找到 PA 代理模型，开始训练代理 PA (H=32)...")
        pa_optimizer = torch.optim.AdamW(pa_surrogate.parameters(), lr=5e-3)
        pa_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pa_optimizer, mode='min', factor=0.5, patience=15)

        best_pa_loss = float('inf')
        for epoch in range(150):  # 代理模型收敛很快，150轮足以压穿 -40dB
            pa_surrogate.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                pa_optimizer.zero_grad()
                preds = pa_surrogate(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                pa_optimizer.step()

            pa_surrogate.eval()
            total_err, total_tgt = 0.0, 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                    preds = pa_surrogate(batch_x)
                    total_err += torch.sum((preds - batch_y) ** 2).item()
                    total_tgt += torch.sum(batch_y ** 2).item()
            val_nmse = 10 * np.log10(total_err / total_tgt)
            pa_scheduler.step(val_nmse)
            print(f"PA Training | Epoch {epoch + 1}/150 | Val NMSE: {val_nmse:.2f} dB")

            if val_nmse < best_pa_loss:
                best_pa_loss = val_nmse
                torch.save(pa_surrogate.state_dict(), surrogate_path)

        pa_surrogate.load_state_dict(torch.load(surrogate_path, map_location=DEVICE))
        print(f">>> [Stage 1] PA 代理模型训练完成！锁定极值精度: {best_pa_loss:.2f} dB")

    # ====================================================================
    # 阶段 2：直接学习架构 (DLA) 下的级联 DPD 训练
    # ====================================================================
    print(f"\n>>> [Stage 2] 开始 DLA 端到端级联训练 DPD (776 MACs)...")

    # 冻结代理 PA 模型的所有梯度
    pa_surrogate.eval()
    for param in pa_surrogate.parameters():
        param.requires_grad = False

    dpd_optimizer = torch.optim.AdamW(dpd_model.parameters(), lr=5e-3, weight_decay=1e-6)
    dpd_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dpd_optimizer, mode='min', factor=0.5, patience=15)

    EPOCHS_DLA = 500
    best_dla_loss = float('inf')
    history_epochs, history_dla_nmse = [], []
    start_time = time.time()

    for epoch in range(EPOCHS_DLA):
        dpd_model.train()
        total_train_error_power, total_train_target_power = 0.0, 0.0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)

            # 生成系统级理想线性目标：我们希望级联后的系统表现为理想的线性增益 G_opt
            target_y = get_ideal_linear_target(batch_x, pa_g_real, pa_g_imag).to(DEVICE)

            dpd_optimizer.zero_grad()

            # 前向传播一：原始信号经过轻量级 DPD 生成预失真信号 U
            u = dpd_model(batch_x)

            # 前向传播二：预失真信号 U 喂入冻结的高精度物理代理 PA
            y_pred = pa_surrogate(u)

            # 系统级重构损失
            loss = criterion(y_pred, target_y)
            loss.backward()
            nn.utils.clip_grad_norm_(dpd_model.parameters(), max_norm=1.0)
            dpd_optimizer.step()

            total_train_error_power += torch.sum((y_pred.detach() - target_y) ** 2).item()
            total_train_target_power += torch.sum(target_y ** 2).item()

        train_nmse = 10 * np.log10(total_train_error_power / total_train_target_power)

        # 验证集评估
        dpd_model.eval()
        total_val_error_power, total_val_target_power = 0.0, 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                target_y = get_ideal_linear_target(batch_x, pa_g_real, pa_g_imag).to(DEVICE)

                u = dpd_model(batch_x)
                y_pred = pa_surrogate(u)

                total_val_error_power += torch.sum((y_pred - target_y) ** 2).item()
                total_val_target_power += torch.sum(target_y ** 2).item()

        val_nmse = 10 * np.log10(total_val_error_power / total_val_target_power)
        dpd_scheduler.step(val_nmse)
        current_lr = dpd_optimizer.param_groups[0]['lr']

        history_epochs.append(epoch + 1)
        history_dla_nmse.append(val_nmse)

        elapsed_time = time.time() - start_time
        mins, secs = int(elapsed_time // 60), int(elapsed_time % 60)

        is_pass = "✅ PASS" if val_nmse < -40.0 else "❌ WAIT"
        print(f"DLA Epoch [{epoch + 1}/{EPOCHS_DLA}] | Time: {mins}m {secs}s | LR: {current_lr:.2e} | "
              f"Train NMSE: {train_nmse:.2f} dB | Cascaded System Val NMSE: {val_nmse:.2f} dB | {is_pass}")

        if val_nmse < best_dla_loss:
            best_dla_loss = val_nmse
            torch.save(dpd_model.state_dict(), 'best_dla_dpd_model.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_dla_nmse, label='Cascaded System Validation NMSE', color='b', linewidth=1.5)
    plt.axhline(y=min(history_dla_nmse), color='r', linestyle='--',
                label=f'Best System Val: {min(history_dla_nmse):.2f} dB')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('System NMSE (dB)', fontsize=12)
    plt.title('DLA (Direct Learning Architecture) Convergence (DPD MACs=776)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('dla_training_convergence.png', dpi=300)
    print("\n>>> DLA 端到端训练完成！")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()