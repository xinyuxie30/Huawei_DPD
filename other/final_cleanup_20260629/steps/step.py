import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 400
BATCH_SIZE = 64
FRAME_LENGTH = 800
STRIDE_LENGTH = 32


# ==========================================
# EMG-DPD：包络记忆门控架构 (782 MACs 极限特化版)
# ==========================================
class EMGDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(EMGDPDNet, self).__init__()
        self.hidden = 12

        # 1. 静态物理旁路 (6 MACs)
        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        # 2. 动态记忆路径 (提取包络特征)
        # 输入仅为幅度包络特征: amp, amp2, amp3 (3通道)
        self.in_conv = nn.Conv1d(3, self.hidden, 1, bias=False, dtype=torch.float64)  # 36 MACs
        self.act = nn.SiLU()

        self.tcn_blocks = nn.ModuleList()
        # K=7 配合 D=[1,3,9]，构建高达 79 采样点的物理记忆感受野
        for dilation in [1, 3, 9]:
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(self.hidden, self.hidden, kernel_size=7, padding=3 * dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                nn.SiLU()
            ))  # 每层 228 MACs * 3 = 684 MACs

        # 输出 4 个通道，对应当前时刻与上一时刻的动态复数抽头 (W1_r, W1_i, W2_r, W2_i)
        self.out_conv = nn.Conv1d(self.hidden, 4, 1, bias=False, dtype=torch.float64)  # 48 MACs
        nn.init.constant_(self.out_conv.weight, 0.0)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        # --- 静态旁路计算 ---
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3

        x3_i = i_x * amp2
        x3_q = q_x * amp2
        lin3_i = x3_i * self.g3_real - x3_q * self.g3_imag
        lin3_q = x3_i * self.g3_imag + x3_q * self.g3_real

        static_out = torch.cat([lin_i + lin3_i, lin_q + lin3_q], dim=1)

        # --- 动态记忆网络计算 ---
        # 仅将包络特征送入深层网络
        env_features = torch.cat([amp, amp2, amp3], dim=1)
        feat = self.act(self.in_conv(env_features))

        for block in self.tcn_blocks:
            feat = feat + block(feat)

        dynamic_taps = self.out_conv(feat)  # Shape: [batch, 4, seq_len]

        w1_r, w1_i = dynamic_taps[:, 0:1, :], dynamic_taps[:, 1:2, :]
        w2_r, w2_i = dynamic_taps[:, 2:3, :], dynamic_taps[:, 3:4, :]

        # --- 动态重构 (应用复数抽头) --- (8 MACs)
        # 获取上一时刻的载波
        i_prev = torch.roll(i_x, shifts=1, dims=2)
        q_prev = torch.roll(q_x, shifts=1, dims=2)
        # 边缘效应消除
        i_prev[:, :, 0] = 0.0
        q_prev[:, :, 0] = 0.0

        # 当前时刻复数乘法
        dyn_i_1 = i_x * w1_r - q_x * w1_i
        dyn_q_1 = i_x * w1_i + q_x * w1_r

        # 上一时刻复数乘法 (消除宽带一阶延迟)
        dyn_i_2 = i_prev * w2_r - q_prev * w2_i
        dyn_q_2 = i_prev * w2_i + q_prev * w2_r

        dynamic_out = torch.cat([dyn_i_1 + dyn_i_2, dyn_q_1 + dyn_q_2], dim=1)

        # 终极输出：静态骨架 + 动态记忆残差
        return static_out + dynamic_out


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


def apply_complex_gain(signal_iq, g_real, g_imag):
    i_x, q_x = signal_iq[:, 0:1, :], signal_iq[:, 1:2, :]
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

    G_opt = np.vdot(x_c, y_c) / np.vdot(x_c, x_c)
    G_inv = 1.0 / G_opt
    dpd_g_real, dpd_g_imag = np.real(G_inv), np.imag(G_inv)

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Target_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]

    train_loader = DataLoader(FramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(FramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 实例化网络，使用 G_inv 初始化骨架
    model = EMGDPDNet(dpd_g_real, dpd_g_imag).to(DEVICE)
    criterion = nn.MSELoss()

    # 高效的纯 ILA 训练，可以使用激进的学习率快速收敛
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float('inf')
    history_epochs, history_val_nmse = [], []

    print(f"\n>>> 放弃 DLA，重返纯净 ILA！启动 EMG-DPD (包络记忆门控网络)")
    print(f">>> 物理感受野: 79点 | 算力消耗: 782 MACs 极限拉满！")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_err, total_tgt = 0.0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)

            # ILA 前向：输入归一化测量 Y，拟合目标干净 X
            batch_y_norm = apply_complex_gain(batch_y, dpd_g_real, dpd_g_imag)

            optimizer.zero_grad()
            u_pred = model(batch_y_norm)
            loss = criterion(u_pred, batch_x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_err += torch.sum((u_pred.detach() - batch_x) ** 2).item()
            total_tgt += torch.sum(batch_x ** 2).item()

        train_nmse = 10 * np.log10(total_err / total_tgt)

        model.eval()
        total_val_err, total_val_tgt = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
                batch_y_norm = apply_complex_gain(batch_y, dpd_g_real, dpd_g_imag)

                u_pred = model(batch_y_norm)

                total_val_err += torch.sum((u_pred - batch_x) ** 2).item()
                total_val_tgt += torch.sum(batch_x ** 2).item()

        val_nmse = 10 * np.log10(total_val_err / total_val_tgt)
        scheduler.step(val_nmse)
        current_lr = optimizer.param_groups[0]['lr']

        history_epochs.append(epoch + 1)
        history_val_nmse.append(val_nmse)

        elapsed = time.time() - start_time
        print(
            f"Ep [{epoch + 1}/{EPOCHS}] | Time: {int(elapsed // 60)}m {int(elapsed % 60)}s | LR: {current_lr:.2e} | Train: {train_nmse:.2f} dB | Val: {val_nmse:.2f} dB")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_emg_dpd.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_val_nmse, label='Validation NMSE', color='b', linewidth=1.5)
    plt.axhline(y=min(history_val_nmse), color='r', linestyle='--', label=f'Best: {min(history_val_nmse):.2f} dB')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title('EMG-DPD Convergence (Pure ILA, 782 MACs)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('emg_dpd_convergence.png', dpi=300)
    print(f"\n>>> 训练大满贯完成！最强物理极限已突破！")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()