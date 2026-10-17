import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 128
FRAME_LENGTH = 200
STRIDE_LENGTH = 32


# ==========================================
# SE 注意力模块 (96 MACs)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=3):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False, dtype=torch.float64),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# ==========================================
# 终极架构：Linear Bypass + Attention-TCN (772 MACs)
# ==========================================
class LinearResidualAttentionTCN(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(LinearResidualAttentionTCN, self).__init__()
        self.hidden = 12

        # 1. 物理线性增益参数 (4 MACs)
        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))

        # 2. 神经网络分支 (总计 768 MACs)
        self.in_conv = nn.Conv1d(3, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()

        self.tcn_blocks = nn.ModuleList()
        for dilation in [1, 2, 4]:
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(self.hidden, self.hidden, 5, padding=2 * dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                nn.SiLU()
            ))

        self.se = SEBlock(self.hidden, reduction=3)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)

        # 核心修正：零初始化输出层，确保初始状态为纯线性最优解
        nn.init.constant_(self.out_conv.weight, 0.0)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        # 1. 计算线性主路径
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real
        lin_out = torch.cat([lin_i, lin_q], dim=1)

        # 2. 计算非线性残差
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        x_3ch = torch.cat([x_iq, amp], dim=1)

        feat = self.act(self.in_conv(x_3ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat)

        feat = self.se(feat)
        residual = self.out_conv(feat)

        # 3. 完美结合
        return lin_out + residual


# ==========================================
# 数据处理与训练
# ==========================================
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


def main():
    data = scipy.io.loadmat('DPD_data.mat')
    raw_x, raw_y = data['x'].flatten(), data['y'].flatten()

    # 独立解耦缩放
    SCALE_X = np.max(np.abs(raw_x))
    SCALE_Y = np.max(np.abs(raw_y))
    x_c = raw_x / SCALE_X
    y_c = raw_y / SCALE_Y

    # 最小二乘法计算初始物理增益
    G_opt = np.vdot(x_c, y_c) / np.vdot(x_c, x_c)
    g_real_init = np.real(G_opt)
    g_imag_init = np.imag(G_opt)
    print(f">>> Initializing Optimal Linear Gain: {G_opt:.4f}")

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Target_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]

    train_loader = DataLoader(FramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH), batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(FramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH), batch_size=BATCH_SIZE,
                            shuffle=False)

    model = LinearResidualAttentionTCN(g_real_init, g_imag_init).to(DEVICE)
    criterion = nn.MSELoss()

    # 保持激进的优化策略
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        total_error_power, total_target_power = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_x)
                total_error_power += torch.sum((preds - batch_y) ** 2).item()
                total_target_power += torch.sum(batch_y ** 2).item()

        val_nmse = 10 * np.log10(total_error_power / total_target_power)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | LR: {current_lr:.2e} | Val NMSE: {val_nmse:.2f} dB")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_competition_model.pth')


if __name__ == '__main__':
    main()