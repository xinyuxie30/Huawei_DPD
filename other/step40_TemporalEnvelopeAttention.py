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
# 修正：时间包络注意力 (Temporal Envelope Attention)
# 不折叠时间维度，逐点根据包络强度施加动态通道加权
# ==========================================
class TemporalEnvelopeAttention(nn.Module):
    def __init__(self, channels):
        super(TemporalEnvelopeAttention, self).__init__()
        # 利用包络信号 (1通道) 映射到隐藏通道 (12通道)，感受野 K=3
        self.env_conv = nn.Conv1d(1, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp):
        # amp 的维度是 [Batch, 1, Time]
        # 生成随时间变化的权重矩阵 [Batch, 12, Time]
        weights = self.sigmoid(self.env_conv(amp))
        # 逐点动态加权
        return x * weights


# ==========================================
# 终极架构：Linear Bypass + Physics-Prep + Attention-TCN (736 MACs)
# ==========================================
class OptimalDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(OptimalDPDNet, self).__init__()
        self.hidden = 12

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))

        # 修正：输入通道增加到 4，接收显式的三阶物理特征
        self.in_conv = nn.Conv1d(4, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()

        self.tcn_blocks = nn.ModuleList()
        for dilation in [1, 2, 4]:
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(self.hidden, self.hidden, 5, padding=2 * dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                nn.SiLU()
            ))

        self.tea = TemporalEnvelopeAttention(self.hidden)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)

        # 零初始化，确保第一轮为纯最小二乘线性基准
        nn.init.constant_(self.out_conv.weight, 0.0)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        # 1. 绝对最优线性主路径
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real
        lin_out = torch.cat([lin_i, lin_q], dim=1)

        # 2. 修正预处理：直接生成基带与三阶主失真项
        amp2 = i_x ** 2 + q_x ** 2
        amp = torch.sqrt(amp2 + 1e-12)
        x_4ch = torch.cat([i_x, q_x, i_x * amp2, q_x * amp2], dim=1)

        feat = self.act(self.in_conv(x_4ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat)

        # 将瞬时包络送入注意力模块生成时间掩码
        feat = self.tea(feat, amp)
        residual = self.out_conv(feat)

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

    SCALE_X = np.max(np.abs(raw_x))
    SCALE_Y = np.max(np.abs(raw_y))
    x_c = raw_x / SCALE_X
    y_c = raw_y / SCALE_Y

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

    model = OptimalDPDNet(g_real_init, g_imag_init).to(DEVICE)
    criterion = nn.MSELoss()

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