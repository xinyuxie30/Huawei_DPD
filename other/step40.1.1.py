import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 64
FRAME_LENGTH = 1600
STRIDE_LENGTH = 32

# ==========================================
# 优化：双包络动态注意力 (Dual-Envelope Attention)
# 同时感知线性区包络与深度饱和区包络
# ==========================================
class DualEnvelopeAttention(nn.Module):
    def __init__(self, channels):
        super(DualEnvelopeAttention, self).__init__()
        # 接收 amp 和 amp3 两个特征通道 (72 MACs)
        self.env_conv = nn.Conv1d(2, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp, amp3):
        # 在通道维度拼接两种包络状态
        env_features = torch.cat([amp, amp3], dim=1) # [Batch, 2, Time]
        weights = self.sigmoid(self.env_conv(env_features)) # [Batch, 12, Time]
        return x * weights

# ==========================================
# 终极架构：静态旁路 + 6-Ch 输入 + DETA (796 MACs)
# ==========================================
class PaperDiagramDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(PaperDiagramDPDNet, self).__init__()
        self.hidden = 12

        # 恢复物理线性增益参数
        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))

        # 严格对接论文图中的 6 通道输入
        self.in_conv = nn.Conv1d(6, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()

        self.tcn_blocks = nn.ModuleList()
        for dilation in [1, 2, 4]:
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(self.hidden, self.hidden, 5, padding=2 * dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                nn.SiLU()
            ))

        # 植入升格后的双包络注意力
        self.deta = DualEnvelopeAttention(self.hidden)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)

        # 零初始化，确保第一轮为纯最小二乘线性基准
        nn.init.constant_(self.out_conv.weight, 0.0)

    def forward(self, x_iq):
        # x_iq 维度: [Batch, 2, Time]
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        # 1. 绝对最优线性主路径
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real
        lin_out = torch.cat([lin_i, lin_q], dim=1)

        # 2. 构造论文图中的 6 维特征矩阵 (phi_t)
        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)

        # 获取幅度和三阶幅度
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp3 = amp ** 3

        # [I_t, Q_t, I_t+1, Q_t+1, |x_t|, |x_t|^3]
        x_6ch = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)

        # 3. 残差网络推演
        feat = self.act(self.in_conv(x_6ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat)

        # 显式传入 amp 和 amp3 驱动注意力门控
        feat = self.deta(feat, amp, amp3)
        residual = self.out_conv(feat)

        return lin_out + residual

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

    model = PaperDiagramDPDNet(g_real_init, g_imag_init).to(DEVICE)
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