import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 256
FRAME_LENGTH = 200
STRIDE_LENGTH = 32

# ==========================================
# SE 注意力模块
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=3):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 12 -> 4 -> 12，消耗 96 MACs
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
# 极致算力架构：Attention-TCN (<800 MACs)
# ==========================================
class AttentionTCN(nn.Module):
    def __init__(self):
        super(AttentionTCN, self).__init__()
        self.hidden = 12

        # 1. 输入映射 (3 -> 12)
        self.in_conv = nn.Conv1d(3, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()

        # 2. 深度可分离 TCN 层
        self.tcn_blocks = nn.ModuleList()
        for dilation in [1, 2, 4]:
            self.tcn_blocks.append(nn.Sequential(
                # DW
                nn.Conv1d(self.hidden, self.hidden, 5, padding=2*dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                # PW
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                nn.SiLU()
            ))

        # 3. 通道注意力
        self.se = SEBlock(self.hidden, reduction=3)

        # 4. 输出残差 (12 -> 2)
        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)

    def forward(self, x_iq):
        # 提取幅度特征辅助网络
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]
        amp = torch.sqrt(i_x**2 + q_x**2 + 1e-12)
        x_3ch = torch.cat([x_iq, amp], dim=1) # [B, 3, T]

        # 主干网络
        feat = self.act(self.in_conv(x_3ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat) # 内部残差

        feat = self.se(feat) # 注意力加权
        residual = self.out_conv(feat)

        # 线性直通
        return x_iq + residual

# ==========================================
# 数据处理与训练
# ==========================================
class FramedDataset(Dataset):
    def __init__(self, features, targets, frame_length=200, stride_length=32):
        self.features = torch.as_tensor(features, dtype=torch.float64)
        self.targets = torch.as_tensor(targets, dtype=torch.float64)
        self.frame_length, self.stride_length = frame_length, stride_length
        self.num_frames = (self.features.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self): return self.num_frames
    def __getitem__(self, idx):
        start = idx * self.stride_length
        # 格式转换为 Conv1d 需要的 [Channels, Time]
        return self.features[start:start+self.frame_length, :].T, self.targets[start:start+self.frame_length, :].T

def main():
    data = scipy.io.loadmat('DPD_data.mat')
    raw_x, raw_y = data['x'].flatten(), data['y'].flatten()
    SCALE_VAL = np.max(np.abs(raw_x))

    X_np = np.stack([np.real(raw_x/SCALE_VAL), np.imag(raw_x/SCALE_VAL)], axis=1)
    Target_np = np.stack([np.real(raw_y/SCALE_VAL), np.imag(raw_y/SCALE_VAL)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]

    train_loader = DataLoader(FramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionTCN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

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
        scheduler.step(val_nmse)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Val NMSE: {val_nmse:.2f} dB")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_competition_model.pth')

if __name__ == '__main__':
    main()