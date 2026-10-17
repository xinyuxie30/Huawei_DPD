import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import time

# ==========================================
# 0. 全局配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPOCHS = 200
BATCH_SIZE = 64
FRAME_LENGTH = 2048  # 论文中解决梯度消失的短帧长度
STRIDE_LENGTH = 32  # 50% 重叠滑窗


# ==========================================
# 1. 华为赛题适配：数据截帧加载器
# ==========================================
class HuaweiFramedDataset(Dataset):
    def __init__(self, features, targets, frame_length=1024, stride_length=512):
        self.features = torch.as_tensor(features, dtype=torch.float64)
        self.targets = torch.as_tensor(targets, dtype=torch.float64)
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.sequence_length = self.features.shape[0]
        self.num_frames = (self.sequence_length - self.frame_length) // self.stride_length + 1

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        start_idx = idx * self.stride_length
        end_idx = start_idx + self.frame_length
        frame_feature = self.features[start_idx:end_idx, :]
        frame_target = self.targets[start_idx:end_idx, :]

        # 转换为 Conv1D 需要的 (Channels, Length) -> (18, 2048)
        frame_feature = frame_feature.transpose(0, 1)
        frame_target = frame_target.transpose(0, 1)
        return frame_feature, frame_target


# ==========================================
# 2. 华为赛题适配：满血版 TCN 算力引擎 (<800 乘法器)
# ==========================================
class HuaweiTCN_Sniper(nn.Module):
    def __init__(self, in_channels=18, hidden_channels=20):
        super(HuaweiTCN_Sniper, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = 2
        self.kernel_size = 5

        self.network = nn.Sequential(
            nn.Conv1d(self.in_channels, self.hidden_channels, 1, bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 1, dilation=1, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 2, dilation=2, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 4, dilation=4, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, stride=1,
                      padding=(self.kernel_size - 3) * 8, dilation=8, groups=self.hidden_channels, bias=False,
                      dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(self.hidden_channels, self.out_channels, 1, bias=False, dtype=torch.float64),
        )

    def forward(self, x_18ch):
        residual = self.network(x_18ch)
        original_iq = x_18ch[:, 0:2, :]  # 提取前两通道(线性I和Q)
        return original_iq + residual  # 线性直通 + 记忆残差


# ==========================================
# 3. 主程序：数据处理与训练循环
# ==========================================
def main():
    print(">>> 1. Loading and Extracting 18-Channel Features...")
    data = scipy.io.loadmat('DPD_data.mat')
    raw_x = data['x'].flatten()
    raw_y = data['y'].flatten()

    x_norm = raw_x / np.max(np.abs(raw_x))
    y_norm = raw_y / np.max(np.abs(raw_y))

    # === 在 train_pa.py 中，把提取特征的对象换成 y_norm ===
    u = y_norm  # 核心修改：以解药作为 PA 的输入提取 18 通道特征
    real = np.real(u)
    imag = np.imag(u)
    mag = np.abs(u)



    mag2 = mag ** 2
    mag4 = mag2 ** 2
    mag6 = mag4 * mag2
    mag8 = mag4 ** 2
    mag10 = mag8 * mag2
    mag12 = mag6 ** 2

    u_delay = np.concatenate([[0], u[:-1]])
    diff_u = u - u_delay
    diff_r = np.real(diff_u)
    diff_i = np.imag(diff_u)
    diff_mag2_r = diff_r * mag2
    diff_mag2_i = diff_i * mag2

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

    # 拼接出 PA 模型的输入特征
    PA_Input_18ch = np.stack(channels, axis=1)

    # PA 模型的拟合目标换成原始信号 x
    PA_Target_np = np.stack([np.real(x_norm), np.imag(x_norm)], axis=1)

    # 切分训练集、验证集，跑 150 个 Epoch，将最佳模型保存为 'best_pa_model.pth'



    # ✂️ 核心修改：抛弃 manually 滑窗！直接在 axis=1 拼接为二维矩阵 [N, 18]
    # 修改：直接使用原始的归一化通道特征，维持真实的物理非线性比例
    X_np = np.stack(channels, axis=1)
    Target_np = np.stack([np.real(y_norm), np.imag(y_norm)], axis=1)  # [N, 2]

    min_len = min(len(X_np), len(Target_np))
    X_np = X_np[:min_len]
    Target_np = Target_np[:min_len]

    # 80% 训练，20% 验证
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:], Target_np[split_idx:]

    print(f"✅ Data Ready! X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

    print(">>> 2. Building Framed DataLoaders...")
    # 训练集：重叠截帧 (stride=1024, frame=2048)
    train_dataset = HuaweiFramedDataset(X_train, Y_train, frame_length=FRAME_LENGTH, stride_length=STRIDE_LENGTH)
    # 验证集：非重叠截帧 (保证验证时没有数据穿越)
    val_dataset = HuaweiFramedDataset(X_val, Y_val, frame_length=FRAME_LENGTH, stride_length=FRAME_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(">>> 3. Initializing Model, Loss, and Optimizer...")
    model = HuaweiTCN_Sniper(in_channels=18, hidden_channels=20).to(DEVICE)
    criterion = nn.MSELoss()
    # 使用 AdamW 优化器，自带权重衰减抗过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 学习率调度器：如果 loss 停滞，降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print(">>> 4. Starting BPTT End-to-End Training...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            preds = model(batch_x)

            # 纯时域 MSE 贴合华为 NMSE 目标
            loss = criterion(preds, batch_y)
            loss.backward()

            # 梯度裁剪防爆炸 (非常关键的保护)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        # ====== Validation ======
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        scheduler.step(avg_val_loss)

        # 将 MSE 转换为对数域的大致 NMSE 估算 (10 * log10(MSE))
        train_nmse_est = 10 * np.log10(avg_train_loss)
        val_nmse_est = 10 * np.log10(avg_val_loss)

        time_taken = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Time: {time_taken:.1f}s | Train MSE: {avg_train_loss:.2e} ({train_nmse_est:.1f} dB) | Val MSE: {avg_val_loss:.2e} ({val_nmse_est:.1f} dB)")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_huawei_tcn.pth')
            print("   🌟 Best model saved!")


if __name__ == '__main__':
    main()