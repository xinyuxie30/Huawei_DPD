import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
# 继承 best 配置
BATCH_SIZE = 16
FRAME_LENGTH = 1000
STRIDE_LENGTH = 32


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
# 微型记忆残差网络 (Micro Residual TCN)
# 总算力: 770 实数乘法器 (Real Multipliers)
# ==========================================
class MicroResidualTCN(nn.Module):
    def __init__(self):
        super(MicroResidualTCN, self).__init__()
        self.hidden = 10

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

        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp3 = amp ** 3
        amp7 = amp ** 7

        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)

        x_6ch = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)

        feat = self.act(self.in_conv(x_6ch))

        for block in self.tcn_blocks:
            feat = feat + block(feat)

        feat = self.attn(feat, amp, amp3, amp7)
        pred_res = self.out_conv(feat)

        return pred_res


class ResidualDataset(Dataset):
    def __init__(self, x, y_res, y_lut, y_true, frame_length, stride_length):
        self.x = torch.as_tensor(x, dtype=torch.float64)
        self.y_res = torch.as_tensor(y_res, dtype=torch.float64)
        self.y_lut = torch.as_tensor(y_lut, dtype=torch.float64)
        self.y_true = torch.as_tensor(y_true, dtype=torch.float64)

        self.frame_length = frame_length
        self.stride_length = stride_length
        self.num_frames = (self.x.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self): return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        end = start + self.frame_length
        return (self.x[start:end, :].T,
                self.y_res[start:end, :].T,
                self.y_lut[start:end, :].T,
                self.y_true[start:end, :].T)


def main():
    print(">>> 正在加载纯残差数据集并进行 RMS 归一化...")
    data = scipy.io.loadmat('DPD_residual_data.mat')

    raw_x = data['x'].flatten()
    raw_y_res = data['y_res'].flatten()
    raw_y_lut = data['y_lut'].flatten()
    raw_y_true = raw_y_res + raw_y_lut

    # ==========================================
    # RMS 归一化逻辑，防止梯度爆炸
    # ==========================================
    RMS_X = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    RMS_Y = np.sqrt(np.mean(np.abs(raw_y_true) ** 2))

    x_c = raw_x / RMS_X
    y_res_c = raw_y_res / RMS_Y
    y_lut_c = raw_y_lut / RMS_Y
    y_true_c = raw_y_true / RMS_Y

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Y_res_np = np.stack([np.real(y_res_c), np.imag(y_res_c)], axis=1)
    Y_lut_np = np.stack([np.real(y_lut_c), np.imag(y_lut_c)], axis=1)
    Y_true_np = np.stack([np.real(y_true_c), np.imag(y_true_c)], axis=1)

    min_len = len(X_np)
    split_idx = int(min_len * 0.8)

    train_loader = DataLoader(
        ResidualDataset(X_np[:split_idx], Y_res_np[:split_idx], Y_lut_np[:split_idx], Y_true_np[:split_idx],
                        FRAME_LENGTH, STRIDE_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(
        ResidualDataset(X_np[split_idx:], Y_res_np[split_idx:], Y_lut_np[split_idx:], Y_true_np[split_idx:],
                        FRAME_LENGTH, FRAME_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    model = MicroResidualTCN().to(DEVICE)
    criterion = nn.MSELoss()

    # 继承 best 配置: weight_decay=0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    # 继承 best 配置: patience=10
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_loss = float('inf')
    history_epochs = []
    history_train_nmse = []
    history_val_nmse = []

    print(f"\n>>> 开始训练 LUT + 纯残差网络 (770 实数乘法器 | BS={BATCH_SIZE} | 双轨监控)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_train_error_power, total_train_target_power = 0.0, 0.0

        for batch_x, batch_y_res, batch_y_lut, batch_y_true in train_loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y_res = batch_y_res.to(DEVICE, non_blocking=True)
            batch_y_lut = batch_y_lut.to(DEVICE, non_blocking=True)
            batch_y_true = batch_y_true.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            pred_res = model(batch_x)

            # 损失函数仅针对残差进行优化
            loss = criterion(pred_res, batch_y_res)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算 Train 阶段的系统总 NMSE
            final_train_pred = batch_y_lut + pred_res.detach()
            total_train_error_power += torch.sum((final_train_pred - batch_y_true) ** 2).item()
            total_train_target_power += torch.sum(batch_y_true ** 2).item()

        train_linear_nmse = total_train_error_power / total_train_target_power
        train_nmse = 10 * np.log10(train_linear_nmse)

        model.eval()
        total_val_error_power, total_val_target_power = 0.0, 0.0
        with torch.no_grad():
            for batch_x, _, batch_y_lut, batch_y_true in val_loader:
                batch_x = batch_x.to(DEVICE, non_blocking=True)
                batch_y_lut = batch_y_lut.to(DEVICE, non_blocking=True)
                batch_y_true = batch_y_true.to(DEVICE, non_blocking=True)

                pred_res = model(batch_x)
                # 系统最终输出 = LUT查表静态输出 + 网络预测的动态残差
                final_val_pred = batch_y_lut + pred_res

                total_val_error_power += torch.sum((final_val_pred - batch_y_true) ** 2).item()
                total_val_target_power += torch.sum(batch_y_true ** 2).item()

        val_linear_nmse = total_val_error_power / total_val_target_power
        val_nmse = 10 * np.log10(val_linear_nmse)

        scheduler.step(val_nmse)
        current_lr = optimizer.param_groups[0]['lr']

        history_epochs.append(epoch + 1)
        history_train_nmse.append(train_nmse)
        history_val_nmse.append(val_nmse)

        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)

        gap = train_nmse - val_nmse
        is_pass = "✅ PASS" if val_linear_nmse < 1e-4 else "❌ WAIT"

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Time: {elapsed_mins}m {elapsed_secs}s | LR: {current_lr:.2e} | Train: {train_nmse:.2f} dB | Val: {val_nmse:.2f} dB | Gap: {gap:.2f} dB | {is_pass}")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_lut_residual_model.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_train_nmse, label='Train System NMSE', color='g', linestyle='-.', linewidth=1.5)
    plt.plot(history_epochs, history_val_nmse, label='Validation System NMSE', color='b', linewidth=1.5)
    plt.axhline(y=min(history_val_nmse), color='r', linestyle='--', label=f'Best Val: {min(history_val_nmse):.2f} dB')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('System NMSE (dB)', fontsize=12)
    plt.title(f'LUT + MicroResidualTCN (Real MACs=770)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'lut_residual_convergence_dual.png', dpi=300)
    print(f"\n>>> 训练完成！收敛曲线已保存。")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()