import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 240
BATCH_SIZE = 32
FRAME_LENGTH = 1000
STRIDE_LENGTH = 32


# ==========================================
# 核心差分层：强制 Float64 精度，防除错崩溃
# ==========================================
class CustomDeltaGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, thx=0.02, thh=0.02):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.thx = thx
        self.thh = thh
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size, dtype=torch.float64))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size, dtype=torch.float64))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size, dtype=torch.float64))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        x_p = torch.zeros(batch_size, self.input_size, dtype=x.dtype, device=x.device)
        h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        h_p = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)

        dm_0 = torch.zeros(batch_size, 3 * self.hidden_size, dtype=x.dtype, device=x.device)
        dm_0[:, :2 * self.hidden_size] = (self.bias_ih + self.bias_hh)[:2 * self.hidden_size]
        dm_0[:, 2 * self.hidden_size:] = self.bias_ih[2 * self.hidden_size:]
        dm = dm_0
        dm_nh = self.bias_hh[2 * self.hidden_size:].unsqueeze(0).expand(batch_size, -1).clone()

        outputs = []
        total_dx_zeros, total_dh_zeros = 0, 0
        total_elements_x = batch_size * seq_len * self.input_size
        total_elements_h = batch_size * seq_len * self.hidden_size

        for t in range(seq_len):
            xt = x[:, t, :]

            dx = xt - x_p
            dh = h - h_p
            dx_abs, dh_abs = torch.abs(dx), torch.abs(dh)

            mask_x = (dx_abs >= self.thx).to(x.dtype)
            mask_h = (dh_abs >= self.thh).to(x.dtype)

            dx, dh = dx * mask_x, dh * mask_h

            if not self.training:
                total_dx_zeros += torch.sum(1.0 - mask_x).item()
                total_dh_zeros += torch.sum(1.0 - mask_h).item()

            x_p = torch.where(mask_x.bool(), xt, x_p)
            h_p = torch.where(mask_h.bool(), h, h_p)

            mac_x = F.linear(dx, self.weight_ih) + dm
            mac_h = F.linear(dh, self.weight_hh)

            mac_x_r, mac_x_z, mac_x_n = mac_x.chunk(3, dim=-1)
            mac_h_r, mac_h_z, mac_h_n = mac_h.chunk(3, dim=-1)

            dm_r, dm_z = mac_x_r + mac_h_r, mac_x_z + mac_h_z
            dm_n, dm_nh_t = mac_x_n, mac_h_n + dm_nh

            dm = torch.cat((dm_r, dm_z, dm_n), dim=-1)
            r, z = torch.sigmoid(dm_r), torch.sigmoid(dm_z)
            n = torch.tanh(dm_n + r * dm_nh_t)

            h = (1 - z) * n + z * h
            outputs.append(h)
            dm_nh = dm_nh_t

        outputs = torch.stack(outputs, dim=1)
        sp_x = total_dx_zeros / total_elements_x if not self.training else 0.0
        sp_h = total_dh_zeros / total_elements_h if not self.training else 0.0

        return outputs, sp_x, sp_h

class TResDeltaGRUDPD(nn.Module):
    def __init__(self):
        super(TResDeltaGRUDPD, self).__init__()
        self.input_size = 6
        self.hidden_size = 19  # 从15扩容至19
        self.tcn = nn.Sequential(
            nn.Conv1d(2, 3, kernel_size=3, padding=16, dilation=16, bias=False, dtype=torch.float64),
            nn.Hardswish(),
            nn.Conv1d(3, 2, kernel_size=1, bias=False, dtype=torch.float64),
            nn.Hardswish(),
        )

        self.gru = CustomDeltaGRULayer(self.input_size, self.hidden_size, thx=0.0, thh=0.0)
        self.fc_out = nn.Linear(self.hidden_size, 2, bias=False, dtype=torch.float64)
        nn.init.constant_(self.fc_out.weight, 0.0)

    def forward(self, x_iq):
        skip_x = self.tcn(x_iq)

        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]
        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp3 = amp ** 3

        x_features = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)
        x_features_rnn = x_features.transpose(1, 2)

        # 3. 差分迭代
        gru_out, sp_x, sp_h = self.gru(x_features_rnn)

        res_out = self.fc_out(gru_out)
        res_out = res_out.transpose(1, 2)

        # TCN旁路 + RNN深层记忆
        return skip_x + res_out, sp_x, sp_h


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

    RMS_X = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    RMS_Y = np.sqrt(np.mean(np.abs(raw_y) ** 2))

    x_c = raw_x / RMS_X
    y_c = raw_y / RMS_Y

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Target_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]

    train_loader = DataLoader(FramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(FramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = TResDeltaGRUDPD().to(DEVICE)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    best_val_loss = float('inf')
    history_epochs = []
    history_val_nmse = []
    history_active_macs = []

    print(f"\n>>> 开始训练 TRes-DeltaGRU (H=19 | 峰值 1487 MACs | 前30轮完全稠密)...")
    start_time = time.time()

    for epoch in range(EPOCHS):

        # ==========================================
        # 动态退火稀疏策略 (Curriculum Sparsity)
        # ==========================================
        if epoch == 30:
            print(f"\n[!] 激活差分稀疏门控 -> 强行将 1487 MACs 压至 800 以下！")
            # 微调阈值，0.02 通常能卡出 40% ~ 60% 的稀疏度
            model.gru.thx = 0.02
            model.gru.thh = 0.02

        model.train()
        total_train_error_power, total_train_target_power = 0.0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            preds, _, _ = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_error_power += torch.sum((preds.detach() - batch_y) ** 2).item()
            total_train_target_power += torch.sum(batch_y ** 2).item()

        train_linear_nmse = total_train_error_power / total_train_target_power
        train_nmse = 10 * np.log10(train_linear_nmse)

        model.eval()
        total_val_error_power, total_val_target_power = 0.0, 0.0
        val_sp_x, val_sp_h = 0.0, 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)

                preds, sp_x, sp_h = model(batch_x)
                total_val_error_power += torch.sum((preds - batch_y) ** 2).item()
                total_val_target_power += torch.sum(batch_y ** 2).item()
                val_sp_x += sp_x
                val_sp_h += sp_h

        val_sp_x /= len(val_loader)
        val_sp_h /= len(val_loader)

        val_linear_nmse = total_val_error_power / total_val_target_power
        val_nmse = 10 * np.log10(val_linear_nmse)

        # 动态计算活跃算力: GRU活跃通道数 + TCN(24) + FC(38)
        H = model.hidden_size
        active_gru_macs = 3 * ((6 * H * (1 - val_sp_x)) + (H * H * (1 - val_sp_h)))
        active_total_macs = active_gru_macs + (H * 2) + 24

        scheduler.step(val_nmse)
        current_lr = optimizer.param_groups[0]['lr']

        history_epochs.append(epoch + 1)
        history_val_nmse.append(val_nmse)
        history_active_macs.append(active_total_macs)

        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)

        gap = train_nmse - val_nmse
        print(f"Ep [{epoch + 1}/{EPOCHS}] | LR: {current_lr:.1e} | "
              f"Val: {val_nmse:.2f} dB (Gap:{gap:.2f}) | Sp_X: {val_sp_x * 100:.1f}% Sp_H: {val_sp_h * 100:.1f}% | Active MACs: {int(active_total_macs)}")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_tres_deltagru.pth')

    # 双轴绘图：同时展示 NMSE 下降与 MACs 崩塌的奇观
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(history_epochs, history_val_nmse, 'b-', linewidth=1.5, label='Validation NMSE')
    ax1.axvline(x=30, color='r', linestyle=':', label='Sparsity Triggered (Ep 30)')
    ax1.axhline(y=min(history_val_nmse), color='b', linestyle='--', alpha=0.5)

    ax2.plot(history_epochs, history_active_macs, 'g-', linewidth=1.5, alpha=0.7, label='Active MACs')
    ax2.axhline(y=800, color='g', linestyle='--', linewidth=2, label='Competition Limit (800)')

    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('NMSE (dB)', color='b', fontsize=12)
    ax2.set_ylabel('Active Real Multipliers', color='g', fontsize=12)
    plt.title(f'TRes-DeltaGRU (H=19): Performance vs Sparsity', fontsize=14)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    ax1.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'tres_deltagru_convergence.png', dpi=300)
    print(f"\n>>> 训练完成！双轴趋势图已保存。")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()