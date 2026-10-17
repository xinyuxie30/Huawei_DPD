import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 2000
BATCH_SIZE = 16
FRAME_LENGTH = 800
STRIDE_LENGTH = 32

# ==========================================
# 改进1: 注意力模块升级为4通道 (amp, amp^3, amp^5, amp^7)
# 原版3通道: 10*3*3 = 90 → 新版4通道: 10*4*3 = 120 (+30乘法器)
# ==========================================
class OptimalSpectrumAttention(nn.Module):
    def __init__(self, channels):
        super(OptimalSpectrumAttention, self).__init__()
        self.env_conv = nn.Conv1d(4, channels, kernel_size=3, padding=1, bias=False, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, amp, amp3, amp5, amp7):
        env_features = torch.cat([amp, amp3, amp5, amp7], dim=1)
        weights = self.sigmoid(self.env_conv(env_features))
        return x * weights


# ==========================================
# 改进2: 增加 dilation=16 TCN层 (5层: 1,2,4,8,16)
# 原版4层: (100+50)*4 = 600 → 新版5层: (100+50)*5 = 750 (+150乘法器)
# 感受野: (5-1)*(1+2+4+8+16)+1 = 125 采样点
#
# 改进3: 增加5阶旁路 (g5_real, g5_imag) (+4乘法器)
# 改进4: 输入从6ch扩展到9ch (+30乘法器)
#   [I_t, Q_t, I_{t+1}, Q_{t+1}, I_{t-1}, Q_{t-1}, |x|, |x|^3, |x|^5]
# 总乘法器: 12(旁路) + 90(in_conv) + 750(TCN) + 120(attn) + 20(out_conv) = 992
# ==========================================

class DiagramDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(DiagramDPDNet, self).__init__()
        self.hidden = 10

        # 一阶线性增益
        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))

        # 三阶增益 (初始化为0，让网络逐步学习)
        self.g3_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g3_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        # 五阶增益 (初始化为0)
        self.g5_real = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.g5_imag = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        # 9通道输入 → hidden
        self.in_conv = nn.Conv1d(9, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()
        self.tcn_blocks = nn.ModuleList()

        # 5层膨胀卷积 (扩张率 1,2,4,8,16)
        for dilation in [1, 2, 4, 8, 16]:
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

        # 1. 物理多项式旁路: 1阶 + 3阶 + 5阶
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real

        # 包络特征 (层次化计算，避免重复运算)
        amp = torch.sqrt(i_x ** 2 + q_x ** 2 + 1e-12)
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp4 = amp2 ** 2
        amp5 = amp ** 5
        amp7 = amp ** 7

        # 三阶项
        x3_i = i_x * amp2
        x3_q = q_x * amp2
        lin3_i = x3_i * self.g3_real - x3_q * self.g3_imag
        lin3_q = x3_i * self.g3_imag + x3_q * self.g3_real

        # 五阶项
        x5_i = i_x * amp4
        x5_q = q_x * amp4
        lin5_i = x5_i * self.g5_real - x5_q * self.g5_imag
        lin5_q = x5_i * self.g5_imag + x5_q * self.g5_real

        # 合并旁路
        lin_out = torch.cat([lin_i + lin3_i + lin5_i, lin_q + lin3_q + lin5_q], dim=1)

        # 2. 构造9维输入特征
        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)
        i_past = torch.roll(i_x, shifts=1, dims=2)
        q_past = torch.roll(q_x, shifts=1, dims=2)

        x_9ch = torch.cat([i_x, q_x, i_future, q_future, i_past, q_past, amp, amp3, amp5], dim=1)

        # 3. TCN残差支路
        feat = self.act(self.in_conv(x_9ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat)

        # 4. 注意力加权 + 输出
        feat = self.attn(feat, amp, amp3, amp5, amp7)
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


# ==========================================
# 改进5: EMA (指数移动平均) 权重管理
# ==========================================
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """将EMA权重复制到模型中 (用于验证)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self, backup):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name])


def main():
    data = scipy.io.loadmat('DPD_data.mat')
    raw_x, raw_y = data['x'].flatten(), data['y'].flatten()

    RMS_X = np.sqrt(np.mean(np.abs(raw_x) ** 2))
    RMS_Y = np.sqrt(np.mean(np.abs(raw_y) ** 2))

    x_c = raw_x / RMS_X
    y_c = raw_y / RMS_Y

    G_opt = np.vdot(x_c, y_c) / np.vdot(x_c, x_c)
    g_real_init = np.real(G_opt)
    g_imag_init = np.imag(G_opt)

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

    model = DiagramDPDNet(g_real_init, g_imag_init).to(DEVICE)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-6)

    # ==========================================
    # 改进6: CosineAnnealingWarmRestarts 替代 ReduceLROnPlateau
    # T_0=50, T_mult=2 → 重启周期: 50, 150, 350, 750, 1550
    # 每次重启以高学习率重新探索，避免陷入局部最优
    # ==========================================
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    ema = EMAModel(model, decay=0.999)

    best_val_loss = float('inf')
    best_ema_loss = float('inf')
    history_epochs = []
    history_nmse = []
    history_ema_nmse = []

    print(f"\n>>> 开始训练 EA-TCN-v2 (5层TCN + 1/3/5阶旁路 + 9ch输入) (~992乘法器)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # ---- 训练 ----
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        ema.update()

        # ---- 标准权重验证 ----
        model.eval()
        total_error_power, total_target_power = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
                preds = model(batch_x)
                total_error_power += torch.sum((preds - batch_y) ** 2).item()
                total_target_power += torch.sum(batch_y ** 2).item()

        linear_nmse = total_error_power / total_target_power
        val_nmse = 10 * np.log10(linear_nmse)

        # ---- EMA权重验证 ----
        backup = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        ema.apply_shadow()
        total_error_power_ema, total_target_power_ema = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
                preds = model(batch_x)
                total_error_power_ema += torch.sum((preds - batch_y) ** 2).item()
                total_target_power_ema += torch.sum(batch_y ** 2).item()
        ema.restore(backup)

        ema_nmse = 10 * np.log10(total_error_power_ema / total_target_power_ema)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history_epochs.append(epoch + 1)
        history_nmse.append(val_nmse)
        history_ema_nmse.append(ema_nmse)

        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Time: {elapsed_mins}m {elapsed_secs}s | LR: {current_lr:.2e} | "
            f"Val: {val_nmse:.2f} dB | EMA: {ema_nmse:.2f} dB"
        )

        # 保存标准权重最优
        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_competition_model.pth')

        # 保存EMA权重最优
        if ema_nmse < best_ema_loss:
            best_ema_loss = ema_nmse
            ema.apply_shadow()
            torch.save(model.state_dict(), 'best_ema_model.pth')
            ema.restore(backup)

    # ---- 绘制收敛曲线 ----
    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_nmse, label='Validation NMSE (raw)', color='b', linewidth=1.0, alpha=0.7)
    plt.plot(history_epochs, history_ema_nmse, label='Validation NMSE (EMA)', color='r', linewidth=1.5)
    plt.axhline(y=min(history_ema_nmse), color='r', linestyle='--',
                label=f'Best EMA: {min(history_ema_nmse):.2f} dB')
    plt.axhline(y=min(history_nmse), color='b', linestyle='--',
                label=f'Best Raw: {min(history_nmse):.2f} dB')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title('EA-TCN-v2 (5 blocks + EMA + CosineAnnealing)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('EA-TCN-v2.png', dpi=300)
    print(f"\n>>> 训练完成！Best Raw: {best_val_loss:.2f} dB | Best EMA: {best_ema_loss:.2f} dB")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
