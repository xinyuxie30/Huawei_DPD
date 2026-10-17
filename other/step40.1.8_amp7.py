import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 64
FRAME_LENGTH = 800
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


class PaperDiagramDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(PaperDiagramDPDNet, self).__init__()
        # 拓宽至 H=16，确立绝对性能上限，算力约 1622 MACs
        self.hidden = 16

        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))
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
        amp2 = amp ** 2
        amp3 = amp ** 3
        amp7 = amp ** 7

        x3_i = i_x * amp2
        x3_q = q_x * amp2

        lin3_i = x3_i * self.g3_real - x3_q * self.g3_imag
        lin3_q = x3_i * self.g3_imag + x3_q * self.g3_real

        lin_out = torch.cat([lin_i + lin3_i, lin_q + lin3_q], dim=1)

        i_future = torch.roll(i_x, shifts=-1, dims=2)
        q_future = torch.roll(q_x, shifts=-1, dims=2)

        x_6ch = torch.cat([i_x, q_x, i_future, q_future, amp, amp3], dim=1)
        feat = self.act(self.in_conv(x_6ch))

        for block in self.tcn_blocks:
            feat = feat + block(feat)

        feat = self.attn(feat, amp, amp3, amp7)
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

    model = PaperDiagramDPDNet(g_real_init, g_imag_init).to(DEVICE)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    )

    best_val_loss = float('inf')
    history_epochs = []
    history_train_nmse = []
    history_val_nmse = []

    print(f"\n>>> 开始训练 H=16 稠密版 TCN (算力边界测试: ~1622 MACs)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_train_error_power, total_train_target_power = 0.0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_error_power += torch.sum((preds.detach() - batch_y) ** 2).item()
            total_train_target_power += torch.sum(batch_y ** 2).item()

        train_linear_nmse = total_train_error_power / total_train_target_power
        train_nmse = 10 * np.log10(train_linear_nmse)

        model.eval()
        total_val_error_power, total_val_target_power = 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
                preds = model(batch_x)
                total_val_error_power += torch.sum((preds - batch_y) ** 2).item()
                total_val_target_power += torch.sum(batch_y ** 2).item()

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

        is_pass = "✅ PASS" if val_linear_nmse < 1e-4 else "❌ WAIT"
        gap = train_nmse - val_nmse

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Time: {elapsed_mins}m {elapsed_secs}s | LR: {current_lr:.2e} | Train: {train_nmse:.2f} dB | Val: {val_nmse:.2f} dB | Gap: {gap:.2f} dB | {is_pass}")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_upper_bound_model.pth')

    plt.figure(figsize=(10, 6))
    plt.plot(history_epochs, history_train_nmse, label='Train NMSE', color='g', linestyle='-.', linewidth=1.5)
    plt.plot(history_epochs, history_val_nmse, label='Validation NMSE', color='b', linewidth=1.5)
    plt.axhline(y=min(history_val_nmse), color='r', linestyle='--', label=f'Best Val: {min(history_val_nmse):.2f} dB')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title(f'TCN H=16 Dense Upper Bound (MACs ≈ 1622)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'dense_tcn16_upperbound.png', dpi=300)
    print(f"\n>>> 训练完成。")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()