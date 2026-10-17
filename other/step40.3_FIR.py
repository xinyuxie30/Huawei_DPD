import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
BATCH_SIZE = 128
FRAME_LENGTH = 800
STRIDE_LENGTH = 32


class ComplexFIR(nn.Module):
    def __init__(self, num_taps, init_g_real, init_g_imag):
        super(ComplexFIR, self).__init__()
        self.num_taps = num_taps
        self.pad = num_taps // 2

        w_real = torch.zeros(1, 1, num_taps, dtype=torch.float64)
        w_imag = torch.zeros(1, 1, num_taps, dtype=torch.float64)
        w_real[0, 0, self.pad] = init_g_real
        w_imag[0, 0, self.pad] = init_g_imag

        self.w_real = nn.Parameter(w_real)
        self.w_imag = nn.Parameter(w_imag)

    def forward(self, i_x, q_x):
        i_pad = torch.nn.functional.pad(i_x, (self.pad, self.pad))
        q_pad = torch.nn.functional.pad(q_x, (self.pad, self.pad))

        i_out = torch.nn.functional.conv1d(i_pad, self.w_real) - torch.nn.functional.conv1d(q_pad, self.w_imag)
        q_out = torch.nn.functional.conv1d(i_pad, self.w_imag) + torch.nn.functional.conv1d(q_pad, self.w_real)
        return torch.cat([i_out, q_out], dim=1)


class HighOrderDPDNet(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(HighOrderDPDNet, self).__init__()
        self.hidden = 10  # 压缩隐藏层以换取更深的记忆

        self.linear_fir = ComplexFIR(num_taps=5, init_g_real=init_g_real, init_g_imag=init_g_imag)

        # 核心：接收 8 通道高阶物理基底 (1st, 3rd, 5th, 7th order)
        self.in_conv = nn.Conv1d(8, self.hidden, 1, bias=False, dtype=torch.float64)
        self.act = nn.SiLU()

        self.tcn_blocks = nn.ModuleList()
        # 更深的记忆跨度：Dilation = 1, 2, 4, 8
        for dilation in [1, 2, 4, 8]:
            self.tcn_blocks.append(nn.Sequential(
                nn.Conv1d(self.hidden, self.hidden, 5, padding=2 * dilation, dilation=dilation,
                          groups=self.hidden, bias=False, dtype=torch.float64),
                nn.Conv1d(self.hidden, self.hidden, 1, bias=False, dtype=torch.float64),
                nn.SiLU()
            ))

        self.out_conv = nn.Conv1d(self.hidden, 2, 1, bias=False, dtype=torch.float64)
        nn.init.constant_(self.out_conv.weight, 0.0)

    def forward(self, x_iq):
        i_x, q_x = x_iq[:, 0:1, :], x_iq[:, 1:2, :]

        # 1. 线性主干
        lin_out = self.linear_fir(i_x, q_x)

        # 2. 零 MACs 消耗的终极高阶特征衍生
        amp2 = i_x ** 2 + q_x ** 2
        amp4 = amp2 ** 2
        amp6 = amp4 * amp2

        x_8ch = torch.cat([
            i_x, q_x,  # 1阶
            i_x * amp2, q_x * amp2,  # 3阶 IMD
            i_x * amp4, q_x * amp4,  # 5阶 IMD
            i_x * amp6, q_x * amp6  # 7阶 IMD
        ], dim=1)

        # 3. TCN 纯粹的时域记忆混合
        feat = self.act(self.in_conv(x_8ch))
        for block in self.tcn_blocks:
            feat = feat + block(feat)

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

    model = HighOrderDPDNet(g_real_init, g_imag_init).to(DEVICE)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-6)
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