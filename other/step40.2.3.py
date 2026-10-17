import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time
from typing import List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 400
BATCH_SIZE = 64
FRAME_LENGTH = 800
STRIDE_LENGTH = 32


@torch.jit.script
def bojanet_rnn_loop(L: torch.Tensor, h_0: torch.Tensor,
                     W_fi_w: torch.Tensor, W_fi_b: torch.Tensor, W_fh_w: torch.Tensor,
                     W_gi_w: torch.Tensor, W_gi_b: torch.Tensor, W_gh_w: torch.Tensor) -> torch.Tensor:
    seq_len = L.size(1)
    h_n = h_0
    h_seq = torch.jit.annotate(List[torch.Tensor], [])

    for t in range(seq_len):
        L_n = L[:, t, :]
        fi_out = torch.nn.functional.linear(L_n, W_fi_w, W_fi_b)
        fh_out = torch.nn.functional.linear(h_n, W_fh_w, None)
        f_n = torch.sigmoid(fi_out + fh_out)

        gi_out = torch.nn.functional.linear(L_n, W_gi_w, W_gi_b)
        gh_out = torch.nn.functional.linear(h_n, W_gh_w, None)
        g_n = torch.tanh(gi_out + gh_out)

        h_n = f_n * h_n + (1.0 - f_n) * g_n
        h_seq.append(h_n)

    return torch.stack(h_seq, dim=1)


# ==========================================
# 核心修复：引入 9 阶复数 FIR 吸收线性频率失真
# ==========================================
class ComplexFIR(nn.Module):
    def __init__(self, num_taps, init_g_real, init_g_imag):
        super(ComplexFIR, self).__init__()
        self.pad = num_taps // 2
        w_real = torch.zeros(1, 1, num_taps, dtype=torch.float64)
        w_imag = torch.zeros(1, 1, num_taps, dtype=torch.float64)
        w_real[0, 0, self.pad] = init_g_real
        w_imag[0, 0, self.pad] = init_g_imag
        self.w_real = nn.Parameter(w_real)
        self.w_imag = nn.Parameter(w_imag)

    def forward(self, x_iq):
        # x_iq shape: (batch, seq_len, 2)
        i_x = x_iq[:, :, 0].unsqueeze(1)
        q_x = x_iq[:, :, 1].unsqueeze(1)

        i_pad = torch.nn.functional.pad(i_x, (self.pad, self.pad))
        q_pad = torch.nn.functional.pad(q_x, (self.pad, self.pad))

        i_out = torch.nn.functional.conv1d(i_pad, self.w_real) - torch.nn.functional.conv1d(q_pad, self.w_imag)
        q_out = torch.nn.functional.conv1d(i_pad, self.w_imag) + torch.nn.functional.conv1d(q_pad, self.w_real)

        return torch.cat([i_out.transpose(1, 2), q_out.transpose(1, 2)], dim=-1)


class BOJANET(nn.Module):
    def __init__(self, hidden_size=11, output_size=2, bias=True):
        super(BOJANET, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.window_size = 16
        self.num_vd_units = 6
        self.bias = bias

        self.fir_I = nn.Linear(self.window_size, self.num_vd_units, bias=False)
        self.fir_Q = nn.Linear(self.window_size, self.num_vd_units, bias=False)

        self.W_fi = nn.Linear(self.num_vd_units * 2, hidden_size, bias=bias)
        self.W_fh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_gi = nn.Linear(self.num_vd_units * 2, hidden_size, bias=bias)
        self.W_gh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_out_I = nn.Linear(hidden_size, 1, bias=bias)
        self.W_out_Q = nn.Linear(hidden_size, 1, bias=bias)

        self.reset_parameters()

    def vd_module(self, I, Q):
        magnitude_squared = torch.pow(I, 2) + torch.pow(Q, 2)
        magnitude = torch.sqrt(magnitude_squared + 1e-12)
        sin_theta = Q / magnitude
        cos_theta = I / magnitude
        return magnitude, magnitude_squared, sin_theta, cos_theta

    def pr_block(self, h, sin_theta, cos_theta, hidden_size, vd_units):
        if vd_units >= hidden_size:
            cos_theta = cos_theta[:, :, :hidden_size]
            sin_theta = sin_theta[:, :, :hidden_size]
        elif hidden_size > vd_units and hidden_size <= vd_units * 2:
            cos_theta = torch.cat([cos_theta, cos_theta[:, :, :hidden_size - vd_units]], dim=-1)
            sin_theta = torch.cat([sin_theta, sin_theta[:, :, :hidden_size - vd_units]], dim=-1)
        elif hidden_size > vd_units * 2:
            cos_theta = torch.cat([cos_theta, cos_theta, cos_theta[:, :, :hidden_size - 2 * vd_units]], dim=-1)
            sin_theta = torch.cat([sin_theta, sin_theta, sin_theta[:, :, :hidden_size - 2 * vd_units]], dim=-1)
        return h * cos_theta, h * sin_theta

    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.shape
        if h_0 is None:
            h_n = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_n = h_0

        feature_size = x.size(2)

        pad = torch.zeros_like(x[:, -(self.window_size - 1):, :])
        x_pad = torch.cat((pad, x), dim=1)
        windows = x_pad.unfold(dimension=1, size=self.window_size, step=1).transpose(2, 3)
        windows = torch.unsqueeze(windows, dim=2)
        windows = windows.contiguous().view(batch_size, seq_len, self.window_size, feature_size)

        I_fir = self.fir_I(windows[:, :, :, 0]) - self.fir_Q(windows[:, :, :, 1])
        Q_fir = self.fir_Q(windows[:, :, :, 0]) + self.fir_I(windows[:, :, :, 1])

        magnitude, magnitude_squared, sin_theta, cos_theta = self.vd_module(I_fir, Q_fir)
        L = torch.stack([magnitude, magnitude_squared], dim=2)
        L = L.view(batch_size, seq_len, self.num_vd_units * 2)

        h_seq_tensor = bojanet_rnn_loop(
            L, h_n,
            self.W_fi.weight, self.W_fi.bias, self.W_fh.weight,
            self.W_gi.weight, self.W_gi.bias, self.W_gh.weight
        )

        I_rot, Q_rot = self.pr_block(h_seq_tensor, sin_theta, cos_theta, self.hidden_size, self.num_vd_units)
        out_I = self.W_out_I(I_rot) - self.W_out_Q(Q_rot)
        out_Q = self.W_out_Q(Q_rot) + self.W_out_I(I_rot)
        out = torch.cat([out_I, out_Q], dim=-1)
        return out

    def reset_parameters(self):
        for module in [self.fir_I, self.fir_Q]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
        for module in [self.W_fi, self.W_gi]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
        for module in [self.W_fh, self.W_gh]:
            nn.init.orthogonal_(module.weight, gain=1.0)


class ResidualBOJANET(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(ResidualBOJANET, self).__init__()

        # 抛弃标量旁路，换装 9 阶 FIR 滤波器
        self.linear_bypass = ComplexFIR(9, init_g_real, init_g_imag)
        self.bojanet = BOJANET(hidden_size=11, output_size=2, bias=True)

        nn.init.constant_(self.bojanet.W_out_I.weight, 0.0)
        nn.init.constant_(self.bojanet.W_out_Q.weight, 0.0)
        if self.bojanet.W_out_I.bias is not None:
            nn.init.constant_(self.bojanet.W_out_I.bias, 0.0)
            nn.init.constant_(self.bojanet.W_out_Q.bias, 0.0)

    def forward(self, x_iq):
        # 1. FIR 完美吸收所有线性形变
        lin_out = self.linear_bypass(x_iq)
        # 2. BO-JANET 专注拟合纯非线性残差
        residual = self.bojanet(x_iq)
        return lin_out + residual


class RNNFramedDataset(Dataset):
    def __init__(self, features, targets, frame_length, stride_length):
        self.features = torch.as_tensor(features, dtype=torch.float64)
        self.targets = torch.as_tensor(targets, dtype=torch.float64)
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.num_frames = (self.features.shape[0] - self.frame_length) // self.stride_length + 1

    def __len__(self): return self.num_frames

    def __getitem__(self, idx):
        start = idx * self.stride_length
        return self.features[start:start + self.frame_length, :], self.targets[start:start + self.frame_length, :]


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

    train_loader = DataLoader(RNNFramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    val_loader = DataLoader(RNNFramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = ResidualBOJANET(g_real_init, g_imag_init).to(DEVICE).double()
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    best_val_loss = float('inf')
    history_epochs = []
    history_nmse = []

    print("\n>>> 开始训练 完全体 FIR-BO-JANET...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE, non_blocking=True), batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

        scheduler.step(linear_nmse)

        history_epochs.append(epoch + 1)
        history_nmse.append(val_nmse)

        current_lr = optimizer.param_groups[0]['lr']
        elapsed_time = time.time() - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time % 60)

        is_pass = "✅ PASS (<1e-4)" if linear_nmse < 1e-4 else "❌ WAIT"
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | Time: {elapsed_mins}m {elapsed_secs}s | LR: {current_lr:.2e} | NMSE: {val_nmse:.2f} dB | Linear NMSE: {linear_nmse:.4e} {is_pass}")

        if val_nmse < best_val_loss:
            best_val_loss = val_nmse
            torch.save(model.state_dict(), 'best_competition_model.pth')


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()