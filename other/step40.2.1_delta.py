import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 400
BATCH_SIZE = 128
FRAME_LENGTH = 800
STRIDE_LENGTH = 32


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

        h_seq = []
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

        for t in range(seq_len):
            L_n = L[:, t, :]
            f_n = torch.sigmoid(self.W_fi(L_n) + self.W_fh(h_n))
            g_n = torch.tanh(self.W_gi(L_n) + self.W_gh(h_n))
            h_n = f_n * h_n + (1 - f_n) * g_n
            h_seq.append(h_n)

        h_seq = torch.stack(h_seq, dim=1)

        I_rot, Q_rot = self.pr_block(h_seq, sin_theta, cos_theta, self.hidden_size, self.num_vd_units)
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


class DeltaResidualBOJANET(nn.Module):
    def __init__(self, init_g_real, init_g_imag):
        super(DeltaResidualBOJANET, self).__init__()
        self.g_real = nn.Parameter(torch.tensor(init_g_real, dtype=torch.float64))
        self.g_imag = nn.Parameter(torch.tensor(init_g_imag, dtype=torch.float64))

        self.bojanet = BOJANET(hidden_size=11, output_size=2, bias=True)

        nn.init.constant_(self.bojanet.W_out_I.weight, 0.0)
        nn.init.constant_(self.bojanet.W_out_Q.weight, 0.0)
        if self.bojanet.W_out_I.bias is not None:
            nn.init.constant_(self.bojanet.W_out_I.bias, 0.0)
            nn.init.constant_(self.bojanet.W_out_Q.bias, 0.0)

    def forward(self, x_iq):
        # 1. 线性主路径：处理绝对幅值
        i_x, q_x = x_iq[:, :, 0:1], x_iq[:, :, 1:2]
        lin_i = i_x * self.g_real - q_x * self.g_imag
        lin_q = i_x * self.g_imag + q_x * self.g_real
        lin_out = torch.cat([lin_i, lin_q], dim=-1)

        # 2. 构造差分特征 (Delta Features)
        delta_x = x_iq.clone()
        # 计算一阶前向差分：x_t - x_{t-1}
        delta_x[:, 1:, :] = x_iq[:, 1:, :] - x_iq[:, :-1, :]
        # 序列首个元素差值设为0，防止越界异常
        delta_x[:, 0, :] = 0.0

        # 3. 将差分特征喂给 BO-JANET
        residual = self.bojanet(delta_x)
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

    SCALE_X = np.max(np.abs(raw_x))
    SCALE_Y = np.max(np.abs(raw_y))
    x_c = raw_x / SCALE_X
    y_c = raw_y / SCALE_Y

    G_opt = np.vdot(x_c, y_c) / np.vdot(x_c, x_c)
    g_real_init = np.real(G_opt)
    g_imag_init = np.imag(G_opt)
    print(f">>> Calculated Optimal Linear Gain: {G_opt:.4f}")

    X_np = np.stack([np.real(x_c), np.imag(x_c)], axis=1)
    Target_np = np.stack([np.real(y_c), np.imag(y_c)], axis=1)

    min_len = min(len(X_np), len(Target_np))
    split_idx = int(min_len * 0.8)
    X_train, Y_train = X_np[:split_idx], Target_np[:split_idx]
    X_val, Y_val = X_np[split_idx:min_len], Target_np[split_idx:min_len]

    train_loader = DataLoader(RNNFramedDataset(X_train, Y_train, FRAME_LENGTH, STRIDE_LENGTH), batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(RNNFramedDataset(X_val, Y_val, FRAME_LENGTH, FRAME_LENGTH), batch_size=BATCH_SIZE,
                            shuffle=False)

    model = DeltaResidualBOJANET(g_real_init, g_imag_init).to(DEVICE).double()
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    history_epochs = []
    history_nmse = []

    print("\n>>> 开始训练...")
    start_time = time.time()

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

        linear_nmse = total_error_power / total_target_power
        val_nmse = 10 * np.log10(linear_nmse)

        scheduler.step()

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
    main()