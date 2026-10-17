import torch
import torch.nn as nn


class SimpleRVTDCNN(nn.Module):
    def __init__(self, in_channels, sequence_length, output_dim=2):
        """
        初始化一个简单的 RVTDCNN 模型。

        参数:
            in_channels (int): 输入数据的通道数（例如：I路、Q路、包络等特征数量）。
            sequence_length (int): 输入序列的时间步长（记忆深度）。
            output_dim (int): 输出数据的维度（默认为2，用于预测 I 和 Q 路信号）。
        """
        super(SimpleRVTDCNN, self).__init__()

        # --- 卷积特征提取层 ---
        # 第一个卷积层：将输入通道映射到更高维的特征空间
        # kernel_size=3 表示卷积核覆盖3个时间步
        # padding=1 用于保持序列长度不变 (sequence_length)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # 第二个卷积层：进一步提取特征
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # --- 全连接拟合层 ---
        # 计算扁平化后的特征维度。
        # 由于 padding=1 且 stride=1，卷积层输出的序列长度保持为 sequence_length。
        # 因此，展平后的维度是 out_channels * sequence_length。
        flattened_dim = 32 * sequence_length

        self.flatten = nn.Flatten()

        # 隐藏全连接层
        self.fc1 = nn.Linear(flattened_dim, 64)
        self.relu3 = nn.ReLU()

        # 输出层：映射到最终的输出维度
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        定义模型的前向传播路径。

        参数:
            x (torch.Tensor): 输入数据，形状为 (batch_size, in_channels, sequence_length)。

        返回:
            torch.Tensor: 模型输出，形状为 (batch_size, output_dim)。
        """
        # 卷积块 1
        x = self.conv1(x)
        x = self.relu1(x)

        # 卷积块 2
        x = self.conv2(x)
        x = self.relu2(x)

        # 扁平化
        x = self.flatten(x)

        # 全连接块
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


# --- 模型使用示例 ---

# 1. 定义超参数
batch_size = 8  # 批次大小
in_channels = 3  # 假设输入特征为 I, Q, 包络
sequence_length = 10  # 记忆深度为10个时间步
output_dim = 2  # 输出预测的 I, Q

# 2. 实例化模型
model = SimpleRVTDCNN(in_channels=in_channels, sequence_length=sequence_length, output_dim=output_dim)

# 打印模型结构
print("模型结构:")
print(model)

# 3. 创建一个随机输入张量来测试模型
# 输入形状: (batch_size, in_channels, sequence_length)
dummy_input = torch.randn(batch_size, in_channels, sequence_length)
print(f"\n输入张量形状: {dummy_input.shape}")

# 4. 执行前向传播
output = model(dummy_input)
print(f"输出张量形状: {output.shape}")

# 验证输出形状是否符合预期 (batch_size, output_dim)
assert output.shape == (batch_size, output_dim)
print("\n前向传播测试成功！")