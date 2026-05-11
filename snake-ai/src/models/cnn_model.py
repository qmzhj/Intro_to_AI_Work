"""
卷积神经网络模型
用于处理贪吃蛇游戏的图像状态 (12x12 小网格)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    轻量卷积神经网络模型
    输入: (batch_size, 3, height, width)   — 如 (B, 3, 12, 12)
    输出: (batch_size, n_actions)
    
    设计说明：对于 12x12 小网格，使用 2 层卷积 + 小全连接层，
    相比原来 3 层深卷积 + 256 隐藏层，参数量减少约 5 倍。
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_actions: int = 4,
        hidden_dim: int = 128
    ):
        """
        初始化轻量CNN模型

        Args:
            input_channels: 输入通道数（3: 蛇身、食物、蛇头）
            n_actions: 动作空间大小
            hidden_dim: 隐藏层维度（从 256 降至 128）
        """
        super(CNNModel, self).__init__()

        # 2 层卷积（原来 3 层），通道数减半
        # 输入 (3, H, W)
        # conv1: 3→16, kernel=3, stride=2, pad=1  -> (16, H/2, W/2)
        # conv2: 16→32, kernel=3, stride=2, pad=1 -> (32, H/4, W/4)
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)

        # 用于动态计算 fc 输入维度
        self._calculate_fc_input_dim(12, 12)

        # 全连接层（隐藏层从 256 降至 128）
        self.fc1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def _calculate_fc_input_dim(self, height: int, width: int) -> None:
        """
        根据输入尺寸计算卷积后的特征图大小
        
        Args:
            height: 输入高度
            width: 输入宽度
        """
        def conv_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1
        
        h = height
        w = width
        
        # conv1: stride=2
        h = conv_output_size(h, 3, 2, padding=1)
        w = conv_output_size(w, 3, 2, padding=1)
        
        # conv2: stride=2
        h = conv_output_size(h, 3, 2, padding=1)
        w = conv_output_size(w, 3, 2, padding=1)
        
        self.fc_input_dim = 32 * h * w
        
        print(f"特征图最终尺寸: ({h}, {w}), fc_input_dim: {self.fc_input_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, 3, height, width)

        Returns:
            Q值，形状为 (batch_size, n_actions)
        """
        # 卷积层 + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 展平
        x = x.reshape(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        根据当前状态选择动作（epsilon-greedy策略）

        Args:
            state: 当前状态
            epsilon: 探索率

        Returns:
            选择的动作
        """
        if torch.rand(1) < epsilon:
            # 随机探索
            return torch.randint(0, self.fc2.out_features, (1,)).item()
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=1).item()


if __name__ == "__main__":
    """测试模型"""
    import numpy as np

    # 创建轻量模型
    model = CNNModel(input_channels=3, n_actions=4, hidden_dim=128)

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    # 测试前向传播
    batch_size = 8
    height, width = 12, 12
    x = torch.randn(batch_size, 3, height, width)

    print(f"\n输入形状: {x.shape}")

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")

    # 测试动作选择
    state = torch.randn(1, 3, height, width)
    action = model.get_action(state, epsilon=0.0)
    print(f"选择的动作 (贪婪): {action}")

    action_random = model.get_action(state, epsilon=1.0)
    print(f"选择的动作 (随机): {action_random}")
