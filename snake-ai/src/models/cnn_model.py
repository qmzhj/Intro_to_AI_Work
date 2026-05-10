"""
卷积神经网络模型
用于处理贪吃蛇游戏的图像状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    卷积神经网络模型
    输入: (batch_size, 3, height, width)
    输出: (batch_size, n_actions)
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_actions: int = 4,
        hidden_dim: int = 256
    ):
        """
        初始化CNN模型

        Args:
            input_channels: 输入通道数（3: 蛇身、食物、蛇头）
            n_actions: 动作空间大小
            hidden_dim: 隐藏层维度
        """
        super(CNNModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 计算卷积后的特征图大小
        # 假设输入为 (3, 20, 30)
        # conv1: (32, 4, 6)
        # conv2: (64, 1, 2)
        # conv3: (64, 1, 2)
        self.fc_input_dim = 64 * 1 * 2

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, 3, height, width)

        Returns:
            Q值，形状为 (batch_size, n_actions)
        """
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平
        x = x.view(x.size(0), -1)

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
    # 创建模型
    model = CNNModel(input_channels=3, n_actions=4, hidden_dim=256)

    # 打印模型结构
    print("模型结构:")
    print(model)

    # 测试前向传播
    batch_size = 8
    height, width = 20, 30
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