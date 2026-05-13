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
    
    架构说明：
      - 1 层 stride-2 卷积（保持 6×6 空间分辨率，不压缩到 3×3）
      - 与之前 2 层 stride-2 卷积（→3×3）相比，空间信息更完整
      - 蛇头、食物、墙壁的位置在 6×6 特征图上仍然可分辨
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_actions: int = 4,
        hidden_dim: int = 256
    ):
        """
        初始化轻量CNN模型

        Args:
            input_channels: 输入通道数（3: 蛇身、食物、蛇头）
            n_actions: 动作空间大小
            hidden_dim: 隐藏层维度
        """
        super(CNNModel, self).__init__()

        # 单层 stride-2 卷积，保留 6×6 空间分辨率
        # 12x12 → stride=2, pad=1 → 6x6
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)

        # 计算 fc 输入维度
        self._calculate_fc_input_dim(12, 12)

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def _calculate_fc_input_dim(self, height: int, width: int) -> None:
        """
        根据输入尺寸计算卷积后的特征图大小
        """
        def conv_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1
        
        h = conv_output_size(height, 3, 2, padding=1)
        w = conv_output_size(width, 3, 2, padding=1)
        
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
        # 卷积 → ReLU → 展平 → 全连接
        x = F.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
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
            return torch.randint(0, self.fc2.out_features, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=1).item()


if __name__ == "__main__":
    """测试模型"""
    model = CNNModel(input_channels=3, n_actions=4, hidden_dim=128)

    print("\n模型结构:")
    print(model)

    batch_size = 8
    x = torch.randn(batch_size, 3, 12, 12)
    output = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"输出形状: {output.shape}")
    print(f"总参数数量: {total_params:,}")
    print(f"空间分辨率: 12×12 → 6×6 （每格对应 2×2 像素）")
