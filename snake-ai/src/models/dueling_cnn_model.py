"""
Dueling CNN 模型
将 Q 值分解为状态价值 V(s) 和动作优势 A(s,a)，减少值函数估计的方差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingCNNModel(nn.Module):
    """
    Dueling 卷积神经网络模型

    输入: (batch_size, 3, height, width)   — 如 (B, 3, 12, 12)
    输出: (batch_size, n_actions)

    架构说明：
      - 共享卷积层提取特征（与 CNNModel 相同的单层 stride-2 卷积）
      - 分离为价值流 V(s) 和优势流 A(s,a)
      - 通过 Q(s,a) = V(s) + A(s,a) - mean(A(s,a)) 组合
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_actions: int = 4,
        hidden_dim: int = 128
    ):
        """
        初始化 Dueling CNN 模型

        Args:
            input_channels: 输入通道数（3: 蛇身、食物、蛇头）
            n_actions: 动作空间大小
            hidden_dim: 隐藏层维度
        """
        super(DuelingCNNModel, self).__init__()

        self.n_actions = n_actions

        # 共享卷积层（与 CNNModel 一致）
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)

        # 计算卷积输出维度
        self._calculate_fc_input_dim(12, 12)

        # === 价值流 V(s) ===
        self.value_fc1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

        # === 优势流 A(s,a) ===
        self.advantage_fc1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.advantage_fc2 = nn.Linear(hidden_dim, n_actions)

    def _calculate_fc_input_dim(self, height: int, width: int) -> None:
        """计算卷积后的特征图大小"""
        def conv_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        h = conv_output_size(height, 3, 2, padding=1)
        w = conv_output_size(width, 3, 2, padding=1)
        self.fc_input_dim = 32 * h * w

        print(f"[DuelingCNN] 特征图尺寸: ({h}, {w}), fc_input_dim: {self.fc_input_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, 3, height, width)

        Returns:
            Q值，形状为 (batch_size, n_actions)
        """
        # 共享卷积特征提取
        x = F.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)

        # 价值流
        v = F.relu(self.value_fc1(x))
        v = self.value_fc2(v)  # (B, 1)

        # 优势流
        a = F.relu(self.advantage_fc1(x))
        a = self.advantage_fc2(a)  # (B, n_actions)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # 减去平均值使优势函数具有零均值的可辨识性
        q = v + a - a.mean(dim=1, keepdim=True)

        return q

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
            return torch.randint(0, self.n_actions, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=1).item()


if __name__ == "__main__":
    """测试 Dueling 模型"""
    model = DuelingCNNModel(input_channels=3, n_actions=4, hidden_dim=128)

    print("\nDueling CNN 模型结构:")
    print(model)

    batch_size = 8
    x = torch.randn(batch_size, 3, 12, 12)
    output = model(x)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"输出形状: {output.shape}")
    print(f"总参数数量: {total_params:,}")
    print(f"空间分辨率: 12×12 → 6×6")

    # 验证 V 和 A 是否独立可访问
    print("\n分解验证:")
    print(f"Q[0]: {output[0]}")
