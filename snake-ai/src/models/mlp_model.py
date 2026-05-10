"""
多层感知机模型
用于处理贪吃蛇游戏的向量状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    多层感知机模型
    输入: (batch_size, state_dim)
    输出: (batch_size, n_actions)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 4,
        hidden_dims: list = [256, 256]
    ):
        """
        初始化MLP模型

        Args:
            state_dim: 状态维度
            n_actions: 动作空间大小
            hidden_dims: 隐藏层维度列表
        """
        super(MLPModel, self).__init__()

        # 构建全连接层
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(input_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, state_dim)

        Returns:
            Q值，形状为 (batch_size, n_actions)
        """
        return self.network(x)

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
            return torch.randint(0, self.network[-1].out_features, (1,)).item()
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=1).item()


if __name__ == "__main__":
    """测试模型"""
    # 创建模型
    model = MLPModel(state_dim=600, n_actions=4, hidden_dims=[256, 256])

    # 打印模型结构
    print("模型结构:")
    print(model)

    # 测试前向传播
    batch_size = 8
    state_dim = 600  # 20 * 30 * 1 (展平的单通道图像)
    x = torch.randn(batch_size, state_dim)

    print(f"\n输入形状: {x.shape}")

    # 前向传播
    output = model(x)
    print(f"输出形状: {output.shape}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")

    # 测试动作选择
    state = torch.randn(1, state_dim)
    action = model.get_action(state, epsilon=0.0)
    print(f"选择的动作 (贪婪): {action}")

    action_random = model.get_action(state, epsilon=1.0)
    print(f"选择的动作 (随机): {action_random}")