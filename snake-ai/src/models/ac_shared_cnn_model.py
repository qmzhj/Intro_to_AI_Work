"""
Actor-Critic 共享 CNN 模型
Actor 输出动作概率分布（Softmax，4 个动作），Critic 输出状态价值（标量）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticSharedCNN(nn.Module):
    """
    Actor-Critic 共享卷积神经网络

    输入: (batch_size, 3, height, width)   — 如 (B, 3, 12, 12)
    Actor 输出: (batch_size, n_actions)    — 动作概率（Softmax）
    Critic 输出: (batch_size, 1)           — 状态价值 V(s)

    架构说明：
      - 共享卷积层提取空间特征（与 CNNModel 相同的单层 stride-2 卷积）
      - Actor 头：特征 → 全连接 → n_actions → Softmax
      - Critic 头：特征 → 全连接 → 1（标量价值）
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_actions: int = 4,
        hidden_dim: int = 128,
    ):
        """
        初始化 Actor-Critic 共享模型

        Args:
            input_channels: 输入通道数（3: 蛇身、食物、蛇头）
            n_actions: 动作空间大小
            hidden_dim: 隐藏层维度
        """
        super(ActorCriticSharedCNN, self).__init__()

        self.n_actions = n_actions

        # 共享卷积层（与 CNNModel 一致）
        # 12x12 → stride=2, pad=1 → 6x6
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)

        # 计算 fc 输入维度
        self._calculate_fc_input_dim(12, 12)

        # 共享特征后的全连接层
        self.shared_fc = nn.Linear(self.fc_input_dim, hidden_dim)

        # === Actor 头：输出动作概率 ===
        self.actor_fc = nn.Linear(hidden_dim, n_actions)

        # === Critic 头：输出状态价值 ===
        self.critic_fc = nn.Linear(hidden_dim, 1)

    def _calculate_fc_input_dim(self, height: int, width: int) -> None:
        """计算卷积后的特征图大小"""
        def conv_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        h = conv_output_size(height, 3, 2, padding=1)
        w = conv_output_size(width, 3, 2, padding=1)
        self.fc_input_dim = 32 * h * w

    def forward(self, x: torch.Tensor):
        """
        前向传播，同时输出动作概率和价值

        Args:
            x: 输入张量 (batch_size, 3, height, width) 或 (B, H, W, C)

        Returns:
            action_probs: 动作概率 (batch_size, n_actions)
            value: 状态价值 (batch_size, 1)
        """
        # 输入格式转换 (B, H, W, C) → (B, C, H, W) 如果需要
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)

        # 共享卷积特征提取
        x = F.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.shared_fc(x))

        # Actor 头：动作概率
        action_logits = self.actor_fc(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic 头：状态价值
        value = self.critic_fc(x)

        return action_probs, value

    def get_action_and_value(self, state: np.ndarray, device: torch.device):
        """
        给定单个状态，返回动作、log 概率和价值

        Args:
            state: (H, W, C) numpy 数组
            device: 计算设备

        Returns:
            action: 采样的动作 int
            log_prob: 动作的 log 概率
            value: 状态价值标量
        """
        state_tensor = (
            torch.FloatTensor(state).unsqueeze(0).to(device)
        )
        # state_tensor: (1, H, W, C) → 在 forward 中自动 permute
        action_probs, value = self.forward(state_tensor)

        # 从概率分布采样
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """
        计算给定状态下指定动作的 log 概率、熵和价值

        Args:
            states: (batch_size, C, H, W)
            actions: (batch_size,) 动作索引

        Returns:
            log_probs: (batch_size,) log π(a|s)
            entropy: (batch_size,) 策略熵
            values: (batch_size,) V(s)
        """
        action_probs, values = self.forward(states)  # (B, n_actions), (B, 1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = values.squeeze(1)  # (B,) -> scalar per sample
        return log_probs, entropy, values


if __name__ == "__main__":
    """测试 Actor-Critic 模型"""
    model = ActorCriticSharedCNN(input_channels=3, n_actions=4, hidden_dim=128)

    print("\nActor-Critic 共享 CNN 模型结构:")
    print(model)

    batch_size = 8
    x = torch.randn(batch_size, 3, 12, 12)
    action_probs, values = model(x)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n输入形状: {x.shape}")
    print(f"动作概率形状: {action_probs.shape}")
    print(f"价值形状: {values.shape}")
    print(f"总参数数量: {total_params:,}")
    print(f"动作概率 (第1样本): {action_probs[0].detach().numpy()}")
    print(f"价值 (第1样本): {values[0].item():.4f}")

    # 测试采样
    device = torch.device('cpu')
    state = np.random.randn(12, 12, 3).astype(np.float32)
    action, log_prob, value = model.get_action_and_value(state, device)
    print(f"\n采样动作: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
