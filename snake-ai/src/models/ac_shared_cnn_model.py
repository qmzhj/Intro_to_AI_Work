"""
Actor-Critic 共享 CNN 模型（v2：加深网络）
3 层卷积 + BatchNorm，增强空间特征提取能力
Actor 输出动作概率分布（Softmax，4 个动作），Critic 输出状态价值（标量）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticSharedCNN(nn.Module):
    """
    Actor-Critic 共享卷积神经网络（v2）

    架构：
      Conv(3→32, s=2) → BN → ReLU      # 12x12 → 6x6
      Conv(32→64, s=2) → BN → ReLU     # 6x6 → 3x3
      Conv(64→64, s=1) → BN → ReLU     # 3x3 → 3x3
      Flatten → FC(576, 256) → ReLU
      ├─ Actor: FC(256, n_actions) → Softmax
      └─ Critic: FC(256, 1)
    """

    def __init__(
        self,
        input_channels: int = 3,
        n_actions: int = 4,
        hidden_dim: int = 256,
    ):
        super(ActorCriticSharedCNN, self).__init__()
        self.n_actions = n_actions

        # 共享卷积层（3 层 Conv + BN）
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # 计算 fc 输入维度
        self._calculate_fc_input_dim(12, 12)

        self.shared_fc = nn.Linear(self.fc_input_dim, hidden_dim)

        # === Actor 头 ===
        self.actor_fc = nn.Linear(hidden_dim, n_actions)

        # === Critic 头 ===
        self.critic_fc = nn.Linear(hidden_dim, 1)

    def _calculate_fc_input_dim(self, height: int, width: int) -> None:
        """计算三层卷积后的特征图大小"""
        def conv_output_size(input_size, kernel_size, stride, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        h = conv_output_size(height, 3, 2, padding=1)
        h = conv_output_size(h, 3, 2, padding=1)
        h = conv_output_size(h, 3, 1, padding=1)
        w = conv_output_size(width, 3, 2, padding=1)
        w = conv_output_size(w, 3, 2, padding=1)
        w = conv_output_size(w, 3, 1, padding=1)
        self.fc_input_dim = 64 * h * w

    def forward(self, x: torch.Tensor):
        # (B, H, W, C) → (B, C, H, W)
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)

        # 共享卷积特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.shared_fc(x))

        # Actor 头
        action_logits = self.actor_fc(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic 头
        value = self.critic_fc(x)

        return action_probs, value

    def get_action_and_value(self, state: np.ndarray, device: torch.device):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, value = self.forward(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        action_probs, values = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = values.squeeze(1)
        return log_probs, entropy, values


if __name__ == "__main__":
    """测试 Actor-Critic 模型（v2）"""
    model = ActorCriticSharedCNN(input_channels=3, n_actions=4, hidden_dim=256)

    print("\nActor-Critic 共享 CNN 模型结构 (v2):")
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
