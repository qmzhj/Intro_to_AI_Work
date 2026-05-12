"""
DQN智能体
实现深度Q网络算法及其改进变体（Double DQN、Dueling DQN）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
import copy

from ..models.cnn_model import CNNModel
from ..models.dueling_cnn_model import DuelingCNNModel
from .replay_buffer import ReplayBuffer


SUPPORTED_ALGOS = ["dqn", "double_dqn", "dueling_dqn"]


class DQNAgent:
    """
    DQN智能体（支持标准DQN / Double DQN / Dueling DQN）

    通过 algo 参数选择不同变体：
      - "dqn":        标准 DQN，目标网络直接取 max Q
      - "double_dqn": Double DQN，用策略网络选动作、目标网络评估
      - "dueling_dqn": Dueling DQN，使用 V + A 的网络结构（且默认启用 double 目标计算）
    """

    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str = None,
        algo: str = "dqn",
    ):
        """
        初始化DQN智能体

        Args:
            state_shape: 状态形状 (H, W, C)
            n_actions: 动作空间大小
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率（步数）
            device: 计算设备
            algo: 算法变体，可选 "dqn" / "double_dqn" / "dueling_dqn"
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_steps = 0

        # 算法选择
        assert algo in SUPPORTED_ALGOS, f"不支持的算法: {algo}，可选: {SUPPORTED_ALGOS}"
        self.algo = algo

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")
        print(f"算法: {self.algo}")

        # 根据 algo 选择网络模型
        input_channels = state_shape[2]

        if algo == "dueling_dqn":
            model_class = DuelingCNNModel
        else:
            model_class = CNNModel

        self.policy_net = model_class(
            input_channels=input_channels,
            n_actions=n_actions
        ).to(self.device)

        self.target_net = model_class(
            input_channels=input_channels,
            n_actions=n_actions
        ).to(self.device)

        # 复制参数到目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # 损失函数
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        选择动作

        Args:
            state: 当前状态 (H, W, C)
            eval_mode: 是否为评估模式（不探索）

        Returns:
            选择的动作
        """
        if eval_mode or np.random.random() > self.epsilon:
            # 贪婪策略
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                state_tensor = state_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
        else:
            # 随机探索
            action = np.random.randint(0, self.n_actions)

        return action

    def _compute_target_q(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        计算目标 Q 值，根据 algo 选择不同方式

        - dqn:        target = r + γ * max_a' Q_target(s', a')
        - double_dqn: target = r + γ * Q_target(s', argmax_a' Q_policy(s', a'))
        - dueling_dqn: 默认使用 double 方式（与 double_dqn 相同）
        """
        with torch.no_grad():
            if self.algo in ("double_dqn", "dueling_dqn"):
                # Double DQN: 策略网络选择动作，目标网络评估
                next_q_policy = self.policy_net(next_states)      # (B, n_actions)
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)  # (B, 1)
                next_q_target = self.target_net(next_states)       # (B, n_actions)
                next_q_value = next_q_target.gather(1, next_actions).squeeze(1)  # (B,)
            else:
                # 标准 DQN: 目标网络直接取 max
                next_q_value = self.target_net(next_states).max(1)[0]

            target_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value

        return target_q_value

    def train_step(self) -> Optional[float]:
        """
        执行一步训练

        Returns:
            损失值
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 采样批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 转换状态格式: (B, H, W, C) -> (B, C, H, W)
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)

        # 计算当前Q值
        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1))

        # 计算目标Q值（按 algo 区分）
        target_q_value = self._compute_target_q(rewards, next_states, dones)

        # 计算损失
        loss = self.criterion(q_value, target_q_value.unsqueeze(1))

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新目标网络
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def update_epsilon(self, episode: int, total_episodes: int):
        """
        根据训练进度更新探索率

        Args:
            episode: 当前回合
            total_episodes: 总回合数
        """
        progress = episode / total_episodes
        self.epsilon = self.epsilon_start * (1 - progress) + self.epsilon_end * progress

    def save(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        torch.save({
            'algo': self.algo,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
        print(f"模型已保存到: {filepath}")

    def load(self, filepath: str):
        """
        加载模型

        Args:
            filepath: 加载路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        if 'algo' in checkpoint:
            self.algo = checkpoint['algo']
        print(f"模型已从 {filepath} 加载 (algo={self.algo})")

    def get_info(self) -> dict:
        """获取智能体信息"""
        return {
            'algo': self.algo,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device)
        }


if __name__ == "__main__":
    """测试三种算法变体的 DQN 智能体"""
    for algo in SUPPORTED_ALGOS:
        print(f"\n{'='*50}")
        print(f"测试算法: {algo}")
        print(f"{'='*50}")

        # 创建智能体
        agent = DQNAgent(
            state_shape=(12, 12, 3),
            n_actions=4,
            learning_rate=1e-4,
            buffer_size=1000,
            batch_size=32,
            algo=algo,
        )

        print("智能体信息:")
        info = agent.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        # 测试动作选择
        state = np.random.randn(12, 12, 3)
        action = agent.select_action(state, eval_mode=False)
        print(f"\n选择的动作: {action}")

        # 测试训练步骤
        print("\n添加经验到缓冲区并训练...")
        for i in range(100):
            state = np.random.randn(12, 12, 3)
            next_state = np.random.randn(12, 12, 3)
            action = np.random.randint(0, 4)
            reward = np.random.randn()
            done = np.random.choice([True, False])
            agent.replay_buffer.push(state, action, reward, next_state, done)

        print(f"缓冲区大小: {len(agent.replay_buffer)}")

        for i in range(5):
            loss = agent.train_step()
            if loss is not None:
                print(f"  步骤 {i+1}, 损失: {loss:.4f}")

        # 测试保存和加载
        print("\n测试保存和加载...")
        agent.save(f"test_model_{algo}.pth")
        agent.load(f"test_model_{algo}.pth")
