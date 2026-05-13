"""
PPO 智能体
实现 Proximal Policy Optimization 算法（Actor-Critic 架构）

核心特性：
  - Actor-Critic 双网络（共享 CNN 骨干）
  - PPO 裁剪目标函数（ε=0.2）
  - 广义优势估计 GAE（λ=0.95）
  - 熵正则化（系数 0.01）
  - 多 epoch 小批量更新
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List

from ..models.ac_shared_cnn_model import ActorCriticSharedCNN


class RolloutBuffer:
    """
    PPO 轨迹缓冲区
    存储一个或多个完整 episode 的 transition 数据，
    供后续 GAE 计算和多 epoch 策略更新使用
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """清空缓冲区"""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

        # GAE 计算结果
        self.returns: List[float] = []
        self.advantages: List[float] = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """添加一条 transition（按时间顺序）"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.states)


class PPOAgent:
    """
    PPO 智能体

    基于 Actor-Critic 架构，使用 PPO 裁剪目标、GAE 优势估计和熵正则化。
    训练策略：逐 episode 收集轨迹 → episode 结束时计算 GAE → 积累
    到一定量后运行多 epoch 小批量更新。
    """

    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        n_actions: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        update_freq: int = 5,
        max_grad_norm: float = 0.5,
        device: str = None,
    ):
        """
        初始化 PPO 智能体

        Args:
            state_shape: 状态形状 (H, W, C)
            n_actions: 动作空间大小
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 参数 λ
            clip_epsilon: PPO 裁剪参数 ε
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            ppo_epochs: 每次更新时对数据做几轮优化
            mini_batch_size: 小批量大小
            update_freq: 每 N 个 episode 更新一次策略
            max_grad_norm: 梯度裁剪最大范数
            device: 计算设备
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.update_freq = update_freq
        self.max_grad_norm = max_grad_norm

        self.episode_count = 0
        self.training_steps = 0

        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")
        print(f"算法: PPO")

        # Actor-Critic 网络（共享 CNN 骨干）
        self.model = ActorCriticSharedCNN(
            input_channels=state_shape[2],
            n_actions=n_actions,
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 轨迹缓冲区
        self.rollout = RolloutBuffer()

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        选择动作（仅返回动作，兼容 DQN 测试接口）

        内部会更新 self.rollout 的状态，但不会 push - 
        调用者需同时使用 get_action_info() 或手动 push。

        为保持与 DQN 相同的 select_action 接口，
        额外提供了 get_action_info() 返回完整信息。
        """
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            action_probs, _ = self.model(state_tensor)

            if eval_mode:
                # 评估模式：取概率最大的动作（确定性策略）
                action = action_probs.argmax(dim=1).item()
            else:
                # 训练模式：按概率采样（随机策略）
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

        return action

    def get_action_info(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        选择动作，同时返回 log_prob 和价值（供训练使用）

        Args:
            state: (H, W, C) numpy 数组

        Returns:
            action: 采样的动作
            log_prob: 动作的 log 概率
            value: 状态价值标量
        """
        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            action_probs, value = self.model(state_tensor)

            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """存储一条 transition 到轨迹缓冲区"""
        self.rollout.push(state, action, reward, done, log_prob, value)

    def end_episode(self):
        """
        episode 结束时调用，计算 GAE 优势和回报（Returns）
        """
        rollout = self.rollout
        T = len(rollout)
        if T == 0:
            return

        # 将数据转为 tensor
        rewards = torch.tensor(rollout.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(rollout.dones, dtype=torch.float32, device=self.device)
        values = torch.tensor(rollout.values, dtype=torch.float32, device=self.device)

        # 计算 GAE
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        returns = torch.zeros(T, dtype=torch.float32, device=self.device)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                # 最后一步：如果 done，V(s_{T+1}) = 0，否则为当前 value
                next_value = 0.0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]

            # δ_t = r_t + γ * V(s_{t+1}) * (1-done_t) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            # A_t = δ_t + (γλ) * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        # Return_t = A_t + V(s_t)
        returns = advantages + values

        # 存入 rollout buffer
        rollout.returns = returns.cpu().tolist()
        rollout.advantages = advantages.cpu().tolist()

        self.episode_count += 1

    def update(self) -> Optional[dict]:
        """
        执行 PPO 策略更新（当积累足够 episode 时）

        返回损失信息字典，若未满足更新条件则返回 None。
        """
        if self.episode_count < self.update_freq:
            return None

        rollout = self.rollout
        T = len(rollout)
        if T == 0:
            return None

        # 将 rollout 数据转为 tensor
        states = torch.FloatTensor(np.array(rollout.states)).to(self.device)  # (T, H, W, C)
        actions = torch.tensor(rollout.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(rollout.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout.returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(rollout.advantages, dtype=torch.float32, device=self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多 epoch 小批量更新
        indices = np.arange(T)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            # 小批量迭代
            for start in range(0, T, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # 计算新策略的 log prob、熵、价值
                log_probs, entropy, values = self.model.evaluate_actions(mb_states, mb_actions)

                # --- PPO 裁剪损失（Actor） ---
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- 价值损失（Critic） ---
                value_loss = nn.MSELoss()(values, mb_returns)

                # --- 熵正则化（最大化熵 → 最小化 -熵）---
                entropy_loss = -entropy.mean()

                # --- 总损失 ---
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                n_updates += 1
                self.training_steps += 1

        # 清空 rollout，准备下一轮数据收集
        self.rollout.clear()
        self.episode_count = 0

        avg_policy_loss = total_policy_loss / n_updates
        avg_value_loss = total_value_loss / n_updates
        avg_entropy_loss = total_entropy_loss / n_updates
        avg_loss = total_loss / n_updates

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'total_loss': avg_loss,
            'n_updates': n_updates,
            'approx_kl': None,  # 可选KL散度监控
        }

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'algo': 'ppo',
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
        }, filepath)
        print(f"PPO 模型已保存到: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        algo = checkpoint.get('algo', 'ppo')
        print(f"PPO 模型已从 {filepath} 加载 (algo={algo})")

    def get_info(self) -> dict:
        """获取智能体信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'algo': 'ppo',
            'training_steps': self.training_steps,
            'rollout_size': len(self.rollout),
            'episode_count': self.episode_count,
            'device': str(self.device),
            'total_params': total_params,
        }


if __name__ == "__main__":
    """测试 PPO 智能体"""
    print("=" * 50)
    print("测试 PPO 智能体")
    print("=" * 50)

    # 创建智能体
    agent = PPOAgent(
        state_shape=(12, 12, 3),
        n_actions=4,
        learning_rate=3e-4,
        update_freq=2,  # 每 2 个 episode 更新一次（测试用）
    )

    print("\n智能体信息:")
    info = agent.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试采样
    print("\n测试动作采样...")
    state = np.random.randn(12, 12, 3).astype(np.float32)
    action, log_prob, value = agent.get_action_info(state)
    print(f"  动作: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    # 模拟几个 episode 的数据收集和更新
    print("\n模拟训练循环...")
    for ep in range(4):
        # 模拟一个 episode
        for step in range(20):
            state = np.random.randn(12, 12, 3).astype(np.float32)
            action, log_prob, value = agent.get_action_info(state)
            reward = np.random.randn() * 0.1
            done = (step == 19)
            agent.store_transition(state, action, reward, done, log_prob, value)
            if done:
                break

        # Episode 结束
        agent.end_episode()

        # 尝试更新
        result = agent.update()
        if result is not None:
            print(f"  回合 {ep + 1}: 更新完成")
            print(f"    policy_loss={result['policy_loss']:.4f}, "
                  f"value_loss={result['value_loss']:.4f}, "
                  f"entropy_loss={result['entropy_loss']:.4f}")
        else:
            print(f"  回合 {ep + 1}: 等待更多数据 (ep_count={agent.episode_count})")

    # 测试保存和加载
    print("\n测试保存和加载...")
    agent.save("test_model_ppo.pth")
    agent.load("test_model_ppo.pth")

    import os
    os.remove("test_model_ppo.pth")

    print("\nPPO 测试完成!")
