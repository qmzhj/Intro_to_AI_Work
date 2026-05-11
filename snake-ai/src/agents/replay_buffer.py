"""
经验回放缓冲区
用于存储和采样训练数据
"""

import random
from collections import deque
from typing import NamedTuple, List
import numpy as np
import torch


class Transition(NamedTuple):
    """经验回放的转换元组"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 1000):
        """
        初始化经验回放缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        添加一条经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> tuple:
        """
        随机采样一批经验

        Args:
            batch_size: 批次大小

        Returns:
            批次数据
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    
        # 预分配张量
        batch = [self.buffer[idx] for idx in indices]
    
        # 获取第一个状态来获取形状
        first_state = batch[0].state
    
        # 预分配张量
        states = torch.empty((batch_size, *first_state.shape), dtype=torch.float32)
        actions = torch.empty(batch_size, dtype=torch.long)
        rewards = torch.empty(batch_size, dtype=torch.float32)
        next_states = torch.empty((batch_size, *first_state.shape), dtype=torch.float32)
        dones = torch.empty(batch_size, dtype=torch.bool)
    
        # 填充张量
        for i, t in enumerate(batch):
            states[i] = torch.from_numpy(t.state)
            actions[i] = torch.tensor(t.action, dtype=torch.long)
            rewards[i] = torch.tensor(t.reward, dtype=torch.float32)
            next_states[i] = torch.from_numpy(t.next_state)
            dones[i] = torch.tensor(t.done, dtype=torch.bool)
    
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        初始化优先经验回放缓冲区

        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """添加一条经验"""
        max_priority = self.priorities.max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = Transition(state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple:
        """
        根据优先级采样

        Args:
            batch_size: 批次大小
            beta: 重要性采样权重指数

        Returns:
            批次数据、索引和权重
        """
        if self.size == 0:
            return None

        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)

        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        # 获取数据
        transitions = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor(np.array([t.state for t in transitions]))
        actions = torch.LongTensor(np.array([t.action for t in transitions]))
        rewards = torch.FloatTensor(np.array([t.reward for t in transitions]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions]))
        dones = torch.BoolTensor(np.array([t.done for t in transitions]))
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return self.size


if __name__ == "__main__":
    """测试经验回放缓冲区"""
    # 测试普通经验回放
    print("测试普通经验回放缓冲区:")
    buffer = ReplayBuffer(capacity=1000)

    # 添加一些经验
    for i in range(100):
        state = np.random.randn(3, 20, 30)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(3, 20, 30)
        done = np.random.choice([True, False])

        buffer.push(state, action, reward, next_state, done)

    print(f"缓冲区大小: {len(buffer)}")

    # 采样一批数据
    if len(buffer) >= 32:
        states, actions, rewards, next_states, dones = buffer.sample(32)
        print(f"采样批次大小: {states.shape[0]}")
        print(f"状态形状: {states.shape}")
        print(f"动作形状: {actions.shape}")

    # 测试优先经验回放
    print("\n测试优先经验回放缓冲区:")
    prio_buffer = PrioritizedReplayBuffer(capacity=1000)

    # 添加一些经验
    for i in range(100):
        state = np.random.randn(3, 20, 30)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        next_state = np.random.randn(3, 20, 30)
        done = np.random.choice([True, False])

        prio_buffer.push(state, action, reward, next_state, done)

    print(f"缓冲区大小: {len(prio_buffer)}")

    # 采样一批数据
    if len(prio_buffer) >= 32:
        result = prio_buffer.sample(32)
        if result is not None:
            states, actions, rewards, next_states, dones, indices, weights = result
            print(f"采样批次大小: {states.shape[0]}")
            print(f"状态形状: {states.shape}")
            print(f"权重形状: {weights.shape}")