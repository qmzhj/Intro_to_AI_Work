"""快速验证三种算法变体能否正常初始化和运行"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.cnn_model import CNNModel
from src.models.dueling_cnn_model import DuelingCNNModel
from src.agents.dqn_agent import DQNAgent, SUPPORTED_ALGOS
from src.envs.snake_env import SnakeEnv, Action
from config import Config
import numpy as np

print("=== 验证导入 ===")
for name, cls in [("CNNModel", CNNModel), ("DuelingCNNModel", DuelingCNNModel)]:
    print(f"  ✓ {name} 导入成功")
print(f"  支持的算法: {SUPPORTED_ALGOS}")

print("\n=== 验证三种算法初始化 ===")
env = SnakeEnv(grid_width=12, grid_height=12, cell_size=20, render_mode=None)

for algo in SUPPORTED_ALGOS:
    agent = DQNAgent(
        state_shape=env.observation_space_shape,
        n_actions=env.action_space,
        algo=algo,
        buffer_size=1000,
        batch_size=32,
    )
    info = agent.get_info()
    assert info['algo'] == algo, f"algo 不匹配: {info['algo']} != {algo}"
    print(f"  ✓ {algo}: policy_net={type(agent.policy_net).__name__}, "
          f"target_net={type(agent.target_net).__name__}")

print("\n=== 验证动作选择和训练 ===")
for algo in SUPPORTED_ALGOS:
    agent = DQNAgent(
        state_shape=env.observation_space_shape,
        n_actions=env.action_space,
        algo=algo,
        buffer_size=500,
        batch_size=32,
        learning_rate=1e-3,
    )
    # 填充缓冲区
    state = env.reset()
    for _ in range(200):
        a = np.random.randint(0, 4)
        ns, r, d, info = env.step(Action(a))
        agent.replay_buffer.push(state, a, r, ns, d)
        state = env.reset() if d else ns

    # 动作选择
    action = agent.select_action(state, eval_mode=True)
    assert 0 <= action < 4

    # 训练几步
    losses = []
    for _ in range(10):
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
    print(f"  ✓ {algo}: 动作={action}, 损失={np.mean(losses):.4f} (共{len(losses)}步)")

# 验证 save/load
print("\n=== 验证保存和加载 ===")
agent_dqn = DQNAgent(
    state_shape=env.observation_space_shape,
    n_actions=env.action_space,
    algo="dqn",
    buffer_size=100,
    batch_size=32,
)
agent_dqn.save("test_verify_dqn.pth")

agent_loaded = DQNAgent(
    state_shape=env.observation_space_shape,
    n_actions=env.action_space,
    algo="dqn",
    buffer_size=100,
    batch_size=32,
)
agent_loaded.load("test_verify_dqn.pth")
print(f"  ✓ 保存/加载: algo={agent_loaded.algo}")

# 清理
os.remove("test_verify_dqn.pth")

print("\n=== 全部验证通过 ===")
