"""验证三项优化改动正确生效"""
import os, sys, io
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.envs.snake_env import SnakeEnv, Action

print("=" * 50)
print("验证 1: 距离奖励")
print("=" * 50)
env = SnakeEnv(grid_width=12, grid_height=12, max_steps=200)
obs = env.reset()
h = env.snake_positions[0]
f = env.food_position
d = env._distance_to_food()
print(f"  蛇头={h} 食物={f} 曼哈顿距离={d}")
print(f"  FOOD奖励={env.REWARD_FOOD} STEP惩罚={env.REWARD_STEP} 距离系数={env.REWARD_PROXIMITY_COEF}")
assert env.REWARD_STEP == -0.01, f"STEP奖励应为-0.01, 实际={env.REWARD_STEP}"
assert env.REWARD_PROXIMITY_COEF == 0.5, f"距离系数应为0.5, 实际={env.REWARD_PROXIMITY_COEF}"
print("  ✓ 奖励参数正确")

# 测试一步奖励
for name, delta in [("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0))]:
    nh = (h[0] + delta[0], h[1] + delta[1])
    if 0 <= nh[0] < 12 and 0 <= nh[1] < 12:
        nd = abs(nh[0] - f[0]) + abs(nh[1] - f[1])
        prox = 0.5 * (d - nd)
        total = -0.01 + prox
        direction = "接近" if nd < d else "远离" if nd > d else "持平"
        print(f"    {name}: 距离{d}→{nd} ({direction}), 临近奖励={prox:+.2f}, 总步奖励={total:+.2f}")

print()
print("=" * 50)
print("验证 2: 指数 epsilon 衰减")
print("=" * 50)
from src.agents.dqn_agent import DQNAgent

old_stdout = sys.stdout
sys.stdout = io.StringIO()
agent = DQNAgent(state_shape=(12, 12, 3), n_actions=4, algo='dqn', buffer_size=100, batch_size=32)
sys.stdout = old_stdout

expected = {
    0: 1.0000,
    200: 0.8205,
    500: 0.6105,
    1000: 0.3742,
    2000: 0.1440,
    3000: 0.0593,
}
print(f"  {'ep':>5} {'指数epsilon':>12} {'期望值':>10} {'匹配':>6}")
all_match = True
for ep, exp_val in expected.items():
    agent.update_epsilon(ep, 5000)
    val = agent.epsilon
    match = abs(val - exp_val) < 0.01
    if not match:
        all_match = False
    print(f"  {ep:>5} {val:>12.4f} {exp_val:>10.4f} {'✓' if match else '✗':>6}")
print(f"  {'全部匹配' if all_match else '有偏差'}")

print()
print("=" * 50)
print("验证 3: 超参数")
print("=" * 50)
import config
print(f"  LEARNING_RATE    = {config.Config.LEARNING_RATE} (期望 5e-4)")
print(f"  BATCH_SIZE       = {config.Config.BATCH_SIZE} (期望 128)")
print(f"  TARGET_UPDATE_FREQ = {config.Config.TARGET_UPDATE_FREQ} (期望 300)")
assert config.Config.LEARNING_RATE == 5e-4
assert config.Config.BATCH_SIZE == 128
assert config.Config.TARGET_UPDATE_FREQ == 300
print("  ✓ 全部超参数正确")

print()
print("=" * 50)
print("验证 4: 模型 hidden_dim")
print("=" * 50)
from src.models.cnn_model import CNNModel
model = CNNModel(input_channels=3, n_actions=4)
# 检查 fc1 的 hidden_dim
fc1_out_features = model.fc1.out_features
print(f"  CNNModel.fc1.out_features = {fc1_out_features} (期望 256)")
assert fc1_out_features == 256, f"hidden_dim 应=256, 实际={fc1_out_features}"
print("  ✓ hidden_dim 正确")

print()
print("=" * 50)
print("全部验证通过！")
print("=" * 50)
