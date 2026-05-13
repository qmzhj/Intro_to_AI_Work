# 贪吃蛇AI项目

基于深度强化学习的贪吃蛇AI实现，使用 Python + PyTorch + Pygame 技术栈。

## 项目概述

本项目旨在实现一个能够自主学习并玩好贪吃蛇游戏的AI智能体。通过深度强化学习算法（DQN、Double DQN、Dueling DQN、PPO），智能体能够通过与环境交互，不断优化自己的策略，最终达到较高的游戏水平。

项目参考了经典的 DQN 算法及其改进变体，并实现了 PPO（Proximal Policy Optimization）算法，提供了多种算法的对比实验框架。

## 项目结构

```
snake-ai/
├── src/                      # 源代码目录
│   ├── __init__.py           # 包初始化文件
│   ├── envs/                 # 游戏环境模块
│   │   └── snake_env.py      # 贪吃蛇环境实现
│   ├── agents/               # 智能体模块
│   │   ├── __init__.py
│   │   ├── dqn_agent.py      # DQN/Double DQN/Dueling DQN 智能体
│   │   ├── ppo_agent.py      # PPO 智能体（Actor-Critic）
│   │   └── replay_buffer.py  # 经验回放缓冲区
│   ├── models/               # 神经网络模型模块
│   │   ├── __init__.py
│   │   ├── cnn_model.py          # 标准 CNN 模型（处理图像输入）
│   │   ├── dueling_cnn_model.py  # Dueling CNN 模型（价值-优势分离）
│   │   ├── ac_shared_cnn_model.py # Actor-Critic 共享 CNN 模型
│   │   └── mlp_model.py          # MLP 模型（处理向量输入）
│   └── utils/                # 工具函数模块
│       └── __init__.py
├── checkpoints/              # 模型检查点目录
├── experiments/              # 实验结果目录（自动按算法名归档）
├── logs/                     # 训练日志目录
├── visualization/            # 可视化目录
├── training/                 # 训练相关目录
├── config.py                 # 配置文件
├── train.py                  # 训练脚本
├── requirements.txt          # 依赖包列表
├── README.md                 # 项目说明文档
└── .gitignore                # Git忽略文件配置
```

## 主要特性

### 游戏环境 (snake_env.py)
- 基于原有 `snake_game.py` 重构，保持游戏逻辑一致
- 遵循 Gymnasium 接口规范，便于与其他强化学习框架集成
- 支持图像状态表示（3通道：蛇身、食物、蛇头）
- 实现了适合 AI 训练的奖励函数设计
- 可配置的游戏参数（网格大小、奖励值等）

### DQN 系列智能体 (dqn_agent.py) — 三种变体
智能体通过 `algo` 参数在三种算法间切换：

- **标准 DQN** (`algo="dqn"`)：经典深度 Q 网络，目标网络直接取 `max Q'` 计算目标值
- **Double DQN** (`algo="double_dqn"`)：用策略网络选择最优动作、目标网络评估该动作的值，有效缓解 Q 值过高估计问题
- **Dueling DQN** (`algo="dueling_dqn"`)：将 Q 值分解为状态价值 V(s) 和动作优势 A(s,a)，架构层面减少值函数估计方差；目标值计算默认采用 double 方式

所有变体共享以下机制：经验回放、目标网络、Epsilon-Greedy 探索策略、模型保存/加载。

### PPO 智能体 (ppo_agent.py)
基于 **Actor-Critic 架构** 的 on-policy 强化学习算法：

- **Actor 网络**：共享 CNN 骨干 + Softmax 输出层，输出 4 个动作的概率分布
- **Critic 网络**：共享 CNN 骨干 + 标量输出，估计状态价值 V(s)
- **PPO 裁剪目标**：`L^CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)`，ε=0.2
- **广义优势估计 GAE**：`A_t = δ_t + (γλ)·A_{t+1}`，λ=0.95
- **熵正则化**：最大化策略熵（系数 0.01），鼓励探索
- **多 epoch 小批量更新**：每次收集 N 个 episode 后做 K 轮优化

### 神经网络 (models/)
- **CNN 模型**: 标准卷积神经网络，处理图像输入，提取空间特征
- **Dueling CNN 模型**: 价值-优势分离架构，分别输出 V(s) 和 A(s,a)，组合为 Q(s,a)
- **Actor-Critic 共享 CNN 模型**: 共享卷积骨干，Actor 头输出动作概率，Critic 头输出状态价值
- **MLP 模型**: 全连接网络，处理向量输入
- 模块化设计，可灵活切换和扩展

### 训练系统 (train.py)
- 通过 `--algo` 参数选择算法：`dqn` / `double_dqn` / `dueling_dqn` / `ppo`
- DQN 系列：off-policy 训练，逐步收集/训练
- PPO：on-policy 训练，逐 episode 收集轨迹 → GAE 计算 → 多 epoch 更新
- 定期模型评估和自动保存最佳模型
- 详细的训练日志记录（包含策略损失、价值损失、熵等）
- 支持训练和测试两种模式
- 测试时自动识别 checkpoint 中的算法类型

## 安装

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

主要依赖包括：
- torch: 深度学习框架
- pygame: 游戏渲染
- numpy: 数值计算

## 使用

### 训练模型（选择算法）

```bash
# 标准 DQN（默认）
python train.py --mode train --algo dqn

# Double DQN
python train.py --mode train --algo double_dqn

# Dueling DQN
python train.py --mode train --algo dueling_dqn

# PPO（Actor-Critic）
python train.py --mode train --algo ppo
```

训练过程会：
- 在 `experiments/` 目录下创建以算法名命名的实验文件夹（如 `ppo_20260512_153000`）
- 自动保存最佳模型和定期检查点
- 记录训练日志到文件
- 显示实时训练进度

### 测试训练好的模型

```bash
python train.py --mode test --model experiments/ppo_xxxxxx/best_model.pth
python train.py --mode test --model experiments/double_dqn_xxxxxx/best_model.pth
```

测试模式会：
- 自动识别模型训练时使用的算法（checkpoint 中记录 `algo` 字段）
- 以可视化方式展示游戏过程
- 显示游戏得分和步数

## 配置

所有训练参数都在 `config.py` 中配置，可以根据需要调整：

### 环境配置
- `GRID_WIDTH`: 网格宽度（默认 12）
- `GRID_HEIGHT`: 网格高度（默认 12）
- `CELL_SIZE`: 单元格大小（默认 20）

### 训练配置
- `TOTAL_EPISODES`: 总训练回合数（默认 5000）
- `MAX_STEPS_PER_EPISODE`: 每回合最大步数（默认 200）
- `SAVE_FREQ`: 保存频率（每 N 个回合保存，默认 200）
- `EVAL_FREQ`: 评估频率（每 N 个回合评估，默认 100）
- `EVAL_EPISODES`: 评估回合数（默认 10）

### DQN 配置
- `LEARNING_RATE`: 学习率（默认 3e-4）
- `GAMMA`: 折扣因子（默认 0.99）
- `EPSILON_START`: 初始探索率（默认 1.0）
- `EPSILON_END`: 最终探索率（默认 0.01）
- `BUFFER_SIZE`: 经验回放缓冲区大小（默认 50000）
- `BATCH_SIZE`: 批次大小（默认 64）
- `TARGET_UPDATE_FREQ`: 目标网络更新频率（默认 500 步）
- `ALGO`: 默认算法（可由命令行 `--algo` 覆盖）

### PPO 配置
- `PPO_LEARNING_RATE`: 学习率（默认 3e-4）
- `GAE_LAMBDA`: GAE 参数 λ（默认 0.95）
- `PPO_CLIP_EPSILON`: PPO 裁剪参数 ε（默认 0.2）
- `PPO_ENTROPY_COEF`: 熵正则化系数（默认 0.01）
- `PPO_VALUE_COEF`: 价值损失系数（默认 0.5）
- `PPO_EPOCHS`: 每次更新对数据做几轮优化（默认 4）
- `PPO_MINI_BATCH_SIZE`: 小批量大小（默认 64）
- `PPO_UPDATE_FREQ`: 每 N 个 episode 更新一次（默认 5）
- `PPO_MAX_GRAD_NORM`: 梯度裁剪最大范数（默认 0.5）

## 快速开始

### 1. 测试游戏环境

```bash
python src/envs/snake_env.py
```

这会启动一个简单的测试，验证游戏环境是否正常工作。

### 2. 测试神经网络

```bash
python src/models/cnn_model.py              # 标准 CNN
python src/models/dueling_cnn_model.py      # Dueling CNN
python src/models/ac_shared_cnn_model.py    # Actor-Critic 共享 CNN
```

测试模型的前向传播和参数统计。

### 3. 测试 DQN 系列智能体

```bash
python -m src.agents.dqn_agent
```

会依次测试 dqn / double_dqn / dueling_dqn 三种变体的初始化和训练。

### 4. 测试 PPO 智能体

```bash
python -m src.agents.ppo_agent
```

会测试 PPO 智能体的初始化和训练流程。

### 5. 开始训练

```bash
python train.py --mode train --algo ppo
```

开始训练你的贪吃蛇 AI！

## 技术栈

- **Python 3.9+**: 编程语言
- **PyTorch**: 深度学习框架
- **Pygame**: 游戏渲染和交互
- **NumPy**: 数值计算
- **Gymnasium**: 强化学习环境接口

## 算法

### DQN (Deep Q-Network)
基础 off-policy 算法。使用深度神经网络近似 Q 函数，目标值 `target = r + γ·max Q_target(s', a')`。通过经验回放和目标网络稳定训练。

### Double DQN
解决标准 DQN 中 Q 值过高估计问题。策略网络负责选择动作，目标网络负责评估：
`target = r + γ·Q_target(s', argmax Q_policy(s', a'))`。

### Dueling DQN
将 Q 网络分为两路：状态价值 V(s) 和动作优势 A(s,a)。组合为 `Q(s,a) = V(s) + A(s,a) - mean(A(s,a))`，在动作价值相近时能更准确估计状态价值。目标值计算默认使用 double 方式。

### PPO (Proximal Policy Optimization)
基于 Actor-Critic 架构的 on-policy 算法。核心损失函数为裁剪后的替代目标：

**PPO 裁剪损失**（Actor）：
```
L^CLIP = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]
```
其中 `r(θ) = π_θ(a|s) / π_old(a|s)` 为重要性采样比率，ε=0.2 为裁剪范围。该目标限制策略更新的步长，防止单次更新过大导致性能坍塌。

**价值损失**（Critic）：
```
L^VF = MSE(V(s), Return)
```

**广义优势估计 GAE**：
```
δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
A_t = δ_t + (γλ)·A_{t+1}                  # λ=0.95
```

### 通用技巧
- **经验回放 (Experience Replay)**：DQN 系列使用，打破样本相关性
- **目标网络 (Target Network)**：DQN 系列使用，周期性更新稳定目标
- **Epsilon-Greedy 探索**：DQN 系列线性衰减
- **熵正则化**：PPO 使用，鼓励策略保持随机性
- **GAE 优势估计**：PPO 使用，平衡偏差与方差

## 项目进度

- [x] 项目框架搭建
- [x] 游戏环境实现
- [x] DQN 算法实现
- [x] Double DQN 算法实现
- [x] Dueling DQN 算法实现
- [x] PPO 算法实现（Actor-Critic + GAE + 裁剪目标）
- [x] 神经网络模型实现（CNN + Dueling CNN + Actor-Critic CNN）
- [x] 训练系统实现（支持四种算法参数）
- [ ] 训练和参数调优
- [ ] 可视化系统完善
- [ ] 实验分析和对比
- [ ] 文档完善

## 作者

- **陈琳**: 算法实现、深度学习
- **曹云龙**: 环境开发、可视化

## 许可

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 参考

- [Deep Q-Learning with Atari Games](https://www.nature.com/articles/nature14236)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [PyTorch Documentation](https://pytorch.org/docs/)
