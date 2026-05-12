# 贪吃蛇AI项目

基于深度强化学习（DQN）的贪吃蛇AI实现，使用 Python + PyTorch + Pygame 技术栈。

## 项目概述

本项目旨在实现一个能够自主学习并玩好贪吃蛇游戏的AI智能体。通过深度强化学习算法（DQN、Double DQN、Dueling DQN），智能体能够通过与环境交互，不断优化自己的策略，最终达到较高的游戏水平。

项目参考了经典的 DQN 算法以及其改进变体，并结合贪吃蛇游戏的特点进行了优化设计。

## 项目结构

```
snake-ai/
├── src/                      # 源代码目录
│   ├── __init__.py           # 包初始化文件
│   ├── envs/                 # 游戏环境模块
│   │   └── snake_env.py      # 贪吃蛇环境实现（基于原有代码重构）
│   ├── agents/               # 智能体模块
│   │   ├── __init__.py
│   │   ├── dqn_agent.py      # DQN/Double DQN/Dueling DQN 智能体
│   │   └── replay_buffer.py  # 经验回放缓冲区
│   ├── models/               # 神经网络模型模块
│   │   ├── __init__.py
│   │   ├── cnn_model.py          # 标准 CNN 模型（处理图像输入）
│   │   ├── dueling_cnn_model.py  # Dueling CNN 模型（价值-优势分离）
│   │   └── mlp_model.py          # MLP模型（处理向量输入）
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

### 智能体 (dqn_agent.py) — 三种算法变体
智能体通过 `algo` 参数在三种算法间切换：

- **标准 DQN** (`algo="dqn"`)：经典深度 Q 网络，目标网络直接取 `max Q'` 计算目标值
- **Double DQN** (`algo="double_dqn"`)：用策略网络选择最优动作、目标网络评估该动作的值，有效缓解 Q 值过高估计问题
- **Dueling DQN** (`algo="dueling_dqn"`)：将 Q 值分解为状态价值 V(s) 和动作优势 A(s,a)，架构层面减少值函数估计方差；目标值计算默认采用 double 方式

所有变体共享以下机制：
- 经验回放，提高样本利用效率
- 目标网络，稳定训练过程
- Epsilon-Greedy 探索策略，平衡探索与利用
- 支持模型保存和加载

### 神经网络 (models/)
- **CNN 模型**: 标准卷积神经网络，处理图像输入，提取空间特征
- **Dueling CNN 模型**: 价值-优势分离架构，分别输出 V(s) 和 A(s,a)，组合为 Q(s,a)
- **MLP 模型**: 全连接网络，处理向量输入，适合简单状态表示
- 模块化设计，可灵活切换和扩展

### 训练系统 (train.py)
- 完整的训练循环，包含数据收集、训练和评估
- 通过 `--algo` 参数选择算法变体
- 定期模型评估和自动保存最佳模型
- 详细的训练日志记录
- 支持训练和测试两种模式
- 进度条显示，实时监控训练状态

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
- stable-baselines3: 强化学习框架（可选）
- tensorboard: 训练可视化

## 使用

### 训练模型（选择算法）

```bash
# 标准 DQN（默认）
python train.py --mode train --algo dqn

# Double DQN
python train.py --mode train --algo double_dqn

# Dueling DQN
python train.py --mode train --algo dueling_dqn
```

训练过程会：
- 在 `experiments/` 目录下创建以算法名命名的实验文件夹（如 `double_dqn_20260512_153000`）
- 自动保存最佳模型和定期检查点
- 记录训练日志到文件
- 显示实时训练进度

### 测试训练好的模型

```bash
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
- `ALGO`: 默认算法（`"dqn"` / `"double_dqn"` / `"dueling_dqn"`，可由命令行 `--algo` 覆盖）

## 快速开始

### 1. 测试游戏环境

```bash
python src/envs/snake_env.py
```

这会启动一个简单的测试，验证游戏环境是否正常工作。

### 2. 测试神经网络

```bash
python src/models/cnn_model.py       # 标准 CNN
python src/models/dueling_cnn_model.py  # Dueling CNN
```

测试模型的前向传播和参数统计。

### 3. 测试智能体

```bash
python -m src.agents.dqn_agent
```

会依次测试 dqn / double_dqn / dueling_dqn 三种变体的初始化和训练。

### 4. 开始训练

```bash
python train.py --mode train --algo double_dqn
```

开始训练你的贪吃蛇 AI！

## 技术栈

- **Python 3.9+**: 编程语言
- **PyTorch**: 深度学习框架
- **Pygame**: 游戏渲染和交互
- **NumPy**: 数值计算
- **Gymnasium**: 强化学习环境接口

## 算法

本项目实现了以下强化学习算法和技巧：

### DQN (Deep Q-Network)
基础算法。使用深度神经网络近似 Q 函数，目标值 `target = r + γ·max Q_target(s', a')`。

### Double DQN
解决标准 DQN 中 Q 值过高估计问题。策略网络负责选择动作，目标网络负责评估：
`target = r + γ·Q_target(s', argmax Q_policy(s', a'))`。

### Dueling DQN
将 Q 网络分为两路：状态价值 V(s) 和动作优势 A(s,a)。组合为 `Q(s,a) = V(s) + A(s,a) - mean(A(s,a))`，在动作价值相近时能更准确估计状态价值。目标值计算默认使用 double 方式。

### 通用技巧
- **经验回放 (Experience Replay)**：存储过往经验，随机采样训练，打破相关性
- **目标网络 (Target Network)**：周期性更新的稳定目标，减少训练震荡
- **Epsilon-Greedy 探索**：线性衰减，从高探索逐步过渡到高利用

## 项目进度

- [x] 项目框架搭建
- [x] 游戏环境实现
- [x] DQN 算法实现
- [x] Double DQN 算法实现
- [x] Dueling DQN 算法实现
- [x] 神经网络模型实现（CNN + Dueling CNN）
- [x] 训练系统实现（支持三种算法参数）
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
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [PyTorch Documentation](https://pytorch.org/docs/)
