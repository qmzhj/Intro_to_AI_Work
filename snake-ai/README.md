# 贪吃蛇AI项目

基于深度强化学习（DQN）的贪吃蛇AI实现，使用 Python + PyTorch + Pygame 技术栈。

## 项目概述

本项目旨在实现一个能够自主学习并玩好贪吃蛇游戏的AI智能体。通过深度强化学习算法（DQN），智能体能够通过与环境交互，不断优化自己的策略，最终达到较高的游戏水平。

项目参考了经典的 DQN 算法，并结合贪吃蛇游戏的特点进行了优化设计。

## 项目结构

```
snake-ai/
├── src/                      # 源代码目录
│   ├── __init__.py           # 包初始化文件
│   ├── envs/                 # 游戏环境模块
│   │   └── snake_env.py      # 贪吃蛇环境实现（基于原有代码重构）
│   ├── agents/               # 智能体模块
│   │   ├── __init__.py
│   │   ├── dqn_agent.py      # DQN智能体实现
│   │   └── replay_buffer.py  # 经验回放缓冲区
│   ├── models/               # 神经网络模型模块
│   │   ├── __init__.py
│   │   ├── cnn_model.py      # CNN模型（处理图像输入）
│   │   └── mlp_model.py      # MLP模型（处理向量输入）
│   └── utils/                # 工具函数模块
│       └── __init__.py
├── checkpoints/              # 模型检查点目录
├── experiments/              # 实验结果目录
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

### DQN 算法 (dqn_agent.py)
- 完整的 DQN 实现，包含核心算法逻辑
- 经验回放机制，提高样本利用效率
- 目标网络更新，稳定训练过程
- Epsilon-Greedy 探索策略，平衡探索与利用
- 支持模型保存和加载

### 神经网络 (models/)
- CNN 模型：专门处理图像输入，提取空间特征
- MLP 模型：处理向量输入，适合简单状态表示
- 模块化设计，可灵活切换和扩展

### 训练系统 (train.py)
- 完整的训练循环，包含数据收集、训练和评估
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

### 训练模型

```bash
python train.py --mode train
```

训练过程会：
- 在 `experiments/` 目录下创建实验文件夹
- 自动保存最佳模型和定期检查点
- 记录训练日志到文件
- 显示实时训练进度

### 测试训练好的模型

```bash
python train.py --mode test --model experiments/dqn_xxxxxx/best_model.pth
```

测试模式会：
- 加载指定的模型文件
- 以可视化方式展示游戏过程
- 显示游戏得分和步数

## 配置

所有训练参数都在 `config.py` 中配置，可以根据需要调整：

### 环境配置
- `GRID_WIDTH`: 网格宽度（默认 30）
- `GRID_HEIGHT`: 网格高度（默认 20）
- `CELL_SIZE`: 单元格大小（默认 20）

### 训练配置
- `TOTAL_EPISODES`: 总训练回合数（默认 10000）
- `MAX_STEPS_PER_EPISODE`: 每回合最大步数（默认 1000）
- `SAVE_FREQ`: 保存频率（每 N 个回合保存）
- `EVAL_FREQ`: 评估频率（每 N 个回合评估）
- `EVAL_EPISODES`: 评估回合数（默认 10）

### DQN 配置
- `LEARNING_RATE`: 学习率（默认 1e-4）
- `GAMMA`: 折扣因子（默认 0.99）
- `EPSILON_START`: 初始探索率（默认 1.0）
- `EPSILON_END`: 最终探索率（默认 0.01）
- `EPSILON_DECAY`: 探索率衰减（默认 0.995）
- `BUFFER_SIZE`: 经验回放缓冲区大小（默认 100000）
- `BATCH_SIZE`: 批次大小（默认 64）
- `TARGET_UPDATE_FREQ`: 目标网络更新频率（默认 1000）

## 快速开始

### 1. 测试游戏环境

```bash
python src/envs/snake_env.py
```

这会启动一个简单的测试，验证游戏环境是否正常工作。

### 2. 测试神经网络

```bash
python src/models/cnn_model.py
```

这会测试 CNN 模型的前向传播和参数统计。

### 3. 测试智能体

```bash
python src/agents/dqn_agent.py
```

这会测试 DQN 智能体的基本功能。

### 4. 开始训练

```bash
python train.py --mode train
```

开始训练你的贪吃蛇 AI！

## 技术栈

- **Python 3.9+**: 编程语言
- **PyTorch**: 深度学习框架
- **Pygame**: 游戏渲染和交互
- **NumPy**: 数值计算
- **Gymnasium**: 强化学习环境接口
- **TensorBoard**: 训练可视化（可选）

## 算法

本项目实现了以下强化学习算法和技巧：

- **DQN (Deep Q-Network)**: 深度 Q 网络
- **经验回放 (Experience Replay)**: 打破样本相关性
- **目标网络 (Target Network)**: 稳定训练过程
- **Epsilon-Greedy 探索**: 平衡探索与利用
- **优先经验回放 (PER)**: 提高样本效率（可选）

## 项目进度

- [x] 项目框架搭建
- [x] 游戏环境实现
- [x] DQN 算法实现
- [x] 神经网络模型实现
- [x] 训练系统实现
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
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [PyTorch Documentation](https://pytorch.org/docs/)