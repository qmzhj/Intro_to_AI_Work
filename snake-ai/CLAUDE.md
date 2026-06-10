# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# 训练（选择算法）
python train.py --mode train --algo dqn
python train.py --mode train --algo double_dqn
python train.py --mode train --algo dueling_dqn
python train.py --mode train --algo ppo

# 测试训练好的模型（自动识别算法类型）
python train.py --mode test --model experiments/<实验目录>/best_model.pth

# 绘制训练曲线
python vis_curve.py                          # 自动扫描所有实验
python vis_curve.py experiments/<实验目录>    # 指定单个实验

# 测试各组件
python src/envs/snake_env.py                 # 测试游戏环境
python -m src.agents.dqn_agent              # 测试 DQN 系列
python -m src.agents.ppo_agent              # 测试 PPO
python src/models/cnn_model.py              # 测试 CNN 模型
python src/models/dueling_cnn_model.py      # 测试 Dueling CNN
python src/models/ac_shared_cnn_model.py    # 测试 Actor-Critic 共享 CNN

# 可视化回放（训练后逐帧回放，不拖慢训练）
python visualize.py experiments/<实验目录1> experiments/<实验目录2> ...
python visualize.py experiments/dqn_*              # 通配符传入多个
python visualize.py experiments/dqn_* --output demo.mp4  # 导出视频（无头模式，WSLg 兼容）

# 训练时会自动每 RECORD_INTERVAL 回合记录逐帧数据到 experiments/xxx/recordings/

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Project Architecture

基于深度强化学习的贪吃蛇AI项目，使用 **Python + PyTorch + Pygame** 技术栈，实现了 DQN / Double DQN / Dueling DQN / PPO 四种算法。

### 目录结构

```
snake-ai/
├── train.py              # CLI 入口：训练/测试/评估
├── config.py             # 所有超参数集中配置
├── vis_curve.py          # 训练曲线绘制（自动扫描 experiments/）
├── visualize.py          # 训练阶段演进回放（并排多算法对比 + 视频导出）
├── debug_window.py       # Pygame 窗口测试（WSLg 环境诊断）
├── src/
│   ├── envs/
│   │   └── snake_env.py              # 贪吃蛇 Gymnasium 环境
│   ├── agents/
│   │   ├── dqn_agent.py              # DQN 系列智能体
│   │   ├── ppo_agent.py              # PPO 智能体（含 RolloutBuffer）
│   │   └── replay_buffer.py          # 经验回放（普通 + 优先）
│   └── models/
│       ├── cnn_model.py              # 标准 CNN（DQN）
│       ├── dueling_cnn_model.py      # Dueling CNN（价值-优势分离）
│       ├── ac_shared_cnn_model.py    # Actor-Critic 共享 CNN（PPO）
│       └── mlp_model.py              # MLP（备用）
└── experiments/          # 训练结果自动归档（按算法名+时间戳）
    └── {algo}_{timestamp}/
        ├── episode_rewards.npy       # 每回合奖励
        ├── episode_lengths.npy       # 每回合步数
        ├── best_model.pth            # 最佳模型
        ├── final_model.pth           # 最终模型
        ├── training_curve.png        # 训练曲线图
        └── recordings/               # 逐帧录制（可视化回放用）
            └── episode_NNNNNN.npz    # 单局逐帧数据
```

### 核心设计

- **snake_env.py**: 12×12 网格，3 通道图像状态（蛇身/食物/蛇头），动作空间 4 方向。奖励函数由食物奖励(+30)、死亡惩罚(-20)、步惩罚(-0.01)和距离变化奖励组成。
- **dqn_agent.py**: 通过 `algo` 参数在三种变体间切换——dqn（标准）、double_dqn（缓解过估计）、dueling_dqn（V+A 分解）。共享经验回放、目标网络、epsilon 指数衰减。
- **ppo_agent.py**: Actor-Critic 架构，PPO 裁剪目标 + GAE 优势估计 + 熵正则化。On-policy 训练：逐 episode 收集轨迹 → GAE 计算 → 多 epoch 小批量更新。
- **config.py**: 所有超参数统一管理（网格尺寸、学习率、缓冲区大小、PPO 参数等）。
- **visualize.py**: 训练阶段演进回放。录制-回放分离设计："训练时记录逐帧数据 → 训练后可视化渲染"，不影响训练速度。交互窗口模式（空格/加减速/逐帧）和视频导出模式（`--output`，OpenCV + dummy SDL 驱动，兼容 WSLg 无头环境）。
- **录制数据格式** (`.npz`)：每个录制文件包含 `positions`、`food_positions`、`actions`、`rewards`、`scores`、`action_values`（DQN 的 Q 值 / PPO 的动作概率）、`dones`。录制频率由 `config.py` 中的 `RECORD_INTERVAL` 控制。

### 数据流

- **DQN 系列**: 每一步 `select_action` → `env.step` → `replay_buffer.push` → `train_step`（采样→计算Q值→优化→更新目标网络）
- **PPO**: 每步 `get_action_info` → `env.step` → `store_transition` → episode结束时 `end_episode`(GAE) → 积累 N 个 episode 后多 epoch 更新

### 实验结果

实验按 `{algo}_{timestamp}` 命名保存在 `experiments/` 中，每个实验包含：`episode_rewards.npy`、`episode_lengths.npy`、`best_model.pth`、`final_model.pth`、检查点文件和学习曲线图。`.gitignore` 会忽略 `experiments/` 和 `*.pth` 文件。
