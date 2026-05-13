"""
项目配置文件
"""

import os


class Config:
    """项目配置"""

    # 项目路径
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, 'experiments')

    # 确保目录存在
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # 游戏环境配置
    # 缩小网格（12x12=144格），大幅减少CNN输入尺寸和探索空间
    GRID_WIDTH = 12
    GRID_HEIGHT = 12
    CELL_SIZE = 20

    # 训练配置
    TOTAL_EPISODES = 10000       # 每轮更快，总轮数可适当减少
    MAX_STEPS_PER_EPISODE = 200  # 12x12网格，200步足够找到食物
    SAVE_FREQ = 200             # 每N个回合保存一次
    EVAL_FREQ = 100             # 每N个回合评估一次
    EVAL_EPISODES = 10          # 评估回合数

    # 算法选择（仅做默认值，实际由命令行 --algo 覆盖）
    ALGO = "dqn"  # 可选: dqn / double_dqn / dueling_dqn

    # DQN配置
    LEARNING_RATE = 5e-4        # 略微提高学习率加速收敛
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995       # 保留但不再使用（由指数衰减替代）
    BUFFER_SIZE = 50000
    BATCH_SIZE = 128            # 增大批次大小，梯度更稳定
    TARGET_UPDATE_FREQ = 300    # 降低目标网络更新间隔（每 300 步更新）

    # PPO 配置
    PPO_LEARNING_RATE = 3e-4       # PPO 学习率
    GAE_LAMBDA = 0.95              # GAE 参数 λ
    PPO_CLIP_EPSILON = 0.2         # PPO 裁剪参数 ε
    PPO_ENTROPY_COEF = 0.01        # 熵正则化系数
    PPO_VALUE_COEF = 0.5           # 价值损失系数
    PPO_EPOCHS = 4                 # 每次更新对数据做几轮优化
    PPO_MINI_BATCH_SIZE = 64       # 小批量大小
    PPO_UPDATE_FREQ = 5            # 每 N 个 episode 更新一次
    PPO_MAX_GRAD_NORM = 0.5        # 梯度裁剪最大范数

    # 日志配置
    LOG_INTERVAL = 10           # 每N个回合打印一次日志
