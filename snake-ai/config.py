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
    TOTAL_EPISODES = 5000       # 每轮更快，总轮数可适当减少
    MAX_STEPS_PER_EPISODE = 200  # 12x12网格，200步足够找到食物
    SAVE_FREQ = 200             # 每N个回合保存一次
    EVAL_FREQ = 100             # 每N个回合评估一次
    EVAL_EPISODES = 10          # 评估回合数

    # 算法选择（仅做默认值，实际由命令行 --algo 覆盖）
    ALGO = "dqn"  # 可选: dqn / double_dqn / dueling_dqn

    # DQN配置
    LEARNING_RATE = 3e-4        # 略微提高，加快收敛
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    # epsilon_decay 改为线性衰减（由 update_epsilon 控制），此参数不再使用
    EPSILON_DECAY = 0.995       # 保留但不再用于 train_step 内乘性衰减
    BUFFER_SIZE = 50000         # 缩小缓冲区，节省内存
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 500    # 目标网络更新频率（步数）

    # 日志配置
    LOG_INTERVAL = 10           # 每N个回合打印一次日志
