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
    GRID_WIDTH = 30
    GRID_HEIGHT = 20
    CELL_SIZE = 20

    # 训练配置
    TOTAL_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 1000
    SAVE_FREQ = 100  # 每N个回合保存一次
    EVAL_FREQ = 50   # 每N个回合评估一次
    EVAL_EPISODES = 10  # 评估回合数

    # DQN配置
    LEARNING_RATE = 1e-4
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 1000

    # 日志配置
    LOG_INTERVAL = 10  # 每N个回合打印一次日志