"""绘制训练曲线
自动扫描 experiments/ 目录下所有有完整数据的实验，生成：
  1. 每个实验的独立学习曲线 → {实验目录}/learning_curve.png
  2. 全部实验的对比图 → experiments/comparison.png

用法:
  python3.10 vis_curve.py                     # 自动扫描所有实验
  python3.10 vis_curve.py experiments/xxx     # 指定单个实验
"""

import os, sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')


def algo_from_dirname(name: str) -> str:
    """从目录名提取算法名称，如 'dqn_20260511_221723' → 'dqn'"""
    match = re.match(r'^([a-z_]+?)(?:_\d{8}_\d{6})$', name)
    if match:
        raw = match.group(1)
        # 'double_dqn' / 'dueling_dqn' 保持原样
        return raw
    # 退化：取第一个下划线之前的部分
    return name.split('_')[0]


def smooth(x, w=100):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_individual(exp_path: str, rewards: np.ndarray, lengths: np.ndarray,
                    algo: str, name: str):
    """为单个实验生成独立学习曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # --- 奖励曲线 ---
    ax = axes[0]
    ax.plot(rewards, alpha=0.15, color='steelblue', lw=0.5, label='raw')
    smoothed = smooth(rewards)
    ax.plot(smoothed, color='steelblue', lw=2,
            label=f'smoothed(n={100})' if len(smoothed) < len(rewards) else 'raw')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel('Reward')
    ax.legend(fontsize=8)
    ax.set_title(f'{algo} — Training Reward ({name})')

    # --- 步数曲线 ---
    ax = axes[1]
    ax.plot(lengths, alpha=0.15, color='forestgreen', lw=0.5, label='raw')
    smoothed_l = smooth(lengths)
    ax.plot(smoothed_l, color='forestgreen', lw=2,
            label=f'smoothed(n={100})' if len(smoothed_l) < len(lengths) else 'raw')
    ax.set_ylabel('Steps')
    ax.legend(fontsize=8)
    ax.set_title('Episode Length')

    # --- 滚动平均奖励 ---
    ax = axes[2]
    window = min(200, len(rewards) // 5) if len(rewards) > 100 else 20
    rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
    ax.plot(rolling, color='coral', lw=2)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Avg Reward')
    ax.set_title(f'Rolling Average Reward (window={window})')

    plt.tight_layout()
    save_path = os.path.join(exp_path, 'learning_curve.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {save_path}')


def plot_comparison(all_data: dict):
    """生成所有实验的对比图"""
    if not all_data:
        print('  (无数据可对比)')
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (algo, runs) in enumerate(sorted(all_data.items())):
        color = color_cycle[idx % len(color_cycle)]
        for run_name, rewards, lengths in runs:
            smoothed_r = smooth(rewards)
            # 奖励对比
            axes[0].plot(smoothed_r, color=color, lw=1.5, alpha=0.8,
                         label=f'{algo} ({len(rewards)} ep)')
            # 长度对比
            smoothed_l = smooth(lengths)
            axes[1].plot(smoothed_l, color=color, lw=1.5, alpha=0.8,
                         label=f'{algo} ({len(lengths)} ep)')

    axes[0].axhline(0, color='gray', ls='--', lw=0.5)
    axes[0].set_ylabel('Reward (smoothed)')
    axes[0].set_title('Training Reward — 算法对比')
    axes[0].legend(fontsize=8, loc='lower right')

    axes[1].set_ylabel('Steps (smoothed)')
    axes[1].set_xlabel('Episode')
    axes[1].set_title('Episode Length — 算法对比')
    axes[1].legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(EXPERIMENTS_DIR, 'comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  ✓ {save_path}')


def main():
    # 支持手动指定单个实验目录（兼容旧用法）
    if len(sys.argv) > 1:
        exp_path = sys.argv[1]
        name = os.path.basename(exp_path)
        algo = algo_from_dirname(name)

        rewards = np.load(os.path.join(exp_path, 'episode_rewards.npy'))
        lengths = np.load(os.path.join(exp_path, 'episode_lengths.npy'))
        print(f'[{algo}] {name} — {len(rewards)} episodes')
        plot_individual(exp_path, rewards, lengths, algo, name)
        return

    # 自动扫描 experiments/
    if not os.path.isdir(EXPERIMENTS_DIR):
        print(f'错误: 目录不存在 {EXPERIMENTS_DIR}')
        sys.exit(1)

    all_data = defaultdict(list)  # algo → [(run_name, rewards, lengths), ...]
    found = 0

    for entry in sorted(os.listdir(EXPERIMENTS_DIR)):
        exp_path = os.path.join(EXPERIMENTS_DIR, entry)
        if not os.path.isdir(exp_path):
            continue

        reward_file = os.path.join(exp_path, 'episode_rewards.npy')
        length_file = os.path.join(exp_path, 'episode_lengths.npy')
        if not (os.path.isfile(reward_file) and os.path.isfile(length_file)):
            continue

        try:
            rewards = np.load(reward_file)
            lengths = np.load(length_file)
        except Exception as e:
            print(f'  ⚠ {entry}: 加载失败 ({e})')
            continue

        algo = algo_from_dirname(entry)
        print(f'[{algo}] {entry} — {len(rewards)} episodes')

        # 生成个体曲线
        plot_individual(exp_path, rewards, lengths, algo, entry)

        all_data[algo].append((entry, rewards, lengths))
        found += 1

    # 生成对比图
    print(f'\n生成算法对比图 ({found} 个实验)...')
    plot_comparison(all_data)

    print(f'\n完成! 共处理 {found} 个实验.')


if __name__ == '__main__':
    main()
