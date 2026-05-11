"""绘制训练曲线"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

exp = sys.argv[1] if len(sys.argv) > 1 else 'experiments/dqn_20260511_221723'

rewards = np.load(f'{exp}/episode_rewards.npy')
lengths = np.load(f'{exp}/episode_lengths.npy')

def smooth(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

axes[0].plot(rewards, alpha=0.2, color='steelblue', label='raw')
axes[0].plot(smooth(rewards), color='steelblue', lw=2, label=f'smoothed({100})')
axes[0].axhline(0, color='gray', ls='--')
axes[0].set_ylabel('Reward')
axes[0].legend(fontsize=9)
axes[0].set_title('Training Reward')

axes[1].plot(lengths, alpha=0.2, color='forestgreen', label='raw')
axes[1].plot(smooth(lengths), color='forestgreen', lw=2, label=f'smoothed({100})')
axes[1].set_ylabel('Steps')
axes[1].legend(fontsize=9)
axes[1].set_title('Episode Length')

# 滚动平均值
window = 200
rolling_reward = np.convolve(rewards, np.ones(window)/window, mode='valid')
axes[2].plot(rolling_reward, color='coral', lw=2)
axes[2].axhline(0, color='gray', ls='--')
axes[2].set_xlabel('Episode')
axes[2].set_ylabel(f'Avg Reward ({window}-ep)')
axes[2].set_title(f'Rolling Average Reward (window={window})')

plt.tight_layout()
plt.savefig(f'{exp}/learning_curve.png', dpi=150)
print(f'图表已保存: {exp}/learning_curve.png')
