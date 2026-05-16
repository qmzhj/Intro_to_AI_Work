"""Visualize training curves
Usage:
  python vis_curve.py                              # Auto-scan all experiments
  python vis_curve.py experiments/xxx              # Specify single experiment
  python vis_curve.py -c exp1 exp2 exp3           # Compare multiple experiments
  python vis_curve.py --compare exp1 exp2 exp3    # Compare multiple experiments
"""

import os, sys, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict


EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')


def parse_training_log(log_path: str):
    """Parse reward and steps data from training_log.txt"""
    episodes = []
    rewards = []
    lengths = []
    
    if not os.path.exists(log_path):
        return None, None
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()  # 一次性读取所有内容
        
        lines = content.split('\n')
        
        for line in lines:
            # Match format: 回合 10/5000 | 奖励: -50.20 | 分数: 0.0 | 步数: 20.9 | Epsilon: 0.998
            match = re.search(r'回合\s+(\d+)/\d+\s*\|\s*奖励:\s*([-\d.]+).*?步数:\s*([\d.]+)', line)
            if match:
                episode = int(match.group(1))
                reward = float(match.group(2))
                steps = float(match.group(3))
                
                # 记录所有回合数据，不限制为10的倍数
                episodes.append(episode)
                rewards.append(reward)
                lengths.append(steps)
                continue
        
        if not episodes:
            print(f"  ⚠ No valid training data found")
            return None, None
        
        rewards_array = np.array(rewards)
        lengths_array = np.array(lengths)
        
        print(f"    Parsed {len(rewards_array)} data points from log")
        
        return rewards_array, lengths_array
        
    except Exception as e:
        print(f"  ⚠ Failed to parse log: {e}")
        return None, None

def algo_from_dirname(name: str) -> str:
    """Extract algorithm name from directory name, e.g., 'dqn_20260511_221723' → 'dqn'"""
    match = re.match(r'^([a-z_]+?)(?:_\d{8}_\d{6})$', name)
    if match:
        raw = match.group(1)
        return raw
    return name.split('_')[0]


def get_exp_name(exp_dir: str) -> str:
    """Get experiment name from full path"""
    return os.path.basename(exp_dir.rstrip('/'))


def load_experiment_data(exp_path: str):
    """Load data for a single experiment"""
    name = get_exp_name(exp_path)
    algo = algo_from_dirname(name)
    
    # First try to load npy files
    reward_file = os.path.join(exp_path, 'episode_rewards.npy')
    length_file = os.path.join(exp_path, 'episode_lengths.npy')
    
    if os.path.isfile(reward_file) and os.path.isfile(length_file):
        try:
            rewards = np.load(reward_file)
            lengths = np.load(length_file)
            print(f"  📊 Loaded from npy: {name}")
            return name, algo, rewards, lengths
        except Exception as e:
            print(f'  ⚠ {name}: Failed to load npy files ({e})')
    
    # Try to parse from log file
    log_file = os.path.join(exp_path, 'training_log.txt')
    if os.path.isfile(log_file):
        print(f'  📄 Parsing from log: {name}')
        rewards, lengths = parse_training_log(log_file)
        if rewards is not None and lengths is not None:
            return name, algo, rewards, lengths
    
    print(f'  ❌ Failed to load experiment data: {name}')
    return None, None, None, None


def smooth(x, w=100):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode='valid')


def plot_individual(exp_path: str, rewards: np.ndarray, lengths: np.ndarray,
                    algo: str, name: str):
    """Generate individual learning curve for a single experiment"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # --- Reward curve ---
    ax = axes[0]
    ax.plot(rewards, alpha=0.15, color='steelblue', lw=0.5, label='Raw data')
    
    mean_reward = rewards.mean()
    ax.axhline(mean_reward, color='red', ls='--', lw=1, alpha=0.7, 
               label=f'Mean: {mean_reward:.2f}')
    
    if len(rewards) > 10:
        smoothed = smooth(rewards, w=min(50, len(rewards)//10))
        x_smoothed = np.arange(len(smoothed)) + (len(rewards) - len(smoothed)) / 2
        ax.plot(x_smoothed, smoothed, color='steelblue', lw=2,
                label=f'Smoothed (window={min(50, len(rewards)//10)})')
    
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel('Average Reward')
    ax.set_xlabel('Episode')
    ax.legend(fontsize=8)
    ax.set_title(f'{algo.upper()} — Training Reward Curve ({name})')
    ax.grid(True, alpha=0.3)
    
    if len(rewards) >= 10:
        last_10_percent = rewards[-max(1, len(rewards)//10):]
        avg_last = last_10_percent.mean()
        ax.text(0.02, 0.05, f'Last {len(last_10_percent)} episodes avg: {avg_last:.2f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Steps curve ---
    ax = axes[1]
    ax.plot(lengths, alpha=0.15, color='forestgreen', lw=0.5, label='Raw data')
    
    mean_length = lengths.mean()
    ax.axhline(mean_length, color='darkorange', ls='--', lw=1, alpha=0.7,
               label=f'Mean: {mean_length:.1f}')
    
    if len(lengths) > 10:
        smoothed_l = smooth(lengths, w=min(50, len(lengths)//10))
        x_smoothed_l = np.arange(len(smoothed_l)) + (len(lengths) - len(smoothed_l)) / 2
        ax.plot(x_smoothed_l, smoothed_l, color='forestgreen', lw=2,
                label=f'Smoothed (window={min(50, len(lengths)//10)})')
    
    ax.set_ylabel('Average Steps')
    ax.set_xlabel('Episode')
    ax.legend(fontsize=8)
    ax.set_title('Episode Length')
    ax.grid(True, alpha=0.3)

    # --- Reward distribution histogram ---
    ax = axes[2]
    n_bins = min(30, len(rewards) // 5)
    if n_bins > 5:
        ax.hist(rewards, bins=n_bins, color='coral', edgecolor='black', alpha=0.7)
        ax.axvline(mean_reward, color='red', ls='--', lw=2, label=f'Mean: {mean_reward:.2f}')
        ax.set_xlabel('Reward Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution Histogram')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough data for histogram',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Reward Distribution (Insufficient Data)')

    plt.tight_layout()
    save_path = os.path.join(exp_path, 'learning_curve3.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  ✓ Saved: {save_path}')


def plot_comparison(experiments_data, output_path=None, title="Training Curves Comparison"):
    """Generate comparison plot for specified experiments"""
    if not experiments_data:
        print('  (No data for comparison)')
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Use different colors and line styles
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    line_styles = ['-', '--', '-.', ':']
    
    for idx, (exp_name, algo, rewards, lengths) in enumerate(experiments_data):
        color = colors[idx % len(colors)]
        line_style = line_styles[(idx // len(colors)) % len(line_styles)]
        
        # Create shorter label
        if len(experiments_data) <= 3:
            label = f"{algo.upper()} ({exp_name})"
        else:
            # For many experiments, use shorter labels
            label = f"{algo.upper()}_{idx+1}"
        
        # Reward curve
        if len(rewards) > 10:
            smoothed_r = smooth(rewards, w=min(50, len(rewards)//10))
            x_smoothed = np.arange(len(smoothed_r)) + (len(rewards) - len(smoothed_r)) / 2
            axes[0].plot(x_smoothed, smoothed_r, color=color, linestyle=line_style, 
                         lw=2, alpha=0.8, label=label)
        else:
            axes[0].plot(rewards, color=color, linestyle=line_style, 
                         lw=2, alpha=0.8, label=label)
        
        # Steps curve
        if len(lengths) > 10:
            smoothed_l = smooth(lengths, w=min(50, len(lengths)//10))
            x_smoothed_l = np.arange(len(smoothed_l)) + (len(lengths) - len(smoothed_l)) / 2
            axes[1].plot(x_smoothed_l, smoothed_l, color=color, linestyle=line_style, 
                         lw=2, alpha=0.8, label=label)
        else:
            axes[1].plot(lengths, color=color, linestyle=line_style, 
                         lw=2, alpha=0.8, label=label)
    
    axes[0].axhline(0, color='gray', ls='--', lw=0.5)
    axes[0].set_ylabel('Average Reward (Smoothed)')
    axes[0].set_xlabel('Episode')
    axes[0].set_title(f'{title} - Reward')
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel('Average Steps (Smoothed)')
    axes[1].set_xlabel('Episode')
    axes[1].set_title(f'{title} - Steps')
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = os.path.join(EXPERIMENTS_DIR, 'comparison3.png')
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # 只有在目录名不为空时才创建目录
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ Saved comparison: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize training curves')
    parser.add_argument('experiments', nargs='*', help='Experiment paths')
    parser.add_argument('--compare', '-c', action='store_true', 
                       help='Comparison mode: compare multiple experiments')
    parser.add_argument('--output', '-o', default='comparison.png',
                       help='Output image path (default: comparison.png)')
    parser.add_argument('--title', '-t', default='Training Curves Comparison',
                       help='Comparison plot title')
    
    args = parser.parse_args()
    
    # If no experiment specified, auto-scan all
    if not args.experiments:
        if args.compare:
            print("Error: Comparison mode requires at least one experiment path")
            return
        
        # Default mode: scan all experiments
        print("Scanning all experiments...")
        all_data = []
        
        if not os.path.isdir(EXPERIMENTS_DIR):
            print(f'Error: Directory does not exist {EXPERIMENTS_DIR}')
            sys.exit(1)
        
        for entry in sorted(os.listdir(EXPERIMENTS_DIR)):
            exp_path = os.path.join(EXPERIMENTS_DIR, entry)
            if not os.path.isdir(exp_path):
                continue
            
            result = load_experiment_data(exp_path)
            if result[0] is not None:  # Successfully loaded
                name, algo, rewards, lengths = result
                all_data.append((name, algo, rewards, lengths))
                
                # Generate individual curve
                print(f'[{algo.upper()}] {name} — {len(rewards)} data points')
                plot_individual(exp_path, rewards, lengths, algo, name)
        
        if all_data:
            print(f'\nGenerating comparison plot ({len(all_data)} experiments)...')
            plot_comparison(all_data, os.path.join(EXPERIMENTS_DIR, args.output), args.title)
        else:
            print('\n⚠ No experiment data found')
        
        print(f'\nDone! Processed {len(all_data)} experiments.')
        return
    
    # Process specified experiments
    experiments_data = []
    
    for exp_arg in args.experiments:
        # Handle relative and absolute paths
        if os.path.isabs(exp_arg):
            exp_path = exp_arg
        else:
            # Try as relative path
            exp_path = exp_arg
            if not os.path.exists(exp_path):
                # Try to find in experiments directory
                exp_path = os.path.join(EXPERIMENTS_DIR, exp_arg)
        
        if not os.path.exists(exp_path):
            print(f"Warning: Experiment path does not exist {exp_arg}")
            continue
        
        result = load_experiment_data(exp_path)
        if result[0] is not None:  # Successfully loaded
            name, algo, rewards, lengths = result
            experiments_data.append((name, algo, rewards, lengths))
            print(f'[{algo.upper()}] {name} — {len(rewards)} data points')
    
    if not experiments_data:
        print("Error: No valid experiment data found")
        return
    
    if len(experiments_data) == 1 and not args.compare:
        # Single experiment mode: generate individual curve only
        name, algo, rewards, lengths = experiments_data[0]
        exp_path = args.experiments[0]
        if not os.path.isabs(exp_path):
            exp_path = os.path.join(EXPERIMENTS_DIR, args.experiments[0])
        plot_individual(exp_path, rewards, lengths, algo, name)
    else:
        # Comparison mode: generate comparison plot
        print(f'\nGenerating comparison plot ({len(experiments_data)} experiments)...')
        
        # Generate title
        if args.title == 'Training Curves Comparison':
            exp_names = [data[0] for data in experiments_data]
            if len(exp_names) <= 3:
                title = f"Comparison: {', '.join(exp_names)}"
            else:
                title = f"Comparison of {len(exp_names)} Experiments"
        else:
            title = args.title
        
        plot_comparison(experiments_data, args.output, title)
        
        # Optional: also generate individual curves
        for (name, algo, rewards, lengths), exp_arg in zip(experiments_data, args.experiments):
            exp_path = exp_arg if os.path.isabs(exp_arg) else os.path.join(EXPERIMENTS_DIR, exp_arg)
            if os.path.exists(exp_path):
                plot_individual(exp_path, rewards, lengths, algo, name)


if __name__ == '__main__':
    main()