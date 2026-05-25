"""
可视化回放程序
读取训练时录制的逐帧数据，同步回放多个模型的游戏过程 + 训练曲线对比

用法:
  python visualize.py experiments/dqn_20260510_132448 experiments/double_dqn_20260512_231315 ...
"""

import os
import sys
import re
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import pygame


# ── 常量 ──
GRID_W = 12
GRID_H = 12

BG_COLOR = (18, 18, 28)
PANEL_BG = (26, 26, 38)
CONTROL_BG = (36, 36, 50)
TEXT_COLOR = (210, 210, 210)
DIM_TEXT = (140, 140, 140)

ALGO_COLORS_HEX = ['#1E88E5', '#43A047', '#F4511E', '#8E24AA',
                   '#00ACC1', '#FFB300', '#E53935', '#5E35B1']


def hex_to_rgb(h):
    return [int(h[i:i+2], 16) for i in (1, 3, 5)]


# ── 数据加载 ──

def load_experiment(exp_dir):
    """加载一个实验的录制数据和训练曲线"""
    rewards_path = os.path.join(exp_dir, 'episode_rewards.npy')
    lengths_path = os.path.join(exp_dir, 'episode_lengths.npy')
    if not os.path.isfile(rewards_path):
        print(f'  ⚠ 未找到 {rewards_path}')
        return None
    rewards = np.load(rewards_path)
    lengths = np.load(lengths_path) if os.path.isfile(lengths_path) else None

    rec_dir = os.path.join(exp_dir, 'recordings')
    if not os.path.isdir(rec_dir):
        print(f'  ⚠ 未找到录制目录 {rec_dir}')
        return None
    rec_files = sorted(f for f in os.listdir(rec_dir) if f.endswith('.npz'))
    if not rec_files:
        print(f'  ⚠ {rec_dir} 中没有录制文件')
        return None
    latest_rec = os.path.join(rec_dir, rec_files[-1])
    rec_data = np.load(latest_rec)

    # 从目录名提取算法名
    dirname = os.path.basename(exp_dir)
    match = re.match(r'^([a-z_]+?)(?:_\d{8}_\d{6})$', dirname)
    algo = match.group(1) if match else dirname

    return {
        'algo': algo,
        'name': dirname,
        'rewards': rewards,
        'lengths': lengths,
        'recording': rec_data,
        'num_steps': len(rec_data['rewards']),
        'exp_dir': exp_dir,
    }


# ── 游戏画面渲染 ──

def render_game_surface(step_data, cell_size):
    """根据一步的录制数据渲染游戏画面"""
    grid_w, grid_h = GRID_W, GRID_H
    surf_w = grid_w * cell_size
    surf_h = grid_h * cell_size
    surf = pygame.Surface((surf_w, surf_h))
    surf.fill((0, 0, 0))

    # 网格线
    for x in range(0, surf_w, cell_size):
        pygame.draw.line(surf, (40, 40, 50), (x, 0), (x, surf_h))
    for y in range(0, surf_h, cell_size):
        pygame.draw.line(surf, (40, 40, 50), (0, y), (surf_w, y))

    head_pos = None
    snake_positions = step_data['snake_positions']
    food_pos = step_data['food_position']
    action_values = step_data.get('action_values')

    # 食物
    if food_pos[0] >= 0:
        fx, fy = food_pos
        fr = pygame.Rect(fx * cell_size + 1, fy * cell_size + 1,
                         cell_size - 2, cell_size - 2)
        pygame.draw.rect(surf, (255, 60, 60), fr)
        pygame.draw.rect(surf, (200, 30, 30), fr, 1)

    # 蛇身（从尾到头绘制，让蛇头在最上层）
    for i in range(len(snake_positions) - 1, -1, -1):
        pos = snake_positions[i]
        if pos[0] < 0:
            continue
        if i == 0:
            head_pos = pos
            color_head = (50, 255, 50)
        else:
            t = i / max(len(snake_positions) - 1, 1)
            g = int(120 + 135 * (1 - t))
            b = int(255 - 200 * t)
            color_head = (20, g, b)
        r = pygame.Rect(pos[0] * cell_size + 1, pos[1] * cell_size + 1,
                         cell_size - 2, cell_size - 2)
        pygame.draw.rect(surf, color_head, r)
        if i > 0:
            pygame.draw.rect(surf, (10, 60, 120), r, 1)

    # 决策热力图（蛇头周围四个方向的色块）
    if action_values is not None and head_pos is not None and head_pos[0] >= 0:
        directions = [(0, -1, '↑'), (0, 1, '↓'), (-1, 0, '←'), (1, 0, '→')]
        vmin, vmax = action_values.min(), action_values.max()
        vrange = vmax - vmin
        for (dx, dy, _), val in zip(directions, action_values):
            nx, ny = head_pos[0] + dx, head_pos[1] + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                norm = (val - vmin) / vrange if vrange > 1e-8 else 0.5
                r = int(255 * norm)
                b = int(255 * (1 - norm))
                overlay = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                overlay.fill((r, 40, b, 140))
                surf.blit(overlay, (nx * cell_size, ny * cell_size))

    return surf


# ── Matplotlib → Pygame Surface ──

def fig_to_surface(fig):
    """将 matplotlib figure 渲染为 pygame Surface"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    return pygame.image.frombuffer(buf, (w, h), 'RGBA')


def smooth_curve(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


# ── 训练曲线构建 ──

def build_individual_curve(algo, rewards, record_episode, width, height, dpi=80):
    """单个模型的训练曲线"""
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1e1e2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1e1e2e')

    ep = np.arange(len(rewards))
    ax.plot(ep, rewards, alpha=0.2, color='#4FC3F7', lw=0.4)
    if len(rewards) > 20:
        s = smooth_curve(rewards)
        ax.plot(np.arange(len(s)), s, color='#4FC3F7', lw=1.5)

    if record_episode < len(rewards):
        ax.axvline(x=record_episode, color='#FF6B6B', ls='--', lw=0.8)
        ax.plot(record_episode, rewards[record_episode], 'o',
                color='#FF6B6B', ms=3)

    ax.axhline(0, color='gray', ls='--', lw=0.4)
    ax.set_xlabel('Episode', color='gray', fontsize=5.5)
    ax.set_ylabel('Reward', color='gray', fontsize=5.5)
    ax.tick_params(colors='gray', labelsize=4.5)
    for s in ax.spines.values():
        s.set_color('#444')
    ax.set_title(algo, color='white', fontsize=7, pad=2)
    fig.tight_layout(pad=0.3)
    return fig


def build_comparison_curve(runs_data, width, height, dpi=80):
    """多条训练曲线叠加对比"""
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1e1e2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1e1e2e')

    for idx, run in enumerate(runs_data):
        rewards = run['rewards']
        color = ALGO_COLORS_HEX[idx % len(ALGO_COLORS_HEX)]
        s = smooth_curve(rewards)
        ax.plot(np.arange(len(s)), s, color=color, lw=1.5,
                label=f"{run['algo']} ({len(rewards)} ep)")

    ax.axhline(0, color='gray', ls='--', lw=0.4)
    ax.set_xlabel('Episode', color='gray', fontsize=6)
    ax.set_ylabel('Reward', color='gray', fontsize=6)
    ax.tick_params(colors='gray', labelsize=5.5)
    ax.legend(fontsize=5.5, loc='lower right',
              facecolor='#2a2a3e', edgecolor='#444', labelcolor='white')
    for s in ax.spines.values():
        s.set_color('#444')
    ax.set_title('Training Reward Comparison', color='white', fontsize=8, pad=2)
    fig.tight_layout(pad=0.3)
    return fig


# ── 主可视化类 ──

class Visualizer:
    def __init__(self, exp_dirs):
        self.runs = []
        for d in exp_dirs:
            run = load_experiment(d)
            if run is not None:
                print(f'  ✓ {run["algo"]} — {run["num_steps"]} 步')
                self.runs.append(run)

        if not self.runs:
            print('错误: 没有加载到任何有效数据')
            sys.exit(1)

        self.n_runs = len(self.runs)
        self.max_steps = max(r['num_steps'] for r in self.runs)
        self.current_step = 0
        self.paused = True
        self.speed = 1
        self.fps = 30
        self.clock = pygame.time.Clock()

        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('microsoftyahei', 13)
        self.font_small = pygame.font.SysFont('microsoftyahei', 11)
        if self.font is None:
            self.font = pygame.font.Font(None, 13)
            self.font_small = pygame.font.Font(None, 11)

        # 布局
        self.window_width = min(1600, max(1000, self.n_runs * 320))
        self.window_height = 860
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('贪吃蛇AI 训练可视化对比')

        self.control_h = 36
        self.comp_curve_h = 180    # 底部总对比曲线高度
        self.bottom_margin = 8
        avail_h = self.window_height - self.control_h - self.comp_curve_h - self.bottom_margin
        self.col_w = self.window_width // self.n_runs

        # 游戏画面尺寸
        max_game_h = int(avail_h * 0.58)
        max_game_w = self.col_w - 20
        cell_by_h = max_game_h // GRID_H
        cell_by_w = max_game_w // GRID_W
        self.cell_size = max(10, min(cell_by_h, cell_by_w, 32))
        self.game_h = self.cell_size * GRID_H
        self.game_w = self.cell_size * GRID_W
        self.curve_h = avail_h - self.game_h - 35

        print(f'布局: {self.n_runs}列, cell={self.cell_size}, '
              f'游戏={self.game_w}×{self.game_h}, 曲线高={self.curve_h}')

        # 预渲染曲线
        print('渲染训练曲线...')
        self._pre_render_curves()

        # 预缓存游戏帧
        print(f'缓存游戏帧 ({self.max_steps} 步 × {self.n_runs} 模型)...')
        self._cache_frames()

        print('就绪!  空格=播放/暂停  +/-=速度  ←→=单步(暂停时)  R=重置  ESC=退出')

    def _pre_render_curves(self):
        dpi = 80
        self.indiv_curve_surfs = []
        for run in self.runs:
            ep = int(run['recording']['episode'])
            fig = build_individual_curve(run['algo'], run['rewards'], ep,
                                         self.col_w - 10, self.curve_h, dpi)
            self.indiv_curve_surfs.append(fig_to_surface(fig))
            plt.close(fig)

        comp_fig = build_comparison_curve(self.runs, self.window_width - 20,
                                          self.comp_curve_h - 10, dpi)
        self.comp_curve_surf = fig_to_surface(comp_fig)
        plt.close(comp_fig)

    def _cache_frames(self):
        self.frames = []
        for run in self.runs:
            rec = run['recording']
            n = run['num_steps']
            run_frames = []
            for step in range(n):
                step_data = {
                    'snake_positions': rec['positions'][step, :rec['snake_lengths'][step]],
                    'food_position': rec['food_positions'][step],
                    'action_values': rec['action_values'][step],
                }
                run_frames.append(render_game_surface(step_data, self.cell_size))
            self.frames.append(run_frames)

    def run(self):
        running = True
        while running:
            self.clock.tick(self.fps)
            running = self._handle_events()
            if not self.paused:
                self.current_step = min(self.current_step + self.speed,
                                        self.max_steps - 1)
                if self.current_step >= self.max_steps - 1:
                    self.paused = True
            self._render()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.speed = min(16, self.speed * 2)
                elif event.key == pygame.K_MINUS:
                    self.speed = max(1, self.speed // 2)
                elif event.key == pygame.K_RIGHT:
                    if self.paused:
                        self.current_step = min(self.current_step + 1,
                                                self.max_steps - 1)
                elif event.key == pygame.K_LEFT:
                    if self.paused:
                        self.current_step = max(self.current_step - 1, 0)
                elif event.key == pygame.K_r:
                    self.current_step = 0
                    self.paused = False
        return True

    def _render(self):
        self.screen.fill(BG_COLOR)

        # ── 顶部控制栏 ──
        pygame.draw.rect(self.screen, CONTROL_BG,
                         (0, 0, self.window_width, self.control_h))
        status = '▶ 播放中' if not self.paused else '⏸ 暂停'
        text = (f'{status}  速度: {self.speed}x  '
                f'步: {self.current_step + 1}/{self.max_steps}  |  '
                f'SPACE=暂停  +/-=速度  ←→=单步  R=重置')
        sf = self.font.render(text, True, TEXT_COLOR)
        self.screen.blit(sf, (10, (self.control_h - sf.get_height()) // 2))

        # ── 各列: 游戏 + 曲线 ──
        for i, run in enumerate(self.runs):
            x = i * self.col_w
            y = self.control_h + 4

            # 算法名 + 当前分数
            idx = min(self.current_step, run['num_steps'] - 1)
            score = int(run['recording']['scores'][idx])
            label = f'{run["algo"]}  分数: {score}'
            sf = self.font.render(label, True, TEXT_COLOR)
            self.screen.blit(sf, (x + 6, y))
            y += 20

            # 游戏画面（居中）
            game_x = x + (self.col_w - self.game_w) // 2
            frame = self.frames[i][min(self.current_step, len(self.frames[i]) - 1)]
            pygame.draw.rect(self.screen, CONTROL_BG,
                             (game_x - 2, y - 2, self.game_w + 4, self.game_h + 4),
                             border_radius=2)
            self.screen.blit(frame, (game_x, y))
            y += self.game_h + 4

            # 进度条
            prog = min(1.0, (self.current_step + 1) / max(run['num_steps'], 1))
            bar_x, bar_y = x + 8, y
            bar_w, bar_h = self.col_w - 16, 5
            pygame.draw.rect(self.screen, (50, 50, 65),
                             (bar_x, bar_y, bar_w, bar_h), border_radius=2)
            if prog > 0:
                rgb = hex_to_rgb(ALGO_COLORS_HEX[i % len(ALGO_COLORS_HEX)])
                pygame.draw.rect(self.screen, rgb,
                                 (bar_x, bar_y, int(bar_w * prog), bar_h),
                                 border_radius=2)
            y += 10

            # 步数/分数信息
            info = f'步 {idx+1}/{run["num_steps"]}  奖励 {run["recording"]["rewards"][idx]:.1f}'
            sf = self.font_small.render(info, True, DIM_TEXT)
            self.screen.blit(sf, (x + 8, y))
            y += 14

            # 训练曲线
            if i < len(self.indiv_curve_surfs):
                cs = self.indiv_curve_surfs[i]
                self.screen.blit(cs, (x + 5, y))

        # ── 底部总对比曲线 ──
        comp_y = self.window_height - self.comp_curve_h - self.bottom_margin
        pygame.draw.rect(self.screen, PANEL_BG,
                         (0, comp_y, self.window_width, self.comp_curve_h))
        self.screen.blit(self.comp_curve_surf,
                         (10, comp_y + (self.comp_curve_h - self.comp_curve_surf.get_height()) // 2))

        pygame.display.flip()


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description='贪吃蛇AI 训练可视化回放')
    parser.add_argument('exp_dirs', nargs='+',
                        help='实验目录路径（可传入多个，如 experiments/dqn_xxx experiments/ppo_xxx）')
    args = parser.parse_args()

    print(f'加载 {len(args.exp_dirs)} 个实验...')
    viz = Visualizer(args.exp_dirs)
    viz.run()


if __name__ == '__main__':
    main()
