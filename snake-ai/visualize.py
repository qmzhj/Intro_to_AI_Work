"""
可视化回放程序
按训练阶段顺序播放所有录制，展示 AI 从笨到聪明的完整过程。
每到一个新阶段，训练曲线会向前延伸，游戏画面切换到该阶段的 checkpoint。

用法:
  python visualize.py experiments/dqn_xxx experiments/ppo_xxx
  python visualize.py experiments/dqn_xxx --output demo.mp4
"""

import os, sys, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame

# ── constants ──

GRID_W, GRID_H = 12, 12
BG = (18, 18, 28)
PANEL_BG = (26, 26, 38)
CTRL_BG = (36, 36, 50)
TEXT_COLOR = (210, 210, 210)
DIM = (140, 140, 140)
ALGO_COLORS = ['#1E88E5', '#43A047', '#F4511E', '#8E24AA',
               '#00ACC1', '#FFB300', '#E53935', '#5E35B1']


def hex_rgb(h):
    return [int(h[i:i+2], 16) for i in (1, 3, 5)]


def smooth(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode='valid') if len(x) >= w else x


def fig2surf(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return pygame.image.frombuffer(
        canvas.buffer_rgba(), canvas.get_width_height(), 'RGBA')


# ── data model ──

class Recording:
    """一个录制文件 = 一局游戏的逐帧数据"""
    def __init__(self, path):
        d = np.load(path)
        self.data = d
        self.episode = int(d['episode'])
        self.num_steps = len(d['rewards'])

    def step_data(self, s):
        s = min(s, self.num_steps - 1)
        return {
            'snake_positions': self.data['positions'][s, :self.data['snake_lengths'][s]],
            'food_position': self.data['food_positions'][s],
            'action_values': self.data['action_values'][s],
        }


class RunData:
    """一个实验的全部数据"""
    def __init__(self, exp_dir):
        # training curve
        rp = os.path.join(exp_dir, 'episode_rewards.npy')
        self.rewards = np.load(rp)

        # recordings sorted by episode
        rd = os.path.join(exp_dir, 'recordings')
        fs = sorted(os.path.join(rd, f) for f in os.listdir(rd) if f.endswith('.npz'))
        self.recordings = [Recording(f) for f in fs]
        self.num_recs = len(self.recordings)

        # algo name from dirname
        m = re.match(r'^([a-z_]+?)(?:_\d{8}_\d{6})$', os.path.basename(exp_dir))
        self.algo = m.group(1) if m else os.path.basename(exp_dir)

        self.episodes = [r.episode for r in self.recordings]


# ── game frame renderer ──

def render_frame(sd, cs):
    w, h = GRID_W * cs, GRID_H * cs
    surf = pygame.Surface((w, h))
    surf.fill((0, 0, 0))
    # grid
    for x in range(0, w, cs):
        pygame.draw.line(surf, (40, 40, 50), (x, 0), (x, h))
    for y in range(0, h, cs):
        pygame.draw.line(surf, (40, 40, 50), (0, y), (w, y))
    pos, food, av = sd['snake_positions'], sd['food_position'], sd['action_values']
    # food
    if food[0] >= 0:
        r = pygame.Rect(food[0]*cs+1, food[1]*cs+1, cs-2, cs-2)
        pygame.draw.rect(surf, (255, 60, 60), r)
        pygame.draw.rect(surf, (200, 30, 30), r, 1)
    # snake (tail → head)
    head = None
    for i in range(len(pos) - 1, -1, -1):
        p = pos[i]
        if p[0] < 0:
            continue
        if i == 0:
            head = p
            c = (50, 255, 50)
        else:
            t = i / max(len(pos) - 1, 1)
            c = (20, int(120 + 135 * (1 - t)), int(255 - 200 * t))
        r = pygame.Rect(p[0]*cs+1, p[1]*cs+1, cs-2, cs-2)
        pygame.draw.rect(surf, c, r)
    # heatmap
    if av is not None and head is not None and head[0] >= 0:
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        lo, hi = av.min(), av.max()
        rg = hi - lo
        for (dx, dy), v in zip(dirs, av):
            nx, ny = head[0]+dx, head[1]+dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                n = (v - lo) / rg if rg > 1e-8 else .5
                o = pygame.Surface((cs, cs), pygame.SRCALPHA)
                o.fill((int(255*n), 40, int(255*(1-n)), 140))
                surf.blit(o, (nx*cs, ny*cs))
    # score HUD overlay (top-left corner)
    score, score_font = sd.get('score'), sd.get('_score_font')
    if score is not None and score_font is not None:
        t = score_font.render(str(score), True, (255, 255, 100))
        bg = pygame.Surface((t.get_width()+10, t.get_height()+4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        surf.blit(bg, (3, 3))
        surf.blit(t, (8, 5))
    return surf


# ── curve builders ──

def build_indiv_curve(algo, rewards, upto_ep, w, h, dpi=80):
    """单个训练曲线，只画到 upto_ep 为止"""
    fig = Figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.patch.set_facecolor('#1e1e2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1e1e2e')
    ep = min(upto_ep, len(rewards))
    x = np.arange(ep)
    y = rewards[:ep]
    ax.plot(x, y, alpha=.2, color='#4FC3F7', lw=.4)
    if len(y) > 20:
        s = smooth(y)
        ax.plot(np.arange(len(s)), s, color='#4FC3F7', lw=1.5)
    if ep > 0:
        ax.axvline(x=ep-1, color='#FF6B6B', ls='--', lw=.8)
        ax.plot(ep-1, y[-1], 'o', color='#FF6B6B', ms=3)
    ax.axhline(0, color='gray', ls='--', lw=.4)
    ax.set_xlim(0, len(rewards)-1)
    ax.set_xlabel('Episode', color='gray', fontsize=5.5)
    ax.set_ylabel('Reward', color='gray', fontsize=5.5)
    ax.tick_params(colors='gray', labelsize=4.5)
    for s in ax.spines.values():
        s.set_color('#444')
    ax.set_title(f'{algo}  ep {upto_ep}', color='white', fontsize=7, pad=2)
    fig.tight_layout(pad=.3)
    return fig


def build_comparison(runs, w, h, dpi=80):
    fig = Figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.patch.set_facecolor('#1e1e2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1e1e2e')
    for i, run in enumerate(runs):
        c = ALGO_COLORS[i % len(ALGO_COLORS)]
        s = smooth(run.rewards)
        ax.plot(np.arange(len(s)), s, color=c, lw=1.5,
                label=f"{run.algo} ({len(run.rewards)} ep)")
    ax.axhline(0, color='gray', ls='--', lw=.4)
    ax.set_xlabel('Episode', color='gray', fontsize=6)
    ax.set_ylabel('Reward', color='gray', fontsize=6)
    ax.tick_params(colors='gray', labelsize=5.5)
    ax.legend(fontsize=5.5, loc='lower right',
              facecolor='#2a2a3e', edgecolor='#444', labelcolor='white')
    for s in ax.spines.values():
        s.set_color('#444')
    ax.set_title('Training Reward Comparison', color='white', fontsize=8, pad=2)
    fig.tight_layout(pad=.3)
    return fig


# ── visualizer ──

class Visualizer:
    def __init__(self, exp_dirs, output_path=None):
        self.output_path = output_path

        # load
        self.runs = []
        for d in exp_dirs:
            try:
                r = RunData(d)
                eps = ','.join(str(e) for e in r.episodes)
                print(f'  ✓ {r.algo} — {r.num_recs} 录 ({eps})')
                self.runs.append(r)
            except Exception as e:
                print(f'  ⚠ {d}: {e}')
        if not self.runs:
            print('错误: 无有效数据'); sys.exit(1)

        self.n = len(self.runs)

        # recording groups (index-aligned across runs)
        self.n_groups = max(r.num_recs for r in self.runs)
        self.group_steps = []
        for g in range(self.n_groups):
            mx = 0
            for r in self.runs:
                if g < r.num_recs:
                    mx = max(mx, r.recordings[g].num_steps)
            self.group_steps.append(mx)
        # global step → (group, step_in_group)
        self.step_map = []
        for g, ns in enumerate(self.group_steps):
            for s in range(ns):
                self.step_map.append((g, s))
        self.total = len(self.step_map)

        self.pos = 0          # global step
        self.group = 0
        self.paused = True
        self.speed = 1

        # pygame init
        if output_path:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont('microsoftyahei', 13)
        self.font_sm = pygame.font.SysFont('microsoftyahei', 11)
        self.font_title = pygame.font.SysFont('microsoftyahei', 16)
        self.score_font = pygame.font.SysFont('microsoftyahei', 20, bold=True)

        # layout
        self.win_w = min(1600, max(1000, self.n * 320))
        self.win_h = 860
        self.ch = 36           # control bar height
        self.bh = 180          # bottom comparison height
        self.bm = 8
        ah = self.win_h - self.ch - self.bh - self.bm
        self.col_w = self.win_w // self.n
        mgh = int(ah * .58)
        mgw = self.col_w - 20
        self.cs = max(10, min(mgh//GRID_H, mgw//GRID_W, 32))
        self.gh = self.cs * GRID_H
        self.gw = self.cs * GRID_W
        self.curve_h = ah - self.gh - 35
        print(f'布局: {self.n}列, cell={self.cs}, '
              f'groups={self.n_groups}, frames={self.total}')

        # screen
        self.screen = (pygame.display.set_mode((self.win_w, self.win_h))
                       if not output_path else pygame.Surface((self.win_w, self.win_h)))
        if not output_path:
            pygame.display.set_caption('贪吃蛇AI — 训练阶段演进')
            self.screen.fill(BG); pygame.display.flip()

        # cache game frames [run][rec][step]
        self.frames = []
        for run in self.runs:
            rf = []
            for rec in run.recordings:
                ff = []
                for s in range(rec.num_steps):
                    sd = rec.step_data(s)
                    sd['score'] = int(rec.data['scores'][s])
                    sd['_score_font'] = self.score_font
                    ff.append(render_frame(sd, self.cs))
                rf.append(ff)
            self.frames.append(rf)

        # comparison curve (static)
        print('渲染总对比曲线...')
        fig = build_comparison(self.runs, self.win_w - 20, self.bh - 10)
        self.comp_surf = fig2surf(fig); plt.close(fig)

        # lazy individual curve cache: group → [surface, ...]
        self.curve_cache = {}
        self._ensure_curves(0)

    def _ensure_curves(self, g):
        if g in self.curve_cache:
            return
        surfs = []
        for run in self.runs:
            ri = min(g, run.num_recs - 1)
            ep = run.recordings[ri].episode
            fig = build_indiv_curve(run.algo, run.rewards, ep, self.col_w - 10, self.curve_h)
            surfs.append(fig2surf(fig)); plt.close(fig)
        self.curve_cache[g] = surfs

    # ── interactive ──

    def _run_interactive(self):
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(30)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    run = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        run = False
                    elif e.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif e.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        self.speed = min(16, self.speed * 2)
                    elif e.key == pygame.K_MINUS:
                        self.speed = max(1, self.speed // 2)
                    elif e.key == pygame.K_RIGHT:
                        if self.paused:
                            self.pos = min(self.pos + 1, self.total - 1)
                            self.group, _ = self.step_map[self.pos]
                    elif e.key == pygame.K_LEFT:
                        if self.paused:
                            self.pos = max(self.pos - 1, 0)
                            self.group, _ = self.step_map[self.pos]
                    elif e.key == pygame.K_r:
                        self.pos = 0; self.group = 0; self.paused = False
            if not self.paused:
                self.pos = min(self.pos + self.speed, self.total - 1)
                g, _ = self.step_map[self.pos]
                if g != self.group:
                    self.group = g; self._ensure_curves(g)
                if self.pos >= self.total - 1:
                    self.paused = True
            self._draw()
            pygame.display.flip()
        pygame.quit()

    # ── video export ──

    def _export_video(self):
        import cv2
        fps = 10
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (self.win_w, self.win_h))
        total = self.total
        for step in range(total):
            self.pos = step
            self.group, _ = self.step_map[step]
            self._ensure_curves(self.group)
            self._draw()
            f = cv2.cvtColor(pygame.surfarray.array3d(self.screen).swapaxes(0, 1),
                             cv2.COLOR_RGB2BGR)
            out.write(f)
            if (step + 1) % 50 == 0:
                print(f'  渲染: {step+1}/{total} ({(step+1)/total*100:.0f}%)')
        out.release()
        print(f'视频已保存: {self.output_path}')

    # ── draw one frame ──

    def _draw(self):
        self.screen.fill(BG)
        g, sg = self.step_map[self.pos]

        # ── top bar ──
        pygame.draw.rect(self.screen, CTRL_BG, (0, 0, self.win_w, self.ch))
        if not self.output_path:
            st = '▶ 播放中' if not self.paused else '⏸ 暂停'
            txt = f'{st}  速度: {self.speed}x  阶段 {g+1}/{self.n_groups}  帧 {self.pos+1}/{self.total}'
        else:
            txt = f'导出视频  阶段 {g+1}/{self.n_groups}  帧 {self.pos+1}/{self.total}'
        sf = self.font.render(txt, True, TEXT_COLOR)
        self.screen.blit(sf, (10, (self.ch - sf.get_height()) // 2))

        # ── per-run columns ──
        for i, run in enumerate(self.runs):
            x = i * self.col_w
            y = self.ch + 4

            ri = min(g, run.num_recs - 1)
            rec = run.recordings[ri]
            ss = min(sg, rec.num_steps - 1)

            # label (algo name, large)
            sf = self.font_title.render(f'{run.algo}  ep{rec.episode}', True, TEXT_COLOR)
            self.screen.blit(sf, (x + 6, y))
            y += 22

            # game
            gx = x + (self.col_w - self.gw) // 2
            f = self.frames[i][ri][ss]
            pygame.draw.rect(self.screen, CTRL_BG, (gx - 2, y - 2, self.gw + 4, self.gh + 4))
            self.screen.blit(f, (gx, y))
            y += self.gh + 4

            # progress bar
            p = min(1., (ss + 1) / max(rec.num_steps, 1))
            bx, by = x + 8, y
            bw, bh = self.col_w - 16, 5
            pygame.draw.rect(self.screen, (50, 50, 65), (bx, by, bw, bh))
            if p > 0:
                rgb = hex_rgb(ALGO_COLORS[i % len(ALGO_COLORS)])
                pygame.draw.rect(self.screen, rgb, (bx, by, int(bw * p), bh))
            y += 10

            # info
            sf = self.font_sm.render(f'{ss+1}/{rec.num_steps}  rw:{rec.data["rewards"][ss]:.1f}',
                                     True, DIM)
            self.screen.blit(sf, (x + 8, y))
            y += 14

            # curve
            cv = self.curve_cache.get(g) or self.curve_cache.get(0, [])
            if i < len(cv):
                self.screen.blit(cv[i], (x + 5, y))

        # ── bottom comparison ──
        cy = self.win_h - self.bh - self.bm
        pygame.draw.rect(self.screen, PANEL_BG, (0, cy, self.win_w, self.bh))
        cs = self.comp_surf
        self.screen.blit(cs, (10, cy + (self.bh - cs.get_height()) // 2))


# ── CLI ──

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='贪吃蛇AI — 训练阶段演进回放')
    p.add_argument('exp_dirs', nargs='+', help='实验目录路径')
    p.add_argument('--output', '-o', help='导出视频路径 (如 out.mp4)')
    args = p.parse_args()
    viz = Visualizer(args.exp_dirs, args.output)
    if args.output:
        viz._export_video()
    else:
        viz._run_interactive()
