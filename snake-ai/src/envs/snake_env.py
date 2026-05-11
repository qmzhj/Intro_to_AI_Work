"""
贪吃蛇游戏环境
遵循Gymnasium接口规范，适用于深度强化学习训练
"""

import numpy as np
import pygame
from enum import Enum
from typing import Tuple, Optional


class Action(Enum):
    """动作空间定义"""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeEnv:
    """贪吃蛇游戏环境"""

    def __init__(
        self,
        grid_width: int = 12,
        grid_height: int = 12,
        cell_size: int = 20,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
    ):
        """
        初始化贪吃蛇环境

        Args:
            grid_width: 网格宽度（格子数）
            grid_height: 网格高度（格子数）
            cell_size: 每个格子的像素大小
            render_mode: 渲染模式 ('human', None)
            max_steps: 每回合最大步数（默认 grid_width * grid_height）
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.render_mode = render_mode

        # 游戏参数
        self.action_space = 4  # 上下左右4个动作
        self.observation_space_shape = (grid_height, grid_width, 3)  # 3通道图像

        # 颜色定义
        self.COLOR_BG = (0, 0, 0)
        self.COLOR_SNAKE_HEAD = (0, 255, 0)
        self.COLOR_SNAKE_BODY = (0, 120, 255)
        self.COLOR_FOOD = (255, 0, 0)
        self.COLOR_GRID = (40, 40, 40)

        # 奖励参数
        self.REWARD_FOOD = 10
        self.REWARD_DEATH = -100
        self.REWARD_STEP = -0.1  # 鼓励快速完成任务

        # 方向映射
        self.direction_map = {
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0)
        }

        # Pygame初始化
        self.screen = None
        self.clock = None
        self.font = None

        # 游戏状态
        self.snake_positions = None
        self.direction = None
        self.food_position = None
        self.score = 0
        self.steps = 0
        self.max_steps = max_steps if max_steps is not None else (grid_width * grid_height)

    def reset(self) -> np.ndarray:
        """
        重置游戏环境

        Returns:
            初始观察状态
        """
        # 初始化蛇的位置（在屏幕中央）
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.snake_positions = [(center_x, center_y)]
        self.direction = Action.RIGHT
        self.score = 0
        self.steps = 0

        # 生成食物
        self.food_position = self._generate_food()

        # 初始化渲染
        if self.render_mode == 'human' and self.screen is None:
            self._init_render()

        return self._get_observation()

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步动作

        Args:
            action: 执行的动作

        Returns:
            observation: 新的观察状态
            reward: 奖励值
            done: 游戏是否结束
            info: 额外信息
        """
        self.steps += 1

        # 更新方向（防止直接反向）
        current_dir = self.direction_map[self.direction]
        new_dir = self.direction_map[action]
        if (new_dir[0] * -1, new_dir[1] * -1) != current_dir:
            self.direction = action

        # 计算新头部位置
        head = self.snake_positions[0]
        new_x = (head[0] + self.direction_map[self.direction][0]) % self.grid_width
        new_y = (head[1] + self.direction_map[self.direction][1]) % self.grid_height
        new_head = (new_x, new_y)

        # 检查是否撞到自己
        done = False
        reward = self.REWARD_STEP

        if new_head in self.snake_positions[1:]:
            # 撞到自己
            done = True
            reward = self.REWARD_DEATH
        elif self.steps >= self.max_steps:
            # 超过最大步数
            done = True
        else:
            # 移动蛇
            self.snake_positions.insert(0, new_head)

            # 检查是否吃到食物
            if new_head == self.food_position:
                # 吃到食物
                reward = self.REWARD_FOOD
                self.score += 1
                self.food_position = self._generate_food()
            else:
                # 没吃到食物，移除尾部
                self.snake_positions.pop()

        # 渲染
        if self.render_mode == 'human':
            self._render()

        observation = self._get_observation()
        info = {
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake_positions)
        }

        return observation, reward, done, info

    def _generate_food(self) -> Tuple[int, int]:
        """生成食物位置"""
        while True:
            food_x = np.random.randint(0, self.grid_width)
            food_y = np.random.randint(0, self.grid_height)
            food_pos = (food_x, food_y)
            if food_pos not in self.snake_positions:
                return food_pos

    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察状态

        Returns:
            形状为 (grid_height, grid_width, 3) 的numpy数组
            通道0: 蛇身
            通道1: 食物
            通道2: 蛇头
        """
        obs = np.zeros(self.observation_space_shape, dtype=np.float32)

        # 绘制蛇身
        for i, pos in enumerate(self.snake_positions):
            if i == 0:
                # 蛇头
                obs[pos[1], pos[0], 2] = 1.0
            else:
                # 蛇身
                obs[pos[1], pos[0], 0] = 1.0

        # 绘制食物
        obs[self.food_position[1], self.food_position[0], 1] = 1.0

        return obs

    def _init_render(self):
        """初始化渲染"""
        pygame.init()
        screen_width = self.grid_width * self.cell_size
        screen_height = self.grid_height * self.cell_size
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("贪吃蛇AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)

    def _render(self):
        """渲染游戏界面"""
        self.screen.fill(self.COLOR_BG)

        # 绘制网格
        for x in range(0, self.grid_width * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.grid_height * self.cell_size))
        for y in range(0, self.grid_height * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.grid_width * self.cell_size, y))

        # 绘制食物
        food_rect = pygame.Rect(
            self.food_position[0] * self.cell_size,
            self.food_position[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_FOOD, food_rect)

        # 绘制蛇
        for i, pos in enumerate(self.snake_positions):
            color = self.COLOR_SNAKE_HEAD if i == 0 else self.COLOR_SNAKE_BODY
            rect = pygame.Rect(
                pos[0] * self.cell_size,
                pos[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, color, rect)

        # 显示分数
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


if __name__ == "__main__":
    """测试环境"""
    env = SnakeEnv(render_mode='human')
    obs = env.reset()

    print(f"环境信息:")
    print(f"  网格大小: {env.grid_width} x {env.grid_height}")
    print(f"  观察空间形状: {env.observation_space_shape}")
    print(f"  动作空间大小: {env.action_space}")
    print(f"  初始观察形状: {obs.shape}")

    # 测试几步
    import random
    for _ in range(100):
        action = random.choice(list(Action))
        obs, reward, done, info = env.step(action)
        print(f"步骤: {info['steps']}, 奖励: {reward}, 分数: {info['score']}, 蛇长: {info['snake_length']}")
        if done:
            print("游戏结束!")
            obs = env.reset()

    env.close()