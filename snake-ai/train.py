"""
DQN训练脚本
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.envs.snake_env import SnakeEnv, Action
from src.agents.dqn_agent import DQNAgent
from config import Config


def train_agent():
    """训练DQN智能体"""

    # 创建环境
    env = SnakeEnv(
        grid_width=Config.GRID_WIDTH,
        grid_height=Config.GRID_HEIGHT,
        cell_size=Config.CELL_SIZE,
        render_mode=None,  # 训练时不渲染
        max_steps=Config.MAX_STEPS_PER_EPISODE,
    )
    # 创建智能体
    agent = DQNAgent(
        state_shape=env.observation_space_shape,
        n_actions=env.action_space,
        learning_rate=Config.LEARNING_RATE,
        gamma=Config.GAMMA,
        epsilon_start=Config.EPSILON_START,
        epsilon_end=Config.EPSILON_END,
        epsilon_decay=Config.EPSILON_DECAY,
        buffer_size=Config.BUFFER_SIZE,
        batch_size=Config.BATCH_SIZE,
        target_update_freq=Config.TARGET_UPDATE_FREQ
    )

    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    best_eval_reward = float('-inf')

    # 创建实验目录
    experiment_name = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(Config.EXPERIMENT_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(experiment_dir, 'training_log.txt')

    print(f"开始训练 DQN 智能体")
    print(f"实验目录: {experiment_dir}")
    print(f"总回合数: {Config.TOTAL_EPISODES}")
    print(f"使用设备: {agent.device}")

    # 训练循环
    start_time = time.time()

    for episode in tqdm(range(Config.TOTAL_EPISODES), desc="训练进度"):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_score = 0

        # 更新探索率
        agent.update_epsilon(episode, Config.TOTAL_EPISODES)

        # 回合循环
        for step in range(Config.MAX_STEPS_PER_EPISODE):
            # 选择动作
            action_int = agent.select_action(state, eval_mode=False)
            action = Action(action_int)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储经验
            agent.replay_buffer.push(state, action_int, reward, next_state, done)

            # 训练智能体
            loss = agent.train_step()

            # 更新状态
            state = next_state
            episode_reward += reward
            episode_length += 1
            episode_score = info['score']

            if done:
                break

        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(episode_score)

        # 打印日志
        if (episode + 1) % Config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-Config.LOG_INTERVAL:])
            avg_length = np.mean(episode_lengths[-Config.LOG_INTERVAL:])
            avg_score = np.mean(episode_scores[-Config.LOG_INTERVAL:])

            log_message = (
                f"回合 {episode + 1}/{Config.TOTAL_EPISODES} | "
                f"奖励: {avg_reward:.2f} | "
                f"分数: {avg_score:.1f} | "
                f"步数: {avg_length:.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"缓冲区: {len(agent.replay_buffer)} | "
                f"损失: {loss if loss else 0:.4f}"
            )
            print(log_message)

            # 写入日志文件
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')

        # 评估智能体
        if (episode + 1) % Config.EVAL_FREQ == 0:
            eval_reward = evaluate_agent(env, agent, Config.EVAL_EPISODES)

            with open(log_file, 'a') as f:
                f.write(f"  评估 | 平均奖励: {eval_reward:.2f}\n")

            # 保存最佳模型
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_model_path = os.path.join(experiment_dir, 'best_model.pth')
                agent.save(best_model_path)
                print(f"  保存最佳模型: {best_model_path}")

        # 定期保存模型
        if (episode + 1) % Config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_episode_{episode+1}.pth')
            agent.save(checkpoint_path)

    # 训练结束
    total_time = time.time() - start_time

    # 最后评估
    final_eval_reward = evaluate_agent(env, agent, Config.EVAL_EPISODES)

    print("\n训练完成!")
    print(f"总训练时间: {total_time / 3600:.2f} 小时")
    print(f"最佳评估奖励: {best_eval_reward:.2f}")
    print(f"最终评估奖励: {final_eval_reward:.2f}")

    # 保存最终模型
    final_model_path = os.path.join(experiment_dir, 'final_model.pth')
    agent.save(final_model_path)

    # 保存训练统计
    np.save(os.path.join(experiment_dir, 'episode_rewards.npy'), np.array(episode_rewards))
    np.save(os.path.join(experiment_dir, 'episode_lengths.npy'), np.array(episode_lengths))

    print(f"模型已保存到: {experiment_dir}")

    return agent, experiment_dir


def evaluate_agent(env: SnakeEnv, agent: DQNAgent, n_episodes: int = 10) -> float:
    """
    评估智能体性能

    Args:
        env: 游戏环境
        agent: 智能体
        n_episodes: 评估回合数

    Returns:
        平均奖励
    """
    total_rewards = []
    total_scores = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(Config.MAX_STEPS_PER_EPISODE):
            action_int = agent.select_action(state, eval_mode=True)
            action = Action(action_int)

            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            state = next_state

            if done:
                total_scores.append(info['score'])
                break

        total_rewards.append(episode_reward)

    avg_score = np.mean(total_scores) if total_scores else 0
    # 额外打印分数信息
    print(f"  [评估] 平均奖励: {np.mean(total_rewards):.2f} | 平均分数: {avg_score:.2f}")
    return np.mean(total_rewards)


def test_trained_agent(model_path: str):
    """
    测试训练好的智能体

    Args:
        model_path: 模型路径
    """
    # 创建环境
    env = SnakeEnv(
        grid_width=Config.GRID_WIDTH,
        grid_height=Config.GRID_HEIGHT,
        cell_size=Config.CELL_SIZE,
        render_mode='human'
    )

    # 创建智能体
    agent = DQNAgent(
        state_shape=env.observation_space_shape,
        n_actions=env.action_space
    )

    # 加载模型
    agent.load(model_path)

    print(f"测试模型: {model_path}")
    print("按 ESC 退出测试")

    # 测试循环
    while True:
        state = env.reset()
        total_reward = 0

        for step in range(Config.MAX_STEPS_PER_EPISODE):
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return

            # 选择动作
            action_int = agent.select_action(state, eval_mode=True)
            action = Action(action_int)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                print(f"回合结束! 奖励: {total_reward:.2f}, 步数: {step}, 分数: {info['score']}")
                time.sleep(1)
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='贪吃蛇DQN训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train 或 test')
    parser.add_argument('--model', type=str, default=None,
                        help='测试模式下的模型路径')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent()
    elif args.mode == 'test':
        if args.model is None:
            print("错误: 测试模式需要指定模型路径")
            print("使用方法: python train.py --mode test --model checkpoints/best_model.pth")
        else:
            import pygame
            test_trained_agent(args.model)
