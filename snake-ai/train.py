"""
训练脚本
支持 DQN / Double DQN / Dueling DQN / PPO
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
from src.agents.ppo_agent import PPOAgent
from config import Config


# ── 改装的 evaluate + select_action 统一接口 ──────────────────────────

def evaluate_agent(env, agent, n_episodes: int = 10, tag: str = "eval"):
    """
    评估智能体性能（兼容 DQNAgent 和 PPOAgent）

    Args:
        env: 游戏环境
        agent: 智能体
        n_episodes: 评估回合数
        tag: 日志标签

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
    avg_reward = np.mean(total_rewards)
    print(f"  [{tag} 评估] 平均奖励: {avg_reward:.2f} | 平均分数: {avg_score:.2f}")
    return avg_reward


# ── DQN 系列训练 ─────────────────────────────────────────────────────

def train_agent(algo: str = "dqn"):
    """
    训练 DQN 系列智能体（dqn / double_dqn / dueling_dqn）
    """
    env = SnakeEnv(
        grid_width=Config.GRID_WIDTH,
        grid_height=Config.GRID_HEIGHT,
        cell_size=Config.CELL_SIZE,
        render_mode=None,
        max_steps=Config.MAX_STEPS_PER_EPISODE,
    )

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
        target_update_freq=Config.TARGET_UPDATE_FREQ,
        algo=algo,
    )

    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    best_eval_reward = float('-inf')

    experiment_name = f"{algo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(Config.EXPERIMENT_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, 'training_log.txt')
    recordings_dir = os.path.join(experiment_dir, 'recordings')
    os.makedirs(recordings_dir, exist_ok=True)

    print(f"\n开始训练 {algo} 智能体")
    print(f"实验目录: {experiment_dir}")
    print(f"总回合数: {Config.TOTAL_EPISODES}")
    print(f"使用设备: {agent.device}")

    start_time = time.time()

    for episode in tqdm(range(Config.TOTAL_EPISODES), desc="训练进度"):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_score = 0

        agent.update_epsilon(episode, Config.TOTAL_EPISODES)

        # 录制逐帧数据
        recording = (episode + 1) % Config.RECORD_INTERVAL == 0
        if recording:
            step_records = []

        for step in range(Config.MAX_STEPS_PER_EPISODE):
            action_int = agent.select_action(state, eval_mode=False)
            action = Action(action_int)
            next_state, reward, done, info = env.step(action)

            if recording:
                action_values = agent.get_action_values(state)
                step_records.append({
                    'snake_positions': env.snake_positions.copy(),
                    'food_position': env.food_position,
                    'action': action_int,
                    'reward': reward,
                    'score': info['score'],
                    'action_values': action_values,
                    'done': done,
                })

            agent.replay_buffer.push(state, action_int, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            episode_reward += reward
            episode_length += 1
            episode_score = info['score']

            if done:
                break

        # 保存录制数据
        if recording:
            n_steps = len(step_records)
            max_len = max(len(r['snake_positions']) for r in step_records)
            positions = np.full((n_steps, max_len, 2), -1, dtype=np.int32)
            snake_lengths = np.zeros(n_steps, dtype=np.int32)
            food_positions = np.zeros((n_steps, 2), dtype=np.int32)
            actions = np.zeros(n_steps, dtype=np.int32)
            rewards = np.zeros(n_steps, dtype=np.float32)
            scores = np.zeros(n_steps, dtype=np.int32)
            action_values = np.zeros((n_steps, 4), dtype=np.float32)
            dones = np.zeros(n_steps, dtype=bool)
            for i, r in enumerate(step_records):
                pos_arr = np.array(r['snake_positions'], dtype=np.int32)
                positions[i, :len(pos_arr)] = pos_arr
                snake_lengths[i] = len(pos_arr)
                food_positions[i] = r['food_position']
                actions[i] = r['action']
                rewards[i] = r['reward']
                scores[i] = r['score']
                action_values[i] = r['action_values']
                dones[i] = r['done']
            record_path = os.path.join(recordings_dir, f'episode_{episode+1:06d}.npz')
            np.savez_compressed(record_path,
                algo=algo, episode=episode+1,
                positions=positions, snake_lengths=snake_lengths,
                food_positions=food_positions, actions=actions,
                rewards=rewards, scores=scores,
                action_values=action_values, dones=dones,
                grid_width=Config.GRID_WIDTH, grid_height=Config.GRID_HEIGHT)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(episode_score)

        if (episode + 1) % Config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-Config.LOG_INTERVAL:])
            avg_length = np.mean(episode_lengths[-Config.LOG_INTERVAL:])
            avg_score = np.mean(episode_scores[-Config.LOG_INTERVAL:])

            log_msg = (
                f"[{algo}] 回合 {episode + 1}/{Config.TOTAL_EPISODES} | "
                f"奖励: {avg_reward:.2f} | 分数: {avg_score:.1f} | "
                f"步数: {avg_length:.1f} | Epsilon: {agent.epsilon:.3f} | "
                f"缓冲区: {len(agent.replay_buffer)} | 损失: {loss if loss else 0:.4f}"
            )
            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')

        if (episode + 1) % Config.EVAL_FREQ == 0:
            eval_reward = evaluate_agent(env, agent, Config.EVAL_EPISODES, algo)
            with open(log_file, 'a') as f:
                f.write(f"  [{algo} 评估] 平均奖励: {eval_reward:.2f}\n")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(experiment_dir, 'best_model.pth'))

        if (episode + 1) % Config.SAVE_FREQ == 0:
            agent.save(os.path.join(experiment_dir, f'checkpoint_episode_{episode+1}.pth'))

    total_time = time.time() - start_time
    final_eval_reward = evaluate_agent(env, agent, Config.EVAL_EPISODES, algo)

    print(f"\n{algo} 训练完成! 耗时: {total_time / 3600:.2f}h")
    print(f"最佳评估奖励: {best_eval_reward:.2f} | 最终: {final_eval_reward:.2f}")

    agent.save(os.path.join(experiment_dir, 'final_model.pth'))
    np.save(os.path.join(experiment_dir, 'episode_rewards.npy'), np.array(episode_rewards))
    np.save(os.path.join(experiment_dir, 'episode_lengths.npy'), np.array(episode_lengths))

    print(f"模型已保存到: {experiment_dir}")
    return agent, experiment_dir


# ── PPO 训练 ──────────────────────────────────────────────────────────

def train_ppo():
    """
    训练 PPO 智能体

    PPO 为 on-policy 算法，与 DQN 系列训练流程不同：
      1. 逐 episode 收集完整轨迹
      2. Episode 结束时计算 GAE 优势
      3. 积累 Config.PPO_UPDATE_FREQ 个 episode 后做多 epoch 小批量更新
    """
    env = SnakeEnv(
        grid_width=Config.GRID_WIDTH,
        grid_height=Config.GRID_HEIGHT,
        cell_size=Config.CELL_SIZE,
        render_mode=None,
        max_steps=Config.MAX_STEPS_PER_EPISODE,
    )

    agent = PPOAgent(
        state_shape=env.observation_space_shape,
        n_actions=env.action_space,
        learning_rate=Config.PPO_LEARNING_RATE,
        gamma=Config.GAMMA,
        gae_lambda=Config.GAE_LAMBDA,
        clip_epsilon=Config.PPO_CLIP_EPSILON,
        entropy_coef=Config.PPO_ENTROPY_COEF,
        value_coef=Config.PPO_VALUE_COEF,
        ppo_epochs=Config.PPO_EPOCHS,
        mini_batch_size=Config.PPO_MINI_BATCH_SIZE,
        update_freq=Config.PPO_UPDATE_FREQ,
        max_grad_norm=Config.PPO_MAX_GRAD_NORM,
    )

    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    best_eval_reward = float('-inf')

    experiment_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = os.path.join(Config.EXPERIMENT_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    log_file = os.path.join(experiment_dir, 'training_log.txt')
    recordings_dir = os.path.join(experiment_dir, 'recordings')
    os.makedirs(recordings_dir, exist_ok=True)

    print(f"\n开始训练 PPO 智能体")
    print(f"实验目录: {experiment_dir}")
    print(f"总回合数: {Config.TOTAL_EPISODES}")
    print(f"使用设备: {agent.device}")
    print(f"PPO 参数: clip_ε={Config.PPO_CLIP_EPSILON}, "
          f"GAE λ={Config.GAE_LAMBDA}, "
          f"熵系数={Config.PPO_ENTROPY_COEF}, "
          f"epochs={Config.PPO_EPOCHS}, "
          f"更新频率={Config.PPO_UPDATE_FREQ}ep")

    start_time = time.time()

    for episode in tqdm(range(Config.TOTAL_EPISODES), desc="训练进度"):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_score = 0

        # 录制逐帧数据
        recording = (episode + 1) % Config.RECORD_INTERVAL == 0
        if recording:
            step_records = []

        # 收集一个 episode 的轨迹
        for step in range(Config.MAX_STEPS_PER_EPISODE):
            action_int, log_prob, value = agent.get_action_info(state)
            action = Action(action_int)
            next_state, reward, done, info = env.step(action)

            if recording:
                action_probs = agent.get_action_probs(state)
                step_records.append({
                    'snake_positions': env.snake_positions.copy(),
                    'food_position': env.food_position,
                    'action': action_int,
                    'reward': reward,
                    'score': info['score'],
                    'action_values': action_probs,
                    'done': done,
                })

            agent.store_transition(state, action_int, reward, done, log_prob, value)

            state = next_state
            episode_reward += reward
            episode_length += 1
            episode_score = info['score']

            if done:
                break

        # 保存录制数据
        if recording:
            n_steps = len(step_records)
            max_len = max(len(r['snake_positions']) for r in step_records)
            positions = np.full((n_steps, max_len, 2), -1, dtype=np.int32)
            snake_lengths = np.zeros(n_steps, dtype=np.int32)
            food_positions = np.zeros((n_steps, 2), dtype=np.int32)
            actions = np.zeros(n_steps, dtype=np.int32)
            rewards = np.zeros(n_steps, dtype=np.float32)
            scores = np.zeros(n_steps, dtype=np.int32)
            action_values = np.zeros((n_steps, 4), dtype=np.float32)
            dones = np.zeros(n_steps, dtype=bool)
            for i, r in enumerate(step_records):
                pos_arr = np.array(r['snake_positions'], dtype=np.int32)
                positions[i, :len(pos_arr)] = pos_arr
                snake_lengths[i] = len(pos_arr)
                food_positions[i] = r['food_position']
                actions[i] = r['action']
                rewards[i] = r['reward']
                scores[i] = r['score']
                action_values[i] = r['action_values']
                dones[i] = r['done']
            record_path = os.path.join(recordings_dir, f'episode_{episode+1:06d}.npz')
            np.savez_compressed(record_path,
                algo='ppo', episode=episode+1,
                positions=positions, snake_lengths=snake_lengths,
                food_positions=food_positions, actions=actions,
                rewards=rewards, scores=scores,
                action_values=action_values, dones=dones,
                grid_width=Config.GRID_WIDTH, grid_height=Config.GRID_HEIGHT)

        # Episode 结束 → 计算 GAE
        agent.end_episode()

        # 记录统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(episode_score)

        # 尝试 PPO 更新（内部按 update_freq 判断是否积累足够数据）
        update_result = agent.update()

        # 日志
        if (episode + 1) % Config.LOG_INTERVAL == 0:
            avg_reward = np.mean(episode_rewards[-Config.LOG_INTERVAL:])
            avg_length = np.mean(episode_lengths[-Config.LOG_INTERVAL:])
            avg_score = np.mean(episode_scores[-Config.LOG_INTERVAL:])

            if update_result is not None:
                log_msg = (
                    f"[PPO] 回合 {episode + 1}/{Config.TOTAL_EPISODES} | "
                    f"奖励: {avg_reward:.2f} | 分数: {avg_score:.1f} | "
                    f"步数: {avg_length:.1f} | "
                    f"策略损失: {update_result['policy_loss']:.4f} | "
                    f"价值损失: {update_result['value_loss']:.4f} | "
                    f"熵: {update_result['entropy_loss']:.4f} | "
                    f"更新次数: {update_result['n_updates']}"
                )
            else:
                log_msg = (
                    f"[PPO] 回合 {episode + 1}/{Config.TOTAL_EPISODES} | "
                    f"奖励: {avg_reward:.2f} | 分数: {avg_score:.1f} | "
                    f"步数: {avg_length:.1f} | "
                    f"收集数据中 (ep_buf={agent.episode_count}/{Config.PPO_UPDATE_FREQ})"
                )

            print(log_msg)
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')

        # 评估
        if (episode + 1) % Config.EVAL_FREQ == 0:
            eval_reward = evaluate_agent(env, agent, Config.EVAL_EPISODES, "PPO")
            with open(log_file, 'a') as f:
                f.write(f"  [PPO 评估] 平均奖励: {eval_reward:.2f}\n")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save(os.path.join(experiment_dir, 'best_model.pth'))

        # 定期保存
        if (episode + 1) % Config.SAVE_FREQ == 0:
            agent.save(os.path.join(experiment_dir, f'checkpoint_episode_{episode+1}.pth'))

    total_time = time.time() - start_time
    final_eval_reward = evaluate_agent(env, agent, Config.EVAL_EPISODES, "PPO")

    print(f"\nPPO 训练完成! 耗时: {total_time / 3600:.2f}h")
    print(f"最佳评估奖励: {best_eval_reward:.2f} | 最终: {final_eval_reward:.2f}")

    agent.save(os.path.join(experiment_dir, 'final_model.pth'))
    np.save(os.path.join(experiment_dir, 'episode_rewards.npy'), np.array(episode_rewards))
    np.save(os.path.join(experiment_dir, 'episode_lengths.npy'), np.array(episode_lengths))

    print(f"模型已保存到: {experiment_dir}")
    return agent, experiment_dir


# ── 测试（通用）────────────────────────────────────────────────────────

def test_trained_agent(model_path: str):
    """
    测试训练好的智能体（自动识别算法类型）
    """
    import pygame

    checkpoint = torch.load(model_path, map_location='cpu')
    algo = checkpoint.get('algo', 'dqn')
    print(f"加载模型，算法: {algo}")

    env = SnakeEnv(
        grid_width=Config.GRID_WIDTH,
        grid_height=Config.GRID_HEIGHT,
        cell_size=Config.CELL_SIZE,
        render_mode='human'
    )

    if algo == 'ppo':
        agent = PPOAgent(
            state_shape=env.observation_space_shape,
            n_actions=env.action_space,
        )
    else:
        agent = DQNAgent(
            state_shape=env.observation_space_shape,
            n_actions=env.action_space,
            algo=algo,
        )

    agent.load(model_path)
    print(f"测试模型: {model_path}  (algo={algo})")
    print("按 ESC 退出测试")

    while True:
        state = env.reset()
        total_reward = 0

        for step in range(Config.MAX_STEPS_PER_EPISODE):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return

            action_int = agent.select_action(state, eval_mode=True)
            action = Action(action_int)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                print(f"回合结束! 奖励: {total_reward:.2f}, 步数: {step}, 分数: {info['score']}")
                time.sleep(1)
                break


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='贪吃蛇强化学习训练')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='运行模式: train 或 test')
    parser.add_argument('--algo', type=str, default='dqn',
                        choices=['dqn', 'double_dqn', 'dueling_dqn', 'ppo'],
                        help='算法: dqn / double_dqn / dueling_dqn / ppo')
    parser.add_argument('--model', type=str, default=None,
                        help='测试模式下的模型路径')

    args = parser.parse_args()

    if args.mode == 'train':
        if args.algo == 'ppo':
            train_ppo()
        else:
            train_agent(algo=args.algo)
    elif args.mode == 'test':
        if args.model is None:
            print("错误: 测试模式需要指定模型路径")
            print("使用方法: python train.py --mode test --model <path>")
        else:
            test_trained_agent(args.model)
