import os
import sys
import yaml
import argparse
import torch
import numpy as np
from types import SimpleNamespace
from datetime import datetime

# Ensure src directories are in the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def load_config(path):
    with open(path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_native(args, task="eval"):
    print(f"\n[MODE] Running SUMO Native (Task: {task})...")
    from core.envs.sumo_env import SUMOEnv
    env = SUMOEnv(args)
    _, _ = env.reset()
    done = False
    total_reward = 0
    total_wait = 0
    steps = 0
    
    while not done:
        _, _, reward, done, info = env.step(None)
        total_reward += reward
        total_wait += info.get("total_wait", 0) # Assumes env might provide this
        steps += 1
    
    avg_reward = total_reward / steps
    print(f"[RESULT] Native Finished. Avg Reward: {avg_reward:.2f}")
    env.close()
    return {"avg_reward": avg_reward, "total_steps": steps}

def run_indian(args, task="eval"):
    print(f"\n[MODE] Running Indian Standard (Task: {task})...")
    from core.envs.sumo_env import SUMOEnv
    from algos.indian.webster_controller import WebsterController
    
    env = SUMOEnv(args)
    env.reset()
    controllers = [WebsterController(env, tls_id) for tls_id in env.tls_ids]
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        for c in controllers: c.step()
        _, _, reward, done, _ = env.step(None)
        total_reward += reward
        steps += 1
        
    avg_reward = total_reward / steps
    print(f"[RESULT] Indian Standard Finished. Avg Reward: {avg_reward:.2f}")
    env.close()
    return {"avg_reward": avg_reward, "total_steps": steps}

def run_legacy(args, task="train"):
    print(f"\n[MODE] Running Legacy QMIX (Task: {task})...")
    from core.envs.sumo_env import SUMOEnv
    from algos.legacy.trainer import QMIXTrainer
    
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    env = SUMOEnv(args)
    trainer = QMIXTrainer(
        env=env, n_agents=env.n_agents, state_dim=env.get_state_size(),
        obs_dim=env.obs_size, n_actions=env.n_actions,
        hidden_dim=args.hidden_dim, lr=args.lr, gamma=args.gamma,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        device=device
    )

    model_path = os.path.join(args.save_dir, "legacy_model.pt")
    
    if task == "eval":
        if not trainer.load_model(model_path):
            env.close()
            return None
        trainer.epsilon = 0.0 # No exploration for eval

    n_episodes = args.n_episodes if task == "train" else 1
    best_reward = -float('inf')

    for ep in range(n_episodes):
        state, obs = env.reset()
        done = False
        ep_reward = 0
        step = 0
        
        while not done:
            actions = trainer.select_action(obs)
            next_state, next_obs, reward, done, _ = env.step(actions)
            
            if task == "train":
                trainer.replay_buffer.store(obs, state, actions, reward, next_obs, next_state, done)
                loss = trainer.train_step()
                if step % args.target_update_interval == 0:
                    trainer.update_target_networks()
                    
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / step
        if task == "train":
            print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} | Epsilon: {trainer.epsilon:.2f}")
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] Legacy Eval Finished. Avg Reward: {avg_ep_reward:.2f}")
            
    env.close()
    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}

def run_v2(args, task="train"):
    print(f"\n[MODE] Running Improved QMIX V2 (Task: {task})...")
    from core.envs.sumo_env import SUMOEnv
    from algos.v2.trainer import QMIXTrainerV2
    
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    env = SUMOEnv(args)
    trainer = QMIXTrainerV2(
        env=env, n_agents=env.n_agents, state_dim=env.get_state_size(),
        obs_dim=env.obs_size, n_actions=env.n_actions,
        rnn_hidden_dim=args.rnn_hidden_dim, mixing_hidden_dim=args.mixer_hidden_dim,
        lr=args.lr, gamma=args.gamma,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        device=device
    )

    model_path = os.path.join(args.save_dir, "v2_model.pt")
    
    if task == "eval":
        if not trainer.load_model(model_path):
            env.close()
            return None
        trainer.epsilon = 0.0

    n_episodes = args.n_episodes if task == "train" else 1
    best_reward = -float('inf')

    for ep in range(n_episodes):
        state, obs = env.reset()
        h = trainer.init_hidden()
        done = False
        ep_reward = 0
        step = 0
        
        while not done:
            actions, h = trainer.select_action(obs, h)
            next_state, next_obs, reward, done, _ = env.step(actions)
            
            if task == "train":
                trainer.replay_buffer.store(obs, state, actions, reward, next_obs, next_state, done)
                loss = trainer.train_step()
                if step % args.target_update_interval == 0:
                    trainer.update_target_networks()
            
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / step
        if task == "train":
            if (ep + 1) % args.print_interval == 0:
                print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} | Epsilon: {trainer.epsilon:.2f}")
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] V2 Eval Finished. Avg Reward: {avg_ep_reward:.2f}")

    env.close()
    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}

def main():
    parser = argparse.ArgumentParser(description="SIH 2025 Traffic Benchmark & Training Suite")
    parser.add_argument("--mode", type=str, required=True, choices=["native", "legacy", "v2", "indian", "compare"])
    parser.add_argument("--task", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--config", type=str, help="Custom config path")
    
    args_cmd = parser.parse_args()
    
    config_map = {
        "native": "config/native.yaml",
        "legacy": "config/legacy_qmix.yaml",
        "v2": "config/improved_qmix.yaml",
        "indian": "config/indian_standard.yaml"
    }

    if args_cmd.mode == "compare":
        results = {}
        for mode in ["native", "indian", "legacy", "v2"]:
            cfg = load_config(config_map[mode])
            results[mode] = globals()[f"run_{mode}"](cfg, task="eval")
        
        print("\n" + "="*40)
        print(" FINAL PERFORMANCE SCORECARD ")
        print("="*40)
        print(f"{'Model':<15} | {'Avg Reward':<12}")
        print("-" * 30)
        for mode, res in results.items():
            score = res['avg_reward'] if res else "N/A"
            print(f"{mode.upper():<15} | {score:<12.2f}" if isinstance(score, float) else f"{mode.upper():<15} | {score:<12}")
        print("="*40)
    else:
        config_path = args_cmd.config if args_cmd.config else config_map[args_cmd.mode]
        cfg = load_config(config_path)
        globals()[f"run_{args_cmd.mode}"](cfg, task=args_cmd.task)

if __name__ == "__main__":
    main()
