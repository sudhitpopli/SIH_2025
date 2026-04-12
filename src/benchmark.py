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
    
    avg_reward = total_reward / max(1, steps)
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
        
    avg_reward = total_reward / max(1, steps)
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
            
        avg_ep_reward = ep_reward / max(1, step)
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
    """
    [MECHANISM: MULTI-AGENT TRAINING PIPELINE]
    This function handles the overarching loop connecting the SUMO environment to the Neural Network.
    - `env.step()` advances the physical traffic cars.
    - `trainer.store_transition()` saves the snapshot of what happened.
    - `trainer.train_step()` runs calculus (Backpropagation) to improve the AI's future choices.
    """
    print(f"\n[MODE] Running Improved QMIX V2 (Task: {task})...")
    from core.envs.sumo_env import SUMOEnv
    from algos.v2.trainer import QMIXTrainerV2
    
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    env = SUMOEnv(args)

    # Episode buffer capacity and chunk length
    ep_buffer_size = getattr(args, 'ep_buffer_size', 500)
    chunk_len = getattr(args, 'chunk_len', 50)

    trainer = QMIXTrainerV2(
        env=env, n_agents=env.n_agents, state_dim=env.get_state_size(),
        obs_dim=env.obs_size, n_actions=env.n_actions,
        rnn_hidden_dim=args.rnn_hidden_dim, mixing_hidden_dim=args.mixer_hidden_dim,
        lr=args.lr, gamma=args.gamma,
        buffer_size=ep_buffer_size, batch_size=args.batch_size,
        chunk_len=chunk_len, device=device,
        epsilon_start=getattr(args, 'epsilon_start', 1.0),
        epsilon_min=getattr(args, 'epsilon_min', 0.05),
        epsilon_decay=getattr(args, 'epsilon_decay', 0.995)
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
            # 1. Neural Network evaluates current traffic (Queue depth, speeds) and picks phase
            actions, h = trainer.select_action(obs, h)

            # 2. SUMO Physics Engine executes the phase (accounting for 3-sec yellow delays) 
            # and returns the new traffic state 5 seconds later
            next_state, next_obs, reward, done, _ = env.step(actions)
            
            if task == "train":
                # [MECHANISM: CONTINUOUS EXPERIENCE REPLAY]
                # In RL, an AI learns by re-watching its past "memories" (Replay Buffer).
                # Because we use a GRU (a type of RNN with memory), we MUST store memories 
                # exactly in the chronological order they happened, so the AI can learn temporal
                # wave causality (e.g. "I turned it green, and 20 seconds later the queue cleared").
                trainer.store_transition(obs, state, actions, reward,
                                         next_obs, next_state, done)
            
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / max(1, step)

        if task == "train":
            # [CRITICAL FIX: TRUNCATED BPTT BATCHING]
            # Once the 720-step episode is done, we flush the exact historical trajectory
            # into the Replay Buffer. 
            trainer.flush_episode()

            # [MECHANISM: EPISODE BATCH TRAINING (BPTT)]
            # PyTorch learns best when it looks at massive batches of data at once (parallelism).
            # Instead of stopping the simulation to train after every single second, we wait until
            # an entire 720-step episode finishes. Then, the computer pauses SUMO, grabs 10 massive 
            # random historical chunks of traffic, and optimizes the neural architecture all at once.
            # This is called Truncated Backpropagation Through Time.
            updates_per_episode = getattr(args, 'updates_per_episode', 10)
            loss = 0
            for _ in range(updates_per_episode):
                step_loss = trainer.train_step()
                if step_loss is not None and step_loss > 0:
                    loss = step_loss

            # [MECHANISM: DOUBLE Q-LEARNING (TARGET NETWORK)]
            # If the AI uses its own active brain to score its own actions, it becomes delusional
            # and wildly overestimates how "good" a traffic state is. We keep a 2nd "frozen" copy 
            # of the brain (`target_network`) to objectively score the state. Every N episodes, 
            # we copy the active brain into the frozen brain to update its understanding.
            if (ep + 1) % args.target_update_interval == 0:
                trainer.update_target_networks()

            if (ep + 1) % args.print_interval == 0:
                print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} "
                      f"| Epsilon: {trainer.epsilon:.2f} | Loss: {float(loss):.4f}", flush=True)
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] V2 Eval Finished. Avg Reward: {avg_ep_reward:.2f}", flush=True)

    env.close()
    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}
def run_experimental(args, task="train"):
    """
    [MECHANISM: MULTI-AGENT TRAINING PIPELINE]
    This function handles the overarching loop connecting the SUMO environment to the Neural Network.
    - `env.step()` advances the physical traffic cars.
    - `trainer.store_transition()` saves the snapshot of what happened.
    - `trainer.train_step()` runs calculus (Backpropagation) to improve the AI's future choices.
    """
    print(f"\n[MODE] Running Improved QMIX V2 (Task: {task})...")
    from core.envs.sumo_env import SUMOEnv
    from algos.experimental.trainer import QMIXTrainerExperimental
    
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    env = SUMOEnv(args)

    # Episode buffer capacity and chunk length
    ep_buffer_size = getattr(args, 'ep_buffer_size', 500)
    chunk_len = getattr(args, 'chunk_len', 50)

    trainer = QMIXTrainerExperimental(
        env=env, n_agents=env.n_agents, state_dim=env.get_state_size(),
        obs_dim=env.obs_size, n_actions=env.n_actions,
        rnn_hidden_dim=args.rnn_hidden_dim, mixing_hidden_dim=args.mixer_hidden_dim,
        lr=args.lr, gamma=args.gamma,
        buffer_size=ep_buffer_size, batch_size=args.batch_size,
        chunk_len=chunk_len, device=device,
        epsilon_start=getattr(args, 'epsilon_start', 1.0),
        epsilon_min=getattr(args, 'epsilon_min', 0.05),
        epsilon_decay=getattr(args, 'epsilon_decay', 0.995)
    )

    model_path = os.path.join(args.save_dir, "experimental_model.pt")
    
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
            # 1. Neural Network evaluates current traffic (Queue depth, speeds) and picks phase
            actions, h = trainer.select_action(obs, h)

            # 2. SUMO Physics Engine executes the phase (accounting for 3-sec yellow delays) 
            # and returns the new traffic state 5 seconds later
            next_state, next_obs, reward, done, _ = env.step(actions)
            
            if task == "train":
                # [MECHANISM: CONTINUOUS EXPERIENCE REPLAY]
                # In RL, an AI learns by re-watching its past "memories" (Replay Buffer).
                # Because we use a GRU (a type of RNN with memory), we MUST store memories 
                # exactly in the chronological order they happened, so the AI can learn temporal
                # wave causality (e.g. "I turned it green, and 20 seconds later the queue cleared").
                trainer.store_transition(obs, state, actions, reward,
                                         next_obs, next_state, done)
            
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / max(1, step)

        if task == "train":
            # [CRITICAL FIX: TRUNCATED BPTT BATCHING]
            # Once the 720-step episode is done, we flush the exact historical trajectory
            # into the Replay Buffer. 
            trainer.flush_episode()

            # [MECHANISM: EPISODE BATCH TRAINING (BPTT)]
            # PyTorch learns best when it looks at massive batches of data at once (parallelism).
            # Instead of stopping the simulation to train after every single second, we wait until
            # an entire 720-step episode finishes. Then, the computer pauses SUMO, grabs 10 massive 
            # random historical chunks of traffic, and optimizes the neural architecture all at once.
            # This is called Truncated Backpropagation Through Time.
            updates_per_episode = getattr(args, 'updates_per_episode', 10)
            loss = 0
            for _ in range(updates_per_episode):
                step_loss = trainer.train_step()
                if step_loss is not None and step_loss > 0:
                    loss = step_loss

            # [MECHANISM: DOUBLE Q-LEARNING (TARGET NETWORK)]
            # If the AI uses its own active brain to score its own actions, it becomes delusional
            # and wildly overestimates how "good" a traffic state is. We keep a 2nd "frozen" copy 
            # of the brain (`target_network`) to objectively score the state. Every N episodes, 
            # we copy the active brain into the frozen brain to update its understanding.
            if (ep + 1) % args.target_update_interval == 0:
                trainer.update_target_networks()

            if (ep + 1) % args.print_interval == 0:
                print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} "
                      f"| Epsilon: {trainer.epsilon:.2f} | Loss: {float(loss):.4f}", flush=True)
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] Experimental Eval Finished. Avg Reward: {avg_ep_reward:.2f}", flush=True)

    env.close()
    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}


def main():
    parser = argparse.ArgumentParser(description="SIH 2025 Traffic Benchmark & Training Suite")
    parser.add_argument("--mode", type=str, required=True, choices=["native", "legacy", "v2", "indian", "experimental", "compare"])
    parser.add_argument("--task", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--config", type=str, help="Custom config path")
    
    args_cmd = parser.parse_args()
    
    config_map = {
        "native": "config/native.yaml",
        "legacy": "config/legacy_qmix.yaml",
        "v2": "config/improved_qmix.yaml",
        "indian": "config/indian_standard.yaml",
        "experimental": "config/experimental.yaml"
    }

    if args_cmd.mode == "compare":
        results = {}
        for mode in ["native", "indian", "legacy", "v2", "experimental"]:
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
