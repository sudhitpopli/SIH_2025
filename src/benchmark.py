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

def run_sumo_default(args, task="eval"):
    """
    [MODE] SUMO DEFAULT (Native)
    Uses the hardcoded phase logic inside the .net.xml file.
    No AI, no dynamic adaptation.
    """
    print(f"\n[MODE] Running SUMO Default (Task: {task})...", flush=True)
    from core.envs.sumo_env import SUMOEnv
    env = SUMOEnv(args)
    _, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        _, _, reward, done, _ = env.step(None)
        total_reward += reward
        steps += 1
    
    avg_reward = total_reward / max(1, steps)
    print(f"[RESULT] Sumo Default Finished. Avg Reward: {avg_reward:.2f}", flush=True)
    env.close()
    return {"avg_reward": avg_reward, "total_steps": steps}

def run_indian(args, task="eval"):
    """
    [MODE] INDIAN STANDARD (Webster)
    Uses Webster's Method for static cycle optimization based on flow.
    """
    print(f"\n[MODE] Running Indian Standard (Task: {task})...", flush=True)
    from core.envs.sumo_env import SUMOEnv
    from algos.indian.webster_controller import WebsterController
    
    env = SUMOEnv(args)
    env.reset()
    controllers = [WebsterController(env, tls_id, is_static=True) for tls_id in env.tls_ids]
    
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        for c in controllers: c.step()
        _, _, reward, done, _ = env.step(None)
        total_reward += reward
        steps += 1
        
    avg_reward = total_reward / max(1, steps)
    print(f"[RESULT] Indian Standard Finished. Avg Reward: {avg_reward:.2f}", flush=True)
    env.close()
    return {"avg_reward": avg_reward, "total_steps": steps}

def run_v1(args, task="train"):
    """
    [MODE] v1.0 (LEGACY QMIX)
    Original QMIX implementation. Simple MLP architecture without recurrence.
    """
    print(f"\n[MODE] Running QMIX v1.0 (Task: {task})...", flush=True)
    from core.envs.sumo_env import SUMOEnv
    from algos.v1.trainer import QMIXTrainer
    
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

    model_path = os.path.join(args.save_dir, "v1_model.pt")
    
    if task == "eval":
        if not trainer.load_model(model_path):
            env.close()
            return None
        trainer.epsilon = 0.0

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
                trainer.train_step()
                if step % args.target_update_interval == 0:
                    trainer.update_target_networks()
                    
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / max(1, step)
        if task == "train":
            if (ep + 1) % args.print_interval == 0:
                print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} | Epsilon: {trainer.epsilon:.2f}", flush=True)
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] v1.0 Eval Finished. Avg Reward: {avg_ep_reward:.2f}", flush=True)
            
    env.close()
    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}

def run_v2(args, task="train"):
    """
    [MODE] v2.0 (IMPROVED QMIX)
    Recurrent architecture (GRU) with Truncated BPTT.
    Stable and optimized for GPU.
    """
    print(f"\n[MODE] Running QMIX v2.0 (Task: {task})...", flush=True)
    from core.envs.sumo_env import SUMOEnv
    from algos.v2.trainer import QMIXTrainerV2
    
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    env = SUMOEnv(args)
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
        h = trainer.init_hidden(env.n_agents)
        done = False
        ep_reward = 0
        step = 0
        
        while not done:
            actions, h = trainer.select_action(obs, h)
            next_state, next_obs, reward, done, _ = env.step(actions)
            if task == "train":
                trainer.store_transition(obs, state, actions, reward, next_obs, next_state, done)
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / max(1, step)

        if task == "train":
            trainer.flush_episode()
            updates_per_episode = getattr(args, 'updates_per_episode', 10)
            loss = 0
            for _ in range(updates_per_episode):
                step_loss = trainer.train_step()
                if step_loss is not None and step_loss > 0: loss = step_loss

            if (ep + 1) % args.target_update_interval == 0:
                trainer.update_target_networks()

            if (ep + 1) % args.print_interval == 0:
                print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} "
                      f"| Epsilon: {trainer.epsilon:.2f} | Loss: {float(loss):.4f}", flush=True)
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] v2.0 Eval Finished. Avg Reward: {avg_ep_reward:.2f}", flush=True)

    env.close()
    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}

def run_v3(args, task="train"):
    """
    [MODE] v3.0 (DEEP EXPERIMENTAL)
    Testing deeper architectures and different hidden dimensions.
    Formerly 'experimental'.
    """
    print(f"\n[MODE] Running QMIX v3.0 (Task: {task})...", flush=True)
    from core.envs.sumo_env import SUMOEnv
    from algos.v3.trainer import QMIXTrainerExperimental
    
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    env = SUMOEnv(args)
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

    model_path = os.path.join(args.save_dir, "v3_model.pt")
    
    if task == "eval":
        if not trainer.load_model(model_path):
            env.close()
            return None
        trainer.epsilon = 0.0

    n_episodes = args.n_episodes if task == "train" else 1
    best_reward = -float('inf')

    for ep in range(n_episodes):
        state, obs = env.reset()
        h = trainer.init_hidden(env.n_agents)
        done = False
        ep_reward = 0
        step = 0
        
        while not done:
            actions, h = trainer.select_action(obs, h)
            next_state, next_obs, reward, done, _ = env.step(actions)
            if task == "train":
                trainer.store_transition(obs, state, actions, reward, next_obs, next_state, done)
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / max(1, step)

        if task == "train":
            trainer.flush_episode()
            updates_per_episode = getattr(args, 'updates_per_episode', 10)
            loss = 0
            for _ in range(updates_per_episode):
                step_loss = trainer.train_step()
                if step_loss is not None and step_loss > 0: loss = step_loss

            if (ep + 1) % args.target_update_interval == 0:
                trainer.update_target_networks()

            if (ep + 1) % args.print_interval == 0:
                print(f"Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} "
                      f"| Epsilon: {trainer.epsilon:.2f} | Loss: {float(loss):.4f}", flush=True)
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] v3.0 Eval Finished. Avg Reward: {avg_ep_reward:.2f}", flush=True)

    env.close()
    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}

def run_v4(args, task="train", config_id="v4.0_default"):
    """
    [MODE] v4.0 (SPATIAL SPARK)
    The Ultimate Model with Neighbor-Aware GRC and Automated Sweep support.
    """
    print(f"\n[MODE] Running QMIX v4.0 Spatial (Config: {config_id}, Task: {task})...", flush=True)
    from core.envs.sumo_env import SUMOEnv
    from algos.v4.trainer import QMIXTrainerV4
    from core.utils.logger import CSVLogger
    
    device = get_device()
    env = SUMOEnv(args)
    logger = CSVLogger(args.save_dir, config_name=config_id)

    trainer = QMIXTrainerV4(
        env=env, n_agents=env.n_agents, state_dim=env.get_state_size(),
        obs_dim=env.obs_size, n_actions=env.n_actions,
        rnn_hidden_dim=args.rnn_hidden_dim, mixing_hidden_dim=args.mixer_hidden_dim,
        lr=args.lr, gamma=args.gamma,
        buffer_size=args.ep_buffer_size, batch_size=args.batch_size,
        chunk_len=args.chunk_len, device=device,
        args=args,
        epsilon_start=getattr(args, 'epsilon_start', 1.0),
        epsilon_min=getattr(args, 'epsilon_min', 0.05),
        epsilon_decay=getattr(args, 'epsilon_decay', 0.99)
    )

    model_path = os.path.join(args.save_dir, f"{config_id}_model.pt")
    if task == "eval":
        if not trainer.load_model(model_path):
            env.close()
            return None
        trainer.epsilon = 0.0

    n_episodes = args.n_episodes if task == "train" else 1
    best_reward = -float('inf')

    for ep in range(n_episodes):
        state, obs = env.reset()
        h = trainer.init_hidden(env.n_agents)
        done = False
        ep_reward = 0
        step = 0
        
        while not done:
            actions, h = trainer.select_action(obs, h)
            next_state, next_obs, reward, done, _ = env.step(actions)
            if task == "train":
                trainer.store_transition(obs, state, actions, reward, next_obs, next_state, done)
            obs, state = next_obs, next_state
            ep_reward += reward
            step += 1
            
        avg_ep_reward = ep_reward / max(1, step)

        if task == "train":
            trainer.flush_episode()
            
            updates_per_episode = getattr(args, 'updates_per_episode', 5)
            loss_list = []
            for _ in range(updates_per_episode):
                step_loss = trainer.train_step()
                if step_loss is not None:
                    loss_list.append(step_loss)
            
            avg_loss = np.mean(loss_list) if loss_list else None
            
            if (ep + 1) % args.target_update_interval == 0:
                trainer.update_target_networks()
            
            trainer.decay_epsilon()

            # Log results
            logger.log_episode(ep + 1, avg_ep_reward, avg_loss if avg_loss is not None else 0, trainer.epsilon)

            if (ep + 1) % args.print_interval == 0:
                loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "Buffer Filling"
                print(f"[{config_id}] Episode {ep+1}/{n_episodes} | Avg Reward: {avg_ep_reward:.2f} | Loss: {loss_str} | Eps: {trainer.epsilon:.2f}", flush=True)
            
            if avg_ep_reward > best_reward:
                best_reward = avg_ep_reward
                trainer.save_model(model_path)
        else:
            print(f"[RESULT] v4.0 Eval Finished. Avg Reward: {avg_ep_reward:.2f}", flush=True)

    env.close()

    if task == "train":
        # Auto-generate training dashboard after run completes
        try:
            from core.utils.visualizer import plot_single_run
            plot_single_run(
                os.path.join(args.save_dir, "training_log.csv"),
                config_id=config_id,
                output_dir=os.path.join(args.save_dir, "plots")
            )
        except Exception as e:
            print(f"[PLOT] Could not generate plot: {e}")

    return {"avg_reward": avg_ep_reward if task == "eval" else best_reward}

def run_sweep(args):
    """
    [MECHANISM: AUTO-OPTIMIZER SWEEP]
    Iterates through architectural combinations to find the best model.
    Compares: hidden_dim, num_layers, use_grc (GRU vs Linear communication)
    """
    print("\n" + "="*55)
    print(" STARTING ULTIMATE v4.0 ARCHITECTURE SWEEP")
    print("="*55)

    # --- Search grid ---
    hidden_dims        = [128, 256]
    num_layers_list    = [1, 2]
    use_grc_list       = [True, False]   # GRU-comm vs Linear-comm
    mixer_dims         = [64, 128]        # Mixing network width

    sweep_episodes     = 50              # Short burst per config
    results_leaderboard = []

    for h_dim in hidden_dims:
        for n_layers in num_layers_list:
            for use_grc in use_grc_list:
                for m_dim in mixer_dims:
                    config_id = f"v4_h{h_dim}_L{n_layers}_GRC{use_grc}_M{m_dim}"

                    # Mutate args for this run
                    args.rnn_hidden_dim    = h_dim
                    args.mixer_hidden_dim  = m_dim
                    args.num_layers        = n_layers
                    args.use_grc           = use_grc
                    args.n_episodes        = sweep_episodes

                    print(f"\n[SWEEP] Testing: {config_id}")
                    try:
                        res = run_v4(args, task="train", config_id=config_id)
                        if res:
                            results_leaderboard.append({"id": config_id, "reward": res['avg_reward']})
                    except Exception as e:
                        print(f"[ERROR] Sweep run {config_id} failed: {e}")

    # --- Final Leaderboard ---
    print("\n" + "="*55)
    print(" SWEEP COMPLETE — LEADERBOARD")
    print("="*55)
    results_leaderboard.sort(key=lambda x: x['reward'], reverse=True)
    for i, res in enumerate(results_leaderboard):
        marker = " *** WINNER ***" if i == 0 else ""
        print(f"#{i+1}: {res['id']:<40} | Reward: {res['reward']:.4f}{marker}")
    print("="*55)

    # --- Auto-graph all results ---
    from core.utils.visualizer import plot_training_results
    plot_training_results(
        os.path.join(args.save_dir, "training_log.csv"),
        output_dir=os.path.join(args.save_dir, "plots")
    )

def main():
    parser = argparse.ArgumentParser(description="SIH 2025 Traffic Benchmark & Training Suite")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["sumo_default", "v1.0", "v2.0", "v3.0", "v4.0", "indian", "compare", "sweep"])
    parser.add_argument("--task", type=str, default="eval", choices=["train", "eval", "sweep"])
    parser.add_argument("--config", type=str, help="Custom config path")
    
    args_cmd = parser.parse_args()
    
    config_map = {
        "sumo_default": "config/sumo_default.yaml",
        "v1.0": "config/v1.0.yaml",
        "v2.0": "config/v2.0.yaml",
        "v3.0": "config/v3.0.yaml",
        "v4.0": "config/v4.0.yaml",
        "indian": "config/indian_standard.yaml"
    }

    if args_cmd.mode == "compare":
        results = {}
        for mode in ["sumo_default", "indian", "v1.0", "v2.0", "v3.0"]:
            if mode not in config_map: continue
            cfg = load_config(config_map[mode])
            func_name = f"run_{mode.replace('.', '')}" if 'v' in mode else f"run_{mode}"
            results[mode] = globals()[func_name](cfg, task="eval")
        
        print("\n" + "="*40)
        print(" FINAL PERFORMANCE SCORECARD ")
        print("="*40)
        print(f"{'Model':<15} | {'Avg Reward':<12}")
        print("-" * 30)
        for mode, res in results.items():
            score = res['avg_reward'] if res else "N/A"
            print(f"{mode.upper():<15} | {score:<12.2f}" if isinstance(score, float) else f"{mode.upper():<15} | {score:<12}")
        print("="*40)
    elif args_cmd.mode == "sweep":
        cfg = load_config(config_map["v4.0"])
        run_sweep(cfg)
    else:
        config_path = args_cmd.config if args_cmd.config else config_map[args_cmd.mode]
        cfg = load_config(config_path)
        func_name = f"run_{args_cmd.mode.replace('.0', '').replace('.', '')}"
        if func_name not in globals():
            func_name = f"run_{args_cmd.mode}"
        globals()[func_name](cfg, task=args_cmd.task)

if __name__ == "__main__":
    main()
