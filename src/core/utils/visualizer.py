import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def smooth(series, window=10):
    return series.rolling(window=window, min_periods=1).mean()

def plot_single_run(log_path, config_id="v4.0_default", output_dir=None):
    """
    Generates a comprehensive 4-panel training dashboard for a single run.
    Panels: Avg Reward | Loss | Epsilon Decay | Reward vs Loss (scatter)
    """
    if not os.path.exists(log_path):
        print(f"[PLOT] No log file at {log_path}. Run training first.")
        return

    df = pd.read_csv(log_path)
    if 'Config_ID' in df.columns:
        df = df[df['Config_ID'] == config_id]
    if df.empty:
        print(f"[PLOT] No data for config '{config_id}'")
        return

    if output_dir is None:
        output_dir = os.path.dirname(log_path)
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Training Dashboard — {config_id}", fontsize=16, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ep = df['Episode']
    reward = df['Avg_Reward'].astype(float)
    loss = df['Loss'].astype(float)
    eps = df['Epsilon'].astype(float)

    # Panel 1: Avg Reward
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, reward, alpha=0.25, color='steelblue', linewidth=0.8)
    ax1.plot(ep, smooth(reward), color='steelblue', linewidth=2, label='Smoothed')
    ax1.set_title("Average Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Avg Reward")
    ax1.axhline(y=reward.max(), color='green', linestyle='--', alpha=0.5, label=f'Best: {reward.max():.2f}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Loss
    ax2 = fig.add_subplot(gs[0, 1])
    valid_loss = loss[loss > 0]
    valid_ep = ep[loss > 0]
    ax2.plot(valid_ep, valid_loss, alpha=0.25, color='tomato', linewidth=0.8)
    ax2.plot(valid_ep, smooth(valid_loss), color='tomato', linewidth=2, label='Smoothed')
    ax2.set_title("Training Loss (MSE)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Epsilon (Exploration) Decay
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep, eps, color='orange', linewidth=2)
    ax3.fill_between(ep, eps, alpha=0.1, color='orange')
    ax3.set_title("Exploration Rate (Epsilon Decay)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Reward improvement rate (delta reward)
    ax4 = fig.add_subplot(gs[1, 1])
    delta_reward = smooth(reward).diff().fillna(0)
    colors = ['green' if x >= 0 else 'red' for x in delta_reward]
    ax4.bar(ep, delta_reward, color=colors, alpha=0.6, width=1.0)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    ax4.set_title("Reward Improvement Rate (Δ per Episode)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Δ Avg Reward")
    ax4.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f"{config_id}_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Dashboard saved → {out_path}")


def plot_training_results(log_path, output_dir=None):
    """
    Generates a multi-config comparison plot (for sweeps).
    Also generates individual dashboards for each config found in the log.
    """
    if not os.path.exists(log_path):
        print(f"[PLOT] No log file at {log_path}")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        print("[PLOT] Training log is empty.")
        return

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(log_path), "plots")
    os.makedirs(output_dir, exist_ok=True)

    configs = df['Config_ID'].unique() if 'Config_ID' in df.columns else ['default']

    # --- Multi-config Reward Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("v4.0 Sweep — Architecture Comparison", fontsize=14, fontweight='bold')

    for config in configs:
        subset = df[df['Config_ID'] == config] if 'Config_ID' in df.columns else df
        r = subset['Avg_Reward'].astype(float)
        l = subset['Loss'].astype(float)
        ep = subset['Episode']
        axes[0].plot(ep, smooth(r), label=config, linewidth=1.5)
        axes[1].plot(ep[l > 0], smooth(l[l > 0]), label=config, linewidth=1.5)

    axes[0].set_title("Avg Reward (Smoothed)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Reward")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Loss (Smoothed)")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Loss (MSE)")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    comp_path = os.path.join(output_dir, "sweep_comparison.png")
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Sweep comparison saved → {comp_path}")

    # --- Individual dashboards ---
    for config in configs:
        plot_single_run(log_path, config_id=config, output_dir=output_dir)


if __name__ == "__main__":
    plot_training_results("./models/v4/training_log.csv")
