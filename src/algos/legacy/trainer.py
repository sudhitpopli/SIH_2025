import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from core.utils.replay_buffer import ReplayBuffer
from algos.legacy.networks import AgentQNetwork, MixingNetwork

class QMIXTrainer:
    """
    Standard QMIX Trainer for legacy model comparison.
    Handles experience replay and centralized training of decentralized policies.
    """
    def __init__(self, env, n_agents, state_dim, obs_dim, n_actions,
                 hidden_dim=64, lr=0.0005, gamma=0.99,
                 buffer_size=10000, batch_size=32, device="cpu"):

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size, obs_dim=obs_dim, state_dim=state_dim,
            n_agents=n_agents, n_actions=n_actions, device=self.device
        )

        # Individual agent networks (shared parameters)
        self.agent_q = AgentQNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_agent_q = AgentQNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())

        # Mixing network
        self.mixing_net = MixingNetwork(n_agents, state_dim, hidden_dim).to(self.device)
        self.target_mixing_net = MixingNetwork(n_agents, state_dim, hidden_dim).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(list(self.agent_q.parameters()) +
                                    list(self.mixing_net.parameters()), lr=lr)

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, obs):
        """epsilon-greedy action selection for all agents"""
        actions = []
        for agent_obs in obs:
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.agent_q(obs_tensor)
                actions.append(torch.argmax(q_values).item())
        return actions

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0

        # Sample minibatch
        obs, state, actions, rewards, next_obs, next_state, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q-values
        agent_qs = []
        for i in range(self.n_agents):
            q_values = self.agent_q(obs[:, i, :])
            q_selected = q_values.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            agent_qs.append(q_selected)
        agent_qs = torch.stack(agent_qs, dim=1)

        q_total = self.mixing_net(agent_qs, state)

        # Compute target Q-values
        with torch.no_grad():
            next_agent_qs = []
            for i in range(self.n_agents):
                q_values = self.target_agent_q(next_obs[:, i, :])
                q_selected = q_values.max(1)[0]
                next_agent_qs.append(q_selected)
            next_agent_qs = torch.stack(next_agent_qs, dim=1)

            target_q_total = self.target_mixing_net(next_agent_qs, next_state)
            targets = rewards + self.gamma * (1 - dones) * target_q_total.squeeze(1)

        # Loss
        loss = nn.MSELoss()(q_total.squeeze(1), targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.agent_q.parameters()) +
                                       list(self.mixing_net.parameters()), 10)
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

    def update_target_networks(self):
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

    def save_model(self, path):
        """Save both agent and mixer weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'agent_q': self.agent_q.state_dict(),
            'mixing_net': self.mixing_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"[INFO] Legacy Model saved to {path}")

    def load_model(self, path):
        """Load weights for evaluation or continued training"""
        if not os.path.exists(path):
            print(f"[ERROR] No model found at {path}")
            return False
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_q.load_state_dict(checkpoint['agent_q'])
        self.mixing_net.load_state_dict(checkpoint['mixing_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        print(f"[INFO] Legacy Model loaded from {path}")
        return True
