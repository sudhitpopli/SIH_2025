import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from algos.v2.networks import RNNAgent, MixingNetworkV2
from core.utils.replay_buffer import ReplayBuffer

class QMIXTrainerV2:
    """
    Advanced QMIX Trainer for version 2.
    Supports recurrent agent networks and state-conditioned mixing.
    """
    def __init__(self, env, n_agents, state_dim, obs_dim, n_actions,
                 rnn_hidden_dim=64, mixing_hidden_dim=32, lr=0.0003, 
                 gamma=0.99, buffer_size=10000, batch_size=32, device="cpu"):
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rnn_hidden_dim = rnn_hidden_dim

        # 1. Networks
        # Shared parameters for all agents
        self.agent = RNNAgent(obs_dim, rnn_hidden_dim, n_actions).to(self.device)
        self.target_agent = RNNAgent(obs_dim, rnn_hidden_dim, n_actions).to(self.device)
        self.target_agent.load_state_dict(self.agent.state_dict())

        # Mixer
        self.mixer = MixingNetworkV2(n_agents, state_dim, mixing_hidden_dim).to(self.device)
        self.target_mixer = MixingNetworkV2(n_agents, state_dim, mixing_hidden_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # 2. Optimizer & Storage
        self.params = list(self.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size, obs_dim=obs_dim, state_dim=state_dim,
            n_agents=n_agents, n_actions=n_actions, device=self.device
        )

        # 3. Exploration Params
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def init_hidden(self, batch_size=1):
        """Initialize hidden states for GRU"""
        return torch.zeros(batch_size, self.n_agents, self.rnn_hidden_dim).to(self.device)

    def select_action(self, obs, hidden_state):
        """Select actions using the recurrent agent network"""
        # obs: [n_agents, obs_dim]
        # hidden_state: [1, n_agents, rnn_hidden_dim]
        
        obs_tensor = torch.FloatTensor(obs).to(self.device) # [n_agents, obs_dim]
        h_in = hidden_state.view(-1, self.rnn_hidden_dim) # [n_agents, rnn_hidden_dim]
        
        # Forward through RNN
        with torch.no_grad():
            q_values, h_out = self.agent(obs_tensor, h_in)
            
        # Re-pack hidden state
        h_out = h_out.view(1, self.n_agents, self.rnn_hidden_dim)
        
        # Epsilon-greedy
        actions = []
        q_vals_np = q_values.cpu().numpy()
        
        for i in range(self.n_agents):
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                actions.append(np.argmax(q_vals_np[i]))
        
        return actions, h_out

    def train_step(self):
        """Perform one optimization step on a mini-batch of transitions"""
        if len(self.replay_buffer) < self.batch_size:
            return 0

        # Sample batch
        obs, state, actions, rewards, next_obs, next_state, dones = self.replay_buffer.sample(self.batch_size)

        # 1. Compute current Q-values
        # RNN starts with zero hidden state for each transition
        h_zeros = torch.zeros(self.batch_size * self.n_agents, self.rnn_hidden_dim).to(self.device)
        
        obs_flat = obs.view(-1, self.obs_dim)
        agent_outs, _ = self.agent(obs_flat, h_zeros)
        agent_outs = agent_outs.view(self.batch_size, self.n_agents, -1)
        
        # Gather chosen actions
        chosen_q = torch.gather(agent_outs, dim=2, index=actions.unsqueeze(2)).squeeze(2)
        
        # Mix
        q_total = self.mixer(chosen_q, state)

        # 2. Compute target Q-values
        with torch.no_grad():
            next_obs_flat = next_obs.view(-1, self.obs_dim)
            next_agent_outs, _ = self.target_agent(next_obs_flat, h_zeros)
            next_agent_outs = next_agent_outs.view(self.batch_size, self.n_agents, -1)
            
            # Max action for next step
            cur_max_q = next_agent_outs.max(dim=2)[0]
            
            # Target Mix
            target_q_total = self.target_mixer(cur_max_q, next_state)
            
            # Bellman
            y = rewards + self.gamma * (1 - dones) * target_q_total

        # 3. Loss & Optimize
        loss = nn.MSELoss()(q_total, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()

        # Update exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_networks(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save_model(self, path):
        """Save V2 model with GRU weights and mixer state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'agent': self.agent.state_dict(),
            'mixer': self.mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"[INFO] V2 Model saved to {path}")

    def load_model(self, path):
        """Load for inference or continued training"""
        if not os.path.exists(path):
            print(f"[ERROR] No model found at {path}")
            return False
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent'])
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        print(f"[INFO] V2 Model loaded from {path}")
        return True
