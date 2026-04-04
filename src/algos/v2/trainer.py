import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from algos.v2.networks import RNNAgent, MixingNetworkV2
from core.utils.episode_replay_buffer import EpisodeReplayBuffer


class QMIXTrainerV2:
    """
    [MECHANISM: THE V2 TRAINER ARCHITECTURE]
    This class is the engine that actually updates the weights (learning) of the Neural Network.
    Unlike standard Q-learning which looks at isolated pictures of traffic, this trainer:
    
    1. Grabs sequential blocks of time (BPTT).
    2. 'Unrolls' the GRU memory mathematically across those blocks.
    3. Calculates how 'wrong' the agent's actions were compared to the real traffic flow.
    4. Backpropagates the error (calculus) to adjust the weights.
    """
    def __init__(self, env, n_agents, state_dim, obs_dim, n_actions,
                 rnn_hidden_dim=64, mixing_hidden_dim=32, lr=0.0003, 
                 gamma=0.99, buffer_size=500, batch_size=16,
                 chunk_len=50, device="cpu",
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.998):
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.chunk_len = chunk_len
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
        
        # Bug 1 fix: episode-based buffer instead of transition-based
        self.replay_buffer = EpisodeReplayBuffer(
            capacity=buffer_size, obs_dim=obs_dim, state_dim=state_dim,
            n_agents=n_agents, n_actions=n_actions,
            chunk_len=chunk_len, device=self.device
        )

        # Episode accumulator — collects transitions during an episode
        self._episode_data = {
            'obs': [], 'state': [], 'actions': [],
            'rewards': [], 'next_obs': [], 'next_state': [], 'dones': []
        }

        # 3. Exploration Params (configurable from YAML)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

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

    # ================================================================
    # Episode accumulation API (used by benchmark.py)
    # ================================================================
    def store_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        """Buffer a single transition during an episode."""
        self._episode_data['obs'].append(np.array(obs, dtype=np.float32))
        self._episode_data['state'].append(np.array(state, dtype=np.float32))
        self._episode_data['actions'].append(np.array(actions, dtype=np.int64))
        self._episode_data['rewards'].append(np.array([reward], dtype=np.float32))
        self._episode_data['next_obs'].append(np.array(next_obs, dtype=np.float32))
        self._episode_data['next_state'].append(np.array(next_state, dtype=np.float32))
        self._episode_data['dones'].append(np.array([float(done)], dtype=np.float32))

    def flush_episode(self):
        """Push the accumulated episode into the replay buffer and reset accumulator."""
        if len(self._episode_data['rewards']) == 0:
            return
        
        episode = {
            'obs': np.array(self._episode_data['obs']),
            'state': np.array(self._episode_data['state']),
            'actions': np.array(self._episode_data['actions']),
            'rewards': np.array(self._episode_data['rewards']),
            'next_obs': np.array(self._episode_data['next_obs']),
            'next_state': np.array(self._episode_data['next_state']),
            'dones': np.array(self._episode_data['dones']),
        }
        self.replay_buffer.store_episode(episode)
        
        # Clear accumulator
        self._episode_data = {
            'obs': [], 'state': [], 'actions': [],
            'rewards': [], 'next_obs': [], 'next_state': [], 'dones': []
        }

    # ================================================================
    # [MECHANISM: BPTT TENSOR UNROLLING]
    # This is the most complex math in the project.
    # PyTorch gives us a massive 4D block of memory: [Batch Size, Time Chunk, Agents, Sensories]
    # E.g., [16 random waves, 72 seconds each, 4 traffic lights, 13 sensors]
    # We must loop through the `72` seconds, feed the `13` sensors into the AI, and save the Result.
    # ================================================================
    def train_step(self):
        """Perform one optimization step on a mini-batch of episode chunks using Backpropagation."""
        if not self.replay_buffer.can_sample(self.batch_size):
            return 0

        # Sample: all tensors are [batch, chunk_len, ...]
        (obs, state, actions, rewards, next_obs, next_state,
         dones, mask) = self.replay_buffer.sample(self.batch_size)

        B = obs.shape[0]       # batch size
        T = obs.shape[1]       # chunk length

        # [MECHANISM: TENSOR GATHER CASTING]
        # PyTorch requires 'Actions' (which are indices like Phase 0, Phase 1) to be strict integers (long).
        actions = actions.long()  # [B, T, n_agents]

        # ---- 1. Compute current Q-values by unrolling GRU sequentially ----
        h = torch.zeros(B * self.n_agents, self.rnn_hidden_dim, device=self.device)
        all_q = []

        for t in range(T):
            obs_t = obs[:, t]  # [B, n_agents, obs_dim]
            obs_flat = obs_t.reshape(B * self.n_agents, self.obs_dim)
            q_t, h = self.agent(obs_flat, h)
            q_t = q_t.view(B, self.n_agents, -1)  # [B, n_agents, n_actions]
            all_q.append(q_t)

        all_q = torch.stack(all_q, dim=1)  # [B, T, n_agents, n_actions]

        # Gather chosen action Q-values
        chosen_q = torch.gather(
            all_q, dim=3,
            index=actions.unsqueeze(3)  # [B, T, n_agents, 1]
        ).squeeze(3)  # [B, T, n_agents]

        # Mix per timestep
        q_total_list = []
        for t in range(T):
            q_total_t = self.mixer(chosen_q[:, t], state[:, t])  # [B, 1]
            q_total_list.append(q_total_t)
        q_total = torch.stack(q_total_list, dim=1)  # [B, T, 1]

        # ---- 2. Compute target Q-values (Double DQN + sequential GRU) ----
        with torch.no_grad():
            # Online network: pick best actions
            h_online = torch.zeros(B * self.n_agents, self.rnn_hidden_dim, device=self.device)
            all_next_q_online = []

            for t in range(T):
                next_obs_t = next_obs[:, t]
                next_obs_flat = next_obs_t.reshape(B * self.n_agents, self.obs_dim)
                next_q_t, h_online = self.agent(next_obs_flat, h_online)
                next_q_t = next_q_t.view(B, self.n_agents, -1)
                all_next_q_online.append(next_q_t)

            all_next_q_online = torch.stack(all_next_q_online, dim=1)  # [B, T, n_agents, n_actions]

            # [MECHANISM: DOUBLE DQN ACTION SELECTION]
            # Use the "Smart" Online brain to pick what it thinks is the very best phase for the future state.
            best_actions = all_next_q_online.argmax(dim=3, keepdim=True)  # [B, T, n_agents, 1]

            # Target network: evaluate those actions
            h_target = torch.zeros(B * self.n_agents, self.rnn_hidden_dim, device=self.device)
            all_next_q_target = []

            for t in range(T):
                next_obs_t = next_obs[:, t]
                next_obs_flat = next_obs_t.reshape(B * self.n_agents, self.obs_dim)
                next_q_t, h_target = self.target_agent(next_obs_flat, h_target)
                next_q_t = next_q_t.view(B, self.n_agents, -1)
                all_next_q_target.append(next_q_t)

            all_next_q_target = torch.stack(all_next_q_target, dim=1)  # [B, T, n_agents, n_actions]

            # Gather target Q-values at online-selected actions
            target_chosen_q = torch.gather(
                all_next_q_target, dim=3, index=best_actions
            ).squeeze(3)  # [B, T, n_agents]

            # Mix targets per timestep
            target_q_total_list = []
            for t in range(T):
                tq = self.target_mixer(target_chosen_q[:, t], next_state[:, t])
                target_q_total_list.append(tq)
            target_q_total = torch.stack(target_q_total_list, dim=1)  # [B, T, 1]

            # Bellman target
            y = rewards + self.gamma * (1 - dones) * target_q_total

        # ---- 3. Masked loss & optimize ----
        td_error = (q_total - y) ** 2
        masked_td = td_error * mask  # Zero out padding timesteps
        loss = masked_td.sum() / mask.sum()  # Mean over valid timesteps

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
