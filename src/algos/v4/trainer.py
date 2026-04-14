import torch as th
import torch.nn as thnn
import torch.optim as optim
import numpy as np
import os
import random
from collections import deque
from .networks import RNNAgentV4, MixingNetworkV4

class ReplayBufferV4:
    def __init__(self, capacity, chunk_len, alpha=0.6):
        self.capacity = capacity
        self.chunk_len = chunk_len
        self.alpha = alpha  # Prioritization exponent (0 = uniform, 1 = fully prioritized)
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.current_episode = []
        self._max_priority = 1.0  # New episodes start with max priority

    def store_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        self.current_episode.append((obs, state, actions, reward, next_obs, next_state, done))

    def flush_episode(self):
        if len(self.current_episode) >= self.chunk_len:
            if self.size < self.capacity:
                self.buffer.append(self.current_episode)
            else:
                self.buffer[self.ptr] = self.current_episode
            # New episodes get max priority so they're sampled at least once
            self.priorities[self.ptr] = self._max_priority
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        self.current_episode = []

    def sample(self, batch_size):
        # Compute sampling probabilities from priorities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        self.last_indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        episodes = [self.buffer[i] for i in self.last_indices]
        
        batch = []
        for ep in episodes:
            start = random.randint(0, len(ep) - self.chunk_len)
            batch.append(ep[start:start + self.chunk_len])
        return batch

    def update_priorities(self, indices, td_errors):
        """Update priorities after a training step. td_errors: [batch_size] numpy array."""
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + 1e-6)  # Small epsilon to prevent zero priority
            self.priorities[idx] = p
            self._max_priority = max(self._max_priority, p)

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def __len__(self):
        return self.size

class QMIXTrainerV4:
    def __init__(self, env, n_agents, state_dim, obs_dim, n_actions, 
                 rnn_hidden_dim, mixing_hidden_dim, lr, gamma, buffer_size, batch_size, 
                 chunk_len, device, args, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.99):
        
        self.env = env
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        
        # Args for network initialization
        from types import SimpleNamespace
        self.args = SimpleNamespace(
            n_agents=n_agents, state_dim=state_dim, n_actions=n_actions,
            rnn_hidden_dim=rnn_hidden_dim, mixing_hidden_dim=mixing_hidden_dim,
            num_layers=getattr(args, 'num_layers', 2),
            use_grc=getattr(args, 'use_grc', True)
        )

        # Networks
        self.agent = RNNAgentV4(obs_dim, self.args).to(device)
        self.mixer = MixingNetworkV4(self.args).to(device)
        
        self.target_agent = RNNAgentV4(obs_dim, self.args).to(device)
        self.target_mixer = MixingNetworkV4(self.args).to(device)
        self.update_target_networks()

        self.optimizer = optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()), lr=lr
        )

        self.replay_buffer = ReplayBufferV4(buffer_size, chunk_len)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Map agent indices to their neighbor indices
        self.neighbor_idx_map = self._build_neighbor_idx_map()
        self.adj = self._build_adj_matrix()

    def _build_adj_matrix(self):
        """Construct a static adjacency weight matrix for vectorized mean-pooling."""
        adj = th.zeros(self.n_agents, self.n_agents).to(self.device)
        for i, neighbors in self.neighbor_idx_map.items():
            if neighbors:
                adj[i, neighbors] = 1.0 / len(neighbors)
        return adj

    def _build_neighbor_idx_map(self):
        """Map agent integer ID to a list of neighbor integer IDs."""
        mapping = {}
        tls_to_idx = {tls: i for i, tls in enumerate(self.env.tls_ids)}
        for tls, neighbors in self.env.neighbor_map.items():
            idx = tls_to_idx[tls]
            neighbor_indices = [tls_to_idx[n] for n in neighbors if n in tls_to_idx]
            mapping[idx] = neighbor_indices
        return mapping

    def update_target_networks(self):
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def store_transition(self, obs, state, actions, reward, next_obs, next_state, done):
        self.replay_buffer.store_transition(obs, state, actions, reward, next_obs, next_state, done)

    def flush_episode(self):
        self.replay_buffer.flush_episode()

    def init_hidden(self, batch_size=1):
        return self.agent.init_hidden().expand(batch_size, -1)

    def select_action(self, obs, hidden_states):
        """
        obs: list of per-agent numpy arrays
        hidden_states: [n_agents, hidden_dim]
        """
        obs_tensor = th.FloatTensor(np.array(obs)).to(self.device) # [n_agents, obs_dim]
        
        with th.no_grad():
            # 1. Vectorized Pass: Encode and Aggregate Neighbors
            all_encoded = self.agent.encoder(obs_tensor) # [n_agents, rnn_hidden_dim]
            neighbor_msg = th.matmul(self.adj, all_encoded) # [n_agents, rnn_hidden_dim]
            
            # 2. Run all agents through the network in parallel
            q_vals, next_h = self.agent(obs_tensor, hidden_states, neighbor_msg)

        if random.random() < self.epsilon:
            actions = [random.randint(0, self.n_actions-1) for _ in range(self.n_agents)]
        else:
            actions = q_vals.argmax(dim=-1).tolist()
            
        return actions, next_h

    def decay_epsilon(self):
        """Should be called once per episode, not per step."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        sampled_indices = self.replay_buffer.last_indices
        
        # Batch preparation [batch, chunk, n_agents, ...]
        # For simplicity, we process one step at a time in the chunk for RNN hidden state carry-over
        
        total_loss = 0
        ep_td_errors = np.zeros(self.batch_size, dtype=np.float32)
        h = self.init_hidden(self.batch_size * self.n_agents).view(self.batch_size, self.n_agents, -1)
        target_h = self.init_hidden(self.batch_size * self.n_agents).view(self.batch_size, self.n_agents, -1)

        # Pre-calculate tensors for the whole chunk
        obs_chunk = [] # [chunk, batch, n_agents, obs_dim]
        state_chunk = []
        action_chunk = []
        reward_chunk = []
        done_chunk = []
        
        for t in range(self.chunk_len):
            obs_t = th.FloatTensor(np.array([[ep[t][0][i] for i in range(self.n_agents)] for ep in batch])).to(self.device)
            state_t = th.FloatTensor(np.array([ep[t][1] for ep in batch])).to(self.device)
            action_t = th.LongTensor(np.array([[ep[t][2][i] for i in range(self.n_agents)] for ep in batch])).to(self.device)
            reward_t = th.FloatTensor(np.array([ep[t][3] for ep in batch])).to(self.device)
            done_t = th.FloatTensor(np.array([ep[t][6] for ep in batch])).to(self.device)
            obs_chunk.append(obs_t)
            state_chunk.append(state_t)
            action_chunk.append(action_t)
            reward_chunk.append(reward_t)
            done_chunk.append(done_t)

        # Optimization loop over chunk
        for t in range(self.chunk_len - 1):
            # Current step data
            obs = obs_chunk[t] # [batch, n_agents, obs_dim]
            actions = action_chunk[t]
            state = state_chunk[t]
            
            # Next step data (for target)
            next_obs = obs_chunk[t+1]
            next_state = state_chunk[t+1]
            reward = reward_chunk[t].unsqueeze(1)
            done = done_chunk[t].unsqueeze(1)

            # 1. Vectorized Forward Pass (Agent)
            # Parallel encode all agents in all batches
            obs_reshaped = obs.view(-1, self.obs_dim)
            all_encoded = self.agent.encoder(obs_reshaped).view(self.batch_size, self.n_agents, -1)
            
            # Matrix Multiply to get all neighbor sums in one go
            neighbor_msg = th.matmul(self.adj, all_encoded).view(self.batch_size * self.n_agents, -1)
            
            # Calculate all agent Qs and hidden states at once
            h_in = h.view(self.batch_size * self.n_agents, -1)
            qs, h_next = self.agent(obs_reshaped, h_in, neighbor_msg)
            
            qs = qs.view(self.batch_size, self.n_agents, -1)
            h = h_next.view(self.batch_size, self.n_agents, -1).detach()
            
            chosen_qs = th.gather(qs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1) # [batch, n_agents]
            q_tot = self.mixer(chosen_qs, state)

            # 2. Vectorized Target Pass
            with th.no_grad():
                next_obs_reshaped = next_obs.view(-1, self.obs_dim)
                all_encoded_target = self.target_agent.encoder(next_obs_reshaped).view(self.batch_size, self.n_agents, -1)
                
                neighbor_msg_target = th.matmul(self.adj, all_encoded_target).view(self.batch_size * self.n_agents, -1)
                
                target_h_in = target_h.view(self.batch_size * self.n_agents, -1)
                target_qs, target_h_next = self.target_agent(next_obs_reshaped, target_h_in, neighbor_msg_target)
                
                target_qs = target_qs.view(self.batch_size, self.n_agents, -1)
                target_h = target_h_next.view(self.batch_size, self.n_agents, -1).detach()
                
                max_target_qs = target_qs.max(dim=-1)[0]
                q_tot_target = self.target_mixer(max_target_qs, next_state)
                
                y_dqn = reward + self.gamma * (1 - done) * q_tot_target

            loss = thnn.MSELoss()(q_tot, y_dqn)
            self.optimizer.zero_grad()
            loss.backward()
            thnn.utils.clip_grad_norm_(list(self.agent.parameters()) + list(self.mixer.parameters()), 10)
            self.optimizer.step()
            total_loss += loss.item()
            
            # Track TD-error per sample for PER priority update
            with th.no_grad():
                td_errors = (q_tot - y_dqn).abs().mean(dim=-1).squeeze(-1).cpu().numpy()
                ep_td_errors += td_errors

        # Update PER priorities with mean absolute TD-error over the chunk
        self.replay_buffer.update_priorities(sampled_indices, ep_td_errors / (self.chunk_len - 1))
        return total_loss / self.chunk_len

    def save_model(self, path):
        th.save({
            'agent': self.agent.state_dict(),
            'mixer': self.mixer.state_dict()
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = th.load(path)
            self.agent.load_state_dict(checkpoint['agent'])
            self.mixer.load_state_dict(checkpoint['mixer'])
            return True
        return False
