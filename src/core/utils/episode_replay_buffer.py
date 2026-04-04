# episode_replay_buffer.py
# Stores full episodes for sequential GRU unrolling in QMIX v2.
# Uses truncated BPTT: samples contiguous chunks from episodes.

import numpy as np
import torch


class EpisodeReplayBuffer:
    """
    [MECHANISM: TIME-TRAVEL MEMORY BANK]
    Replay buffer that stores full chronological episodes for recurrent QMIX training.
    
    In standard Reinforcement Learning, the memory mixes isolated moments together (e.g. 
    State A -> Action B -> Reward 5). But our Agent has a GRU memory cell. It needs continuous time.
    Instead of isolated steps, this buffer stores entire 720-second waves of physical traffic flow.
    
    Supports truncated backpropagation through time (TBPTT) by sampling
    contiguous chunks of `chunk_len` timesteps from stored episodes.
    """

    def __init__(self, capacity, obs_dim, state_dim, n_agents, n_actions,
                 chunk_len=50, device="cpu"):
        """
        Args:
            capacity:   Max number of episodes to store.
            obs_dim:    Observation dimension per agent.
            state_dim:  Global state dimension.
            n_agents:   Number of agents.
            n_actions:  Number of actions per agent.
            chunk_len:  Length of contiguous chunks sampled for training (TBPTT).
            device:     Torch device for returned tensors.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.chunk_len = chunk_len
        self.device = device

        # Circular buffer of episode dicts
        self.buffer = []
        self.ptr = 0
        self.size = 0

    def store_episode(self, episode):
        """
        Store a complete episode.
        
        Args:
            episode: dict with keys:
                'obs':        np.array [T, n_agents, obs_dim]
                'state':      np.array [T, state_dim]
                'actions':    np.array [T, n_agents]  (int)
                'rewards':    np.array [T, 1]
                'next_obs':   np.array [T, n_agents, obs_dim]
                'next_state': np.array [T, state_dim]
                'dones':      np.array [T, 1]
        """
        ep = {
            'obs': np.array(episode['obs'], dtype=np.float32),
            'state': np.array(episode['state'], dtype=np.float32),
            'actions': np.array(episode['actions'], dtype=np.int64),
            'rewards': np.array(episode['rewards'], dtype=np.float32),
            'next_obs': np.array(episode['next_obs'], dtype=np.float32),
            'next_state': np.array(episode['next_state'], dtype=np.float32),
            'dones': np.array(episode['dones'], dtype=np.float32),
        }

        if self.size < self.capacity:
            self.buffer.append(ep)
        else:
            self.buffer[self.ptr] = ep

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        [MECHANISM: CONTINUOUS WAVE SAMPLING]
        When the AI wants to learn, we don't give it back the whole 720-second wave (which would crash 
        your GPU's VRAM). Instead, we slice out a random `chunk_len` piece (e.g. 72 seconds).
        The GRU unrolls strictly forward across these 72 seconds, perfectly learning how a traffic queue 
        built up and vanished.
        
        Returns padded chunks if the slice hits the end of an episode early.
        
        Returns:
            obs:        [batch, chunk_len, n_agents, obs_dim]
            state:      [batch, chunk_len, state_dim]
            actions:    [batch, chunk_len, n_agents]       (long)
            rewards:    [batch, chunk_len, 1]
            next_obs:   [batch, chunk_len, n_agents, obs_dim]
            next_state: [batch, chunk_len, state_dim]
            dones:      [batch, chunk_len, 1]
            mask:       [batch, chunk_len, 1]               (1 = valid, 0 = pad)
        """
        # Pick random episodes
        ep_indices = np.random.choice(self.size, batch_size, replace=True)

        batch_obs = []
        batch_state = []
        batch_actions = []
        batch_rewards = []
        batch_next_obs = []
        batch_next_state = []
        batch_dones = []
        batch_mask = []

        for idx in ep_indices:
            ep = self.buffer[idx]
            ep_len = len(ep['rewards'])

            # Pick a random start index for the chunk
            if ep_len <= self.chunk_len:
                start = 0
                actual_len = ep_len
            else:
                start = np.random.randint(0, ep_len - self.chunk_len + 1)
                actual_len = self.chunk_len

            # Extract chunk
            obs_chunk = ep['obs'][start:start + actual_len]
            state_chunk = ep['state'][start:start + actual_len]
            actions_chunk = ep['actions'][start:start + actual_len]
            rewards_chunk = ep['rewards'][start:start + actual_len]
            next_obs_chunk = ep['next_obs'][start:start + actual_len]
            next_state_chunk = ep['next_state'][start:start + actual_len]
            dones_chunk = ep['dones'][start:start + actual_len]

            # Pad if episode is shorter than chunk_len
            pad_len = self.chunk_len - actual_len
            if pad_len > 0:
                obs_chunk = np.pad(obs_chunk,
                    [(0, pad_len), (0, 0), (0, 0)], mode='constant')
                state_chunk = np.pad(state_chunk,
                    [(0, pad_len), (0, 0)], mode='constant')
                actions_chunk = np.pad(actions_chunk,
                    [(0, pad_len), (0, 0)], mode='constant')
                rewards_chunk = np.pad(rewards_chunk,
                    [(0, pad_len), (0, 0)], mode='constant')
                next_obs_chunk = np.pad(next_obs_chunk,
                    [(0, pad_len), (0, 0), (0, 0)], mode='constant')
                next_state_chunk = np.pad(next_state_chunk,
                    [(0, pad_len), (0, 0)], mode='constant')
                dones_chunk = np.pad(dones_chunk,
                    [(0, pad_len), (0, 0)], mode='constant',
                    constant_values=1.0)  # treat padding as "done"

            # Build mask: 1 for real timesteps, 0 for padding
            mask = np.zeros((self.chunk_len, 1), dtype=np.float32)
            mask[:actual_len] = 1.0

            batch_obs.append(obs_chunk)
            batch_state.append(state_chunk)
            batch_actions.append(actions_chunk)
            batch_rewards.append(rewards_chunk)
            batch_next_obs.append(next_obs_chunk)
            batch_next_state.append(next_state_chunk)
            batch_dones.append(dones_chunk)
            batch_mask.append(mask)

        return (
            torch.tensor(np.array(batch_obs), device=self.device),
            torch.tensor(np.array(batch_state), device=self.device),
            torch.tensor(np.array(batch_actions), device=self.device, dtype=torch.long),
            torch.tensor(np.array(batch_rewards), device=self.device),
            torch.tensor(np.array(batch_next_obs), device=self.device),
            torch.tensor(np.array(batch_next_state), device=self.device),
            torch.tensor(np.array(batch_dones), device=self.device),
            torch.tensor(np.array(batch_mask), device=self.device),
        )

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def __len__(self):
        return self.size
