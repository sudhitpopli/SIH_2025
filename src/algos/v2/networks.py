import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    """
    Recurrent Agent Network for QMIX v2.
    Adds memory (GRU) to handle partial observability in traffic flows.
    """
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(RNNAgent, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        # Initial hidden state (batch_size will be provided during forward)
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class MixingNetworkV2(nn.Module):
    """
    Centralized Mixer with Hypernetworks. 
    Enforces monotonicity via non-negative weights generated from the global state.
    """
    def __init__(self, n_agents, state_dim, mixing_hidden_dim=32):
        super(MixingNetworkV2, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_hidden_dim

        # Hypernetworks to generate weights and biases
        self.hyper_w_1 = nn.Linear(state_dim, self.embed_dim * n_agents)
        self.hyper_w_final = nn.Linear(state_dim, self.embed_dim)

        self.hyper_b_1 = nn.Linear(state_dim, self.embed_dim)
        # V(s) - state-dependent baseline
        self.V = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: [batch, n_agents]
        states: [batch, state_dim]
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer weights
        w1 = torch.abs(self.hyper_w_1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)

        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Final layer weights
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias (V)
        v = self.V(states).view(-1, 1, 1)

        # Q_total computation
        y = torch.bmm(hidden, w_final) + v
        q_total = y.view(bs, -1)
        
        return q_total
