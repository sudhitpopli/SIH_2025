import torch as th
import torch.nn as thnn
import torch.nn.functional as F

class RNNAgentV4(thnn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentV4, self).__init__()
        self.args = args
        
        # [MECHANISM: DYNAMIC ENCODER]
        # MLP structure that learns local features before memory/communication
        hidden_dim = args.rnn_hidden_dim
        num_layers = getattr(args, 'num_layers', 2)
        
        layers = []
        in_dim = input_shape
        for i in range(num_layers):
            layers.append(thnn.Linear(in_dim, hidden_dim))
            layers.append(thnn.LayerNorm(hidden_dim))
            layers.append(thnn.ReLU())
            in_dim = hidden_dim
        self.encoder = thnn.Sequential(*layers)
        
        # [MECHANISM: GATED RECURRENT COMMUNICATION (GRC)]
        # Process signals from neighbors to understand inflow/outflow
        self.use_grc = getattr(args, 'use_grc', True)
        if self.use_grc:
            # Neighbor features are same dimension as our encoded local features
            self.comm_gru = thnn.GRUCell(hidden_dim, hidden_dim)
        else:
            self.comm_linear = thnn.Linear(hidden_dim, hidden_dim)
            
        # [MECHANISM: CORE MEMORY (GRU)]
        self.rnn = thnn.GRUCell(hidden_dim, hidden_dim)
        
        # [MECHANISM: ACTION HEAD WITH SKIP CONNECTION]
        # Residual architecture helps deep gradients flow better
        self.fc2 = thnn.Linear(hidden_dim, args.n_actions)
        self.skip = thnn.Linear(input_shape, args.n_actions) # Direct path from input

    def init_hidden(self):
        # make hidden states on same device as model
        return self.encoder[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, neighbor_obs=None):
        """
        inputs: [batch, input_shape]
        hidden_state: [batch, rnn_hidden_dim]
        neighbor_obs: [batch, num_neighbors, hidden_dim] (Encoded neighbor states)
        """
        # 1. Local Encoding
        x = self.encoder(inputs)
        
        # 2. Spatial Communication (GRC)
        if neighbor_obs is not None:
            if neighbor_obs.dim() == 3 and neighbor_obs.shape[1] > 0:
                # Aggregate neighbors (Inference mode)
                neighbor_msg = th.mean(neighbor_obs, dim=1)
            elif neighbor_obs.dim() == 2:
                # Pre-aggregated (Vectorized Training mode)
                neighbor_msg = neighbor_obs
            else:
                neighbor_msg = None
            
            if neighbor_msg is not None:
                if self.use_grc:
                    x = self.comm_gru(neighbor_msg, x)
                else:
                    msg_feat = F.relu(self.comm_linear(neighbor_msg))
                    x = x + msg_feat
        
        # 3. Temporal Memory (GRU)
        h = self.rnn(x, hidden_state)
        
        # 4. Action Scoring (Q-values)
        q = self.fc2(h)
        
        # Skip connection for better stability
        q_skip = self.skip(inputs)
        return q + q_skip, h

class MixingNetworkV4(thnn.Module):
    def __init__(self, args):
        super(MixingNetworkV4, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = args.state_dim
        self.embed_dim = args.mixing_hidden_dim

        # [MECHANISM: GLOBAL STATE ENCODER]
        # QMIX works best when the state is first compressed into a meaningful embedding
        self.state_encoder = thnn.Sequential(
            thnn.Linear(self.state_dim, self.embed_dim),
            thnn.LayerNorm(self.embed_dim),
            thnn.ReLU(),
            thnn.Linear(self.embed_dim, self.embed_dim)
        )

        # Hypernetworks: generate weights for the mixer from the global state
        self.hyper_w_1 = thnn.Linear(self.embed_dim, self.embed_dim * self.n_agents)
        self.hyper_w_final = thnn.Linear(self.embed_dim, self.embed_dim)

        # State-dependent bias
        self.hyper_b_1 = thnn.Linear(self.embed_dim, self.embed_dim)
        self.v = thnn.Sequential(
            thnn.Linear(self.embed_dim, self.embed_dim),
            thnn.ReLU(),
            thnn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # Encode the global state
        state_embed = self.state_encoder(states)

        # First layer weights (Must be non-negative for monotonicity)
        w1 = th.abs(self.hyper_w_1(state_embed))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(state_embed)
        b1 = b1.view(-1, 1, self.embed_dim)

        # Mixing layer 1
        hidden = F.elu(th.matmul(agent_qs, w1) + b1)

        # Final layer weights
        w_final = th.abs(self.hyper_w_final(state_embed))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State bias V(s)
        v = self.v(state_embed).view(-1, 1, 1)

        # Final reward sum
        y = th.matmul(hidden, w_final) + v
        q_tot = y.view(bs, -1)
        return q_tot
