import os
import torch
import yaml
import asyncio
from types import SimpleNamespace
from core.envs.sumo_env import SUMOEnv
from algos.v2.networks import RNNAgent

class DualSimRunner:
    """
    [MECHANISM: THE DUAL-SIM ORCHESTRATOR]
    This class runs two separate SUMO simulations in parallel:
    1. QMIX V2: The AI-controlled version with a GRU memory.
    2. Native: The default SUMO rule-based version.
    
    It steps both simulations, collects vehicle coordinates, and returns
    a unified telemetry packet for the frontend.
    """
    def __init__(self, v2_model_path="./models/v2/v2_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.v2_model_path = v2_model_path
        self.is_running = False
        self.v2_env = None
        self.native_env = None
        self.v2_agent = None
        self.hidden_state = None

    def _load_config(self, path):
        with open(path, "r") as f:
            return SimpleNamespace(**yaml.safe_load(f))

    async def setup(self):
        """Initializes both environments and loads the AI brain."""
        print(f"[DUAL-SIM] Initializing environments on {self.device}...")
        
        # Load Configs
        v2_args = self._load_config("config/improved_qmix.yaml")
        native_args = self._load_config("config/native.yaml")

        # Initialize Envs with unique Ports and Labels
        # Port 8813 = QMIX AI
        # Port 8814 = Native
        self.v2_env = SUMOEnv(v2_args, port=8813, label="v2_ai")
        self.native_env = SUMOEnv(native_args, port=8814, label="native")

        # Initialize AI Agent
        self.v2_agent = RNNAgent(self.v2_env.obs_size, v2_args.rnn_hidden_dim, self.v2_env.n_actions).to(self.device)
        
        if os.path.exists(self.v2_model_path):
            print(f"[DUAL-SIM] Loading trained model from {self.v2_model_path}")
            checkpoint = torch.load(self.v2_model_path, map_location=self.device)
            # Load state dict (handle potential 'agent.' prefix if saved from trainer)
            if "agent_state_dict" in checkpoint:
                self.v2_agent.load_state_dict(checkpoint["agent_state_dict"])
            else:
                self.v2_agent.load_state_dict(checkpoint)
        else:
            print("[WARN] No trained model found! Using random initialization for V2.")

        self.v2_agent.eval()
        self.is_running = True
        print("[DUAL-SIM] Both simulations ready.")

    async def step(self):
        """Advances both simulations by exactly 1 step and collects telemetry."""
        if not self.is_running:
            return None

        # 1. Handle V2 Action Selection (Every 10 steps based on decision_interval)
        # However, for smooth visualization, we step SUMO every 1 second.
        # We only re-calculate actions when needed.
        
        # In this DualRunner, we simplify: we step 1 second at a time.
        # If it's a decision point, we ask the AI.
        
        # [MECHANISM: AI INFERENCE]
        obs = self.v2_env.get_obs() # List of np.arrays
        obs_tensor = torch.FloatTensor(obs).to(self.device).reshape(1, self.v2_env.n_agents, -1)
        
        # Re-init hidden state if it doesn't exist
        if self.hidden_state is None:
            self.hidden_state = self.v2_agent.init_hidden().to(self.device)

        with torch.no_grad():
            # Get Q-values and update hidden state (temporal memory)
            q_values, self.hidden_state = self.v2_agent(obs_tensor, self.hidden_state)
            actions = q_values.max(dim=2)[1].cpu().numpy()[0]

        # 2. Advance Environments
        # Note: We step the physics by 1 second here for max visual smoothness
        # Normally the trainer uses 10s, but for the Frontend we want 'high-framerate'
        self.v2_env.step(actions) 
        self.native_env.step(None) # Native manages itself

        # 3. Collect Telemetry
        telemetry = {
            "v2": {
                "vehicles": self.v2_env.get_vehicle_telemetry(),
                "tls": self.v2_env.get_tls_states(),
                "step": self.v2_env.time,
                "reward": round(float(self.v2_env._compute_reward()), 2)
            },
            "native": {
                "vehicles": self.native_env.get_vehicle_telemetry(),
                "tls": self.native_env.get_tls_states(),
                "step": self.native_env.time,
                "reward": round(float(self.native_env._compute_reward()), 2)
            }
        }
        
        return telemetry

    def stop(self):
        """Gracefully shuts down both SUMO instances."""
        self.is_running = False
        if self.v2_env:
            self.v2_env.close()
        if self.native_env:
            self.native_env.close()
        print("[DUAL-SIM] Simulation engines stopped.")
