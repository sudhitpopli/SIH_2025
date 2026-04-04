import asyncio
import json
import torch
import numpy as np
import traci
import yaml
from types import SimpleNamespace
from core.envs.sumo_env import SUMOEnv
from algos.v2.trainer import QMIXTrainerV2

class SimRunner:
    """
    [MECHANISM: LIVE TELEMETRY BRIDGE]
    This class handles the execution of a single SUMO episode while streaming 
    real-time data to a WebSocket client.
    
    It bridges the synchronous TraCI physics loop with the asynchronous FastAPI 
    WebSocket handler.
    """
    def __init__(self, config_path, model_path, device="cpu"):
        with open(config_path, 'r') as f:
            self.cfg_dict = yaml.safe_load(f)
            self.args = SimpleNamespace(**self.cfg_dict)
        
        self.model_path = model_path
        self.device = torch.device(device)
        self.env = None
        self.trainer = None
        self.is_running = False

    async def start_streaming(self, websocket):
        """
        Runs the simulation loop and feeds data into the websocket.
        """
        print(f"[BACKEND] Starting live simulation with model: {self.model_path}")
        
        # Override config for live streaming (we want high granularity)
        self.args.env_args["use_gui"] = False # Run headless for the backend
        self.args.env_args["decision_interval"] = 1 # Update every single second
        
        try:
            self.env = SUMOEnv(self.args)
            self.trainer = QMIXTrainerV2(
                env=self.env, 
                n_agents=self.env.n_agents, 
                state_dim=self.env.get_state_size(),
                obs_dim=self.env.obs_size, 
                n_actions=self.env.n_actions,
                rnn_hidden_dim=self.args.rnn_hidden_dim, 
                mixing_hidden_dim=self.args.mixer_hidden_dim,
                device=self.device
            )
            
            # Load the best weights
            if not self.trainer.load_model(self.model_path):
                await websocket.send_json({"error": "Failed to load model weights"})
                return

            self.trainer.epsilon = 0.0 # Strict inference
            self.is_running = True
            
            state, obs = self.env.reset()
            h = self.trainer.init_hidden()
            done = False
            
            while self.is_running and not done:
                # 1. AI Decision
                actions, h = self.trainer.select_action(obs, h)
                
                # 2. Physics Step
                next_state, next_obs, reward, done, info = self.env.step(actions)
                
                # 3. Telemetry Payload
                # We calculate some extra metrics for the frontend graphs
                # Total Queue Length (PCU-weighted)
                total_queue = 0
                for tls in self.env.tls_ids:
                    for lane in traci.trafficlight.getControlledLanes(tls):
                        total_queue += traci.lane.getLastStepHaltingNumber(lane)

                # Avg Speed of all vehicles
                all_speeds = [o[self.env.max_lanes : 2*self.env.max_lanes] for o in next_obs]
                avg_speed = np.mean(all_speeds) if all_speeds else 0
                
                payload = {
                    "step": self.env.time,
                    "reward": float(reward),
                    "vehicles": int(info.get("vehicles", 0)),
                    "total_queue": int(total_queue),
                    "avg_speed": float(avg_speed)
                }
                
                # Send to frontend
                await websocket.send_json(payload)
                
                obs = next_obs
                # Small sleep to yield to the event loop so a "Stop" command can be processed
                await asyncio.sleep(0.001)

            print("[BACKEND] Simulation episode complete.")
            await websocket.send_json({"status": "complete"})
            
        except Exception as e:
            print(f"[BACKEND ERROR] {e}")
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
        finally:
            self.stop()

    def stop(self):
        """Cleanly shutdown the simulation"""
        self.is_running = False
        if self.env:
            self.env.close()
            self.env = None
        print("[BACKEND] Simulation stopped and TraCI closed.")
