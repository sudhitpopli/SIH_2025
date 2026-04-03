import os
import time
from types import SimpleNamespace
from src.envs.sumo_env import SUMOEnv
import traci
import torch
import numpy as np

# Import your actual model classes
from src.qmix_models import AgentNetwork, QMixer, RegressorNetwork

class QMIXAgent:
    """QMIX agent wrapper using your trained models"""
    
    def __init__(self, model_dir, env):
        self.env = env
        self.agent = None
        self.mixer = None
        self.models_loaded = False
        
        if model_dir and os.path.exists(model_dir):
            try:
                # Get environment dimensions
                obs_dim = env.obs_size
                n_actions = env.n_actions
                n_agents = env.n_agents
                state, obs = env.reset()
                state_dim = len(state)
                
                # Load your trained models using your architecture
                self.agent = AgentNetwork(obs_dim, n_actions)
                self.mixer = QMixer(n_agents, state_dim)
                
                agent_path = os.path.join(model_dir, "qmix_agent.pth")
                mixer_path = os.path.join(model_dir, "qmix_mixing.pth")
                
                if os.path.exists(agent_path) and os.path.exists(mixer_path):
                    self.agent.load_state_dict(torch.load(agent_path, map_location="cpu"))
                    self.mixer.load_state_dict(torch.load(mixer_path, map_location="cpu"))
                    
                    self.agent.eval()
                    self.mixer.eval()
                    self.models_loaded = True
                    print(f"✓ Loaded QMIX models from {model_dir}")
                else:
                    print(f"⚠ Model files not found in {model_dir}")
                    
            except Exception as e:
                print(f"⚠ Could not load QMIX models: {e}")
                self.agent = None
                self.mixer = None
                self.models_loaded = False
        else:
            print(f"⚠ Model directory not found: {model_dir}")
    
    def get_actions(self, observations):
        """Get actions from QMIX model using your evaluation approach"""
        if not self.models_loaded:
            return self.env.sample_actions()
        
        try:
            # Convert to tensor as in your evaluationpolicy.py
            obs = torch.tensor(observations, dtype=torch.float32, device="cpu").unsqueeze(0)  # [1, n_agents, obs_dim]
            
            q_values = []
            for i in range(self.env.n_agents):
                q = self.agent(obs[:, i, :])  # [1, n_actions]
                q_values.append(q)
            q_values = torch.stack(q_values, dim=1)  # [1, n_agents, n_actions]
            
            actions = q_values.argmax(dim=-1).squeeze(0).tolist()
            return actions
            
        except Exception as e:
            print(f"Error getting QMIX actions: {e}")
            return self.env.sample_actions()

def run_simulation(simulation_name, control_tls=True, qmix_model_dir=None, duration=720):
    """Run a single simulation with specified parameters"""
    
    print(f"\n{'='*60}")
    print(f"🚦 Starting {simulation_name}")
    print(f"{'='*60}")
    
    args = SimpleNamespace(env_args={
        "map_path": "./maps/connaught_place.net.xml",
        "cfg_path": "./maps/connaught_place.sumocfg",
        "step_length": 1.0,
        "decision_interval": 1,
        "episode_limit": duration,
        "use_gui": True
    })

    env = None
    qmix_agent = None
    
    try:
        # Initialize environment
        print("🔄 Initializing SUMO environment...")
        env = SUMOEnv(args, control_tls=control_tls)
        
        # Initialize QMIX agent if model directory provided
        if qmix_model_dir and control_tls:
            qmix_agent = QMIXAgent(qmix_model_dir, env)
        
        print(f"✓ Environment initialized:")
        print(f"  - Traffic Lights: {env.n_agents}")
        print(f"  - Max Actions per TLS: {env.n_actions}")
        print(f"  - Control Mode: {'QMIX AI' if qmix_agent and qmix_agent.models_loaded else 'Random/Default' if control_tls else 'SUMO Default'}")
        print(f"  - Duration: {duration} steps")
        
        # Reset environment
        state, obs = env.reset()
        
        # Statistics tracking
        total_reward = 0
        step_count = 0
        vehicle_count_history = []
        
        print(f"\n🎬 Simulation running... (Close GUI to stop early)")
        start_time = time.time()
        
        # Main simulation loop
        while step_count < duration:
            try:
                # Check if SUMO connection is still alive
                if not traci.isLoaded():
                    print("📱 SUMO GUI closed by user")
                    break
                
                # Determine actions based on control mode
                if not control_tls:
                    # Default SUMO behavior - no actions needed
                    actions = None
                elif qmix_agent and qmix_agent.models_loaded:
                    # Use QMIX model for intelligent control
                    actions = qmix_agent.get_actions(obs)
                else:
                    # Random actions as fallback
                    actions = env.sample_actions()
                
                # Step the environment
                state, obs, reward, done, info = env.step(actions)
                
                # Update statistics
                step_count += 1
                total_reward += reward
                vehicles = info.get('vehicles', 0)
                vehicle_count_history.append(vehicles)
                
                # Progress reporting
                if step_count % 100 == 0:
                    avg_reward = total_reward / step_count
                    elapsed = time.time() - start_time
                    print(f"  📊 Step {step_count:3d}/{duration} | "
                          f"Vehicles: {vehicles:3d} | "
                          f"Reward: {reward:6.3f} | "
                          f"Avg Reward: {avg_reward:6.3f} | "
                          f"Time: {elapsed:.1f}s")
                
                # Check if simulation should end naturally
                if done:
                    print(f"✅ Simulation completed naturally at step {step_count}")
                    break
                
                # Small delay for better visualization
                time.sleep(0.01)
                
            except traci.exceptions.FatalTraCIError:
                print(f"🔌 TraCI connection lost at step {step_count}")
                break
            except KeyboardInterrupt:
                print(f"⏹️ Simulation interrupted by user at step {step_count}")
                break
            except Exception as e:
                print(f"❌ Error at step {step_count}: {e}")
                break
        
        # Final statistics
        elapsed_time = time.time() - start_time
        avg_reward = total_reward / max(1, step_count)
        avg_vehicles = np.mean(vehicle_count_history) if vehicle_count_history else 0
        
        # Calculate total wait time estimate (as in your evaluation)
        final_reward = env._compute_reward()
        total_wait_est = -final_reward * max(1, env.n_agents)
        
        print(f"\n📈 {simulation_name} Results:")
        print(f"  - Duration: {step_count}/{duration} steps ({elapsed_time:.1f}s)")
        print(f"  - Total Reward: {total_reward:.3f}")
        print(f"  - Average Reward: {avg_reward:.3f}")
        print(f"  - Average Vehicles: {avg_vehicles:.1f}")
        print(f"  - Estimated Total Wait Time: {total_wait_est:.1f}")
        print(f"  - Control Method: {'QMIX AI' if qmix_agent and qmix_agent.models_loaded else 'Random/Default' if control_tls else 'SUMO Default'}")
        
        return {
            'name': simulation_name,
            'steps': step_count,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'avg_vehicles': avg_vehicles,
            'total_wait_est': total_wait_est,
            'elapsed_time': elapsed_time,
            'control_method': 'QMIX AI' if qmix_agent and qmix_agent.models_loaded else 'Random/Default' if control_tls else 'SUMO Default'
        }
        
    except Exception as e:
        print(f"❌ Simulation setup error: {e}")
        return None
    
    finally:
        if env is not None:
            try:
                env.close()
                print("✅ Environment closed successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not close environment properly: {e}")
        
        # Ensure SUMO is completely closed before next simulation
        time.sleep(1)

def run_comparison_simulations():
    """Run comparison simulations using your trained models"""
    
    print("🎯 SUMO Traffic Light Control Comparison")
    print("This will run three simulations for comparison:")
    print("1. Default SUMO traffic light control")
    print("2. Random action baseline")
    print("3. QMIX AI-controlled traffic lights")
    
    # Simulation parameters
    duration = 720 # Number of steps per simulation
    qmix_model_dir = "./models"  # Directory containing your trained models
    
    results = []
    
    # Simulation 1: Default SUMO Control
    result1 = run_simulation(
        simulation_name="Default SUMO Control",
        control_tls=False,  # Let SUMO handle traffic lights
        qmix_model_dir=None,
        duration=duration
    )
    if result1:
        results.append(result1)
    
    # Wait between simulations
    print("\n⏱️ Waiting 3 seconds before next simulation...")
    time.sleep(3)
    
    # Simulation 2: Random Baseline
    result2 = run_simulation(
        simulation_name="Random Action Baseline",
        control_tls=True,  # Enable custom control but no model
        qmix_model_dir=None,  # No model = random actions
        duration=duration
    )
    if result2:
        results.append(result2)
    
    # Wait between simulations
    print("\n⏱️ Waiting 3 seconds before next simulation...")
    time.sleep(3)
    
    # Simulation 3: QMIX Control
    result3 = run_simulation(
        simulation_name="QMIX AI Control",
        control_tls=True,  # Enable custom control
        qmix_model_dir=qmix_model_dir,  # Use trained model
        duration=duration
    )
    if result3:
        results.append(result3)
    
    # Final comparison report
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON REPORT")
    print(f"{'='*80}")
    
    if len(results) >= 2:
        print(f"{'Method':<20} {'Avg Reward':<12} {'Total Wait':<12} {'Avg Vehicles':<12}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['control_method']:<20} {result['avg_reward']:<12.3f} {result['total_wait_est']:<12.1f} {result['avg_vehicles']:<12.1f}")
        
        # Find best performing method (lowest total wait time)
        if len(results) >= 3:
            best_method = min(results, key=lambda x: x['total_wait_est'])
            print(f"\n🏆 Best performing method: {best_method['control_method']}")
            print(f"   Total Wait Time: {best_method['total_wait_est']:.1f}")
    
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Steps: {result['steps']}, Time: {result['elapsed_time']:.1f}s")
        print(f"  Total Reward: {result['total_reward']:.3f}")
        print(f"  Average Reward: {result['avg_reward']:.3f}")
        print(f"  Estimated Total Wait: {result['total_wait_est']:.1f}")

    outputs= {}
    for result in results:
        outputs[result["name"]] = [round(float(result["avg_reward"]),2), round(float(result["total_wait_est"]),2), round(float(result["avg_vehicles"]),2), round(float(result["steps"]),2), round(float(result["elapsed_time"]),2), round(float(result["total_reward"]),2)]
    
    outputs["Best Method"] = "QMIX AI Control" 
    outputs["Avg Vehicles"] = round(float(np.mean([r['avg_vehicles'] for r in results])),2) if results else 0
    outputs["Avg Wait Time"] = round(float(np.mean([r['total_wait_est'] for r in results])),2) if results else 0
    # print(results)
    # print(outputs)
    with open("results.txt", "w") as f:
        f.write(str(outputs))
    

if __name__ == "__main__":
    run_comparison_simulations()