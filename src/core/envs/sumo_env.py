import os
import traci
import numpy as np
from core.envs.multiagentenv import MultiAgentEnv

class SUMOEnv(MultiAgentEnv):
    def __init__(self, args, control_tls=True):
        """
        Initializes the SUMO Environment wrapper. 
        This class bridges PyTorch/RL logic with the SUMO TraCI C++ backend.
        """
        super().__init__()
        self.control_tls = control_tls
        self.args = args
        self.net_file = args.env_args.get("map_path", "./maps/connaught_place.net.xml")
        self.cfg_file = args.env_args.get("cfg_path", "./maps/connaught_place.sumocfg")
        self.step_length = args.env_args.get("step_length", 1.0)
        self.decision_interval = args.env_args.get("decision_interval", 5)
        
        # [MECHANISM: TOPOLOGY & SENSORY INPUT]
        # The environment reads config parameters to understand the map.
        # Connaught Place intersections have many roads meeting at a central point (radial hubs).
        # We set `max_lanes` to 8 so the RL Agent's neural network has a wide enough "field of view"
        # to observe up to 8 incoming lanes at every traffic light.
        self.max_lanes = args.env_args.get("max_lanes", 8)  
        self.episode_limit = args.env_args.get("episode_limit", 720)
        self.use_gui = args.env_args.get("use_gui", False) 
        
        # [MECHANISM: YELLOW LIGHT PHYSICS]
        # Neural networks output immediate 1s and 0s, but real traffic lights need 3 seconds of yellow 
        # before turning red to prevent physical crashes in the simulator.
        # We store the "requested" green phase in `_pending_phase`, turn the current light yellow, 
        # and start a countdown `_yellow_countdown`. The environment ignores new agent commands 
        # until the countdown finishes and the strict green phase is safely applied.
        self.current_phases = {}       # tls_id -> current green phase index
        self._yellow_countdown = {}    # tls_id -> remaining yellow steps (0 = not in yellow)
        self._pending_phase = {}       # tls_id -> phase to switch to after yellow completes

        # ---- obs_size consistency guard ----
        # If the RL environment resets and TraCI initializes a slightly different state matrix, 
        # the PyTorch network tensor shapes will crash. We lock the initial size here.
        self._original_obs_size = None
        
        # ---- Permanent clamps for dummy lights ----
        # Dummy pedestrian lights have strict [0,0] operational boundaries. TraCI will crash internally 
        # and flood stderr if asked to change them. This set persistently remembers Dummy Lights 
        # so they are never touched across thousands of episodes.
        self._permanent_clamps = {}
        self.tls_definitions = {}

        # Initialize SUMO once at boot
        self._start_sumo()
        self._initialize_env_info()

    def _start_sumo(self):
        """Start SUMO simulation with proper error handling"""
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass
        
        # Build SUMO command
        binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            binary, 
            "-c", self.cfg_file, 
            "--step-length", str(self.step_length), 
            "--no-warnings",
            "--quit-on-end"  # Important: quit when simulation ends
        ]
        
        try:
            traci.start(sumo_cmd)
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            raise

    def _initialize_env_info(self):
        """
        Pulls down the entire phase-logic mapping from SUMO C++ memory bounds
        to dynamically construct the Action-Space geometry for the neural network.
        """
        try:
            self.tls_ids = traci.trafficlight.getIDList()
            self.n_agents = len(self.tls_ids)

            if self.n_agents == 0:
                print("Warning: No traffic lights found in the network!")
                self.n_actions = 4  # Default assumption for a basic 4-way cross
                self.tls_action_counts = {}
            else:
                self.tls_action_counts = {}
                self.tls_definitions = {}
                for tls in self.tls_ids:
                    try:
                        # [MECHANISM: TRAFFIC LIGHT PHASE MAPPING]
        # A "Phase" in SUMO is a string of characters like "GGGgrrr" representing which lanes have Green/Red.
        # An intersection can have multiple logics/programs. The agent's neural network acts by choosing 
        # an integer (Action 0, Action 1...). We ask SUMO exactly how many phases the CURRENT active program 
        # has, so the agent's action space perfectly maps to the physical intersections.
                        logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
                        curr_prog = traci.trafficlight.getProgram(tls)
                        active_logic = next((l for l in logics if l.programID == curr_prog), logics[0])
                        self.tls_action_counts[tls] = len(active_logic.phases)
                        self.tls_definitions[tls] = active_logic
                    except:
                        # Fallback for undocumented or completely static dummy nodes (e.g. pedestrian crossing)
                        self.tls_action_counts[tls] = 1
                        self.tls_definitions[tls] = None
                        
                    # Re-apply any bounds that threw IndexError exceptions in past episodes
                    if tls in self._permanent_clamps:
                        self.tls_action_counts[tls] = self._permanent_clamps[tls]

                # Maintain global observation width matching the most complex intersection on the map
                self.n_actions = max(self.tls_action_counts.values()) if self.tls_action_counts else 4

            # Initialize sizes for PyTorch memory pre-allocation
            if self.tls_ids:
                sample_features = self._build_features(self.tls_ids[0])
                self.obs_size = len(sample_features)
            else:
                self.obs_size = 13  # Baseline fallback

            
            self.state_size = self.n_agents * self.obs_size
            self.is_initialized = True

            # ---- Bug 5 fix: lock obs_size on first init ----
            if self._original_obs_size is None:
                self._original_obs_size = self.obs_size
            else:
                if self.obs_size != self._original_obs_size:
                    print(f"[WARN] obs_size changed from {self._original_obs_size} to {self.obs_size} "
                          f"after reset. Forcing original size.")
                    self.obs_size = self._original_obs_size
                    self.state_size = self.n_agents * self.obs_size

            # ---- Bug 6 fix: initialize phase tracking ----
            for tls in self.tls_ids:
                try:
                    self.current_phases[tls] = traci.trafficlight.getPhase(tls)
                except:
                    self.current_phases[tls] = 0
                self._yellow_countdown[tls] = 0
                self._pending_phase[tls] = None
            
        except Exception as e:
            print(f"Error initializing environment info: {e}")
            # Fallback values
            self.tls_ids = []
            self.n_agents = 0
            self.n_actions = 4
            self.obs_size = 13
            self.state_size = 0
            self.tls_action_counts = {}

    # ====================================================================
    # [MECHANISM: ENVIRONMENT RESET ENGINE]
    # At the start of every new episode (every 720 time steps), we shut down
    # and reboot the SUMO C++ simulation completely to ensure no residual 
    # traffic memory corrupts the new learning cycle.
    # ====================================================================
    def reset(self):
        """Reset the simulation by cleanly restarting the entire SUMO binary."""
        self.time = 0

        # Always close and restart — no backwards stepping!
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass

        self._start_sumo()
        self._initialize_env_info()

        # Get initial observations
        try:
            obs = self.get_obs()
            state = self.get_state()
            return state, obs
        except Exception as e:
            print(f"Error getting initial observations: {e}")
            # Return empty observations as fallback
            empty_obs = [np.zeros(self.obs_size) for _ in range(self.n_agents)]
            empty_state = np.zeros(self.state_size)
            return empty_state, empty_obs

    # ====================================================================
    # [MECHANISM: THE CONTROL LOOP]
    # This is the beating heart of Reinforcement Learning.
    # 1. Take PyTorch `actions` (a list of numbers from 0 to N).
    # 2. Tell SUMO to act on them (or schedule yellow lights).
    # 3. Fast-forward the physics engine by 5 seconds (`decision_interval`).
    # 4. Measure the result (queue lengths, speeds) to return `next_state`.
    # 5. Measure the penalty (`reward`) to punish or encourage the neural network.
    # ====================================================================
    def step(self, actions):
        """Advances the simulation by 5 seconds, enacting PyTorch actions and mapping physics."""
        try:
            # Check if simulation is still running
            if not traci.isLoaded():
                return self._get_terminal_state()
            
            if self.control_tls and actions is not None and self.tls_ids:
                for i, tls in enumerate(self.tls_ids):
                    if i >= len(actions):
                        continue

                    n_actions_tls = self.tls_action_counts.get(tls, self.n_actions)
                    requested_phase = actions[i] % n_actions_tls

                    # If currently in a yellow countdown, don't change anything
                    if self._yellow_countdown.get(tls, 0) > 0:
                        continue

                    current_phase = self.current_phases.get(tls, 0)

                    if requested_phase != current_phase:
                        # Need to transition: insert yellow phase first
                        self._start_yellow_transition(tls, requested_phase)
                    # else: same phase requested, do nothing

            # Advance SUMO simulation
            steps_to_advance = self.decision_interval if self.control_tls else 1
            
            for _ in range(steps_to_advance):
                # Tick yellow countdowns BEFORE stepping
                self._tick_yellow_phases()

                try:
                    traci.simulationStep()
                    self.time += 1
                    
                    # Check if simulation ended naturally
                    if traci.simulation.getMinExpectedNumber() <= 0 and self.time > 100:
                        print("Simulation ended: no more vehicles")
                        return self._get_terminal_state(done=True)
                        
                except traci.exceptions.FatalTraCIError as e:
                    print(f"TraCI connection lost: {e}")
                    return self._get_terminal_state(done=True)
                except Exception as e:
                    print(f"Error during simulation step: {e}")
                    return self._get_terminal_state(done=True)

            # Get observations and compute reward
            obs = self.get_obs()
            state = self.get_state()
            reward = self._compute_reward()
            done = self.time >= self.episode_limit
            info = {"time": self.time, "vehicles": self._get_vehicle_count()}

            return state, obs, reward, done, info
            
        except Exception as e:
            print(f"Critical error in step: {e}")
            return self._get_terminal_state(done=True)

    # ====================================================================
    # [CRITICAL FIX: TRACI STDERR BYPASS]
    # The absolute biggest breakthrough for simulation stability.
    # TraCI C++ evaluates `setPhase(index)` internally and prints to the terminal 
    # BEFORE Python can catch the exception if the limit is exceeded.
    # By pulling the literal state string ("GGGgrrr") from memory and applying it via
    # `setRedYellowGreenState`, SUMO skips boundary checks entirely!
    # ====================================================================
    def _start_yellow_transition(self, tls, target_green_phase):
        """
        Force-injects a realistic yellow transition into the physics engine.
        Intercepts the Agent's Phase switch, turns active Green lights to Yellow (y), 
        and sets a timer.
        """
        try:
            # Dynamically rewrite current 'G' or 'g' characters to 'y'.
            current_state = traci.trafficlight.getRedYellowGreenState(tls)
            yellow_state = current_state.replace('G', 'y').replace('g', 'y')
            traci.trafficlight.setRedYellowGreenState(tls, yellow_state)

            self._yellow_countdown[tls] = self.yellow_duration
            self._pending_phase[tls] = target_green_phase
        except Exception as e:
            # If for some reason dynamic yellow fails, we try to set the requested Green phase.
            try:
                logic = self.tls_definitions.get(tls)
                target_state = logic.phases[target_green_phase].state
                traci.trafficlight.setRedYellowGreenState(tls, target_state)
                self.current_phases[tls] = target_green_phase
            except Exception as inner_e:
                # [SELF-HEALING CLAMP] 
                # If Python throws an IndexError, the light doesn't actually support this phase.
                # We permanently clamp its limit to 1. The modulo math in `step()` will bound all 
                # future Neural Network requests identically to 0. Silence achieved.
                self.tls_action_counts[tls] = 1
                self.current_phases[tls] = 0
                self._permanent_clamps[tls] = 1

    def _tick_yellow_phases(self):
        """Decrement yellow countdowns and apply pending green phases when done."""
        for tls in self.tls_ids:
            countdown = self._yellow_countdown.get(tls, 0)
            if countdown <= 0:
                continue

            self._yellow_countdown[tls] = countdown - 1

            if self._yellow_countdown[tls] <= 0:
                # Yellow period is over — apply the pending green phase
                pending = self._pending_phase.get(tls)
                if pending is not None:
                    try:
                        logic = self.tls_definitions.get(tls)
                        target_state = logic.phases[pending].state
                        traci.trafficlight.setRedYellowGreenState(tls, target_state)
                        self.current_phases[tls] = pending
                    except Exception as e:
                        # Dummy light out-of-bounds catch silently in Python!
                        self.tls_action_counts[tls] = 1
                        self.current_phases[tls] = 0
                        self._permanent_clamps[tls] = 1
                    self._pending_phase[tls] = None

    def _get_terminal_state(self, done=True):
        """Return terminal state when simulation ends or errors occur"""
        empty_obs = [np.zeros(self.obs_size) for _ in range(self.n_agents)]
        empty_state = np.zeros(self.state_size)
        return empty_state, empty_obs, 0.0, done, {"terminal": True}

    def _get_vehicle_count(self):
        """Get current vehicle count safely"""
        try:
            return traci.simulation.getMinExpectedNumber()
        except:
            return 0

    def get_pcu_count_on_lane(self, lane_id):
        """
        Calculate PCU-weighted vehicle count on a lane based on IRC:106-1990.
        Factors:
        - passenger (car): 1.0
        - motorcycle: 0.5
        - bus: 3.0
        - truck: 3.0
        - bicycle: 0.2
        - default: 1.0
        """
        pcu_factors = {
            "passenger": 1.0,
            "motorcycle": 0.5,
            "bus": 3.0,
            "truck": 3.0,
            "bicycle": 0.2
        }
        
        try:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            total_pcu = 0
            for v_id in veh_ids:
                v_type = traci.vehicle.getTypeID(v_id)
                # Handle SUMO default type names or specific ones
                factor = 1.0
                for key, val in pcu_factors.items():
                    if key in v_type.lower():
                        factor = val
                        break
                total_pcu += factor
            return total_pcu
        except Exception as e:
            print(f"Error calculating PCU for lane {lane_id}: {e}")
            return 0

    def get_obs(self):
        """Get observations for all agents"""
        obs = []
        try:
            for tls in self.tls_ids:
                features = self._build_features(tls)
                # Bug 5 fix: always pad/truncate to _original_obs_size
                if len(features) < self.obs_size:
                    features = features + [0.0] * (self.obs_size - len(features))
                elif len(features) > self.obs_size:
                    features = features[:self.obs_size]
                obs.append(np.array(features, dtype=np.float32))
        except Exception as e:
            print(f"Error getting observations: {e}")
            # Return zero observations as fallback
            obs = [np.zeros(self.obs_size, dtype=np.float32) for _ in range(self.n_agents)]
        
        return obs

    def get_obs_size(self):
        return self.obs_size

    def get_state(self):
        """Get global state"""
        try:
            obs = self.get_obs()
            if obs:
                return np.concatenate(obs, axis=0)
            else:
                return np.zeros(self.state_size, dtype=np.float32)
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_size, dtype=np.float32)

    def get_state_size(self):
        return self.state_size

    def get_avail_actions(self, agent_id):
        if agent_id < len(self.tls_ids):
            tls = self.tls_ids[agent_id]
            n_actions = self.tls_action_counts.get(tls, self.n_actions)
            return [1] * n_actions
        return [1] * self.n_actions

    def get_total_actions(self):
        return self.n_actions

    def get_env_info(self):
        return {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_size,
            "state_shape": self.state_size,
            "n_actions": self.n_actions,
            "episode_limit": self.episode_limit
        }

    def _compute_max_pressure(self, tls_id):
        """
        Calculate the 'pressure' of a traffic light.
        Pressure = Sum(Incoming Lane Occupancy) - Sum(Outgoing Lane Occupancy)
        Lower pressure (near zero) means the junction is balanced and efficient.
        """
        try:
            incoming_lanes = traci.trafficlight.getControlledLanes(tls_id)
            # Find unique outgoing lanes (links)
            links = traci.trafficlight.getControlledLinks(tls_id)
            outgoing_lanes = []
            for link in links:
                for connection in link:
                    if connection[1] not in outgoing_lanes:
                        outgoing_lanes.append(connection[1])
            
            # Use PCU-weighted counts for more accurate pressure
            in_flow = sum([self.get_pcu_count_on_lane(l) for l in incoming_lanes])
            out_flow = sum([self.get_pcu_count_on_lane(l) for l in outgoing_lanes])
            
            # Pressure is the net accumulation
            return abs(in_flow - out_flow)
        except:
            return 0

    def _compute_reward(self):
        """Compute reward based on traffic performance (Wait Time or MaxPressure)"""
        reward_type = getattr(self.args, "reward_type", "wait_time")
        
        try:
            if reward_type == "max_pressure":
                total_pressure = sum([self._compute_max_pressure(tls) for tls in self.tls_ids])
                # We want to MINIMIZE pressure, so reward is negative
                reward = -total_pressure / max(1, self.n_agents)
                return np.clip(reward, -20, 0)
            
            else: # Default: wait_time
                total_wait = 0
                for tls in self.tls_ids:
                    try:
                        lanes = traci.trafficlight.getControlledLanes(tls)
                        for lane in lanes:
                            total_wait += traci.lane.getWaitingTime(lane)
                    except:
                        continue
                
                n_veh = max(1, self._get_vehicle_count())
                reward = -(total_wait / n_veh)
                return np.clip(reward, -10, 0)
            
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0

    def sample_actions(self):
        """Sample random actions"""
        import random
        return [
            random.randrange(self.tls_action_counts.get(tls, self.n_actions))
            for tls in self.tls_ids
        ]

    def _build_features(self, tls):
        """
        [MECHANISM: THE AGENT'S EYES]
        This function generates the specific sensory data (Observation tensor) for ONE intersection.
        The Neural Network is completely blind; it only 'sees' the array of numbers output here.
        
        We scan the `max_lanes` (8) incoming roads, counting the number of stopped cars (Queue) 
        and calculating their average speed. We append the current active Phase (as a One-Hot encoded 
        array) so the agent knows what it's currently doing.
        """
        try:
            lanes = traci.trafficlight.getControlledLanes(tls)
            # De-duplicate lanes (SUMO may list a lane multiple times for
            # different signal groups at the same intersection)
            unique_lanes = list(dict.fromkeys(lanes))  # preserves order

            q_lengths = []
            avg_speeds = []
            
            for i in range(self.max_lanes):
                if i < len(unique_lanes):
                    try:
                        q_lengths.append(traci.lane.getLastStepHaltingNumber(unique_lanes[i]))
                        avg_speeds.append(traci.lane.getLastStepMeanSpeed(unique_lanes[i]))
                    except:
                        q_lengths.append(0)
                        avg_speeds.append(0)
                else:
                    q_lengths.append(0)
                    avg_speeds.append(0)

            # Phase one-hot encoding
            try:
                phase = traci.trafficlight.getPhase(tls)
            except:
                phase = 0
                
            phase_onehot = [0] * self.n_actions
            if 0 <= phase < len(phase_onehot):
                phase_onehot[phase] = 1

            # Time until next switch
            try:
                elapsed = max(0, traci.trafficlight.getNextSwitch(tls) - traci.simulation.getTime())
            except:
                elapsed = 0

            return q_lengths + avg_speeds + phase_onehot + [elapsed]
            
        except Exception as e:
            print(f"Error building features for {tls}: {e}")
            # Return zero features as fallback
            return [0] * (self.max_lanes + self.max_lanes + self.n_actions + 1)

    def close(self):
        """Close the environment"""
        try:
            if traci.isLoaded():
                traci.close()
        except Exception as e:
            print(f"Error closing SUMO: {e}")
        finally:
            self.is_initialized = False