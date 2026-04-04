import traci
import numpy as np

class WebsterController:
    """
    Implements Indian native traffic signal control based on IRC:93-1985 guidelines.
    Uses Webster's Method for optimum cycle time calculation across any junction structure.
    """
    def __init__(self, env, tls_id, update_interval=120):
        self.env = env
        self.tls_id = tls_id
        self.update_interval = update_interval  # Shorter interval for adaptation
        self.timer = 0
        
        # IRC Constants (IRC:106-1990)
        self.saturation_flow = 2000 / 3600  # 2000 PCU/hour per lane -> PCU/second
        self.lost_time_per_phase = 4  # Standard IRC lost time (seconds)
        
        self.phase_lane_map = {} # Phase index -> List of Lane IDs
        self.lane_flows = {}     # Lane ID -> [PCU counts]
        
        self._initialize_geometry()

    def _initialize_geometry(self):
        """Analyze the TLS logic to map lanes to green phases automatically."""
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        
        # Group lanes by which phase they belong to (Green phases)
        for i, phase in enumerate(logic.phases):
            # We only optimize 'major' green phases (those with 'G' or 'g')
            if 'G' in phase.state or 'g' in phase.state:
                lanes_in_phase = []
                for j, state in enumerate(phase.state):
                    if state.lower() == 'g':
                        lanes_in_phase.append(controlled_lanes[j])
                
                if lanes_in_phase:
                    self.phase_lane_map[i] = list(set(lanes_in_phase))
                    for lane in self.phase_lane_map[i]:
                        if lane not in self.lane_flows:
                            self.lane_flows[lane] = []

    def step(self):
        """Collect data and periodically update signal timings."""
        self.timer += 1
        
        # 1. Collect real-time PCU flow data
        for lane in self.lane_flows.keys():
            current_pcu = self.env.get_pcu_count_on_lane(lane)
            self.lane_flows[lane].append(current_pcu)
        
        # 2. Re-calculate and Apply Webster's Method
        if self.timer >= self.update_interval:
            self._optimize_timings()
            self.timer = 0
            # Reset buffers
            for lane in self.lane_flows.keys():
                self.lane_flows[lane] = []

    def _optimize_timings(self):
        """Map flows to phases and calculate optimal Webster splits."""
        # A. Calculate Phase-wise Critical Flows (PCU/sec)
        y_ratios = {}
        Y_sum = 0
        
        for phase_idx, lanes in self.phase_lane_map.items():
            # Find the 'Critical Lane' for this phase (highest flow)
            max_q = 0
            for lane in lanes:
                if self.lane_flows[lane]:
                    avg_q = np.mean(self.lane_flows[lane])
                    max_q = max(max_q, avg_q)
            
            y_i = max_q / self.saturation_flow
            y_ratios[phase_idx] = y_i
            Y_sum += y_i
            
        # B. Calculate Lost Time L
        num_phases = len(y_ratios)
        if num_phases == 0: return # No green phases to optimize
        
        L = (num_phases * self.lost_time_per_phase) + 2 # +2 for all-red buffer
        
        # C. Webster's Optimum Cycle Time
        # Formula: Co = (1.5L + 5) / (1 - Y)
        if Y_sum >= 0.95: # Saturation threshold
            cycle_time = 120
        else:
            cycle_time = (1.5 * L + 5) / (1 - (Y_sum if Y_sum < 0.95 else 0.95))
            cycle_time = int(np.clip(cycle_time, 40, 120))
        
        # D. Apportion Green Time proportionally to y_i
        green_time_pool = cycle_time - L
        
        new_durations = {}
        for phase_idx, y_i in y_ratios.items():
            if Y_sum > 0:
                g_i = int((y_i / Y_sum) * green_time_pool)
            else:
                g_i = green_time_pool // num_phases
            new_durations[phase_idx] = max(5, g_i) # Minimum 5s green for safety
            
        print(f"[Webster Optimizer] {self.tls_id} | Critical Ratio Y: {Y_sum:.3f} | New Cycle: {cycle_time}s")
        
        # E. Apply to TLS Program
        self._apply_logic(new_durations)

    def _apply_logic(self, phase_durations):
        """Inject the new durations into the SUMO Traffic Light logic."""
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        
        for i, phase in enumerate(logic.phases):
            if i in phase_durations:
                phase.duration = phase_durations[i]
            elif 'y' in phase.state or 'Y' in phase.state:
                phase.duration = 3 # Standard Yellow
            else:
                # Keep original or set small default if minor red
                phase.duration = max(3, phase.duration)
                
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.tls_id, logic)
