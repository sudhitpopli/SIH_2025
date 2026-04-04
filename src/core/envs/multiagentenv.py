import random

class MultiAgentEnv:
    """
    [MECHANISM: THE BLUEPRINT (ABSTRACT BASE CLASS)]
    This file is completely conceptual. It is a "Blueprint" (Abstract Base Class) for Reinforcement Learning.
    
    Before we can create a complex Traffic Simulation, PyTorch needs a guarantee that the environment 
    will have standardized buttons: `reset()` (to reboot), `step()` (to advance physics), and 
    `sample_actions()` (to guess blindly). 
    
    `sumo_env.py` inherits this blueprint so PyTorch knows exactly how to talk to SUMO without 
    needing to know how SUMO actually works.
    """

    def __init__(self, n_agents=1, action_space=None, obs_space=None):
        self.n_agents = n_agents
        # action_space, obs_space can be lists (one per agent) or shared
        self.action_space = action_space if action_space else [2] * n_agents  # e.g. 2 discrete actions
        self.obs_space = obs_space if obs_space else [1] * n_agents  # e.g. 1D observation

    def reset(self):
        """
        Reset the environment.
        Must be implemented by subclasses.
        Returns:
            observations (list): list of initial observations
        """
        raise NotImplementedError("reset() must be implemented in subclass")

    def step(self, actions):
        """
        Step the environment with a list of agent actions.
        Must be implemented by subclasses.
        Returns:
            observations (list)
            rewards (list)
            done (bool)
            info (dict)
        """
        raise NotImplementedError("step() must be implemented in subclass")

    def sample_actions(self):
        """
        Default random action sampler.
        Subclasses can override this to use their own action space definition.
        """
        return [random.randrange(space) for space in self.action_space]

    def get_obs_shape(self):
        """Return the shape/dimension of observations for each agent."""
        return self.obs_space

    def get_action_space(self):
        """Return the size of action space for each agent."""
        return self.action_space
