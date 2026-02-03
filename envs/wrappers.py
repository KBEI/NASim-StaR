"""Environment wrappers for NASim."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HierarchicalActionWrapper(gym.Wrapper):
    """Expose hierarchical (type, action) action space for flat NASim envs.

    Keeps underlying environment unchanged (flat actions), but presents a
    MultiDiscrete action space to the agent:
        action = (type_idx, local_idx)
    """

    def __init__(self, env, action_type_order=None):
        super().__init__(env)
        if not getattr(env, "flat_actions", False):
            raise ValueError("HierarchicalActionWrapper requires flat actions.")

        self.actions = env.action_space.actions
        self._build_action_groups(action_type_order)
        self.max_actions_per_type = max(
            len(indices) for indices in self.type_to_indices.values()
        )
        self.action_space = spaces.MultiDiscrete(
            [self.num_types, self.max_actions_per_type]
        )
        self.observation_space = env.observation_space
        self.flat_actions = False
        self.hierarchical_actions = True

    def _build_action_groups(self, action_type_order):
        if action_type_order is None:
            action_type_order = [
                "service_scan",
                "os_scan",
                "subnet_scan",
                "process_scan",
                "exploit",
                "privilege_escalation",
                "noop",
            ]

        type_to_indices = {name: [] for name in action_type_order}
        index_to_type = {}
        index_to_local = {}

        for idx, action in enumerate(self.actions):
            if action.is_noop():
                type_name = "noop"
            elif action.is_service_scan():
                type_name = "service_scan"
            elif action.is_os_scan():
                type_name = "os_scan"
            elif action.is_subnet_scan():
                type_name = "subnet_scan"
            elif action.is_process_scan():
                type_name = "process_scan"
            elif action.is_exploit():
                type_name = "exploit"
            elif action.is_privilege_escalation():
                type_name = "privilege_escalation"
            else:
                raise ValueError(f"Unknown action type: {action}")

            if type_name not in type_to_indices:
                type_to_indices[type_name] = []
            local_idx = len(type_to_indices[type_name])
            type_to_indices[type_name].append(idx)
            index_to_type[idx] = type_name
            index_to_local[idx] = local_idx

        self.type_names = [
            name for name in action_type_order if type_to_indices.get(name)
        ]
        self.type_to_indices = {
            name: type_to_indices[name] for name in self.type_names
        }
        self.type_name_to_idx = {
            name: i for i, name in enumerate(self.type_names)
        }
        self.index_to_type = index_to_type
        self.index_to_local = index_to_local
        self.num_types = len(self.type_names)

    def _normalize_action(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, (list, tuple)) or len(action) != 2:
            raise ValueError(
                "Hierarchical action must be (type_idx, local_idx)."
            )
        type_idx, local_idx = action
        type_idx = int(type_idx) % self.num_types
        type_name = self.type_names[type_idx]
        local_actions = self.type_to_indices[type_name]
        local_idx = int(local_idx) % len(local_actions)
        flat_idx = local_actions[local_idx]
        return type_idx, local_idx, flat_idx

    def get_flat_action(self, action):
        """Map hierarchical action to flat action index."""
        _, _, flat_idx = self._normalize_action(action)
        return flat_idx

    def get_action(self, action):
        """Return Action object for hierarchical action."""
        flat_idx = self.get_flat_action(action)
        return self.env.action_space.get_action(flat_idx)

    def step(self, action):
        flat_idx = self.get_flat_action(action)
        return self.env.step(flat_idx)

    def goal_reached(self):
        """Proxy goal_reached to underlying env."""
        return self.env.goal_reached()

    @property
    def render_mode(self):
        return self.env.render_mode

    @render_mode.setter
    def render_mode(self, value):
        self.env.render_mode = value
