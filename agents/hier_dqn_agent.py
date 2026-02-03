"""Hierarchical DQN Agent with action-type selection.

This agent keeps the environment action space unchanged (flat actions),
but chooses actions in two stages:
1) select an action type (e.g. exploit, scan),
2) select a concrete action within that type.
"""
import random
from pprint import pprint

from gymnasium import error
import numpy as np

import nasim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install dqn_agent dependencies by running "
        "'pip install nasim[dqn]'.)"
    )


class ReplayMemory:

    def __init__(self, capacity, s_dims, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.s_buf[sample_idxs],
            self.a_buf[sample_idxs],
            self.next_s_buf[sample_idxs],
            self.r_buf[sample_idxs],
            self.done_buf[sample_idxs],
        ]
        return [torch.from_numpy(buf).to(self.device) for buf in batch]


class DQN(nn.Module):
    """A simple Deep Q-Network."""

    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))
        self.out = nn.Linear(layers[-1], num_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]


class _ReplayProxy:
    """Proxy to keep compatibility with existing training loops."""

    def __init__(self, agent):
        self.agent = agent

    def store(self, s, a, next_s, r, done):
        self.agent._store_transition(s, a, next_s, r, done)


class HierarchicalDQNAgent:
    """Hierarchical DQN Agent with action-type selection."""

    def __init__(self,
                 env,
                 seed=None,
                 lr=0.001,
                 training_steps=20000,
                 batch_size=32,
                 replay_size=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 hidden_sizes=[64, 64],
                 target_update_freq=1000,
                 verbose=True,
                 action_type_order=None,
                 **kwargs):

        # This DQN implementation supports flat or hierarchical wrapper actions
        if not (getattr(env, "flat_actions", False)
                or getattr(env, "hierarchical_actions", False)):
            raise ValueError(
                "HierarchicalDQNAgent requires flat actions or "
                "HierarchicalActionWrapper."
            )
        self.verbose = verbose
        if self.verbose:
            print("\nRunning Hierarchical DQN with config:")
            pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # environment setup
        self.env = env
        if hasattr(self.env, "actions"):
            self.actions = self.env.actions
        else:
            self.actions = self.env.action_space.actions
        self.obs_dim = self.env.observation_space.shape
        self.uses_hier_action = getattr(self.env, "hierarchical_actions", False)

        # logger setup
        self.logger = SummaryWriter()

        # Training related attributes
        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0,
                                            self.final_epsilon,
                                            self.exploration_steps)
        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0
        self.target_update_freq = target_update_freq

        # action-type grouping
        if self.uses_hier_action and hasattr(self.env, "type_names"):
            self.type_names = list(self.env.type_names)
            self.type_to_indices = dict(self.env.type_to_indices)
            self.type_name_to_idx = dict(self.env.type_name_to_idx)
            self.num_types = len(self.type_names)
            self.index_to_type = dict(self.env.index_to_type)
            self.index_to_local = dict(self.env.index_to_local)
        else:
            self._build_action_groups(action_type_order)

        # Neural Network related attributes
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.type_dqn = DQN(self.obs_dim,
                            hidden_sizes,
                            self.num_types).to(self.device)
        self.target_type_dqn = DQN(self.obs_dim,
                                   hidden_sizes,
                                   self.num_types).to(self.device)

        self.action_dqns = {}
        self.target_action_dqns = {}
        for type_name in self.type_names:
            num_actions = len(self.type_to_indices[type_name])
            self.action_dqns[type_name] = DQN(
                self.obs_dim, hidden_sizes, num_actions
            ).to(self.device)
            self.target_action_dqns[type_name] = DQN(
                self.obs_dim, hidden_sizes, num_actions
            ).to(self.device)

        self.type_optimizer = optim.Adam(self.type_dqn.parameters(), lr=self.lr)
        self.action_optimizers = {
            name: optim.Adam(self.action_dqns[name].parameters(), lr=self.lr)
            for name in self.type_names
        }
        self.loss_fn = nn.SmoothL1Loss()

        # replay setup
        self.type_replay = ReplayMemory(replay_size,
                                        self.obs_dim,
                                        self.device)
        self.action_replays = {
            name: ReplayMemory(replay_size, self.obs_dim, self.device)
            for name in self.type_names
        }
        self.replay = _ReplayProxy(self)
        self._last_type_name = None

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

        # filter out empty types while preserving order
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

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def _select_type(self, o):
        o = torch.from_numpy(o).float().to(self.device)
        return self.type_dqn.get_action(o).cpu().item()

    def _select_action_in_type(self, o, type_idx):
        type_name = self.type_names[type_idx]
        dqn = self.action_dqns[type_name]
        o = torch.from_numpy(o).float().to(self.device)
        local_idx = dqn.get_action(o).cpu().item()
        if self.uses_hier_action:
            return local_idx
        return self.type_to_indices[type_name][local_idx]

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            type_idx = self._select_type(o)
            local_or_flat = self._select_action_in_type(o, type_idx)
            if self.uses_hier_action:
                return (type_idx, local_or_flat)
            return local_or_flat

        type_idx = random.randint(0, self.num_types - 1)
        type_name = self.type_names[type_idx]
        if self.uses_hier_action:
            local_idx = random.randint(0, len(self.type_to_indices[type_name]) - 1)
            return (type_idx, local_idx)
        return random.choice(self.type_to_indices[type_name])

    def _store_transition(self, s, a, next_s, r, done):
        if self.uses_hier_action:
            type_idx, local_idx = a
            type_idx = int(type_idx)
            local_idx = int(local_idx)
            type_name = self.type_names[type_idx]
        else:
            type_name = self.index_to_type[a]
            type_idx = self.type_name_to_idx[type_name]
            local_idx = self.index_to_local[a]
        self.type_replay.store(s, type_idx, next_s, r, done)
        self.action_replays[type_name].store(s, local_idx, next_s, r, done)
        self._last_type_name = type_name

    def _optimize_dqn(self, dqn, target_dqn, optimizer, replay):
        if replay.size < self.batch_size:
            return 0.0, 0.0

        batch = replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        q_vals_raw = dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        with torch.no_grad():
            target_q_val_raw = target_dqn(next_s_batch)
            target_q_val = target_q_val_raw.max(1)[0]
            target = r_batch + self.discount*(1-d_batch)*target_q_val

        loss = self.loss_fn(q_vals, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        return loss.item(), mean_v

    def optimize(self):
        type_loss, type_mean_v = self._optimize_dqn(
            self.type_dqn,
            self.target_type_dqn,
            self.type_optimizer,
            self.type_replay
        )

        action_loss = 0.0
        action_mean_v = 0.0
        if self._last_type_name is not None:
            action_loss, action_mean_v = self._optimize_dqn(
                self.action_dqns[self._last_type_name],
                self.target_action_dqns[self._last_type_name],
                self.action_optimizers[self._last_type_name],
                self.action_replays[self._last_type_name]
            )

        if self.steps_done % self.target_update_freq == 0:
            self.target_type_dqn.load_state_dict(self.type_dqn.state_dict())
            for type_name in self.type_names:
                self.target_action_dqns[type_name].load_state_dict(
                    self.action_dqns[type_name].state_dict()
                )

        loss = type_loss + action_loss
        mean_v = (type_mean_v + action_mean_v) / 2.0
        return loss, mean_v

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
                         render_mode="human"):
        if env is None:
            env = self.env

        original_render_mode = env.render_mode
        env.render_mode = render_mode

        o, _ = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render()
            input("Initial state. Press enter to continue..")

        while not done and not env_step_limit_reached:
            a = self.get_egreedy_action(o, eval_epsilon)
            next_o, r, done, env_step_limit_reached, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render()
                print(f"Reward = {r}")
                print(f"Done = {done}")
                print(f"Step limit reached = {env_step_limit_reached}")
                input("Press enter to continue..")

                if done or env_step_limit_reached:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        env.render_mode = original_render_mode
        return episode_return, steps, env.goal_reached()
