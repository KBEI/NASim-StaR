import numpy as np

from nasim.envs.action import ActionResult
from nasim.envs.utils import get_minimal_hops_to_goal, min_subnet_depth, AccessLevel
import nasim.scenarios.utils as u

# column in topology adjacency matrix that represents connection between
# subnet and public
INTERNET = 0


class Network:
    """A computer network """

    def __init__(self, scenario):
        self.scenario = scenario
        self.hosts = scenario.hosts
        self.host_num_map = scenario.host_num_map
        self.subnets = scenario.subnets
        self.topology = scenario.topology
        self.firewall = scenario.firewall
        self.address_space = scenario.address_space
        self.address_space_bounds = scenario.address_space_bounds
        self.sensitive_addresses = scenario.sensitive_addresses
        self.sensitive_hosts = scenario.sensitive_hosts
        self.alerts_count = 0
        self.isolated_hosts = set()
        self.monitoring_level = 1.0
        self.monitoring_steps_left = 0
        self.firewall_overrides = {}
        self.last_alert_host = None

    def reset(self, state):
        """Reset the network state to initial state """
        self.alerts_count = 0
        self.isolated_hosts = set()
        self.monitoring_level = 1.0
        self.monitoring_steps_left = 0
        self.firewall_overrides = {}
        self.last_alert_host = None
        next_state = state.copy()
        for host_addr in self.address_space:
            host = next_state.get_host(host_addr)
            host.compromised = False
            host.access = AccessLevel.NONE
            host.reachable = self.subnet_public(host_addr[0])
            host.discovered = host.reachable
        return next_state

    def perform_action(self, state, action):
        """Perform the given Action against the network.

        Arguments
        ---------
        state : State
            the current state
        action : Action
            the action to perform

        Returns
        -------
        State
            the state after the action is performed
        ActionObservation
            the result from the action
        """
        tgt_subnet, tgt_id = action.target
        assert 0 < tgt_subnet < len(self.subnets)
        assert tgt_id <= self.subnets[tgt_subnet]

        next_state = state.copy()

        if action.is_noop():
            return next_state, ActionResult(True)

        if action.target in self.isolated_hosts:
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if not state.host_reachable(action.target) \
           or not state.host_discovered(action.target):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        has_req_permission = self.has_required_remote_permission(state, action)
        if action.is_remote() and not has_req_permission:
            result = ActionResult(False, 0.0, permission_error=True)
            return next_state, result

        if action.is_exploit() \
           and not self.traffic_permitted(
                    state, action.target, action.service
           ):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        host_compromised = state.host_compromised(action.target)
        if action.is_privilege_escalation() and not host_compromised:
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if action.is_exploit() and host_compromised:
            # host already compromised so exploits don't fail due to randomness
            pass
        elif np.random.rand() > action.prob:
            return next_state, ActionResult(False, 0.0, undefined_error=True)

        if action.is_subnet_scan():
            next_state, action_obs = self._perform_subnet_scan(
                next_state, action
            )
            self._apply_detection(action, action_obs)
            return next_state, action_obs

        t_host = state.get_host(action.target)
        next_host_state, action_obs = t_host.perform_action(action)
        next_state.update_host(action.target, next_host_state)
        self._update(next_state, action, action_obs)
        self._apply_detection(action, action_obs)
        return next_state, action_obs

    def _perform_subnet_scan(self, next_state, action):
        if not next_state.host_compromised(action.target):
            result = ActionResult(False, 0.0, connection_error=True)
            return next_state, result

        if not next_state.host_has_access(action.target, action.req_access):
            result = ActionResult(False, 0.0, permission_error=True)
            return next_state, result

        discovered = {}
        newly_discovered = {}
        discovery_reward = 0
        target_subnet = action.target[0]
        for h_addr in self.address_space:
            newly_discovered[h_addr] = False
            discovered[h_addr] = False
            if self.subnets_connected(target_subnet, h_addr[0]):
                host = next_state.get_host(h_addr)
                discovered[h_addr] = True
                if not host.discovered:
                    newly_discovered[h_addr] = True
                    host.discovered = True
                    discovery_reward += host.discovery_value

        obs = ActionResult(
            True,
            discovery_reward,
            discovered=discovered,
            newly_discovered=newly_discovered
        )
        return next_state, obs

    def _apply_detection(self, action, action_obs):
        if not self.scenario.detection_enabled:
            return
        if action.is_noop() or not action_obs.success:
            return
        honeypot_cfg = self._get_honeypot_config(action.target)
        if honeypot_cfg is not None and honeypot_cfg.get(
                u.HONEYPOT_FORCE_DETECTED, False
        ):
            self._record_detection(action, action_obs, honeypot_cfg)
            return

        base_prob = self._get_detection_base_prob(action, honeypot_cfg)
        if base_prob <= 0.0:
            return
        if self.monitoring_level > 1.0:
            base_prob = min(1.0, base_prob * self.monitoring_level)
        cost_factor = self.scenario.detection_cost_stealth_factor
        if cost_factor > 0:
            base_prob = base_prob / (1.0 + cost_factor * action.cost)
        if np.random.rand() <= base_prob:
            self._record_detection(action, action_obs, honeypot_cfg)

    def _record_detection(self, action, action_obs, honeypot_cfg):
        action_obs.detected = True
        self.alerts_count += 1
        self.last_alert_host = action.target
        self._maybe_trigger_response(action, action_obs, honeypot_cfg)

    def _get_detection_base_prob(self, action, honeypot_cfg=None):
        if honeypot_cfg is not None and u.HONEYPOT_DETECTION_PROB in honeypot_cfg:
            return honeypot_cfg[u.HONEYPOT_DETECTION_PROB]
        base_prob = self.scenario.detection_base_prob
        if action.is_service_scan():
            return base_prob.get("service_scan", 0.0)
        if action.is_os_scan():
            return base_prob.get("os_scan", 0.0)
        if action.is_subnet_scan():
            return base_prob.get("subnet_scan", 0.0)
        if action.is_process_scan():
            return base_prob.get("process_scan", 0.0)
        if action.is_exploit():
            return base_prob.get("exploit", 0.0)
        if action.is_privilege_escalation():
            return base_prob.get("privilege_escalation", 0.0)
        return 0.0

    def _maybe_trigger_response(self, action, action_obs, honeypot_cfg):
        if not self.scenario.response_enabled:
            return
        threshold_reached = (
            self.alerts_count >= self.scenario.response_alert_threshold
        )
        honeypot_trigger = (
            honeypot_cfg is not None
            and honeypot_cfg.get(u.HONEYPOT_TRIGGER_RESPONSE, False)
        )
        if not (threshold_reached or honeypot_trigger):
            return
        self._apply_response_actions()

    def _apply_response_actions(self):
        for action in self.scenario.response_actions:
            action_type = action.get(u.RESPONSE_ACTION_TYPE)
            if action_type == u.RESPONSE_ACTION_ISOLATE_HOST:
                if self.last_alert_host is not None:
                    self.isolated_hosts.add(self.last_alert_host)
            elif action_type == u.RESPONSE_ACTION_TIGHTEN_FIREWALL:
                connections = action.get(u.RESPONSE_FIREWALL_CONNECTIONS, {})
                for conn, remove_services in connections.items():
                    current = list(self._get_firewall_allowed(*conn))
                    updated = [s for s in current if s not in remove_services]
                    self.firewall_overrides[conn] = updated
            elif action_type == u.RESPONSE_ACTION_INCREASE_MONITORING:
                factor = action.get(u.RESPONSE_MONITORING_FACTOR, 1.0)
                steps = action.get(u.RESPONSE_MONITORING_STEPS, 1)
                if factor > self.monitoring_level:
                    self.monitoring_level = factor
                self.monitoring_steps_left = max(
                    self.monitoring_steps_left, steps
                )

    def tick_monitoring(self):
        if self.monitoring_steps_left > 0:
            self.monitoring_steps_left -= 1
            if self.monitoring_steps_left <= 0:
                self.monitoring_level = 1.0

    def under_increased_monitoring(self):
        return self.monitoring_steps_left > 0

    def any_sensitive_isolated(self):
        for addr in self.sensitive_addresses:
            if addr in self.isolated_hosts:
                return True
        return False

    def _get_honeypot_config(self, host_addr):
        return self.scenario.honeypots.get(host_addr)

    def _update(self, state, action, action_obs):
        if action.is_exploit() and action_obs.success:
            self._update_reachable(state, action.target)

    def _update_reachable(self, state, compromised_addr):
        """Updates the reachable status of hosts on network, based on current
        state and newly exploited host
        """
        comp_subnet = compromised_addr[0]
        for addr in self.address_space:
            if state.host_reachable(addr):
                continue
            if self.subnets_connected(comp_subnet, addr[0]):
                state.set_host_reachable(addr)

    def get_sensitive_hosts(self):
        return self.sensitive_addresses

    def is_sensitive_host(self, host_address):
        return host_address in self.sensitive_addresses

    def subnets_connected(self, subnet_1, subnet_2):
        return self.topology[subnet_1][subnet_2] == 1

    def subnet_traffic_permitted(self, src_subnet, dest_subnet, service):
        if src_subnet == dest_subnet:
            # in same subnet so permitted
            return True
        if not self.subnets_connected(src_subnet, dest_subnet):
            return False
        return service in self._get_firewall_allowed(src_subnet, dest_subnet)

    def _get_firewall_allowed(self, src_subnet, dest_subnet):
        key = (src_subnet, dest_subnet)
        if key in self.firewall_overrides:
            return self.firewall_overrides[key]
        return self.firewall[key]

    def host_traffic_permitted(self, src_addr, dest_addr, service):
        dest_host = self.hosts[dest_addr]
        return dest_host.traffic_permitted(src_addr, service)

    def has_required_remote_permission(self, state, action):
        """Checks attacker has necessary permissions for remote action """
        if self.subnet_public(action.target[0]):
            return True

        for src_addr in self.address_space:
            if not state.host_compromised(src_addr):
                continue
            if action.is_scan() and \
               not self.subnets_connected(src_addr[0], action.target[0]):
                continue
            if action.is_exploit() and \
               not self.subnet_traffic_permitted(
                   src_addr[0], action.target[0], action.service
               ):
                continue
            if state.host_has_access(src_addr, action.req_access):
                return True
        return False

    def traffic_permitted(self, state, host_addr, service):
        """Checks whether the subnet and host firewalls permits traffic to a
        given host and service, based on current set of compromised hosts on
        network.
        """
        for src_addr in self.address_space:
            if not state.host_compromised(src_addr) and \
               not self.subnet_public(src_addr[0]):
                continue
            if not self.subnet_traffic_permitted(
                    src_addr[0], host_addr[0], service
            ):
                continue
            if self.host_traffic_permitted(src_addr, host_addr, service):
                return True
        return False

    def subnet_public(self, subnet):
        return self.topology[subnet][INTERNET] == 1

    def get_number_of_subnets(self):
        return len(self.subnets)

    def all_sensitive_hosts_compromised(self, state):
        for host_addr in self.sensitive_addresses:
            if not state.host_has_access(host_addr, AccessLevel.ROOT):
                return False
        return True

    def get_total_sensitive_host_value(self):
        total = 0
        for host_value in self.sensitive_hosts.values():
            total += host_value
        return total

    def get_total_discovery_value(self):
        total = 0
        for host in self.hosts.values():
            total += host.discovery_value
        return total

    def get_minimal_hops(self):
        return get_minimal_hops_to_goal(
            self.topology, self.sensitive_addresses
        )

    def get_subnet_depths(self):
        return min_subnet_depth(self.topology)

    def __str__(self):
        output = "\n--- Network ---\n"
        output += "Subnets: " + str(self.subnets) + "\n"
        output += "Topology:\n"
        for row in self.topology:
            output += f"\t{row}\n"
        output += "Sensitive hosts: \n"
        for addr, value in self.sensitive_hosts.items():
            output += f"\t{addr}: {value}\n"
        output += "Num_services: {self.scenario.num_services}\n"
        output += "Hosts:\n"
        for m in self.hosts.values():
            output += str(m) + "\n"
        output += "Firewall:\n"
        for c, a in self.firewall.items():
            output += f"\t{c}: {a}\n"
        return output
