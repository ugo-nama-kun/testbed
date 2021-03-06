from collections import deque
from typing import List, Optional

import numpy as np
from attr import attrs, attrib

import torch
from torch import Tensor
import torch.optim as optim

from testbed.ppo.network import ValueNetwork, PolicyNetwork
from testbed.ppo.util import RunningMeanStd


@attrs
class Experience:
    observation = attrib()
    reward = attrib()
    action = attrib()
    is_done = attrib()


class PPOAgent:
    def __init__(self,
                 reward_discount,
                 n_trajectory,
                 max_time_steps,
                 dim_observation=2,
                 dim_action=2,
                 lr_value=0.001,
                 iter_op_vf=20,
                 iter_op_policy=10,
                 lr_policy=0.001,
                 eps_policy_clip=0.2
                 ):
        """

        :param n_trajectory:  number of trajectories
        :param max_time_steps: maximum time step of trajectories
        """
        # RL params
        self._reward_discount = reward_discount

        # DNN params
        self._value_network = ValueNetwork(
            dim_observation=dim_observation,
            hidden_size1=200,
            hidden_size2=200
        )

        self._policy_network = PolicyNetwork(
            dim_observation=dim_observation,
            hidden_size1=200,
            hidden_size2=200,
            dim_action=dim_action
        )
        self._policy_old = PolicyNetwork(
            dim_observation=dim_observation,
            hidden_size1=200,
            hidden_size2=200,
            dim_action=dim_action
        )
        # copy policy into the old-policy network
        self._policy_old.load_state_dict(state_dict=self._policy_network.state_dict())

        # Optimizer
        self._optimizer_vf = optim.Adam(
            params=self._value_network.parameters(),
            lr=lr_value,
            weight_decay=1.0e-6
        )
        self._optimizer_policy = optim.Adam(
            params=self._policy_network.parameters(),
            lr=lr_policy,
            weight_decay=1.0e-6
        )

        # PPO params
        self._n_trajectory = n_trajectory
        self._max_time_steps = max_time_steps
        self._iteration_op_value = iter_op_vf
        self._iteration_op_policy = iter_op_policy
        self._eps_policy_clip = eps_policy_clip

        # Internal state of the system
        self._rms_reward = RunningMeanStd()
        self._rms_observation = RunningMeanStd()
        self._trajectory = deque()
        self._data_buffer = deque(maxlen=n_trajectory)
        self._prev_action = None
        self._prev_obs = None
        self._num_iteration = 0

    @property
    def num_iteration(self):
        return self._num_iteration

    def step(self,
             observation: List,
             reward: Optional[float],
             is_done: bool,
             is_test=False):
        """
        Collect trajectory
        :param is_test:
        :param observation:
        :param reward:
        :param is_done:
        :return:
        """
        # Normalizer update
        self._rms_observation.update(np.array(observation))
        if reward:
            reward = np.clip(reward, a_min=-10, a_max=10)
            self._rms_reward.update(np.array([reward]))

        # Taking Action
        action = self._policy_network.sample(Tensor([observation])).detach()

        # Store data into buffer
        if reward is not None:
            self._add_data_into_buffer(self._prev_obs if self._prev_obs is not None else observation,
                                       self._prev_action if self._prev_action is not None else action,
                                       reward,
                                       is_done)
        self._prev_obs = observation
        self._prev_action = action

        if len(self._data_buffer) == self._n_trajectory and not is_test:
            print(f"{self._num_iteration+1}-th Iteration...")
            self._model_update()
            self._num_iteration += 1
            print("done!")
        return action

    def _model_update(self):
        """
        Update process of Value network and Policy network
        :return:
        """
        advantage, reward_to_go = self._get_advantage()
        # policy update
        self._policy_update(advantage)
        # value update
        self._update_vf(reward_to_go)
        # clear previous trajectory
        self._data_buffer.clear()

    def _policy_update(self, advantage):
        for epoch in range(self._iteration_op_policy):
            loss = torch.zeros(1)
            len_data = 0
            for n, traj in enumerate(self._data_buffer):
                len_data += len(traj)
                # per-batch normalization
                advantage_ = (advantage[n,:] - torch.mean(advantage[n,:])) / torch.std(advantage[n,:])
                for t, experience in enumerate(traj):
                    log_p = self._policy_network.log_prob(Tensor(experience.observation), Tensor(experience.action))
                    log_p_old = self._policy_old.log_prob(Tensor(experience.observation), Tensor(experience.action)).detach()
                    ratio = torch.exp(log_p - log_p_old)
                    tmp = torch.minimum(
                        ratio,
                        torch.clip(ratio, max=1 + self._eps_policy_clip, min=1 - self._eps_policy_clip)
                    )
                    loss += - tmp * advantage_[t]
            loss /= len_data
            print(f"policy loss {epoch + 1}/{self._iteration_op_policy}: {len_data * loss.detach()}")
            self._optimizer_policy.zero_grad()
            loss.backward()
            self._optimizer_policy.step()

        # copy old policy
        self._policy_old.load_state_dict(self._policy_network.state_dict())

    def _add_data_into_buffer(self,
                              observation: List,
                              action: List,
                              reward: float,
                              is_done: bool):
        """
        Trajectory Experience Buffer. The length of trajectories are shorter than than max_time_steps
        :param observation: RAW (non-scaled) observation is registered
        :param reward: RAW (non-scaled) reward is registered
        :param action: RAW (non-scaled from network output) action is registered
        :param is_done:
        :return:
        """
        self._trajectory.append(Experience(observation, reward, action, is_done))
        if len(self._trajectory) == self._max_time_steps or is_done:
            self._data_buffer.append(self._trajectory)
            self._trajectory = deque()

    def _get_advantage(self) -> (Tensor, Tensor):
        # TODO: update to efficient tensor computation
        reward_to_go = torch.zeros((self._n_trajectory, self._max_time_steps))
        advantage = torch.zeros_like(reward_to_go)
        for n, traj in enumerate(self._data_buffer):
            # bootstrap
            obs_final = Tensor(traj[-1].observation)
            value_final = self._evaluate_vf(obs_final).detach()
            for t, _ in enumerate(traj):
                sum_rew = 0
                discount = 1.0
                for experience in list(traj)[(t + 1):]:
                    # TODO: apply Generalized advantage estimation (GAE)
                    sum_rew += discount * self._normalize_reward(experience.reward)
                    discount *= self._reward_discount

                # bootstrap
                if not traj[-1].is_done:
                    sum_rew += discount * value_final
                else:
                    pass
                    #print("is_done == True!")

                # reward to go
                reward_to_go[n, t] = sum_rew

                # advantage
                # print(f"{len(traj)}, {t}, {n}")
                obs_t = Tensor(traj[t].observation)
                value_t = self._evaluate_vf(obs_t).detach()
                advantage[n, t] = reward_to_go[n, t] - value_t

        return advantage, reward_to_go.detach()

    def _update_vf(self, reward_to_go: Tensor):
        # TODO: update to efficient tensor computation
        for epoch in range(self._iteration_op_value):
            loss = torch.zeros(1)
            len_data = 0.
            for n, traj in enumerate(self._data_buffer):
                len_data += len(traj)
                for t, _ in enumerate(traj):
                    obs_t = Tensor(traj[t].observation)
                    value_t = self._evaluate_vf(obs_t)
                    loss += (reward_to_go[n, t] - value_t) ** 2

            loss /= len_data
            print(f"value loss {epoch + 1}/{self._iteration_op_value}: {loss.detach()}")
            self._optimizer_vf.zero_grad()
            loss.backward()
            self._optimizer_vf.step()

    def _evaluate_vf(self, observation: Tensor) -> Tensor:
        return self._value_network.forward(self._normalize_observation(observation))

    def _normalize_reward(self, reward: float):
        return reward / np.sqrt(self._rms_reward.var)

    def _normalize_observation(self, observation: Tensor):
        return ((observation - self._rms_observation.mean)/np.sqrt(self._rms_observation.var)).detach()
