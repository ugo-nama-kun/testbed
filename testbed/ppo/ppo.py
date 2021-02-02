
from collections import deque
from typing import List

from attr import attrs, attrib

import torch
from torch import Tensor
import torch.optim as optim

from testbed.ppo.network import ValueNetwork, PolicyNetwork


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
            hidden_size1=20,
            hidden_size2=20
        )

        self._policy_network = PolicyNetwork(
            dim_observation=dim_observation,
            hidden_size1=20,
            hidden_size2=20,
            dim_action=dim_action
        )
        self._policy_old = PolicyNetwork(
            dim_observation=dim_observation,
            hidden_size1=20,
            hidden_size2=20,
            dim_action=dim_action
        )
        # copy policy into the old-policy network
        self._policy_old.load_state_dict(state_dict=self._policy_network.state_dict())

        # Optimizer
        self._optimizer_vf = optim.Adam(
            params=self._value_network.parameters(),
            lr=lr_value,
        )
        self._optimizer_policy = optim.Adam(
            params=self._policy_network.parameters(),
            lr=lr_policy,
        )

        # PPO params
        self._n_trajectory = n_trajectory
        self._max_time_steps = max_time_steps
        self._iteration_op_value = iter_op_vf
        self._eps_policy_clip = eps_policy_clip

        # Internal state of the system
        self._trajectory = deque()
        self._data_buffer = deque(maxlen=n_trajectory)
        self._prev_action = None

    def step(self,
             observation: List,
             reward: float,
             is_done: bool):
        """
        Collect trajectory
        :param observation:
        :param reward:
        :param is_done:
        :return:
        """
        action = self._policy_network.sample(Tensor(observation)).detach()
        self._add_data_into_buffer(observation,
                                   reward,
                                   action,
                                   is_done)
        return action

    def _add_data_into_buffer(self,
                              observation: List,
                              reward: float,
                              action: List,
                              is_done: bool):
        """
        Trajectory Experience Buffer. The length of trajectories are shorter than than max_time_steps
        :param observation:
        :param reward:
        :param action:
        :param is_done:
        :return:
        """
        self._trajectory.append(Experience(observation, reward, action, is_done))
        if len(self._trajectory) == self._max_time_steps or is_done:
            self._data_buffer.append(self._trajectory)
            self._trajectory = deque()

    def update_policy(self):
        advantage, reward_to_go = self._get_advantage()


    def _get_advantage(self) -> (Tensor, Tensor):
        reward_to_go = torch.zeros((self._n_trajectory, self._max_time_steps))
        advantage = torch.zeros_like(reward_to_go)
        for n, traj in enumerate(self._data_buffer):
            # bootstrap
            obs_final = Tensor(traj[-1].observation)
            value_final = self._evaluate_vf(obs_final).detach()
            for t, _ in enumerate(traj):
                sum_rew = 0
                discount = 1.0
                for experience in list(traj)[(t+1):]:
                    # TODO: apply Generalized advantage estimation (GAE)
                    sum_rew += discount * experience.reward
                    discount *= self._reward_discount

                # bootstrap
                sum_rew += discount * value_final

                # reward to go
                reward_to_go[n, t] = sum_rew

                # advantage
                obs_t = Tensor(traj[t].observation)
                value_t = self._evaluate_vf(obs_t).detach()
                advantage[n, t] = reward_to_go[n, t] - value_t

        return advantage, reward_to_go

    def _update_vf(self):
        # TODO: update to efficient tensor computation
        # TODO: summarize with advantage calculation
        for epoch in range(self._iteration_op_value):
            sum_error = torch.zeros(1)
            len_data = 0.
            for n, traj in enumerate(self._data_buffer):
                len_data += len(traj)
                # for bootstrap
                obs_final = Tensor(traj[-1].observation)
                value_final = self._evaluate_vf(obs_final).detach()
                for t, _ in enumerate(traj):
                    sum_rew = 0.
                    discount = 1.0
                    for experience in list(traj)[(t+1):]:
                        # TODO: apply Generalized advantage estimation (GAE)
                        sum_rew += discount * experience.reward
                        discount *= self._reward_discount

                    # bootstrap
                    sum_rew += discount * value_final

                    # error
                    obs_t = Tensor(traj[t].observation)
                    value_t = self._evaluate_vf(obs_t)
                    sum_error += (sum_rew - value_t)**2  # td error minimization?

            sum_error /= len_data
            # print(f"VF minimization {epoch + 1}/{self._iteration_op_value}: {sum_error.detach().numpy()}")
            self._optimizer_vf.zero_grad()
            sum_error.backward()
            self._optimizer_vf.step()

    def _evaluate_vf(self, observation: Tensor) -> Tensor:
        return self._value_network.forward(observation)
