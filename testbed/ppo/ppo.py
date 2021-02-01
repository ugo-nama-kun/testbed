
from collections import deque
from typing import List

from attr import attrs, attrib

import  torch
from torch import Tensor

from testbed.ppo.value_network import ValueNetwork


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
                 ):
        """

        :param n_trajectory:  number of trajectories
        :param max_time_steps: maximum time step of trajectories
        """
        # RL params
        self._reward_discount = reward_discount

        # DNN params
        self._value_netwotk = ValueNetwork(
            dim_observation=dim_observation,
            hidden_size1=20,
            hidden_size2=20
        )
        self._param_policy = None

        # PPO params
        self._n_trajectory = n_trajectory
        self._max_time_steps = max_time_steps

        # Internal state of the system
        self._trajectory = deque()
        self._data_buffer = deque(maxlen=n_trajectory)
        self._prev_action = None

    def step(self,
             observation,
             reward,
             is_done):
        """
        Collect trajectory
        :param observation:
        :param reward:
        :param is_done:
        :return:
        """
        action = None  # TODO: get action
        self._add_data_into_buffer(observation,
                                   reward,
                                   action,
                                   is_done)
        return action

    def _add_data_into_buffer(self, observation, reward, action, is_done):
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
            for t, _ in enumerate(traj):
                sum_rew = 0
                discount = 1.0
                for experience in list(traj)[(t+1):]:
                    sum_rew += discount * experience.reward
                    discount *= self._reward_discount

                # bootstrap
                obs_final = Tensor(traj[-1].observation)
                value_t = self._evaluate_vf(obs_final).detach()
                sum_rew += discount * value_t

                # reward to go
                reward_to_go[n, t] = sum_rew

                # advantage
                obs_t = Tensor(traj[t].observation)
                value_t = self._evaluate_vf(obs_t).detach()
                advantage[n, t] = reward_to_go[n, t] - value_t

        return advantage, reward_to_go

    def update_vf(self):
        pass

    def _evaluate_vf(self, observation: Tensor) -> Tensor:
        return self._value_netwotk.forward(observation)
