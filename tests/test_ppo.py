import numpy as np
from pytest import approx

from torch import Tensor

from testbed.ppo.ppo import PPOAgent


def test_add_sample_into_trajectory():
    agent = PPOAgent(reward_discount=1.0, n_trajectory=3, max_time_steps=100)
    assert len(agent._data_buffer) == 0

    agent._add_data_into_buffer(observation=[0],
                                reward=1,
                                action=[2],
                                is_done=False)
    assert len(agent._trajectory) == 1
    assert agent._trajectory[0].observation == [0]
    assert agent._trajectory[0].reward == 1
    assert agent._trajectory[0].action == [2]
    assert agent._trajectory[0].is_done is False


def test_save_multiple_trajectories():
    agent = PPOAgent(reward_discount=1.0, n_trajectory=3, max_time_steps=10)
    assert len(agent._data_buffer) == 0

    for i in range(3 * 10):
        agent._add_data_into_buffer(observation=[0],
                                    reward=0,
                                    action=[0],
                                    is_done=False)
    assert len(agent._data_buffer) == 3
    assert len(agent._data_buffer[0]) == 10


def test_save_multiple_trajectories_with_done():
    agent = PPOAgent(reward_discount=1.0, n_trajectory=3, max_time_steps=10)
    assert len(agent._data_buffer) == 0

    for i in range(3):
        for j in range(3 * (i + 1)):
            is_done = j == 3 * (i + 1) - 1
            agent._add_data_into_buffer(observation=[0],
                                        reward=0,
                                        action=[0],
                                        is_done=is_done)
        assert len(agent._data_buffer) == i + 1

    assert len(agent._data_buffer[0]) == 3
    assert len(agent._data_buffer[1]) == 6
    assert len(agent._data_buffer[2]) == 9


def test_get_advantage():
    agent = PPOAgent(reward_discount=1.0,
                     n_trajectory=3,
                     max_time_steps=10,
                     dim_observation=2)
    for i in range(3):
        for j in range(10):
            obs = [0, 0]
            agent._add_data_into_buffer(observation=obs,
                                        reward=1,
                                        action=[0],
                                        is_done=False)

    advantage, reward_to_go = agent._get_advantage()
    assert reward_to_go.size() == (3, 10)

    for n, traj in enumerate(agent._data_buffer):
        for t, expr in enumerate(traj):
            obs_final = Tensor(traj[-1].observation)
            value_t = agent._evaluate_vf(obs_final).detach()[0]

            assert approx((9 + value_t - t).tolist(), reward_to_go[n, t].tolist())
            obs = Tensor(expr.observation)
            assert approx((9 - t + value_t - agent._evaluate_vf(obs)).tolist(), advantage[n, t].tolist())


def test_agent_step():
    agent = PPOAgent(reward_discount=1.0,
                     n_trajectory=3,
                     max_time_steps=10,
                     dim_observation=2,
                     dim_action=2)

    action = agent.step(
        observation=[5.2, 3.1],
        reward=1,
        is_done=False
    )
    assert action.size()[0] == 2


def test_vf_update():
    len_traj = 5
    agent = PPOAgent(reward_discount=1.0,
                     n_trajectory=3,
                     max_time_steps=len_traj,
                     dim_observation=2,
                     lr_value=0.05,
                     iter_op_vf=100,
                     )

    # Data for value estimation
    # Value of all sequence should be one.
    for i in range(3):
        for t in range(len_traj):
            r = 1 if t == len_traj - 1 else 0
            is_done = True if t == len_traj - 1 else False
            agent._add_data_into_buffer(observation=[t/float(len_traj), 1 - t/float(len_traj)],
                                        reward=r,
                                        action=[0, 0],
                                        is_done=is_done)

    advantage, reward_to_go = agent._get_advantage()
    # print(reward_to_go)
    agent._update_vf(reward_to_go)

    for i in range(3):
        for t in range(len_traj):
            val = agent._evaluate_vf(
                observation=Tensor([t/float(len_traj), 1. - t/float(len_traj)])
            ).detach()
            if t == len_traj - 1:
                assert 0 == approx(val, abs=0.2)
            else:
                assert 1 == approx(val, abs=0.2)


def test_policy_update():
    agent = PPOAgent(reward_discount=0.0,
                     n_trajectory=3,
                     max_time_steps=10,
                     dim_observation=2,
                     lr_value=0.1,
                     iter_op_vf=3,
                     lr_policy=0.1,
                     iter_op_policy=3,
                     )

    # Data for value estimation
    # Value of all sequence should be one.
    for i in range(3):
        for t in range(10):
            r = 1 if t == 9 else 0
            is_done = True if t == 9 else False
            agent.step(
                observation=[t / 9., 1 - (t / 9.)],
                reward=r,
                is_done=is_done,
                is_test=True,
            )

    # just a running check
    advantage, reward_to_go = agent._get_advantage()
    agent._policy_update(advantage)


def test_reward_normalizer():
    agent = PPOAgent(reward_discount=0.9,
                     n_trajectory=3,
                     max_time_steps=10,
                     dim_observation=2)

    data = [np.random.randn(1) for i in range(5)]
    for x in data:
        agent.step(
            observation=[0, 1],
            reward=x,
            is_done=False,
            is_test=True,
        )

    assert 10. / np.std(data) == approx(agent._normalize_reward(10.), abs=0.01)
