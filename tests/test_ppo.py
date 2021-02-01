from pytest import approx

from torch import Tensor

from testbed.ppo.ppo import PPOAgent


def test_add_sample_into_trajectory():
    agent = PPOAgent(reward_discount=1.0, n_trajectory=3, max_time_steps=100)
    assert len(agent._data_buffer) == 0

    agent._add_data_into_buffer(observation=0,
                                reward=1,
                                action=2,
                                is_done=False)
    assert len(agent._trajectory) == 1
    assert agent._trajectory[0].observation == 0
    assert agent._trajectory[0].reward == 1
    assert agent._trajectory[0].action == 2
    assert agent._trajectory[0].is_done is False


def test_save_multiple_trajectories():
    agent = PPOAgent(reward_discount=1.0, n_trajectory=3, max_time_steps=10)
    assert len(agent._data_buffer) == 0

    for i in range(3*10):
        agent._add_data_into_buffer(observation=0,
                                    reward=0,
                                    action=0,
                                    is_done=False)
    assert len(agent._data_buffer) == 3
    assert len(agent._data_buffer[0]) == 10


def test_save_multiple_trajectories_with_done():
    agent = PPOAgent(reward_discount=1.0, n_trajectory=3, max_time_steps=10)
    assert len(agent._data_buffer) == 0

    for i in range(3):
        for j in range(3 * (i+1)):
            is_done = j == 3*(i+1) - 1
            agent._add_data_into_buffer(observation=0,
                                        reward=0,
                                        action=0,
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
                                        action=0,
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
