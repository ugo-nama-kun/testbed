import gym
import visdom
import numpy as np

from testbed.ppo.ppo import PPOAgent

vis = visdom.Visdom()

env = gym.make('Pendulum-v0')


def obs_scale(observation):
    x = (observation - env.observation_space.low)/(env.observation_space.high - env.observation_space.low)
    return 2. * x - 1.


print(f"dim_obs: {len(env.observation_space.high)}")
print(f"dim_action: {len(env.action_space.high)}")

agent = PPOAgent(
    reward_discount=0.9,
    n_trajectory=10,
    max_time_steps=20,
    dim_observation=len(env.observation_space.high),
    dim_action=len(env.action_space.high),
    lr_value=0.01,
    iter_op_vf=3,
    iter_op_policy=1,
    lr_policy=0.01,
    eps_policy_clip=0.2)

if __name__ == '__main__':
    episode = 0
    rew_sum_list = []
    while True:
        t = 0
        done = False
        observation = env.reset()
        rew_sum = 0
        while True:
            env.render()
            obs = obs_scale(observation)
            action = agent.step(
                observation=obs,
                reward=0,
                is_done=done
            )

            u = env.action_space.high * action.numpy()
            observation, reward, done, info = env.step(u)
            rew_sum += reward
            if done:
                print(f"Episode finished after {t + 1} timesteps: Reward sum : {rew_sum}")
                episode += 1
                rew_sum_list.append(rew_sum)
                vis.line(X=np.array(range(episode)),
                         Y=np.array(rew_sum_list),
                         win='performance',
                         name='reward_sum')
                break
            t += 1
    env.close()
