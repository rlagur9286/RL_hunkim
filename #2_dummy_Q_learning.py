import numpy as np
import gym
import tensorflow as tf
import random

from matplotlib import pyplot as plt
from gym.envs.registration import register

register(
    id='FrozenLake-v2',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)
env = gym.make('FrozenLake-v2')

Q = np.zeros([env.observation_space.n, env.action_space.n])


def rargmax(vector):
    n = np.amax(vector)
    indices = np.nonzero(vector == n)[0]
    return random.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

num_episodes = 2000

r_list = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = rargmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    r_list.append(rAll)

print('Success rate : ', str(sum(r_list)/num_episodes))
print('Final Q-Table Values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.bar(range(len(r_list)), r_list, color="b")
plt.show()