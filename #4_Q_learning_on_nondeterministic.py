import gym
import numpy as np
from gym.envs.registration import register
from matplotlib import pyplot as plt

learning_rate = 0.85

env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
dis = 0.99

r_list = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    # e-greedy
    # e = 1 / ((i / 100) + 1)

    while not done:
        """
        # 방법1 - e-greedy를 사용하는 방법
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        """
        # 방법2 - noise를 뿌리는 방법
        action = np.argmax(Q[state, :] + np.random.rand(1, env.action_space.n) / (i+1))

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    r_list.append(rAll)

print('Success rate : ', str(sum(r_list) / num_episodes))
print('Final Q-Table Values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.bar(range(len(r_list)), r_list, color="b")
plt.show()