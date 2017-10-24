import gym
import numpy as np
import tensorflow as tf
from gym.envs.registration import register
from matplotlib import pyplot as plt

env = gym.make('CartPole-v0')
env.reset()
random_episode = 0
reward_sum = 0

# Contants defining for neural network
learning_rate = 1e-1
input_size = env.observation_space.shape[0]    # 4
output_size = env.action_space.n    # 2 (<-, ->)

X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
W = tf.get_variable('weight', shape=[input_size, output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(X, W)

Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# Loss func
loss = tf.reduce_sum(tf.square(Qpred - Y))

# Leaerning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

# Value for Q Learning
num_epoch = 2000
dis = 0.9
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epoch):
        e = 1 / ((i / 10) + 1)
        step_count = 0
        state = env.reset()
        done = False

        while not done:
            step_count += 1
            x = np.reshape(state, [1, input_size])

            Qs = sess.run(Qpred, feed_dict={X: x})
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            # Get new state and reward from env
            new_state, reward, done, _ = env.step(action)

            if done:
                Qs[0, action] = -100
            else:
                x_reshape = np.reshape(new_state, [1, input_size])
                Qs1 = sess.run(Qpred, feed_dict={X: x_reshape})
                Qs[0, action] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X: x, Y: Qs})
            state = new_state
        rList.append(step_count)
        print('Episode: {}  step: {}'.format(i, step_count))

        if len(rList) > 10 and np.mean(rList[-10:]) > 500:
            break

    observation = env.reset()
    reward_sum = 0
    while True:
        env.render()

        x = np.reshape(observation, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        action = np.argmax(Qs)

        observation, reward, done, _ = env.step(action)
        reward_sum += reward

        if done:
            print('Total Score : {}'.format(reward_sum))
            break