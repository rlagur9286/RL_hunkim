import tensorflow as tf
import numpy as np
import gym
from matplotlib import pyplot as plt
from gym.envs.registration import register

env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.1

# set Q-learning Params
dis = 0.99
num_epoch = 2000


def one_hot(x):
    return np.identity(16)[x:x+1]

# input and output size based on the Env
input_size = env.observation_space.n # 16
output_size = env.action_space.n # 4

# These line establish the feed-forword part of the network used to choice actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32) # state input
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))   # weight
Qpred = tf.matmul(X, W) # output, Q Prediction
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32) # Y Label

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_epoch):
        state = env.reset()
        e = 1 / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        while not done:
            Qs = sess.run(Qpred, feed_dict={X: one_hot(state)})
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            # Get new state and reward from env
            next_state, reward, done, _ = env.step(action)

            if done:
                # update Q, and no Qs1, since it's a terminal state
                Qs[0, action] = reward
            else:
                # Obtaon Qs1 value by feeding the new state through out network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(next_state)})
                # update Q
                Qs[0, action] = reward + dis * np.max(Qs1)

            # Train out network using target Y and predicted Q value
            sess.run(train, feed_dict={X: one_hot(state), Y: Qs})

            rAll += reward
            state = next_state
        rList.append(rAll)

print('Success rate : ', str(sum(rList) / num_epoch))
print('Final Q-Table Values')
print('LEFT DOWN RIGHT UP')
print(Q)
plt.bar(range(len(rList)), rList, color="b")
plt.show()