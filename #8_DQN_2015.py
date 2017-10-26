import random
import numpy as np
import tensorflow as tf
import gym

from dqn import DQN
from collections import deque

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

REPLAY_MEMORY = 5000
dis = 0.9


def get_copy_var_ops(*, dest_scope_name='name', src_scope_name='main'):
    op_holder = []

    src_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src, dest in zip(src_var, dest_var):
        op_holder.append(dest.assign(src.value()))

    return op_holder


def replay_train(mainDQN, tragetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(tragetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)


def bot_play(mainDQN):
    # See out trained network in action
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print('TOTAL SCORE : {}'.format(reward_sum))
            break

def main():
    max_episode = 5000
    replay_buffer = deque()

    with tf.Session() as sess:
        mainDQN = DQN(sess, input_size, output_size, name='main')
        targetDQN = DQN(sess, input_size, output_size, name='target')
        tf.global_variables_initializer().run()

        copy_ops = get_copy_var_ops(dest_scope_name="taget", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episode):
            e = 1 / ((episode / 10) + 1)
            done = False
            step_count = 0

            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)

                if done and step_count != 200:
                    reward -= 100

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break

            print('Episode[{}] - steps : {}'.format(episode, step_count))
            if step_count > 10000:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print('loss : ', loss)
                # copy q_net -> traget_net
                sess.run(copy_ops)

        bot_play(mainDQN)

if __name__ == '__main__':
    main()