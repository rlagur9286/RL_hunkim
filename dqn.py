import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, input_size, output_size, name='main'):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            # First layer of weight
            W1 = tf.get_variable(name='w1', shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.tanh(tf.matmul(self.X, W1))

            # Second layer of weight
            W2 = tf.get_variable(name='w2', shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            # Q prediction
            self._Qpred = tf.matmul(layer1, W2)

            # We need to define the parts of the network needed for learning a
            # policy
            self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

            # Loss Function
            self._loss = tf.reduce_mean(tf.square(self._Qpred - self._Y))
            # Learning
            self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss=self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self.X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self.X: x_stack, self._Y: y_stack})
