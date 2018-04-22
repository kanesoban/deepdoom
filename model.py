import numpy
import tensorflow as tf
import random


class NeuralNetwork:
    def __init__(self, session, input, layers, output_size, learning_rate, scope='model'):
        self.session = session
        self.input = input
        self.layers = layers
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.target = tf.placeholder(tf.float32, shape=(None, output_size), name='target')
            self.action = tf.placeholder(tf.float32, shape=(1, output_size), name='action')
            self.loss_function = tf.reduce_mean(tf.square(self.action * (self.target - self.layers[-1])))
            self.learning_rate = learning_rate
            self.predict_op = self.layers[-1]
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_function)

    def predict(self, input):
        return self.session.run(
            self.predict_op,
            feed_dict={
                self.input: input
            }
        )

    def update(self, input, target, action):
        return self.session.run(
            self.train_op,
            feed_dict={
                self.input: input,
                self.target: target,
                self.action: action
            }
        )


class DoomNeuralNetwork(NeuralNetwork):
    def __init__(self, session, input_shape, num_actions, learning_rate=0.001, scope='doom_network'):
        self.num_actions = num_actions
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            input_variable = tf.placeholder(tf.float32, shape=input_shape, name='X')
            layers = []
            layers.append(tf.layers.conv2d(inputs=input_variable, filters=32, kernel_size=[5, 5], padding="same",
                                           activation=tf.nn.relu))
            layers.append(tf.layers.max_pooling2d(inputs=layers[-1], pool_size=[2, 2], strides=2))
            pooling_input_size = int(layers[-1].shape[1] * layers[-1].shape[2] * layers[-1].shape[3])
            layers.append(tf.reshape(layers[-1], [-1, pooling_input_size]))
            layers.append(tf.contrib.layers.fully_connected(layers[-1], num_actions))
            super(DoomNeuralNetwork, self).__init__(session, input_variable, layers, num_actions, learning_rate)

    def to_doom_action(self, action):
        doom_action = [False for _ in range(self.num_actions)]
        doom_action[int(numpy.argmax(action))] = True
        return doom_action

    def random_action(self):
        doom_action = [False for _ in range(self.num_actions)]
        doom_action[random.randint(0, self.num_actions-1)] = True
        return doom_action

    def sample_action(self, state, epsilon):
        if numpy.random.random() < epsilon:
            return self.random_action()
        else:
            return self.to_doom_action(numpy.argmax(self.predict(state)))
