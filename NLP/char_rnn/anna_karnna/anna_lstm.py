# -*- coding: utf-8 -*-
"""
使用安娜名著来训练char-rnn
"""

import time
from collections import namedtuple

import numpy as np
import tensorflow as tf


# model parameters
BATCH_SIZE = 100
NUM_STEPS = 100
LSTM_SIZE = 512
NUM_LAYERS = 2
LEARNING_RATE = 0.001
KEEP_PROB = 0.5

# training parameters
EPOCH = 20
PRINT_EVERY_N = 50
SAVE_EVERY_N = 200


class CharRNN(object):
    """char-rnn model intend to read txt file"""

    def __init__(self, filename, batch_size, num_steps, lstm_size,
                 num_layers, learning_rate, grad_clip, sampling=False):
        """initialize model and load file to create dict

        Arguments:
            filename {str} -- the filename
        """

        with open(filename, 'r') as f:
            text = f.read()
        self.vocab = sorted(set(text))
        self.num_classes = len(self.vocab)
        self._vocab_to_int = {c: i for i, c in enumerate(vocab)}
        self._int_to_vocab = dict(enumerate(vocab))
        self.encoded = np.array([self._vocab_to_int[c]
                                 for c in text], dtype=np.int32)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.sampling = sampling

    def _build_input(self):
        self.inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.num_steps], name="inputs")
        self.x_one_hot = tf.one_hot(self.inputs, self.num_classes)
        self.targets = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.num_steps], name="targets")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def _get_batches(self):
        chars_per_batches = self.batch_size * self.num_steps
        n_batches = len(self.encoded) // chars_per_batches
        self.encoded = self.encoded[:n_batches*chars_per_batches]
        self.encoded = self.encoded.reshape((self.batch_size, -1))

        for n in range(0, self.encoded.shape[1], self.num_steps):
            x = self.encoded[:, n:n+num_steps]
            y_temp = arr[:, n+1:n+num_steps]

            y = np.zeros(x.shape, dtype=x.dtype)
            y[:, :y_temp.shape[1]] = y_temp
            yield x, y

    def _build_cell(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    def _build_lstm(self):
        cell = tf.contrib.rnn.MultiRNNCell(
            [self._build_cell(self.lstm_size, keep_prob)
             for _ in range(self.num_layers)]
        )
        initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, self.x_one_hot, initial_state=initial_state)

    def _build_output(self):
        seq_output = tf.concat(self.outputs, axis=1)
        x = tf.reshape(seq_output, [-1, in_size])

        with tf.variable_scope("softmax"):
            softmax_w = tf.get_variable(
                "weight", shape=(self.lstm_size, self.num_classes),
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            softmax_b = tf.get_variable(
                "bias", shape=(self.num_classes),
                initializer=tf.zeros_initializer())

        self.logits = tf.matmul(x, softmax_w) + softmax_b
        self.prediction = tf.nn.softmax(logits, name="predictions")

    def _build_loss(self):
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_shaped = tf.reshape(y_one_hot, self.logits.get_shape())
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=y_shaped)
        self.loss = tf.reduce_mean(loss)

    def _build_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdadeltaOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def build_network(self):
        self._build_input()
        self._build_lstm()
        self._build_output()
        self._build_loss()
        self._build_optimizer()

    def train(self):
        pass


def main():
    model = CharRNN("anna.txt")


if __name__ == '__main__':
    main()
