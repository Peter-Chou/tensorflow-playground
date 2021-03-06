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
GRAD_CLIP = 5


class CharRNN(object):
    """char-rnn model intend to read txt file"""

    def __init__(self, filename, epoch, keep_prob, batch_size, num_steps,
                 lstm_size, num_layers, learning_rate, grad_clip, sampling=False):
        """initialize model and load file to create dict

        Arguments:
            filename {str} -- the filename
        """

        with open(filename, 'r') as f:
            text = f.read()
        self.epoch = epoch
        self.keep_prob = keep_prob if not sampling else 1.
        self.batch_size = batch_size if not sampling else 1
        self.num_steps = num_steps if not sampling else 1
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.sampling = sampling
        self.vocab = sorted(set(text))
        self.num_classes = len(self.vocab)
        self._vocab_to_int = {c: i for i, c in enumerate(self.vocab)}
        self._int_to_vocab = dict(enumerate(self.vocab))
        self.encoded = np.array([self._vocab_to_int[c]
                                 for c in text], dtype=np.int32)
        chars_per_batches = self.batch_size * self.num_steps
        n_batches = len(self.encoded) // chars_per_batches
        self.encoded = self.encoded[:n_batches*chars_per_batches]
        self.encoded = self.encoded.reshape((self.batch_size, -1))

    def _build_input(self):
        self.inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.num_steps], name="inputs")
        self.x_one_hot = tf.one_hot(self.inputs, self.num_classes)
        self.targets = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.num_steps], name="targets")

    @property
    def _get_batches(self):
        for n in range(0, self.encoded.shape[1], self.num_steps):
            x = self.encoded[:, n:n+self.num_steps]
            y_temp = self.encoded[:, n+1:n+self.num_steps+1]
            y = np.zeros(x.shape, dtype=x.dtype)
            y[:, :y_temp.shape[1]] = y_temp
            yield x, y

    def _build_cell(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(
            lstm, output_keep_prob=self.keep_prob)
        return drop

    def _pick_top_n(self, preds, top_n=5):
        p = np.squeeze(preds)
        p[np.argsort(p)[:-top_n]] = 0
        p = p / np.sum(p)
        c = np.random.choice(len(self.vocab), 1, p=p)[0]
        return c

    def _build_lstm(self):
        cell = tf.contrib.rnn.MultiRNNCell(
            [self._build_cell() for _ in range(self.num_layers)])
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell, self.x_one_hot, initial_state=self.initial_state)

    def _build_output(self):
        seq_output = tf.concat(self.outputs, axis=1)
        x = tf.reshape(seq_output, [-1, self.lstm_size])

        with tf.variable_scope("softmax"):
            softmax_w = tf.get_variable(
                "weight", shape=(self.lstm_size, self.num_classes),
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            softmax_b = tf.get_variable(
                "bias", shape=(self.num_classes),
                initializer=tf.zeros_initializer())

        self.logits = tf.matmul(x, softmax_w) + softmax_b
        self.prediction = tf.nn.softmax(self.logits, name="predictions")

    def _build_loss(self):
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_shaped = tf.reshape(y_one_hot, self.logits.get_shape())
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=y_shaped)
        self.loss = tf.reduce_mean(self.loss)

    def _build_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def build_network(self):
        self._build_input()
        self._build_lstm()
        self._build_output()
        self._build_loss()
        self._build_optimizer()

    def train(self, print_every_n, save_every_n):
        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            counter = 0
            for e in range(self.epoch):
                new_state = sess.run(self.initial_state)
                loss = 0
                for x, y in self._get_batches:
                    counter += 1
                    start = time.time()
                    feed = {
                        self.inputs: x,
                        self.targets: y,
                        self.initial_state: new_state
                    }
                    batch_loss, new_state, _ = sess.run([
                        self.loss,
                        self.final_state,
                        self.optimizer
                    ], feed_dict=feed)
                    if (counter % print_every_n == 0):
                        end = time.time()
                        print('Epoch: {}/{}... '.format(e+1, self.epoch),
                              'Training Step: {}... '.format(counter),
                              'Training loss: {:.4f}... '.format(batch_loss),
                              '{:.4f} sec/batch'.format((end-start)))
                    if (counter % save_every_n == 0):
                        saver.save(sess, "checkpoints/i{}_l{}".format(
                            counter, self.lstm_size))

            saver.save(sess, "checkpoints/i{}_l{}".format(
                counter, self.lstm_size))

    def sample(self, checkpoint, n_samples, prime="The "):
        samples = [c for c in prime]
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            new_state = sess.run(self.initial_state)
            for c in prime:
                x = np.zeros((1, 1))
                x[0, 0] = self._vocab_to_int[c]
                feed = {
                    self.inputs: x,
                    self.initial_state: new_state}
                preds, new_state = sess.run(
                    [self.prediction, self.final_state], feed_dict=feed)

            c = self._pick_top_n(preds)
            samples.append(self._int_to_vocab[c])

            for i in range(n_samples):
                x[0, 0] = c
                feed = {
                    self.inputs: x,
                    self.initial_state: new_state}
                preds, new_state = sess.run(
                    [self.prediction, self.final_state], feed_dict=feed)
                c = self._pick_top_n(preds)
                samples.append(self._int_to_vocab[c])
        return "".join(samples)


def main():
    model = CharRNN("anna.txt", EPOCH, KEEP_PROB, BATCH_SIZE, NUM_STEPS,
                    LSTM_SIZE, NUM_LAYERS, LEARNING_RATE,
                    GRAD_CLIP, sampling=False)
    model.build_network()
    model.train(PRINT_EVERY_N, SAVE_EVERY_N)


if __name__ == '__main__':
    main()
