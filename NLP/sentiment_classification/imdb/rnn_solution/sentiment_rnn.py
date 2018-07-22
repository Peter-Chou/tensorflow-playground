# -*- coding: utf-8 -*-

REVIEWS_URL = "https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/reviews.txt"
LABELS_URL = "https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/labels.txt"
REVIEWS_FILE = "reviews.txt"
LABELS_FILE = "labels.txt"

import numpy as np
import tensorflow as tf

import utils

# MODEL PARAMETERS
BATCH_SIZE = 500
EMBED_SIZE = 300
LSTM_LAYERS = 1
LSTM_SIZE = 256

# OPTIMIZER PARAMETERS
LEARNING_RATE = 0.001
KEEP_PROB = 0.5
EPOCHS = 10


class RnnSentiment(object):

    def __init__(self, batch_size, embed_size, n_words, lstm_layers, lstm_size,
                 learning_rate, epochs):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.n_words = n_words
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.lr = learning_rate
        self.epochs = epochs

    def _build_input(self):
        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
            self.labels = tf.placeholder(tf.int32, [None, None], name="labels")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def _build_embedding(self):
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable(
                "embedding", shape=(self.n_words, self.embed_size),
                initializer=tf.random_uniform_initializer(-1, 1))
            self.embed = tf.nn.embedding_lookup(embedding, self.inputs)

    def _lstm_dropout(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm,
                                             output_keep_prob=self.keep_prob)
        return drop

    def _build_rnn(self):
        with tf.variable_scope("rnn"):
            self.cell = tf.contrib.rnn.MultiRNNCell(
                [self._lstm_dropout() for _ in range(self.lstm_layers)])
            self.initial_state = self.cell.zero_state(
                self.batch_size, tf.float32)
            self.outputs, self.final_state = tf.nn.dynamic_rnn(
                self.cell, self.embed, initial_state=self.initial_state)
            self.predictions = tf.contrib.layers.fully_connected(
                self.outputs[:, -1], 1, activation_fn=tf.sigmoid)

    def _build_cost_optimizer(self):
        with tf.variable_scope("CostOptimizer"):
            self.cost = tf.losses.mean_squared_error(self.labels, self.predictions)
            self.optimizer = tf.train.AdamOptimizer(
                self.lr).minimize(self.cost)

    def _build_accuracy(self):
        with tf.variable_scope("accuracy"):
            correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32),
                                    self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def build(self):
        self._build_input()
        self._build_embedding()
        self._build_rnn()
        self._build_cost_optimizer()
        self._build_accuracy()

    def _get_batches(self, x, y):
        n_batches = len(x) // self.batch_size
        x, y = x[:n_batches * self.batch_size], y[:n_batches * self.batch_size]
        for i in range(0, len(x), self.batch_size):
            yield x[i:i+self.batch_size], y[i:i+self.batch_size]

    def train(self, X, Y, val_X, val_Y, keep_prob):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            for e in range(self.epochs):
                state = sess.run(self.initial_state)
                for i, (x, y) in enumerate(self._get_batches(X, Y), 1):
                    feed = {
                        self.inputs: x,
                        self.labels: y[:, None],
                        self.initial_state: state,
                        self.keep_prob: 0.5}
                    loss_, state, _ = sess.run([self.cost, self.final_state,
                                                self.optimizer], feed_dict=feed)

                    if iteration % 5 == 0:
                        print(
                            f"Epoch: {e}/{self.epochs}",
                            f"Iteration: {iteration}",
                            f"Train loss: {loss_:0.3f}"
                        )
                    # TODO: complete validation part
                    if iteration % 25 == 0:
                        val_acc = []
                        val_state = sess.run(self.cell.zero_state(
                            self.batch_size, tf.float32))
                        for x, y in self._get_batches(val_X, val_Y):
                            feed = {
                                self.inputs: x,
                                self.labels: y[:, None],
                                self.initial_state: val_state,
                                self.keep_prob: 1.
                            }
                            batch_acc, val_state = sess.run([
                                self.accuracy, self.final_state],
                                feed_dict=feed)
                            val_acc.append(batch_acc)
                        print(f"val acc: {np.mean(val_acc):0.4f}")
                    iteration += 1
            saver.save(sess, "checkpoints/sentiment")
        print(format("train end", "*^50s"))

    def test(self, X, Y):
        test_acc = []
        saver = tf.train.Saver()
        print(format("Test start", "*^50s"))
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("checkpoints"))
            test_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
            for i, (x, y) in enumerate(self._get_batches(X, Y), 1):
                feed = {
                    self.inputs: x,
                    self.labels: y[:, None],
                    self.keep_prob: 1.,
                    self.initial_state: test_state
                }
                batch_acc, test_state = sess.run([self.accuracy, self.final_state],
                                                 feed_dict=feed)
                test_acc.append(batch_acc)
            print(f"Test accuracy: {np.mean(test_acc):.3f}")


def main():
    utils.download(REVIEWS_URL, REVIEWS_FILE)
    utils.download(LABELS_URL, LABELS_FILE)
    features, labels, N_WORDS = utils.preprocess(REVIEWS_FILE, LABELS_FILE)
    train_x, train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(
        features, labels, 0.8)
    model = RnnSentiment(BATCH_SIZE, EMBED_SIZE, N_WORDS, LSTM_LAYERS,
                         LSTM_SIZE, LEARNING_RATE, EPOCHS)
    model.build()
    model.train(train_x, train_y, val_x, val_y, KEEP_PROB)
    model.test(test_x, test_y)


if __name__ == '__main__':
    main()
