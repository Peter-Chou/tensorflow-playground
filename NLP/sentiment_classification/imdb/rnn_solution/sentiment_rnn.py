# -*- coding: utf-8 -*-

REVIEWS_URL = "https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/reviews.txt"
LABELS_URL = "https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/labels.txt"
REVIEWS_FILE = "reviews.txt"
LABELS_FILE = "labels.txt"

import utils
import tensorflow as tf


class RnnSentiment(object):

    def __init__(self, batch_size, embed_size, n_words, lstm_layers, lstm_size,
                 learning_rate, keep_prob, epochs):
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.n_words = n_words
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.lr = learning_rate
        self.epochs = epochs
        self.keep_prob = keep_prob

    def _build_input(self):
        with tf.variable_scope("input"):
            self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
            self.labels = tf.placeholder(tf.int32, [None, None], name="labels")

    def _build_embedding(self):
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable(
                "embedding", shape=(self.n_words, self.embed_size),
                initializer=tf.random_uniform_initializer(-1, 1))
            embed = tf.nn.embedding_lookup(embedding, self.inputs)

    def _lstm_dropout(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm,
                                             output_keep_prob=self.keep_prob)
        return drop

    def _build_rnn(self):
        with tf.variable_scope("rnn"):
            self.cell = tf.contrib.rnn.MultiRNNCell(
                [self._lstm_dropout() for _ in range(self.lstm_layers)])
            self.initial_state = cell.zero_state(batch_size, tf.float32)
            self.outputs, self.final_state = tf.nn.dynamic_rnn(cell, embed)
            self.predictions = tf.contrib.layers.fully_connected(
                self.outputs[:, -1], 1, activation_fn=tf.sigmoid)

    def _build_cost_optimizer(self):
        with tf.variable_scope("CostOptimizer"):
            cost = tf.losses.mean_squared_error(self.labels, predictions)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)

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


def main():
    utils.download(REVIEWS_URL, REVIEWS_FILE)
    utils.download(LABELS_URL, LABELS_FILE)
    features, labels, n_words = utils.preprocess(REVIEWS_FILE, LABELS_FILE)
    train_x, train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(
        features, labels, 0.8)


if __name__ == '__main__':
    main()
