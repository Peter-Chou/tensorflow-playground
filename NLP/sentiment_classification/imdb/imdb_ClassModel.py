"""
imdb mini-project using tensorflow in class
"""

import os
import sys

import keras
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer

# parameters
NUM_CLASSES = 2
SEED = 42
EPOCHS = 15
BATCH_SIZE = 32
ITER_PATH = "ckpt/iter"
BEST_PATH = "ckpt/best"
GRAPH_PATH = "graphs/imdb-placeholder"
LEARNING_RATE = 0.0005

# set random seed
np.random.seed(SEED)
tf.set_random_seed(SEED)


class Imdb_nn:
    """
    a simple model in class
    """
    def __init__(self, data, output_dim, epochs, batch_size, learning_rate):
        """
        Args:
        """
        self.data = data
        self.output_dim = output_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.graph_path = GRAPH_PATH
        self.iter_path = ITER_PATH
        self.best_path = BEST_PATH
        self.best_test_loss = np.inf

    def _next_batch(self, x_data, y_data, shuffle=False):
        """
        a generator
        """
        if shuffle:
            index = np.arange(len(x_data))
            np.random.shuffle(index)
            x_data = x_data[index]
            y_data = y_data[index]
        n_batches = int(x_data.shape[0] / BATCH_SIZE)
        # generate
        for idx in range(n_batches):
            x = x_data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
            y = y_data[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
            yield x, y

    def _safe_madirs(self,path):
        try:
            os.makedirs(path)    # equal to mkdir -p
        except OSError:
            pass

    def _preprocess(self):
        """
        create folders & retrieve data
        """
        self._safe_madirs(self.iter_path)
        self._safe_madirs(self.best_path)
        self._safe_madirs(self.graph_path)

        (self._x_train, self._y_train), (self._x_test, self._y_test) = \
            self.data.load_data(num_words=1000)

    def _build_data(self):
        """
        data preprocessing & graph input initialization

        args:

            train_dataset: tuple -- (x_train, y_train)
            test_dataset: tuple -- (x_test, y_test)
        """
        # one-hot encode
        tokenizer = Tokenizer(num_words=1000)
        self._x_train = tokenizer.sequences_to_matrix(self._x_train, mode="binary")
        self._y_train = keras.utils.to_categorical(self._y_train, self.output_dim)
        self._x_test = tokenizer.sequences_to_matrix(self._x_test, mode="binary")
        self._y_test = keras.utils.to_categorical(self._y_test, self.output_dim)
        self.data_num = self._x_train.shape[0]

        with tf.name_scope("init"):
            self.x = tf.placeholder(tf.float32, shape=(None, 1000), name="x")
            self.y = tf.placeholder(tf.float32, shape=(None, 2), name="y")
            self.global_step = tf.get_variable("global_step", trainable=False,
                                        initializer=tf.constant(0))

    def _build_network(self):
        """
        build the inference model
        """
        with tf.name_scope("network"):
            net = tf.layers.dense(self.x, units=128, activation=tf.nn.relu)
            net = tf.layers.dropout(net, rate=0.5)
            net = tf.layers.dense(self.x, units=256, activation=tf.nn.relu)
            net = tf.layers.dropout(net, rate=0.5)
            net = tf.layers.dense(self.x, units=512, activation=tf.nn.relu)
            net = tf.layers.dropout(net, rate=0.5)
            self.output = tf.layers.dense(net, units=self.output_dim,
                                     activation=tf.nn.softmax)

    def _build_loss(self):
        """
        set the loss
        """
        with tf.name_scope("build_loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output, labels=self.y, name="loss")
            self.loss = tf.reduce_mean(self.loss)

    def _build_optimizer(self):
        """
        build the optimizer
        """
        with tf.name_scope("optimize"):
            # set the optimizer
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(
                    self.loss, global_step=self.global_step)

    def _build_summary(self):
        """
        tensorboard summary
        """
        with tf.name_scope("summar"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """
        assemble the graph
        """
        self._preprocess()
        self._build_data()
        self._build_network()
        self._build_loss()
        self._build_optimizer()
        self._build_summary()

    def _train_once(self, sess, x, y):
        """
        train one epoch

        Args:

        Returns:
            batch_loss:
        """
        _, loss_batch, = sess.run(
            [self.optimizer, self.loss], feed_dict={self.x: x, self.y: y})
        return loss_batch

    def _test_once(self, sess, x, y):
        """
        test validation one time
        """
        test_loss, test_summary = sess.run(
            [self.loss, self.summary_op], feed_dict={self.x: x, self.y: y})
        return test_loss, test_summary

    def train_model(self):
        """
        train the model
        """
        # if use one saver for two difference save, cause problems
        saver = tf.train.Saver()
        best_saver = tf.train.Saver(max_to_keep=3)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # saver restore from checkpoint file.  dirname get that file path's directory
            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname(self.best_path + "/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            train_writer = tf.summary.FileWriter(self.graph_path + "/train", sess.graph)
            test_writer = tf.summary.FileWriter(self.graph_path + "/test", sess.graph)

            # initialize global_step
            self.global_step.eval()

            n_batches = int(self.data_num / self.batch_size)

            for i in range(self.epochs):
                epoch_train_loss = 0.0
                epoch_validation_loss = 0.0
                for x, y in self._next_batch(self._x_train, self._y_train, shuffle=True):
                    train_loss = self._train_once(sess, x, y)
                    epoch_train_loss += train_loss

                print("epoch {0}:\nAverage train loss: {1}".format(
                    i + 1, epoch_train_loss / n_batches), end='\t')

                train_summary = sess.run(self.summary_op, feed_dict={
                                        self.x: self._x_train, self.y: self._y_train})
                train_writer.add_summary(train_summary, global_step=i)

                if (i + 1) % 2 == 0:
                    saver.save(sess, self.iter_path + "/nn", global_step=self.global_step)
                # test
                test_loss, test_summary = sess.run(
                    [self.loss, self.summary_op], feed_dict={self.x: self._x_test, self.y: self._y_test})
                test_writer.add_summary(test_summary, global_step=i)
                print("Average test loss: {}".format(test_loss))

                if test_loss < self.best_test_loss:
                    best_saver.save(
                        sess, self.best_path + "/best--{:0.5f}".format(test_loss))
                    print("current test loss:{0:0.4f} is better than last best record:{1:0.4f}, saved.".format(
                        test_loss, self.best_test_loss))
                    self.best_test_loss = test_loss
                # ensure print stdout to screen immediately
                sys.stdout.flush()
            train_writer.close()
            test_writer.close()


def main():
    model = Imdb_nn(data=imdb, output_dim=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATE)
    model.build_graph()
    model.train_model()

if __name__ == "__main__":
    main()
