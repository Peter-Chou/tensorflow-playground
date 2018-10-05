import csv
import json
import math
import os
import random
from collections import Counter

import numpy as np
import tensorflow as tf

from utils import set_logger

logger = set_logger(__name__, "./embed_train.log")

TRAIN_FILE = "../data/embed_subsampled_pair.tsv"
WORD_INT_FILE = "../data/W2I.json"
EMBED_VECTORS_SAVED = "../data/embedding_lookup_vectors_normalized.csv"

# model parameters
EMBED_SIZE = 300
NCE_SAMPLE_NUM = 2000
BATCH_SIZE = 512
# BATCH_SIZE = 128
# NCE_SAMPLE_NUM = 1000

# optimizer parameters
LEARNING_RATE = 0.1
FIRST_DECAY_STEPS = 200000
T_MUL = 1.5  # multipled 10 times when epochs close to 10
M_MUL = 0.85  # may be too big cause slow converge
ALPHA = 0.00001  # minimum lr when decay
MOMENTUM = 0.9

# training parameters
EPOCHS = 2
PRINT_INTERVAL = 2000
SAVE_AFTER_N_PRINT = 10


def load_word_int_map(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            word2int = json.load(f)
        return word2int
    else:
        logger.error("file does not exist")


class WordVectors(object):
    """
    A class to train word vectors in the specific dataset.
    """

    def __init__(self, data_path, word2int_table, batch_size, embed_size,
                 nce_sample_num):
        self.word2int = word2int_table
        # self.int2word = {i: word for word, i in word2int_table.items()}
        self.vocab_size = len(word2int_table)
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.nce_sample_num = nce_sample_num
        self.data_path = data_path

    def _build_untrainable_variables(self):
        with tf.variable_scope("untrainable"):
            self.gstep = tf.get_variable("global_step", shape=(), dtype=tf.int32,
                                         initializer=tf.zeros_initializer(),
                                         trainable=False)
            self.min_loss = tf.get_variable("min_loss", shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer(),
                                            trainable=False)
            self.tmp_loss = tf.placeholder(
                dtype=tf.float32, shape=[], name='tmp_loss')
            self.min_loss_assign = tf.assign(self.min_loss, self.tmp_loss)

    def _build_dataset(self):
        with tf.variable_scope("dataset"):
            dataset = tf.data.TextLineDataset(self.data_path)
            dataset = dataset.map(lambda line: tf.decode_csv(
                line, record_defaults=[[1], [1]], field_delim='\t'), num_parallel_calls=4)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(2)
            self.train_iterator = dataset.make_initializable_iterator()
            self.inputs, self.labels = self.train_iterator.get_next()
            self.labels = tf.expand_dims(self.labels, 1)
            self.train_loss = tf.get_variable(
                "train_loss", shape=[], dtype=tf.float32,
                initializer=tf.zeros_initializer, trainable=False)

    def _build_embed(self):
        with tf.variable_scope("embed"):
            self.embedding_vectors = tf.get_variable(
                "embedding", shape=[self.vocab_size, self.embed_size],
                initializer=tf.random_uniform_initializer(-0.1, 0.1))

            self.normalized_embedding_vectors = tf.nn.l2_normalize(
                self.embedding_vectors, axis=1)
            self.embed = tf.nn.embedding_lookup(
                self.embedding_vectors, self.inputs)

    def _build_nce_loss(self):
        with tf.variable_scope("nce_loss"):
            nce_weights = tf.get_variable(
                "nce_weights", shape=[self.vocab_size, self.embed_size],
                initializer=tf.truncated_normal_initializer(
                    stddev=1.0 / math.sqrt(self.embed_size)))
            nce_biases = tf.get_variable(
                "nce_biases", shape=[self.vocab_size],
                initializer=tf.zeros_initializer())
            self.nce_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self.labels,
                               inputs=self.embed,
                               num_sampled=self.nce_sample_num,
                               num_classes=self.vocab_size))

    def _set_lr(self, learning_rate, first_decay_steps, t_mul, m_mul, alpha):
        with tf.variable_scope("set_lr"):
            self.lr = tf.train.cosine_decay_restarts(
                learning_rate, self.gstep, first_decay_steps=first_decay_steps,
                t_mul=t_mul, m_mul=m_mul, alpha=alpha, name="learning_rate")

    def _build_optimizer(self, momentum):
        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.lr, momentum=momentum).minimize(self.nce_loss, global_step=self.gstep)

    def build_network(self, learning_rate, first_decay_steps, t_mul, m_mul,
                      momentum, alpha=0.0):
        self._build_untrainable_variables()
        self._build_dataset()
        self._build_embed()
        self._build_nce_loss()
        self._set_lr(learning_rate, first_decay_steps, t_mul, m_mul, alpha)
        self._build_optimizer(momentum)

    def train(self, epochs, print_interval=100, save_after_n_print=5):
        """训练word vector

        Arguments:
            epochs {int} -- 共训练多少轮

        Keyword Arguments:
            print_interval {int} -- 每隔多少iteration输出loss (default: {100})
            save_after_n_print {int} -- 每隔多少次输出保存best graph (default: {5})

        """

        best_ckpt = "checkpoints/embed_best"
        latest_ckpt = "checkpoints/embed_latest"

        saver_best = tf.train.Saver(max_to_keep=2)
        saver_epoch = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname(f"{latest_ckpt}/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:  # load model if exists
                saver_epoch.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            # min_loss = None
            for e in range(epochs):
                sess.run(self.train_iterator.initializer)

                first_best_save = True
                print_loss = 0
                while True:
                    try:
                        loss_, _ = sess.run(
                            [self.nce_loss, self.optimizer])
                        print_loss += loss_

                        if (iteration + 1) % print_interval == 0:
                            avg_print_loss = print_loss / print_interval
                            logger.info(
                                f"epoch: {e + 1} , iteratoin: {iteration + 1:10,} , loss= {avg_print_loss:.4f}")
                            # save per n print
                            if (iteration + 1) % (save_after_n_print * print_interval) == 0:
                                record_min_loss = self.min_loss.eval()
                                if first_best_save:  # skip first save in each epoch
                                    first_best_save = False
                                    if record_min_loss == 0.:
                                        sess.run(self.min_loss_assign, feed_dict={
                                            self.tmp_loss: avg_print_loss})
                                        logger.info(
                                            f"skip first save round , current loss: {avg_print_loss:.4f}")
                                else:
                                    if avg_print_loss < record_min_loss:
                                        saver_best.save(
                                            sess, f"{best_ckpt}/best-{avg_print_loss}")
                                        sess.run(self.min_loss_assign, feed_dict={
                                            self.tmp_loss: avg_print_loss})
                                        logger.info(
                                            f"save model:  current loss: {avg_print_loss:.4f} ,  previous loss: {record_min_loss:.4f}")
                            print_loss = 0
                        iteration += 1
                    except tf.errors.OutOfRangeError:
                        break
                # save per epoch
                saver_epoch.save(sess, f"{latest_ckpt}/latest-{iteration}")
                logger.info("one epoch complete, save latest model.")

    def get_embedding_weights(self, save_filename):
        """保存 word embedding

        Arguments:
            save_filename {str} -- 保存到指定文件

        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                os.path.dirname("checkpoints/embed_best/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:  # load model if exists
                saver.restore(sess, ckpt.model_checkpoint_path)
            normalized_embedding = sess.run(self.normalized_embedding_vectors)
            normalized_embedding = np.asarray(normalized_embedding)
            np.savetxt(save_filename, normalized_embedding, delimiter=",")


def main():
    os.makedirs("checkpoints", exist_ok=True)
    WORD2INT = load_word_int_map(WORD_INT_FILE)
    model = WordVectors(TRAIN_FILE, WORD2INT, batch_size=BATCH_SIZE,
                        embed_size=EMBED_SIZE, nce_sample_num=NCE_SAMPLE_NUM)
    model.build_network(LEARNING_RATE, FIRST_DECAY_STEPS,
                        T_MUL, M_MUL, MOMENTUM, alpha=ALPHA)
    model.train(epochs=EPOCHS, print_interval=PRINT_INTERVAL,
                save_after_n_print=SAVE_AFTER_N_PRINT)
    # model.get_embedding_weights(EMBED_VECTORS_SAVED)


if __name__ == '__main__':
    main()
