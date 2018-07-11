# -*- coding: utf-8 -*-

"""implementation of embeddings using skip-grams"""
import random
import time
from collections import Counter

import numpy as np
import tensorflow as tf

from utils import download, save_mkdir

# DATASET PARAMETERS
DATASET_FOLDER_PATH = 'data'
DATASET_FILENAME = 'text8.zip'
DATASET_NAME = 'Text8 Dataset'
URL = "http://mattmahoney.net/dc/text8.zip"


# MODEL PARAMETERS
N_COUNTS_TO_KEEP = 5
N_EMBEDDING = 200
N_SAMPLED = 100
WINDOW_SIZE = 10


# TRAINING PARAMETERS
EPOCHS = 10
BATCH_SIZE = 1000


class SkipGrams(object):
    """embeddings using skip-grams"""

    def __init__(self, text, epochs, batch_size, n_embedding, n_sampled,
                 n_counts_to_keep, subsample_threshold=1e-5, window_size=5):
        self.text = text
        self.n_counts_to_keep = n_counts_to_keep
        self.subsample_threshold = subsample_threshold
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_embedding = n_embedding
        self.n_sampled = n_sampled
        self.epochs = epochs

    def _preprocess(self):
        self.text = self.text.lower()
        self.text = self.text.replace('.', ' <PERIOD> ')
        self.text = self.text.replace(',', ' <COMMA> ')
        self.text = self.text.replace('"', ' <QUOTATION_MARK> ')
        self.text = self.text.replace(';', ' <SEMICOLON> ')
        self.text = self.text.replace('!', ' <EXCLAMATION_MARK> ')
        self.text = self.text.replace('?', ' <QUESTION_MARK> ')
        self.text = self.text.replace('(', ' <LEFT_PAREN> ')
        self.text = self.text.replace(')', ' <RIGHT_PAREN> ')
        self.text = self.text.replace('--', ' <HYPHENS> ')
        self.text = self.text.replace('?', ' <QUESTION_MARK> ')
        # self.text = self.text.replace('\n', ' <NEW_LINE> ')
        self.text = self.text.replace(':', ' <COLON> ')

        self.words = self.text.split()
        word_count = Counter(self.words)
        self.words = [word for word in self.words if word_count[word]
                      > self.n_counts_to_keep]
        word_count = Counter(self.words)
        sorted_vocab = sorted(word_count, key=word_count.get, reverse=True)
        self._int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        self._vocab_to_int = {word: ii for ii,
                              word in self._int_to_vocab.items()}
        self.int_words = [self._vocab_to_int[word] for word in self.words]

    def _subsampling(self):
        """
        subsample the word then it's frequency exceeds the threshold, and
        retain unchanged when is under
        """
        int_word_count = Counter(self.int_words)
        total_count = len(self.int_words)
        freqs = {word: count/total_count for word,
                 count in int_word_count.items()}
        p_drop = {word: 1-np.sqrt(self.subsample_threshold/freqs[word])
                  for word in int_word_count}
        self.train_int_words = [
            word for word in int_word_count if random.random() < 1 - p_drop[word]]
        n_batches = len(self.train_int_words) // self.batch_size
        self.train_int_words = self.train_int_words[:n_batches*self.batch_size]

    def _get_target(self, words, index):
        R = np.random.randint(1, self.window_size+1)
        start = index - R if (index-R) > 0 else 0
        stop = index + R
        target_words = set(words[start:index] + words[index+1:stop+1])
        return list(target_words)

    @property
    def _get_batch(self):
        for idx in range(0, len(self.train_int_words), self.batch_size):
            x, y = [], []
            batch = self.train_int_words[idx:idx+self.batch_size]
            for i in range(len(batch)):
                batch_x = batch[i]
                batch_y = self._get_target(batch, i)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            yield x, y

    def _build_input(self):
        self.inputs = tf.placeholder(tf.int32, [None], name='input')
        self.labels = tf.placeholder(tf.int32, [None, None], name='labels')

    def _build_embeddings(self):
        with tf.variable_scope("embeddings"):
            embedding = tf.get_variable(
                "embed matrx",
                shape=[len(self._int_to_vocab), self.n_embedding],
                initializer=tf.random_uniform_initializer(-1, 1))
            self.embed = tf.nn.embedding_lookup(embedding, self.inputs)

    def _build_loss(self):
        with tf.variable_scope("loss"):
            softmax_w = tf.get_variable(
                "weight", shape=[len(self._int_to_vocab), self.n_embedding],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            softmax_b = tf.get_variable(
                "bias", shape=[len(self._int_to_vocab)],
                initializer=tf.zeros_initializer())
            self.loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                                   self.labels, self.embed,
                                                   self.n_sampled, len(self._int_to_vocab))
            self.loss = tf.reduce_mean(loss)

    def _build_optimizer(self):
        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def build(self):
        """build the skip-grams model
        """
        self._preprocess()
        self._subsampling()
        self._build_input()
        self._build_embeddings()
        self._build_loss()
        self._build_optimizer()

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            iteration = 1
            loss = 0
            sess.run(tf.global_variables_initializer())

            for e in range(1, self.epochs+1):
                batch = self._get_batch
                start = time.time()
                for x, y in batch:
                    feed = {
                        self.inputs: x,
                        self.labels: np.array(y)[:, None]}
                    train_loss, _ = sess.run([self.loss, self.optimizer],
                                             feed_dict=feed)

                    loss += train_loss

                    if iteration % 100 == 0:
                        end = time.time()

                        print("Epoch {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss/100),
                              "{:.4f} sec/batch".format((end-start)/100))
                        loss = 0
                        start = time.time()

                    # if iteration % 1000 == 0:
                    #     # note that this is expensive (~20% slowdown if computed every 500 steps)
                    #     sim = similarity.eval()
                    #     for i in range(valid_size):
                    #         valid_word = int_to_vocab[valid_examples[i]]
                    #         top_k = 8 # number of nearest neighbors
                    #         nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    #         log = 'Nearest to %s:' % valid_word
                    #         for k in range(top_k):
                    #             close_word = int_to_vocab[nearest[k]]
                    #             log = '%s %s,' % (log, close_word)
                    #         print(log)
            saver.save(sess, "checkpoints/text8")


def main():
    text = download(URL, DATASET_FOLDER_PATH, DATASET_FILENAME, DATASET_NAME)
    save_mkdir("checkpoints")
    model = SkipGrams(N_COUNTS_TO_KEEP, EPOCHS, BATCH_SIZE, N_EMBEDDING,
    N_SAMPLED,N_COUNTS_TO_KEEP)
    model.build()
    model.train()


if __name__ == '__main__':
    main()
