# -*- coding: utf-8 -*-

"""implementation of embeddings using skip-grams"""

import random
from collections import Counter

import numpy as np
import tensorflow as tf

from utils import download

DATASET_FOLDER_PATH = 'data'
DATASET_FILENAME = 'text8.zip'
DATASET_NAME = 'Text8 Dataset'
URL = "http://mattmahoney.net/dc/text8.zip"


class SkipGrams(object):
    """embeddings using skip-grams"""

    def __init__(self, text, n_counts_to_keep, subsample_threshold=1e-5,
                 window_size=5):
        self.text = text
        self.n_counts_to_keep = n_counts_to_keep
        self.subsample_threshold = subsample_threshold
        self.window_size = window_size

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

    def _get_target(self, words, index):
        R = np.random.randint(1, self.window_size+1)
        start = index - R if (index-R) > 0 else 0
        stop = index + R
        target_words = set(words[start:index] + words[index+1:stop+1])
        return list(target_words)

    # TODO: finish get_batch method


def main():
    text = download(URL, DATASET_FOLDER_PATH, DATASET_FILENAME, DATASET_NAME)
    model = SkipGrams(text, 5)
    model._preprocess()


if __name__ == '__main__':
    main()
