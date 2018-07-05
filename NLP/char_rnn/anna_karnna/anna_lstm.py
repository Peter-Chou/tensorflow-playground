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

    def __init__(self, filename):
        """initialize model and load file to create dict

        Arguments:
            filename {str} -- the filename
        """

        with open(filename, 'r') as f:
            text = f.read()
        vocab = sorted(set(text))
        self._vocab_to_int = {c: i for i, c in enumerate(vocab)}
        self._int_to_vocab = dict(enumerate(vocab))
        self.encoded = np.array([self._vocab_to_int[c]
                                 for c in text], dtype=np.int32)


def main():
    model = CharRNN("anna.txt")


if __name__ == '__main__':
    main()
