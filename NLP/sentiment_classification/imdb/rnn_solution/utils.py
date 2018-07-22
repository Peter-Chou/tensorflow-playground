# -*- coding: utf-8 -*-


import os
from collections import Counter
from os.path import isfile
from string import punctuation
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download(url, filename):
    if not isfile(filename):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc=filename) as pbar:
            urlretrieve(url, filename, pbar.hook)


def preprocess(review_file, label_file, seq_len=200):
    with open(review_file) as f:
        reviews = f.read()
    all_text = "".join(c for c in reviews if c not in punctuation)
    reviews = all_text.split('\n')
    words = all_text.split()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    review_ints = []
    for each in reviews:
        review_ints.append([vocab_to_int[word] for word in each.split()])

    with open(label_file) as f:
        labels = f.read()
    labels = labels.split('\n')
    labels = np.array([1 if each == "positive" else 0 for each in labels])

    non_zero_idx = [ii for ii, review in enumerate(
        review_ints) if len(review) != 0]
    review_ints = [review_ints[ii] for ii in non_zero_idx]
    labels = np.array([labels[ii] for ii in non_zero_idx])
    features = np.zeros((len(review_ints), seq_len), dtype=int)
    for i, row in enumerate(review_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]
    return features, labels, len(vocab_to_int)


def train_val_test_split(features, labels, train_fraction):
    split_idx = int(len(features)*train_fraction)
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]
    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]
    return train_x, train_y, val_x, val_y, test_x, test_y
