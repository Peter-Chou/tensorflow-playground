# -*- coding: utf-8 -*-


import os
from tqdm import tqdm
from urllib.request import urlretrieve


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download():
    with DLProgress(
            unit='B', unit_scale=True, miniters=1, desc="reviews.txt") as pbar:
        urlretrieve(
            "https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/reviews.txt",
            "reviews.txt")
    with DLProgress(
            unit='B', unit_scale=True, miniters=1, desc="labels.txt") as pbar:
        urlretrieve(
            "https://raw.githubusercontent.com/udacity/deep-learning/master/sentiment-network/labels.txt",
            "labels.txt")
