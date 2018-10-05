import csv
import json
import os
import random
from collections import Counter

import numpy as np
from tqdm import tqdm

from utils import flatten

EMBED_SUBSAMPLE_TRAINSET = "../data/embed_subsampled_pair.tsv"
TRAIN_FILE = "../data/embed_whole_data.tsv"
WORD2INT_FILE = "../data/W2I.json"

# model parameters
WINDOWS_SIZE = 3
SUBSAMPLE_THRESHOLD = 0.001
MIN_LENGTH = 5


def load_embedding_transet(filename):
    if os.path.exists(filename):
        with open(filename, 'r', newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')
            return list(reader)


def load_word_int_table(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        return json.load(f)


def _word_keep_prob(word_prob, subsample_threshold):
    return (np.sqrt(word_prob / subsample_threshold) + 1) * (
        subsample_threshold / word_prob)


def _get_words_keep_prob(sentences, subsample_threshold):
    counter = Counter(word for word in flatten(sentences))
    total = sum(counter.values())
    words_keep_rate = {word: _word_keep_prob(value / total, subsample_threshold)
                       for word, value in counter.items()}
    return words_keep_rate


def _subsampling(sentences, subsample_threshold, min_length):
    words_keep_rate = _get_words_keep_prob(sentences, subsample_threshold)
    dataset = []
    for sentence in sentences:
        sentence = [word for word in sentence if random.random()
                    <= words_keep_rate[word]]
        if len(sentence) >= min_length:
            dataset.append(sentence)
    return dataset


def embed_subsample_trainset(sentences, save_file, subsample_threshold, w2i_table,
                             min_length=3, window_size=5):
    sub_sentences = _subsampling(sentences, subsample_threshold, min_length)
    random.shuffle(sub_sentences)
    with open(save_file, 'w', newline='', encoding='utf-8', errors='ignore') as f:
        writer = csv.writer(f, delimiter='\t')
        for sentence in tqdm(sub_sentences):
            for idx, word in enumerate(sentence):
                left_band = max(idx - window_size, 0)
                right_band = min(idx + window_size, len(sentence)) + 1
                for target_word in sentence[left_band: right_band]:
                    if target_word != word:
                        writer.writerow(
                            [w2i_table[word], w2i_table[target_word]])
                        # writer.writerow([word, target_word])


def main():
    sentences = load_embedding_transet(TRAIN_FILE)
    word_2_int = load_word_int_table(WORD2INT_FILE)
    embed_subsample_trainset(
        sentences, EMBED_SUBSAMPLE_TRAINSET, SUBSAMPLE_THRESHOLD, word_2_int,
        min_length=MIN_LENGTH, window_size=WINDOWS_SIZE)


if __name__ == '__main__':
    main()
