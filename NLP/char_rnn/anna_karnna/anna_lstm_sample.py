# -*- coding: utf-8 -*-

import tensorflow as tf


from anna_lstm_train import CharRNN
from anna_lstm_train import (EPOCH, KEEP_PROB, BATCH_SIZE, NUM_STEPS,
                             LSTM_SIZE, NUM_LAYERS, LEARNING_RATE, GRAD_CLIP)


def main():
    model = CharRNN("anna.txt", EPOCH, KEEP_PROB, BATCH_SIZE, NUM_STEPS,
                    LSTM_SIZE, NUM_LAYERS, LEARNING_RATE,
                    GRAD_CLIP, sampling=True)
    model.build_network()
    checkpoint = tf.train.latest_checkpoint("checkpoints")
    sample = model.sample(checkpoint, 2000, prime="Far")
    print(sample)


if __name__ == '__main__':
    main()
