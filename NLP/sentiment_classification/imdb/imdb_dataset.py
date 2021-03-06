"""
imdb mini-project using tensorflow with dataset API
vanilla approach doesn't have class setup
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
LEARNING_RATE = 0.0001

# set random seed
np.random.seed(SEED)
tf.set_random_seed(SEED)


def save_mkdir(path):
    """
    make directory safely

    Args:
        path: relative file path.

    Returns:
        None
    """
    try:
        os.makedirs(path)    # equal to mkdir -p
    except OSError:
        pass


# load the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

# one-hot encode
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode="binary")
x_test = tokenizer.sequences_to_matrix(x_test, mode="binary")
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# TODO(peter): Use dataset api must do this week
# build the dataset object
dx_train = tf.data.Dataset.from_tensor_slices(x_train)
dy_train = tf.data.Dataset.from_tensor_slices(y_train)
train_dataset = tf.data.Dataset.zip(
    (dx_train, dy_train)).shuffle(50000).batch(BATCH_SIZE)

dx_test = tf.data.Dataset.from_tensor_slices(x_test)
dy_test = tf.data.Dataset.from_tensor_slices(y_test)
test_dataset = tf.data.Dataset.zip((dx_test, dy_test)).batch(x_test.shape[0])
# generate iterator
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
print("train_dataset output shapes", train_dataset.output_shapes)

X, Y = iterator.get_next()

# dataset initializer
train_init = iterator.make_initializer(train_dataset)
test_init = iterator.make_initializer(test_dataset)

# build the inference network
net = tf.layers.dense(X, units=512, activation=tf.nn.relu)
net = tf.layers.dropout(net, rate=0.5)
output = tf.layers.dense(net, units=NUM_CLASSES, activation=tf.nn.softmax)

# set the loss
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=output, labels=Y, name="loss")
loss = tf.reduce_mean(loss)

global_step = tf.get_variable("global_step", trainable=False,
                              initializer=tf.constant(0))
# set the optimizer
optimizer = tf.train.AdamOptimizer(
    learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

init_op = tf.global_variables_initializer()

save_mkdir(ITER_PATH)
save_mkdir(BEST_PATH)
save_mkdir(GRAPH_PATH)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    # saver restore from checkpoint file.  dirname get that file path's directory
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(ITER_PATH + "/checkpoint"))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_writer = tf.summary.FileWriter(GRAPH_PATH + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(GRAPH_PATH + "/test", sess.graph)

    # initialize global_step
    global_step.eval()

    n_batches = int(x_train.shape[0] / BATCH_SIZE)
    best_test_loss = None

    for i in range(EPOCHS):
        epoch_train_loss = 0.0
        epoch_validation_loss = 0.0

        # train the network
        sess.run(train_init)
        try:
            while True:
                _, loss_batch = sess.run([optimizer, loss])
                epoch_train_loss += loss_batch
        except tf.errors.OutOfRangeError:
            pass
        print("epoch {0}:\nAverage train loss: {1}".format(
            i + 1, epoch_train_loss / n_batches), end='\t')

        # train_summary = sess.run(summary_op, feed_dict={
        #                          X: x_train, Y: y_train})
        # train_writer.add_summary(train_summary, global_step=i)

        if (i + 1) % 2 == 0:
            saver.save(sess, ITER_PATH + "/nn", global_step=global_step)

        # test
        sess.run(test_init)
        try:
            while True:
                test_loss, test_summary = sess.run([loss, summary_op])
                test_writer.add_summary(test_summary, global_step=i)
        except tf.errors.OutOfRangeError:
            pass
        print("Average test loss: {}".format(test_loss))

        if not best_test_loss:
            best_test_loss = test_loss
        else:
            if test_loss < best_test_loss:
                saver.save(
                    sess, BEST_PATH + "/nn_test_loss-{:0.4f}".format(test_loss))
                print("current test loss:{0:0.4f} is better than last best record:{1:0.4f}, saved.".format(
                    test_loss, best_test_loss))
                best_test_loss = test_loss
        # ensure print stdout to screen immediately
        sys.stdout.flush()

    train_writer.close()
    # test_writer.close()
