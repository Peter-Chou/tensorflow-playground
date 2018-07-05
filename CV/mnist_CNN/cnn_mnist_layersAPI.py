"""
this file is for cnn in mnist data
"""

import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


# global parameters
SEED = 1000

# model hyperparameters
BATCH_SIZE = 200
EPOCH_NUM = 20
LEARNING_RATE = 0.001
SKIP_STEP = 5

mnist = input_data.read_data_sets("MNIST_DATA", validation_size=0)

tf.set_random_seed(SEED)
# reset the default graph before building
tf.reset_default_graph()


class cnn_mnist:
    """
    easy way to build a cnn using property
    :param dataset: tf.data.Dataset object
    :param batch_size: batch size for each training
    :param epoch_num: number of times repeat the whole dataset
    :param learning_rate: learning rate for algorithm
    """
    def __init__(self, dataset, batch_size, epoch_num, learning_rate):
        self._dataset = dataset
        self._batch_size = batch_size
        self._epoch_num = epoch_num
        self._lr = learning_rate
        self.n_classes = 10
        if not self._global_step:
            self._global_step = tf.get_variable("global_step",
                                                initializer=tf.constant(0),
                                                trainable=False)
        self._skip_step = SKIP_STEP
        self._keep_rate = tf.constant(0.75)
        self._n_test = 10000
        self._training = True

    def __getattr__(self, attr):
        return None

    def _create_dataset(self):
        """
        setup train / test dataset object
        create general iterator for batch training.
        """
        if not self._inputs:
            with tf.name_scope("data"):
                # trainset
                self._trainset = tf.data.Dataset.from_tensor_slices(
                    (self._dataset.train.images.reshape([-1, 28, 28, 1]),
                        tf.keras.utils.to_categorical(self._dataset.train.labels, 10)))
                self._trainset = self._trainset.shuffle(10000)
                self._trainset = self._trainset.batch(self._batch_size)
                # testset
                self._testset = tf.data.Dataset.from_tensor_slices(
                    (self._dataset.test.images.reshape([-1, 28, 28, 1]),
                        tf.keras.utils.to_categorical(self._dataset.test.labels, 10)))
                self._testset = self._testset.shuffle(10000)
                self._testset = self._testset.batch(self._batch_size)
                # iteration
                iter = tf.data.Iterator.from_structure(self._trainset.output_types,
                                                       self._trainset.output_shapes)
                self._inputs, self._targets = iter.get_next()
                self._train_init_op = iter.make_initializer(self._trainset)
                self._test_init_op = iter.make_initializer(self._testset)

    def _create_cnn(self):
        """
        create CNN net structure
        """
        if not self._logits:
            with tf.name_scope("cnn"):
                conv1 = tf.layers.conv2d(inputs=self._inputs,
                                         filters=32,
                                         kernel_size=[5, 5],
                                         padding="same",
                                         activation=tf.nn.relu,
                                         name="conv1")
                pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                pool_size=[2, 2],
                                                strides=2,
                                                name="pool1")
                conv2 = tf.layers.conv2d(inputs=pool1,
                                         filters=64,
                                         kernel_size=[5, 5],
                                         padding="same",
                                         activation=tf.nn.relu,
                                         name="conv2")
                pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                name="pool2")

                feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
                pool2 = tf.reshape(pool2, [-1, feature_dim])
                fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name="fc")
                dropout = tf.layers.dropout(fc, rate=self._keep_rate,
                                            training=self._training,
                                            name="dropout1")
                self._logits = tf.layers.dense(dropout, units=self.n_classes,
                                               name="logits")

    def _loss_function(self):
        """
        create loss tensor operator
        """
        if not self._loss:
            with tf.name_scope("loss"):
                entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self._targets,
                    logits=self._logits)
                self._loss = tf.reduce_mean(entropy, name="loss")

    def _optimize(self):
        """
        create optimization using adam
        """
        if not self._opt:
            with tf.name_scope("optimize"):
                self._opt = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(
                    self._loss, global_step=self._global_step)

    def _summary(self):
        """
        create FileWriter for tensorboard
        """
        if not self._summary_op:
            with tf.name_scope("summary"):
                tf.summary.scalar("loss", self._loss)
                tf.summary.scalar("accuracy", self._accuracy)
                tf.summary.scalar("histogram_loss", self._loss)
                self._summary_op = tf.summary.merge_all()

    def _eval(self):
        """
        predict the categories
        """
        if not self._accuracy:
            with tf.name_scope("predict"):
                pred = tf.nn.softmax(self._logits)
                correct_preds = tf.equal(tf.argmax(pred, 1), tf.argmax(self._targets, 1))
                self._accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build_graph(self):
        """
        build the CNN model graph
        """
        self._create_dataset()
        self._create_cnn()
        self._loss_function()
        self._optimize()
        self._eval()
        self._summary()

    def _train_one_epoch(self, sess, saver, init, writer, epoch, step):
        """
        training for one epoch
        :param sess: current model graph
        :param saver: tf.train.Saver object
        :param init: general iterator --- make_initializer method
        :param writer: FireWriter for tensorboard
        :param epoch: current epoch number
        :param step: current iteration step number
        :return: step <==> next iteration step number
        """
        start_time = time.time()
        sess.run(init)
        self._training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self._opt, self._loss, self._summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self._skip_step == 0:
                    print("loss at step {0}: {1}".format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, "checkpoints/convnet_layers/mnist-convet", step)
        print("Average loss at epoch {0}: {1:0.3f}".format(epoch, total_loss/n_batches))
        print("Took: {0} seconds".format(time.time() - start_time))
        return step

    def _eval_once(self, sess, init, writer, epoch, step):
        """
        build one epoch for test validation.
        """
        start_time = time.time()
        sess.run(init)
        self._training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self._accuracy, self._summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print("accuracy at epoch {0}: {1:0.3f}".format(
            epoch, float(total_correct_preds)/self._n_test))
        print("Took: {0} seconds".format(time.time() - start_time))

    def train(self, n_epoch):
        """
        train method to run the whole process
        """
        safe_mkdir("checkpoints")
        safe_mkdir("checkpoints/convnet_layers")
        writer = tf.summary.FileWriter("./graphs/convnet_layers", tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                "checkpoints/convnet_layers/checkpoint"))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self._global_step.eval()

            for epoch in range(n_epoch):
                step = self._train_one_epoch(sess, saver, self._train_init_op,
                                             writer, epoch, step)
                self._eval_once(sess, self._test_init_op, writer, epoch, step)
        writer.close()


if __name__ == "__main__":
    model = cnn_mnist(mnist, BATCH_SIZE, EPOCH_NUM, LEARNING_RATE)
    model.build_graph()
    model.train(n_epoch=10)
