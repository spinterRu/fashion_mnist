# -*- coding: utf-8 -*-
"""
https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses, metrics, activations
from tensorflow.contrib.layers import batch_norm, dropout
from tensorflow.contrib.opt import NadamOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import data_augmentation as da


def RRelu(x):
    rand = tf.Variable(tf.random_uniform([]) * 0.3 + 0.1, dtype=tf.float32)
    alpha = tf.cond(is_train, lambda: rand, lambda: tf.Variable(0.3, dtype=tf.float32))
    return tf.nn.relu(x) - tf.nn.relu(-x)*alpha

def conv_layer(layer, num_filters, k_size=(3, 3), shape=(-1, 28, 28, 1), padding="same"):
    new_layer = Conv2D(num_filters, kernel_size=k_size, strides=(1, 1), padding=padding,
                       kernel_initializer='he_normal', input_shape=shape)(layer)
    new_layer = batch_norm(new_layer, updates_collections=None, center=True, scale=True, is_training=is_train)
    return activations.relu(new_layer)

def fc_layer(layer, num_neurons):
    new_layer = Dense(num_neurons, kernel_initializer='he_normal')(layer)
    new_layer = batch_norm(new_layer, updates_collections=None, center=True, scale=True, is_training=is_train)
    return activations.relu(new_layer)

print(tf.__version__)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
K.set_session(sess)

#mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
mnist_data = input_data.read_data_sets('F:/Downloads/fashion_mnist', reshape=False, one_hot=True, validation_size=0)

#plt.gray() # use this line if you don't want to see it in color
#plt.imshow(train[0].reshape(28, 28))
#plt.show()

is_train = tf.placeholder(tf.bool)
img = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

labels = tf.placeholder(tf.float32, shape=(None, 10))
lr = tf.placeholder(tf.float32)
# x_image = tf.reshape(img, [-1, 28, 28, 1])

# first 3 convolutions approximate Conv(7,7):
layer = conv_layer(img, 64)
layer = conv_layer(layer, 64)
layer = conv_layer(layer, 64)
layer = MaxPooling2D()(layer)
layer = dropout(layer, keep_prob=0.7, is_training=is_train)
layer = conv_layer(layer, 128, shape=(-1, 14, 14, -1))
layer = conv_layer(layer, 128, shape=(-1, 14, 14, -1))
layer = conv_layer(layer, 64, (1, 1), shape=(-1, 14, 14, -1))
layer = MaxPooling2D()(layer)
layer = Flatten()(layer)
layer = dropout(layer, keep_prob=0.7, is_training=is_train)
layer = fc_layer(layer, 2048)
layer = dropout(layer, is_training=is_train)
layer = fc_layer(layer, 512)
layer = dropout(layer, is_training=is_train)
layer = fc_layer(layer, 256)
layer = dropout(layer, is_training=is_train)
layer = Dense(10, kernel_initializer='glorot_normal')(layer)
layer = batch_norm(layer, updates_collections=None, center=True, scale=True, is_training=is_train)
preds = activations.softmax(layer)

lossL2 = tf.add_n(
        [ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name ])

beta = 1e-7
loss = tf.reduce_mean(losses.categorical_crossentropy(labels, preds))
train_step = NadamOptimizer(learning_rate=lr).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

acc_value = tf.reduce_mean(metrics.categorical_accuracy(labels, preds))
def accuracy(data, n):
    l = []
    for i in range(n):
        batch = data.next_batch(100)
        acc = acc_value.eval(feed_dict={img: batch[0], labels: batch[1], is_train: False})
        l.append(acc)
    return np.mean(l)

initial_learning_rate = 0.001
cumulative_loss = 0.0
BATCH_SIZE = 20
TRAIN_SIZE = 60000
NUM_EPOCHS = 100
EPOCH_SIZE = TRAIN_SIZE / BATCH_SIZE
num_iterations = int(NUM_EPOCHS * EPOCH_SIZE)

# Run training loop
with sess.as_default():
    for i in range(1, num_iterations + 1):
        current_learning_rate = initial_learning_rate * (1.0 - i / (num_iterations + 5))
        batch = mnist_data.train.next_batch(BATCH_SIZE)
        train, train_labels = da.augment_data(batch[0], batch[1], use_random_zoom=False, use_random_shift=False)

        _, loss_val = sess.run([train_step, loss], feed_dict={img: train,
                                                              labels: train_labels, is_train: True,
                                                              lr: current_learning_rate})
        cumulative_loss = cumulative_loss + loss_val
        if i % EPOCH_SIZE == 0:
            print(str(cumulative_loss / EPOCH_SIZE))
            cumulative_loss = 0.0
        if i % (10 * EPOCH_SIZE) == 0:
            test_acc = accuracy(mnist_data.test, 100)
            print("Epoch: " + str(i / EPOCH_SIZE) + " Test accuracy: " + str(test_acc))

    train_acc = accuracy(mnist_data.train, 600)
    print("Train accuracy: " + str(train_acc))
    test_acc = accuracy(mnist_data.test, 100)
    print("Test accuracy: " + str(test_acc))

sess.close()