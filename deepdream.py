"""
Created on Mon Jul 19 15:03:44 2019@author: deenashefter
"""
"""
Spyder EditorThis is a temporary script file.
"""
import struct
import numpy as np
import tensorflow as tf
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

N_BATCHES = 100001
BATCH_SZ = 32
N_OUT = 10
N_HID = 128
N_IN = 784
N_STEPS = 100000


def load_mnist(imagefile, labelfile, count):
    with open(imagefile, 'rb') as image_data:
        image_data.read(16)
        images = []
        for _ in range(count):
            bytes = image_data.read(784)
            image = np.asarray(struct.unpack('784B', bytes),
                               dtype=np.float_)
            image /= 255.0
            images.append(image)
    with open(labelfile, 'rb') as label_data:
        label_data.read(8)
        labels = []
        for _ in range(count):
            byte = label_data.read(1)
            labels += struct.unpack('1B', byte)
    return (np.array(images).astype(np.float32),
            np.array(labels).astype(np.int32))


def layer(input_, n_in, n_out, name,
          activation=tf.nn.sigmoid):
    """
    A single layer's feedforward step.
    """
    W_init = truncnorm.rvs(-0.05, 0.05, size=(n_in, n_out))
    W_ = tf.get_variable(
        'W' + name,
        initializer=tf.constant(W_init.astype(np.float32)),
        trainable=True)

    b_init = truncnorm.rvs(-0.05, 0.05, size=n_out)
    b_ = tf.get_variable(
        'b' + name,
        initializer=tf.constant(b_init.astype(np.float32)),
        trainable=True)

    z_ = activation(tf.nn.xw_plus_b(input_, W_, b_))

    return z_, W_, b_


# ---------------------------------------------------------------------
if __name__ == '__main__':
    # ------------------------- set up model --------------------------
    x = tf.placeholder(tf.float32, shape=(BATCH_SZ, N_IN), name='x')
    h_, W1_, b1_ = layer(x, N_IN, N_HID, 'layer1')
    z_, W2_, b2_ = layer(h_, N_HID, N_OUT, 'layer2')

    # --------------------------- training ----------------------------
    c_ = tf.placeholder(tf.int32, shape=(BATCH_SZ,), name='c')
    y_ = tf.one_hot(c_, 10, dtype=tf.float32)
    loss_ = tf.reduce_mean(tf.square(y_ - z_))
    opt = tf.train.MomentumOptimizer(0.05, 0.1)
    train_op = opt.minimize(loss_)

    # --------------------------- deep dream --------------------------
    x_init = np.random.uniform(low=0.0, high=1.0, size=(9, N_IN)).astype(np.float32)
    dd_x = tf.get_variable('input', initializer=x_init)
    dd_h_ = tf.nn.sigmoid(tf.nn.xw_plus_b(dd_x, W1_, b1_))
    dd_z_ = tf.nn.sigmoid(tf.nn.xw_plus_b(dd_h_, W2_, b2_))

    R = tf.reduce_sum(dd_z_[:, 0])
    max_op_ = tf.train.GradientDescentOptimizer(0.01).minimize(-R, var_list=[dd_x])

    # ---------------------------- data -------------------------------
    train_images, train_labels = load_mnist('fashionmnist/train-images-idx3-ubyte',
                                            'fashionmnist/train-labels-idx1-ubyte',
                                            60000)
    test_images, test_labels = load_mnist('fashionmnist/t10k-images-idx3-ubyte',
                                          'fashionmnist/t10k-labels-idx1-ubyte',
                                          10000)

    # ------------------------- testing ops ---------------------------
    test_X_ = tf.constant(np.array(test_images))
    test_C_ = tf.constant(np.array(test_labels))
    test_H_ = tf.nn.sigmoid(tf.nn.xw_plus_b(test_X_, W1_, b1_))
    test_Z_ = tf.nn.sigmoid(tf.nn.xw_plus_b(test_H_, W2_, b2_))
    test_predictions = tf.argmax(test_Z_, axis=1, output_type=tf.int32)
    accuracy_ = tf.reduce_mean(
        tf.cast(
            tf.equal(test_predictions, test_C_),
            tf.float32))

    # ------------------------- run everything ------------------------
    # This op initializes weights and biases
    init_ = tf.global_variables_initializer()
    batch_indices = np.random.randint(60000, size=(N_BATCHES, BATCH_SZ))

    # Launch session
    with tf.Session() as sess:
        # Run initialization
        sess.run(init_)

        # Training loop
        for i, batch_inds in enumerate(batch_indices):
            loss, _ = sess.run([loss_, train_op], feed_dict={
                x: train_images[batch_inds],
                c_: train_labels[batch_inds],
            })
            # if not i % 50000:
            #     print(loss)
        W1 = sess.run(W1_)
        W2 = sess.run(W2_)
        fig, axes = plt.subplots(3, 3)
        for Wi, ax in zip(W1.T, axes.flat):
            ax.imshow(Wi.reshape(28, 28), cmap='gray')
        plt.show()
        fig, axes = plt.subplots(3, 3, squeeze=False)
        for Wi, ax in zip((W1 @ W2).T, axes.flat):
            ax.imshow(Wi.reshape(28, 28), cmap='gray')
        plt.show()

        # Run test op
        accuracy = sess.run(accuracy_)
        print('accuracy', accuracy)
        # fig, axes = plt.subplots(3, 3)
        # for i, ax in zip(random_order, axes.flat):
        #     ax.imshow(train_images[i].reshape(28, 28), cmap='gray')
        # plt.show()

        for i in range(N_STEPS):
            sess.run(max_op_)
            if not i % 5000:
                dd_result = sess.run(dd_x)
                print(dd_result)
                fig, axes = plt.subplots(3, 3, squeeze=False)
                for x, ax in zip(dd_result, axes.flat):
                    ax.imshow(x.reshape(28, 28), cmap='gray')
                plt.show()