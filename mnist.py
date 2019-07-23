"""
Spyder Editor

This is a temporary script file.
"""

import struct
import random
import os
import numpy as np
from scipy.special import expit as sigmoid
from scipy.stats import truncnorm
from tqdm import trange
import matplotlib.pyplot as plt


N_OUT = 10
N_IN = 784


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

    return images, labels


def feedforward(W, x):
    return sigmoid(W @ x)


def backprop(W, x, y):
    z = feedforward(W, x)
    u = 2 * (z - y) * z * (1 - z)
    dW = np.outer(u, x)
    return dW


def onehot(c):
    y = np.zeros(10)
    y[c] = 1
    return y


if __name__ == '__main__':
    train_images, train_labels = load_mnist('train-images-idx3-ubyte',
                                            'train-labels-idx1-ubyte',
                                            60000)
    test_images, test_labels = load_mnist('t10k-images-idx3-ubyte',
                                          't10k-labels-idx1-ubyte',
                                          10000)
    
    W = truncnorm.rvs(-0.1, 0.1, size=(N_OUT, N_IN))
    
    random_order = list(range(60000))
    random.shuffle(random_order)

    for i in random_order:
        x = train_images[i]
        c = train_labels[i]
        y = onehot(c)
        
        dW = backprop(W, x, y)
        W -= 0.05 * dW
        
        if not i % 100:
            fig, axes = plt.subplots(3, 3)
            for Wi, ax in zip(W, axes.flat):
                ax.imshow(Wi.reshape(28,28),cmap='gray')
            plt.show()
            fig, (aa, ab) = plt.subplots(1, 2)
            aa.imshow(x.reshape(28, 28), cmap='gray')
            ab.imshow(feedforward(W, x).reshape(1, 10), cmap='gray')
            plt.show()
        
    
    count_correct = 0
    for i in range(10000):
        x = test_images[i]
        c = test_labels[i]
        z = feedforward(W, x)
        c_hat = np.argmax(z)
        
        if c_hat == c:
            count_correct += 1
        
    print(count_correct/10000)
    
    
    fig, axes = plt.subplots(3, 3)
    for i, ax in zip(random_order, axes.flat):
        ax.imshow(train_images[i].reshape(28,28),cmap='gray')
    plt.show()
    
    
        
        
    
    
    
