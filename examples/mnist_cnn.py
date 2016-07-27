import numpy as np
np.random.seed(42)

import mnist_loader
import util
from network import *
from convolutional import *

net = Network([
    Input([28, 28]),
    Convolution([12, 2, 2]),
    MaxPooling(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation=Softmax()),
], loss=CrossEntropy())

print(net.summary())

(training_data, validation_data, test_data) = mnist_loader.load()
X_train, y_train = training_data
X_valid, y_valid = validation_data
X_test, y_test = test_data

X_train = X_train.reshape(len(X_train), 28, 28)
X_valid = X_valid.reshape(len(X_valid), 28, 28)
X_test = X_test.reshape(len(X_test), 28, 28)

def evaluate(network, X, y):
    y_out = network.output(X)
    accuracy = np.mean(y_out.argmax(axis=1) == y.argmax(axis=1))
    loss = network.loss(X, y)

    return accuracy, loss

batch_size = 30

early_stop_wait = 0
prev_valid_loss = 100000.

import time
t = time.time()
n = 1000
for i in range(n):
    net.feedforward(X_train[0])

print((time.time() - t) / n)

t = time.time()
n = 1000
for i in range(n):
    net.backprop(X_train[0:1], y_train[0:1])

print((time.time() - t) / n)


for epoch in range(50):
    for i in range(len(X_train) // batch_size):
        X_batch, y_batch = util.batch(X_train, y_train, batch_size=batch_size)
        net.train_on_batch(X_batch, y_batch, learning_rate=0.1)

        train_accuracy, train_loss = evaluate(net, X_batch, y_batch)
        valid_accuracy, valid_loss = evaluate(net, X_valid, y_valid)

        print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" % (epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))

    # early stopping
    if valid_loss > prev_valid_loss:
        early_stop_wait += 1

        if early_stop_wait >= 2:
            break

    else:
        early_stop_wait = 0

    prev_valid_loss = valid_loss


print("test_accuracy: %f\ttest_loss: %f" % evaluate(net, X_test, y_test))
