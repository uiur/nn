import numpy as np

import mnist_loader
import util
from network import *

np.random.seed(42)

net = Network([
    Input(784),
    Dense(30),
    Dense(10, activation=Softmax()),
], loss=CrossEntropy())

(training_data, validation_data, test_data) = mnist_loader.load()
X_train, y_train = training_data
X_valid, y_valid = validation_data
X_test, y_test = test_data

def evaluate(network, X, y):
    y_out = network.output(X)
    accuracy = np.mean(y_out.argmax(axis=1) == y.argmax(axis=1))
    loss = network.loss(X, y)

    return accuracy, loss

batch_size = 30
for epoch in range(100):
    for i in range(len(X_train) // batch_size):
        X_batch, y_batch = util.batch(X_train, y_train, batch_size=batch_size)
        net.train_on_batch(X_batch, y_batch, learning_rate=1.)

    train_accuracy, train_loss = evaluate(net, X_batch, y_batch)
    valid_accuracy, valid_loss = evaluate(net, X_valid, y_valid)

    print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" % (epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))

print("test_accuracy: %f\ttest_loss: %f" % evaluate(net, X_test, y_test))
