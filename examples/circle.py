import util
from network import *
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

network = Network([
    Input(2),
    Dense(8),
    Dense(16),
    Dense(1)
])

X = np.random.randn(10000, 2)
y = np.array([(x[0] ** 2) + (x[1] ** 2) < 1.0 ** 2 for x in X]) * 1.0

X_train, y_train, X_valid, y_valid = util.test_split(X, y, rate=0.1)

def evaluate(network, X, y):
    y_out = network.output(X)
    accuracy = np.mean((y_out > 0.5).flatten() == (y == 1.0))
    loss = network.loss(X, y)

    return accuracy, loss


batch_size = 50
for epoch in range(20):
    for i in range(len(X_train) // batch_size):
        X_batch, y_batch = util.batch(X_train, y_train, batch_size=batch_size)
        network.train_on_batch(X_batch, y_batch, learning_rate=0.1)

    train_accuracy, train_loss = evaluate(network, X_batch, y_batch)
    valid_accuracy, valid_loss = evaluate(network, X_valid, y_valid)

    print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" % (epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))

y_result = network.output(X_valid)
for i, x in enumerate(X_valid):
    if y_result[i][0] > 0.5:
        plt.plot(x[0], x[1], "bo")
    else:
        plt.plot(x[0], x[1], "ro")

plt.show()
