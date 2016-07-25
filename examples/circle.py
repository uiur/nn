import mlp
import numpy as np
import matplotlib.pyplot as plt

network = mlp.MLP([2, 8, 16, 1])

X = np.random.randn(10000, network.layers[0])
y = np.array([(x[0] ** 2) + (x[1] ** 2) < 1.0 ** 2 for x in X]) * 1.0

X_train, y_train, X_valid, y_valid = mlp.test_split(X, y, rate=0.1)

def evaluate(network, X, y):
    y_out = network.output(X)
    accuracy = np.mean((y_out > 0.5).flatten() == (y == 1.0))
    loss = network.loss(X, y)

    return accuracy, loss


for epoch in range(4000):
    X_batch, y_batch = mlp.batch(X_train, y_train)
    network.train_on_batch(X_batch, y_batch, learning_rate=0.1)

    if epoch % 100 == 0:
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
