import mnist_loader
import mlp
import numpy as np

(training_data, validation_data, test_data) = mnist_loader.load()
X_train, y_train = training_data
X_valid, y_valid = validation_data
X_test, y_test = test_data

network = mlp.MLP([784, 30, 10])

def evaluate(network, X, y):
    y_out = network.output(X)
    accuracy = np.mean(y_out.argmax(axis=1) == y.argmax(axis=1))
    loss = network.loss(X, y)

    return accuracy, loss

batch_size = 30
for epoch in range(100):
    for i in range(len(X_train) // batch_size):
        X_batch, y_batch = mlp.batch(X_train, y_train, batch_size=batch_size)
        network.train_on_batch(X_batch, y_batch, learning_rate=3.0)

    train_accuracy, train_loss = evaluate(network, X_batch, y_batch)
    valid_accuracy, valid_loss = evaluate(network, X_valid, y_valid)

    print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" % (epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))

print("test_accuracy: %f\ttest_loss: %f" % evaluate(network, X_test, y_test))
