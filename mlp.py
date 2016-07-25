import numpy as np
np.random.seed(42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# multi layer perceptron
class MLP():
    def __init__(self, layers):
        self.layers = layers
        input_num = layers[0]
        self.weights = [None]
        self.biases = [None]
        self.init_weight()

    def init_weight(self):
        for i in range(1, len(self.layers)):
            current_layer = self.layers[i]
            input_layer = self.layers[i-1]
            self.weights.append(np.random.rand(current_layer, input_layer))
            self.biases.append(np.random.rand(current_layer))

    def feedforward(self, x):
        activations = [x]
        weighted_inputs = [None]
        for i in range(1, len(self.layers)):
            w = self.weights[i]
            b = self.biases[i]
            prev_a = activations[i-1]
            activation, weighted_input = self.compute_layer(w, b, prev_a)
            weighted_inputs.append(weighted_input)
            activations.append(activation)

        return activations, weighted_inputs

    def compute_layer(self, w, b, x):
        weighted_input = np.dot(w, x) + b
        activation = sigmoid(weighted_input)
        return activation, weighted_input

    # mean squared error
    def loss(self, X, y):
        outputs = self.output(X)
        return np.mean((y - outputs) ** 2) / 2.0

    def output(self, X):
        return np.array([self.feedforward(x)[0][-1] for x in X])

    def nabla_loss(self, a, y):
        return a - y

    def train_on_batch(self, X, y, learning_rate=0.001):
        batch_num = len(X)
        w_derivatives, b_derivatives = self.backprop(X, y)

        for i in range(batch_num):
            for l in range(1, len(self.layers)):
                w_delta = w_derivatives[i][l]
                b_delta = b_derivatives[i][l]
                self.weights[l] -= learning_rate / batch_num * w_delta
                self.biases[l] -= learning_rate / batch_num * b_delta

    def backprop(self, X, y):
        batch_num = len(X)
        assert len(X) == len(y)

        w_derivatives = []
        b_derivatives = []

        for i, x in enumerate(X):
            errors = [None] * len(self.layers)

            activations, weighted_inputs = self.feedforward(x)
            nabla = self.nabla_loss(activations[-1], y[i])
            errors[-1] = nabla * sigmoid_prime(weighted_inputs[-1])

            for l in range(len(errors)-2, 0, -1):
                layer_error = np.dot(self.weights[l+1].T, errors[l+1]) * sigmoid_prime(weighted_inputs[l+1])

                errors[l] = layer_error

            d_w = [None]
            d_b = [None]

            for l in range(1, len(self.layers)):
                b_derivative = errors[l]
                d_b.append(b_derivative)

                w = self.weights[l]
                w_derivative = np.array([activations[l-1]] * self.layers[l]) * np.array([errors[l]] * self.layers[l-1]).T

                d_w.append(w_derivative)

            w_derivatives.append(d_w)
            b_derivatives.append(d_b)

        return w_derivatives, b_derivatives



def batch(X, y, batch_size=50):
    example_size = len(X)
    batch_index = np.random.choice(np.arange(example_size), size=batch_size)
    return X[batch_index], y[batch_index]

def test_split(X, y, rate=0.0):
    example_size = len(X)
    validation_size = int(rate * example_size)

    validation_index = np.random.choice(np.arange(example_size), size=validation_size)
    train_index = np.array(list(set(np.arange(example_size)) - set(validation_index)))

    return X[train_index], y[train_index], X[validation_index], y[validation_index]

def evaluate(network, X, y):
    y_out = network.output(X)
    accuracy = np.mean((y_out > 0.5) == (y == 1.0))
    loss = network.loss(X, y)

    return accuracy, loss

if __name__ == "__main__":
    network = MLP([2, 4, 1])

    X = np.random.randn(10000, network.layers[0])
    y = np.array([x[0] < x[1] for x in X]) * 1.0 # y > x

    X_train, y_train, X_valid, y_valid = test_split(X, y, rate=0.2)

    for epoch in range(10000):
        X_batch, y_batch = batch(X_train, y_train)
        network.train_on_batch(X_batch, y_batch, learning_rate=0.1)

        if epoch % 100 == 0:
            train_accuracy, train_loss = evaluate(network, X_train, y_train)
            valid_accuracy, valid_loss = evaluate(network, X_valid, y_valid)

            print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" % (epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))
