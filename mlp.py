import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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
            self.weights.append(np.random.randn(current_layer, input_layer))
            self.biases.append(np.random.randn(current_layer))

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
        return 0.5 * np.mean(np.linalg.norm(y - outputs, axis=1) ** 2)

    def output(self, X):
        return np.array([self.feedforward(x)[0][-1] for x in X])

    def nabla_loss(self, a, y):
        return a - y

    def train_on_batch(self, X, y, learning_rate=0.001):
        batch_num = len(X)
        w_derivatives, b_derivatives = self.backprop(X, y)

        w_delta_sum = w_derivatives[0]
        b_delta_sum = b_derivatives[0]

        for i in range(1, batch_num):
            for l in range(1, len(self.layers)):
                w_delta_sum[l] += w_derivatives[i][l]
                b_delta_sum[l] += b_derivatives[i][l]

        for l in range(1, len(self.layers)):
            self.weights[l] -= learning_rate / batch_num * w_delta_sum[l]
            self.biases[l] -= learning_rate / batch_num * b_delta_sum[l]

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
                errors[l] = np.dot(self.weights[l+1].T, errors[l+1]) * sigmoid_prime(weighted_inputs[l])

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



def batch(X, y, batch_size=100):
    example_size = len(X)
    batch_index = np.random.choice(np.arange(example_size), size=batch_size)
    return X[batch_index], y[batch_index]


def test_split(X, y, rate=0.0):
    example_size = len(X)
    validation_size = int(rate * example_size)

    validation_index = np.random.choice(np.arange(example_size), size=validation_size)
    train_index = np.array(list(set(np.arange(example_size)) - set(validation_index)))

    return X[train_index], y[train_index], X[validation_index], y[validation_index]
