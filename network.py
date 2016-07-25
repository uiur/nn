import numpy as np

class Network():
    def __init__(self, layers):
        self.layers = layers
        self.layer_nums = [layer.output_num() for layer in self.layers]
        for l, layer in enumerate(self.layers):
            if l - 1 < 0:
                continue

            layer.build(self.layer_nums[l-1])

    def feedforward(self, x):
        activations = []

        assert x.shape[0] == self.layer_nums[0]
        for l in range(len(self.layers)):
            if l - 1 >= 0:
                prev_activation = activations[l-1]
            else:
                prev_activation = x

            activation, weighted_input = self.layers[l].call(prev_activation)
            activations.append(activation)

        return activations

    # mean squared error
    def loss(self, X, y):
        outputs = self.output(X)
        return 0.5 * np.mean(np.linalg.norm(y - outputs, axis=1) ** 2)

    def output(self, X):
        return np.array([self.feedforward(x)[-1] for x in X])

    def nabla_loss(self, a, y):
        return a - y

    def train_on_batch(self, X, y, learning_rate=0.1):
        batch_num = len(X)
        w_derivatives, b_derivatives = self.backprop(X, y)

        w_delta_sum = w_derivatives[0]
        b_delta_sum = b_derivatives[0]

        for i in range(1, batch_num):
            for l in range(1, len(self.layers)):
                w_delta_sum[l] += w_derivatives[i][l]
                b_delta_sum[l] += b_derivatives[i][l]

        for l in range(1, len(self.layers)):
            layer = self.layers[l]
            layer.weight -= learning_rate / batch_num * w_delta_sum[l]
            layer.bias -= learning_rate / batch_num * b_delta_sum[l]

    def backprop(self, X, y):
        batch_num = len(X)
        assert len(X) == len(y)

        w_derivatives = []
        b_derivatives = []

        for i, x in enumerate(X):
            activations = self.feedforward(x)

            last_layer = self.layers[-1]
            nabla = self.nabla_loss(last_layer.activation, y[i])
            last_layer.error = nabla * sigmoid_prime(last_layer.weighted_input)

            for l in range(len(self.layers)-1, 1, -1):
                layer = self.layers[l]
                self.layers[l-1].error = layer.backprop(self.layers[l-1])

            w_derivative = [None]
            b_derivative = [None]

            for l in range(1, len(self.layers)):
                layer = self.layers[l]
                w_nabla, b_nabla = layer.nabla(self.layers[l-1])
                w_derivative.append(w_nabla)
                b_derivative.append(b_nabla)

            w_derivatives.append(w_derivative)
            b_derivatives.append(b_derivative)

        return w_derivatives, b_derivatives



def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Dense():
    def __init__(self, n):
        self.n = n

    def build(self, input_num):
        self.weight = np.random.randn(self.n, input_num)
        self.bias = np.random.randn(self.n)
        self.input_num = input_num

    def call(self, x):
        weighted_input = np.dot(self.weight, x) + self.bias
        activation = sigmoid(weighted_input)

        self.weighted_input = weighted_input
        self.activation = activation

        return activation, weighted_input

    # returns errors of a previous layer
    def backprop(self, prev_layer):
        return np.dot(self.weight.T, self.error) * sigmoid_prime(prev_layer.weighted_input)

    def nabla(self, prev_layer):
        w_nabla = np.array([prev_layer.activation] * self.n) * np.array([self.error] * self.input_num).T
        b_nabla = self.error

        return w_nabla, b_nabla

    def output_num(self):
        return self.n


class Input():
    def __init__(self, n):
        self.n = n

    def build(self):
        pass

    def call(self, x):
        self.activation = x
        return x, None

    def output_num(self):
        return self.n
