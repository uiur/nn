import numpy as np
import time


class MeanSquaredError():
    def __init__(self):
        pass

    def nabla(self, a, y):
        return a - y

    def call(self, a, y):
        return 0.5 * np.mean(np.linalg.norm(y - a, axis=1) ** 2)


class CrossEntropy():
    def __init__(self):
        pass

    def nabla(self, a, y):
        return -1 * (y - a) / (a * (1 - a))

    def call(self, a, y):
        return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


class Network():
    def __init__(self, layers, loss=MeanSquaredError()):
        self.layers = layers
        self.layer_nums = []
        self.loss_function = loss

        for l, layer in enumerate(self.layers):
            if l - 1 >= 0:
                layer.build(self.layer_nums[l-1])

            self.layer_nums.append(layer.output_num())

    def feedforward(self, x):
        activations = []

        if type(self.layer_nums[0]) == int:
            assert x.shape[0] == self.layer_nums[0]
        else:
            assert x.shape == tuple(self.layer_nums[0])

        times = []
        for l in range(len(self.layers)):
            if l - 1 >= 0:
                prev_activation = activations[l-1]
            else:
                prev_activation = x

            activation = self.layers[l].call(prev_activation)
            activations.append(activation)

        return activations

    # mean squared error
    def loss(self, X, y):
        activation = self.output(X)
        return self.loss_function.call(activation, y)

    def output(self, X):
        return np.array([self.feedforward(x)[-1] for x in X])

    def train_on_batch(self, X, y, learning_rate=0.1):
        batch_num = len(X)
        self.backprop(X, y)

        for layer in self.layers[1:]:
            layer.update_params(-1 * learning_rate / batch_num)

    def backprop(self, X, y):
        batch_num = len(X)
        assert len(X) == len(y)

        for i, x in enumerate(X):
            activations = self.feedforward(x)

            last_layer = self.layers[-1]
            if self.loss_function.__class__.__name__ == 'CrossEntropy':
                last_layer.error = last_layer.activation - y[i]
            else:
                nabla = self.loss_function.nabla(last_layer.activation, y[i])
                last_layer.error = nabla * last_layer.activation_function.prime(last_layer.weighted_input)

            for l in range(len(self.layers)-1, 1, -1):
                layer = self.layers[l]
                self.layers[l-1].error = layer.backprop(self.layers[l-1])

            for layer, prev_layer in zip(self.layers[1:], self.layers[0:-1]):
                layer.update_nabla(prev_layer)


    def summary(self):
        param_count = 0
        for layer in self.layers[1:]:
            param_count += layer.param_num()

        return { 'params': param_count }


class Sigmoid():
    def call(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self, z):
        return self.call(z) * (1 - self.call(z))


class Softmax():
    def call(self, z):
        c = np.max(z)
        return np.exp(z - c) / np.sum(np.exp(z - c))

    def prime(self, z):
        return self.call(z) * (1 - self.call(z))


class ReLU():
    def call(self, z):
        return z * (z > 0)

    def prime(self, z):
        return (z > 0) * 1.


class Layer():
    def activation_prime(self):
        return self.activation_function.prime(self.weighted_input)

    def update_nabla(self, prev_layer):
        self.nabla(prev_layer)

    def nabla(self, prev_layer):
        if self.param_num() == 0:
            return

        raise Exception("nabla must be implemented")

    def update_params(self, c):
        pass


class Dense(Layer):
    def __init__(self, n, activation=Sigmoid()):
        self.n = n
        self.activation_function = activation

    def build(self, input_num):
        assert type(input_num) != tuple, 'the input shape of Dense layer must be 1 dimension'

        self.weight = np.random.randn(self.n, input_num) / np.sqrt(input_num)
        self.bias = np.random.rand(self.n)

        self.weight_nabla = np.zeros_like(self.weight)
        self.bias_nabla = np.zeros_like(self.bias)

        self.input_num = input_num

    def call(self, x):
        weighted_input = np.dot(self.weight, x) + self.bias
        activation = self.activation_function.call(weighted_input)

        self.weighted_input = weighted_input
        self.activation = activation

        return activation

    # returns errors of a previous layer
    def backprop(self, prev_layer):
        return np.dot(self.weight.T, self.error) * prev_layer.activation_prime()

    def nabla(self, prev_layer):
        w_nabla = np.array([prev_layer.activation] * self.n) * np.array([self.error] * self.input_num).T
        b_nabla = self.error

        return w_nabla, b_nabla

    def update_nabla(self, prev_layer):
        w_nabla, b_nabla = self.nabla(prev_layer)

        self.weight_nabla += w_nabla
        self.bias_nabla += b_nabla

    def update_params(self, c):
        self.weight += c * self.weight_nabla
        self.bias += c * self.bias_nabla

        self.weight_nabla *= 0.
        self.bias_nabla *= 0.

    def output_num(self):
        return self.n

    def param_num(self):
        weight_num = self.weight.shape[0] * self.weight.shape[1]
        bias_num = self.bias.shape[0]

        return weight_num + bias_num


class Input(Layer):
    '''
        Input(784)
        Input([28, 28])
    '''
    def __init__(self, shape):
        self.shape = shape

    def build(self):
        pass

    def call(self, x):
        self.weighted_input = x
        self.activation = x
        return x

    def output_num(self):
        return self.shape
