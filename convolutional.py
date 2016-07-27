import numpy as np
from scipy import signal

from network import *


class Convolution(Layer):
    ''' Convolution([32, 3, 3], activation=ReLU())
        border strategy is 'same'
    '''
    def __init__(self, shape, activation=ReLU()):
        assert len(shape) == 3
        self.shape = shape
        self.activation_function = activation

    def build(self, input_shape):
        self.weight = np.random.randn(self.shape[0], self.shape[1], self.shape[2]) / np.sqrt(self.shape[1] * self.shape[2])
        self.bias = np.random.rand(self.shape[0])

        self.weight_nabla = np.zeros_like(self.weight)
        self.bias_nabla = np.zeros_like(self.bias)

        self.input_shape = input_shape

    def call(self, x):
        filter_num = self.shape[0]
        weighted_inputs = []
        activations = []
        for i in range(filter_num):
            w = self.weight[i]
            b = self.bias[i]

            z = signal.correlate2d(x, w, mode='same') + b
            a = self.activation_function.call(z)
            weighted_inputs.append(z)
            activations.append(a)

        self.weighted_input = np.array(weighted_inputs)
        self.activation = np.array(activations)

        return self.activation

    def backprop(self, prev_layer):
        prev_error = np.zeros(self.input_shape)
        out_shape = tuple(self.output_num())

        height = self.shape[1]
        width = self.shape[2]

        for depth, _ in enumerate(self.error):
            for i in range(out_shape[1]):
                for j in range(out_shape[2]):
                    pixel_error = self.error[depth][i][j]
                    nabla = prev_layer.activation_function.prime(prev_layer.weighted_input[i:i+height, j:j+width])

                    prev_error[i:i+height, j:j+width] += pixel_error * self.weight[depth] * nabla

        return prev_error


    def nabla(self, prev_layer):
        filter_num, filter_height, filter_width = self.shape

        b_nabla = [np.sum(e) for e in self.error]

        w_nabla = []
        for depth in range(filter_num):
            a_with_pad = np.pad(
                prev_layer.activation,
                (0, 1),
                mode='constant',
                constant_values=0.
            )
            w_nabla.append(signal.correlate2d(a_with_pad, self.error[depth], mode='valid'))

        w_nabla = np.array(w_nabla)
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
        return [self.shape[0], self.input_shape[-2], self.input_shape[-1]]

    def param_num(self):
        return np.size(self.weight) + np.size(self.bias)


class MaxPooling(Layer):

    # 2d max pooling (2, 2) (border strategy is 'same')
    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size

    def build(self, input_shape):
        self.input_shape = input_shape

    def call(self, x):
        assert len(x.shape) == 3

        layers = []

        max_depth = x.shape[0]
        pool_shape = tuple(
            np.array(x.shape[1:]) // np.array(self.pool_size)
        )

        self.argmax = np.zeros(tuple(self.output_num()) + (2,), dtype=np.int32)

        for depth, layer in enumerate(x):
            patches = []
            for i in range(pool_shape[0]):
                for j in range(pool_shape[1]):
                    y = self.pool_size[0] * i
                    x = self.pool_size[1] * j
                    patch = layer[y:y+self.pool_size[0], x:x+self.pool_size[1]]
                    patches.append(patch)

                    argmax_i = np.argmax(patch)
                    self.argmax[depth][i][j] = np.array([y + (argmax_i // self.pool_size[1]), x + (argmax_i % self.pool_size[0])])

            pool_layer = np.array([np.max(patch) for patch in patches]).reshape(pool_shape)
            layers.append(pool_layer)

        return np.array(layers)

    def backprop(self, prev_layer):
        prev_error = np.zeros(self.input_shape)
        for depth in range(self.error.shape[0]):
            for i in range(self.error.shape[1]):
                for j in range(self.error.shape[2]):
                    (y, x) = self.argmax[depth][i][j]
                    prev_error[depth][y][x] = self.error[depth][i][j]

        return prev_error

    def output_num(self):
        return [
            self.input_shape[0],
            self.input_shape[1] // self.pool_size[0],
            self.input_shape[2] // self.pool_size[1],
        ]

    def param_num(self):
        return 0


class Flatten(Layer):
    def build(self, input_shape):
        self.input_shape = input_shape

    def call(self, x):
        self.activation = self.weighted_input = x.reshape(np.product(self.input_shape))

        return self.activation

    def backprop(self, prev_layer):
        return self.error.reshape(tuple(self.input_shape))

    def activation_prime(self):
        return self.weighted_input

    def output_num(self):
        return np.product(self.input_shape)

    def param_num(self):
        return 0
