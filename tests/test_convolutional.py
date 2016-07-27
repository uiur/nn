from convolutional import *
from network import *
import numpy as np


def test_Convolution():
    layer = Convolution([3, 2, 2])
    layer.build([8, 8])

    out = layer.call(np.arange(8 * 8).reshape(8, 8))
    assert out.shape == (3, 8, 8)


def test_MaxPooling():
    layer = MaxPooling((2, 2))
    input_shape = [3, 4, 4]
    layer.build(input_shape)

    a = np.zeros((3, 4, 4))
    a[0][1][0] = 4

    result = layer.call(a)

    assert result.shape == (3, 2, 2)
    assert result[0][0][0] == 4

    layer.call(np.arange(3 * 4 * 4).reshape(3, 4, 4))
    layer.error = np.ones((3, 2, 2))
    error = layer.backprop(Convolution(input_shape))

    for depth, _ in enumerate(error):
        assert np.all(error[depth] == np.array([[0., 0., 0., 0.],
                                                [0., 1., 0., 1.],
                                                [0., 0., 0., 0.],
                                                [0., 1., 0., 1.]]))


def test_Flatten():
    layer = Flatten()
    layer.build([3, 2, 2])

    seq = np.arange(3 * 2 * 2)
    out = layer.call(seq.reshape(3, 2, 2))
    assert np.all(out == seq)

    layer.error = seq[:]
    error = layer.backprop(MaxPooling())
    assert np.all(error == seq[:].reshape(3, 2, 2))


def test_Integration():
    net = Network([
        Input([4, 4]),
        Convolution([3, 2, 2]),
        MaxPooling((2, 2)),
        Flatten(),
    ])

    out = net.feedforward(np.arange(4 * 4).reshape(4, 4))
    assert out[-1].shape[0] == 3 * 2 * 2

    net = Network([
        Input([4, 4]),
        Convolution([3, 2, 2]),
        MaxPooling((2, 2)),
        Flatten(),
        Dense(1),
    ])

    net.backprop([np.arange(4 * 4).reshape(4, 4)], [1.])
