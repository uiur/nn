import mlp
import numpy as np


def test_sigmoid():
    value = mlp.sigmoid(np.arange(-10, 10, 0.1))
    assert ((0 < value) + (value < 1.0)).all()


def test_sigmoid_prime():
    assert (
        mlp.sigmoid_prime(-10.0) < mlp.sigmoid_prime(0) and
        mlp.sigmoid_prime(0) > mlp.sigmoid_prime(10.0)
    )


def test_mlp():
    # classify y > x
    network = mlp.MLP([2, 1])
    network.weights = [None, np.array([[-10.0, 10.0]])]
    network.biases = [None, np.array([0.])]

    X = 2 * np.random.randn(10, network.layers[0])
    y = np.array([x[1] > x[0] for x in X]) * 1.0 # y > x

    assert ((network.output(X) > 0.5).flatten() == (y == 1.0)).all()

def test_test_split():
    x = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([[0], [1]])

    x_train, y_train, x_valid, y_valid = mlp.test_split(x, y, rate=0.5)
    assert len(x_train) == 1 and len(y_train) == 1
    assert y_train[0][0] != y_valid[0][0]
