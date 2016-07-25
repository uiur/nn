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
