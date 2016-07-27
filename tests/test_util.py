import util
import numpy as np

def test_test_split():
    x = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([[0], [1]])

    x_train, y_train, x_valid, y_valid = util.test_split(x, y, rate=0.5)
    assert len(x_train) == 1 and len(y_train) == 1
    assert y_train[0][0] != y_valid[0][0]
