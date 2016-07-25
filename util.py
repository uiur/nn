import numpy as np

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
