import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def load():
    tr_d, va_d, te_d = load_data()
    training_inputs = np.array(tr_d[0])
    training_results = vectorize_array(tr_d[1])
    training_data = (training_inputs, training_results)

    validation_inputs = np.array(va_d[0])
    validation_results = vectorize_array(va_d[1])
    validation_data = (validation_inputs, validation_results)

    test_inputs = np.array(te_d[0])
    test_data = (test_inputs, vectorize_array(te_d[1]))
    return (training_data, validation_data, test_data)


def vectorize_array(d):
    return np.array([vectorize(y) for y in d])


def vectorize(i):
    v = np.zeros(10)
    v[i] = 1.0
    return v
