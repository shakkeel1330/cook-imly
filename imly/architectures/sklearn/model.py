from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np


def f1(**kwargs):
    p = {
        'first_neuron': 1,
        'activation': 'linear',
        'optimizer': 'adam',
        'losses': 'mse'
    }
    kwargs.setdefault('params', p)
    kwargs.setdefault('x_train', np.array([[1], [2]]))

    model = Sequential()
    model.add(Dense(kwargs['params']['first_neuron'],
                    input_dim=kwargs['x_train'].shape[1],
                    activation=kwargs['params']['activation']))

    model.compile(optimizer=kwargs['params']['optimizer'],
                  loss=kwargs['params']['losses'],
                  metrics=['acc'])
    return model
