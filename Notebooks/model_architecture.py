from keras import Sequential
from keras.layers import Dense
import numpy as np
import marshal
import json


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


model_details = {
    'LinearRegression': marshal.dumps(f1)
}

model_details = json.dumps(model_details)