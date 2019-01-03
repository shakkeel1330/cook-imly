from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
import json

 
def f1(**kwargs):
    params_json = json.load(open('../imly/architectures/sklearn/params.json'))
    params = params_json['params']
    kwargs.setdefault('params', params)
    kwargs.setdefault('x_train', np.array([[1], [2]]))

    model = Sequential()
    model.add(Dense(kwargs['params']['first_neuron'],
                    input_dim=kwargs['x_train'].shape[1],
                    activation=kwargs['params']['activation']))

    model.compile(optimizer=kwargs['params']['optimizer'],
                  loss=kwargs['params']['losses'],
                  metrics=['acc'])
    return model

# try catch