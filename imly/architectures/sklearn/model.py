from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
import json


def glm(**kwargs):  # Should param_name be optional or mandatory?

    # kwargs.setdefault('param_name', 'glm_1')
    params_json = json.load(open('../imly/architectures/sklearn/params.json')) # Remove and make it generic
    params = params_json['params'][kwargs['param_name']]
    kwargs.setdefault('params', params)
    kwargs.setdefault('x_train', np.array([[1], [2]]))

    model = Sequential()
    model.add(Dense(kwargs['params']['first_neuron'], # Change first_neuron to input_size
                    input_dim=kwargs['x_train'].shape[1], # Find a better way to pass input_dim. Through params maybe?
                    activation=kwargs['params']['activation']))

    model.compile(optimizer=kwargs['params']['optimizer'],
                  loss=kwargs['params']['losses'],
                  metrics=['acc'])
    return model

# try catch