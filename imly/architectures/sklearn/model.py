from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np
import json


class create_model:
    def __init__(self, fn_name, param_name, **kwargs):
        self.fn_name = fn_name
        self.param_name = param_name

    def __call__(self, **kwargs):
        try:
            module = __import__('architectures.sklearn.model',
                                fromlist=[self.fn_name])
            function = getattr(module, self.fn_name)
        except KeyError:
            print('Invalid model name passed to mapping_data')

        try:
            model = function(param_name=self.param_name, x_train=kwargs['x_train'])
            print('From try -- ', function)
        except KeyError:
            model = function(param_name=self.param_name)
            print('From except -- ', function)

        return model


def glm(**kwargs):  # Should param_name be optional or mandatory?

    # kwargs.setdefault('param_name', 'glm_1')

    params_json = json.load(open('../imly/architectures/sklearn/params.json')) # Remove and make it generic
    params = params_json['params'][kwargs['param_name']]

    kwargs.setdefault('params', params)
    kwargs.setdefault('x_train', 10) # FIX!

    model = Sequential()
    model.add(Dense(kwargs['params']['first_neuron'],  # Change first_neuron to input_size
                    input_dim=10,  # Find a better way to pass input_dim. Through params maybe?
                    activation=kwargs['params']['activation']))

    model.compile(optimizer=kwargs['params']['optimizer'],
                loss=kwargs['params']['losses'],
                metrics=["accuracy"])  # Dealing with accuracy in regression models
    
    print(model)

    return model
