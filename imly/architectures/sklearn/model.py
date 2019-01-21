from keras.models import Sequential
from keras.layers.core import Dense
import json


class create_model:
    def __init__(self, fn_name, param_name, **kwargs):
        self.fn_name = fn_name
        self.param_name = param_name
        self.x_train = None

    def __call__(self, **kwargs):
        try:
            module = __import__('architectures.sklearn.model',
                                fromlist=[self.fn_name])
            function = getattr(module, self.fn_name)
        except KeyError:
            print('Invalid model name passed to mapping_data')

        try:
            model = function(param_name=self.param_name, 
                             x_train=kwargs['x_train'])
        except KeyError:
            model = function(param_name=self.param_name, x_train=self.x_train) 
            # TODO
            # Error handling missing. What happens if the user
            # sends w/o setting x_train of the object?
        return model


def glm(**kwargs):  # Should param_name be optional or mandatory?

    params_json = json.load(open('../imly/architectures/sklearn/params.json'))
    params = params_json['params'][kwargs['param_name']]

    kwargs.setdefault('params', params)

    model = Sequential()
    model.add(Dense(kwargs['params']['units'],
                    input_dim=kwargs['x_train'].shape[1],
                    activation=kwargs['params']['activation']))

    model.compile(optimizer=kwargs['params']['optimizer'],
                  loss=kwargs['params']['losses'])

    return model
