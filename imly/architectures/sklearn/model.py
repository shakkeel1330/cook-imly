from keras.models import Sequential
from keras.layers.core import Dense
# from utils.losses import lda_loss
from keras.regularizers import l2
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
                             x_train=kwargs['x_train'],
                             params=kwargs['params'])
        except KeyError:
            # TODO
            # 1) Error handling missing. What happens if the user
            # sends w/o setting x_train of the object?
            # 2) Looks like this use case is not meaningful anymore.
            # Cross check and remove.
            model = function(param_name=self.param_name, x_train=self.x_train)
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
                  loss=kwargs['params']['losses'],
                  metrics=['accuracy'])

    return model


def lda(**kwargs):

    params_json = json.load(open('../imly/architectures/sklearn/params.json'))
    params = params_json['params'][kwargs['param_name']]

    kwargs.setdefault('params', params)

    model = Sequential()
    model.add(Dense(kwargs['params']['units'],
                    input_dim=kwargs['x_train'].shape[1],
                    activation=kwargs['params']['activation'][0],
                    kernel_regularizer=l2(1e-5)))
    model.compile(optimizer=kwargs['params']['optimizer'],
                #   loss=lda_loss(n_components=1, margin=1),
                  loss='mse',
                  metrics=['accuracy'])
    # Metrics is usually provided through Talos.
    # Since we are bypassing Talos for LDA, we add the metrics directly.

    return model


# TODO
# reg_par missing for kernal_regularizer
