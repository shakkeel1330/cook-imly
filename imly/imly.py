from utils.model_mapping import get_model_design
from wrappers.sklearn.keras_regressor import SklearnKerasRegressor
from wrappers.sklearn.keras_classifier import SklearnKerasClassifier
from architectures.sklearn.model import create_model

from keras.models import Sequential
from keras.layers.core import Dense

import copy, json


def dope(model, **kwargs):
    model_name = model.__class__.__name__
    kwargs.setdefault('using', 'dnn')
    params_json = json.load(open('../imly/optimizers/talos/params.json'))
    params = params_json['params'][model_name]['config']
    kwargs.setdefault('params', params)
    search_params = {**params, **kwargs['params']}

    kwargs.setdefault('performance_metric', params_json['params'][model_name]['performance_metric'])

    if kwargs['using'] == 'dnn':
        primal = copy.deepcopy(model)

        fn_name, param_name = get_model_design(model_name)  # Should retrun fn_name and not the actual function

        build_fn = create_model(fn_name, param_name)

        # create_model = model_function(param_name=param_name)

        if model.__class__.__name__ == "LogisticRegression":
            model = SklearnKerasClassifier(build_fn=build_fn,
                                        epochs=10, batch_size=10,
                                        verbose=0, primal=primal,
                                        params=search_params,
                                        performance_metric=kwargs['performance_metric'])
        else:
            model = SklearnKerasRegressor(build_fn=create_model,
                                            epochs=10, batch_size=10,
                                            verbose=0, primal=primal,
                                            params=search_params,
                                            performance_metric=kwargs['performance_metric'])

    return model


# Idea of supporting multiple of backends for ONNX
# Restructure the backend aspect
# Add more class(mapping algorithms)
# Importing Keras from a middle package
# Multiple cuts(params, model arch and hyperparams)
