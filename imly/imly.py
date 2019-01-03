from utils.model_mapping import get_model_design
from wrappers.sklearn.sklearn_keras_regressor.keras_regressor import SklearnKerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
import copy, json


def dope(model, **kwargs):
    name = model.__class__.__name__
    kwargs.setdefault('using', 'dnn')
    params_json = json.load(open('../imly/optimizers/talos/params.json'))
    params = params_json['params'][name]['config']
    kwargs.setdefault('params', params)
    search_params = {**params, **kwargs['params']}

    kwargs.setdefault('performance_metric', params_json['params'][name]['performance_metric'])

    if kwargs['using'] == 'dnn':
        primal = copy.deepcopy(model)
        create_model = get_model_design(name)
        model = SklearnKerasRegressor(build_fn=create_model,
                                        epochs=10, batch_size=10,
                                        verbose=0, primal=primal,
                                        params=search_params,
                                        performance_metric=kwargs['performance_metric'])

    return model


# kwargs changes
# multi-layer solution

''' Changes
[X] kwargs - Access to params for the user
[X] Rename talos_optimization()
[X] Accessing model_name
[X] Get rid of best_model
[X] Move Talos to imly/optimizer
[] try/catch
[] Generalization of model structure using ONNX
'''