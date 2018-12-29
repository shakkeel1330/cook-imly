from utils.model_mapping import get_model_design
from wrappers.sklearn.sklearn_keras_regressor.keras_regressor import SklearnKerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
import copy


def dope(model, **kwargs):
    kwargs.setdefault('using', 'dnn')
    if kwargs['using'] == 'dnn':
        primal = copy.deepcopy(model)
        name = model.__class__.__name__
        create_model = get_model_design(name)
        model = SklearnKerasRegressor(build_fn=create_model,
                               epochs=10, batch_size=10,
                               verbose=0, primal=primal)

    return model
