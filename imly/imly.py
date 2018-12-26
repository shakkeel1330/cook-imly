import pickle
import copy
import onnxmltools
import types
from utils.model_mapping import get_model_design
from wrappers.talos.talos import talos_optimization
from keras.wrappers.scikit_learn import KerasRegressor


def dope(obj, **kwargs):
    kwargs.setdefault('using', 'dnn')
    if kwargs['using'] == 'dnn':
        primal_model = copy.deepcopy(obj)
        name = obj.__class__.__name__
        create_model = get_model_design(name)
        model = KerasRegressor(build_fn=create_model, #Create a Keras wrapper and move this out
                               epochs=10, batch_size=10, verbose=0)
        obj = model
        obj.primal = primal_model
        obj.fit = types.MethodType(fit_best, obj) # Move to arch/package-name
        obj.save = types.MethodType(save, obj)
    return obj


def fit_best(self, x_train, y_train, **kwargs):
    kwargs['params']['model_name'] = [self.__class__.__name__]

    primal_model = self.primal
    primal_model.fit(x_train, y_train)
    y_pred = primal_model.predict(x_train)
    kwargs['params']['y_pred'] = y_pred

    self.model = talos_optimization(x_train, y_train, kwargs)
    return self.model.fit(x_train, y_train)


def save(self, using='dnn'):
    if using == 'sklearn':
        filename = 'scikit_model'
        pickle.dump(self.model, open(filename, 'wb'))
    else:
        onnx_model = onnxmltools.convert_keras(self.model)
        return onnx_model


''' Notes -

Things to move out of imly
1) Keras Regressor and it's extension methods
2) get_model_design() to utils
3) f1() and p values
4) talos methods - fit_keras() and talos_model()
    + Params JSON - Rectify the values(None,tuples etc)
5) save() goes with keras regressor

'''

