from keras.wrappers.scikit_learn import KerasRegressor
from optimizers.talos.talos import get_best_model
import pickle, onnxmltools


class SklearnKerasRegressor(KerasRegressor):
        def __init__(self, build_fn, **kwargs):
            super(KerasRegressor, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']

        def fit(self, x_train, y_train, **kwargs):
            primal_model = self.primal
            primal_model.fit(x_train, y_train)
            y_pred = primal_model.predict(x_train)
            kwargs['y_pred'] = y_pred
            kwargs['model_name'] = self.__class__.__name__

            self.model = get_best_model(x_train, y_train, kwargs) # get_best_model - rename
            self.model.fit(x_train, y_train)
            return self.model

        def save(self, using='dnn'):
            if using == 'sklearn':
                filename = 'scikit_model'
                pickle.dump(self.model, open(filename, 'wb'))
            else:
                onnx_model = onnxmltools.convert_keras(self.model)
                return onnx_model
