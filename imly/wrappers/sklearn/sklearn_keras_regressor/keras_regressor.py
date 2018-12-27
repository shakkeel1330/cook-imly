from keras.wrappers.scikit_learn import KerasRegressor
from wrappers.talos.talos import talos_optimization
import pickle, onnxmltools


class SklearnKerasRegressor(KerasRegressor.BaseWrapper):
        def __init__(self, obj):
            # KerasRegressor.__init__(self, build_fn=None)
            # BaseWrapper.__init__(self, build_fn=None)
            super().__init__()
            self.primal = obj

        def fit(self, x_train, y_train, **kwargs):
        # kwargs.setdefault('model_name', self.__class__.__name__)

            primal_model = self.primal
            primal_model.fit(x_train, y_train)
            y_pred = primal_model.predict(x_train)
            kwargs['y_pred'] = y_pred
            kwargs['model_name'] = self.__class__.__name__

            self.model = talos_optimization(x_train, y_train, kwargs)
            return self.model.fit(x_train, y_train)

        def save(self, using='dnn'):
            if using == 'sklearn':
                filename = 'scikit_model'
                pickle.dump(self.model, open(filename, 'wb'))
            else:
                onnx_model = onnxmltools.convert_keras(self.model)
                return onnx_model
