from keras.wrappers.scikit_learn import KerasClassifier
from optimizers.talos.talos import get_best_model
import pickle, onnxmltools


class SklearnKerasClassifier(KerasClassifier):
        def __init__(self, build_fn, **kwargs):
            super(KerasClassifier, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.performance_metric = kwargs['performance_metric']

        def fit(self, x_train, y_train, **kwargs):
            print('Keras classifier chosen')
            primal_model = self.primal
            primal_model.fit(x_train, y_train)
            y_pred = primal_model.predict(x_train)
            primal_data = {
                'y_pred': y_pred,
                'model_name': primal_model.__class__.__name__
            }

            self.model = get_best_model(x_train, y_train, primal_data=primal_data, params=self.params, 
                                        performance_metric=self.performance_metric) 
            # self.model.fit(x_train, y_train)
            super(KerasClassifier, self).fit(x_train, y_train) # Why? - 'classes_' missing
            return self.model

        def save(self, using='dnn'):
            if using == 'sklearn':
                filename = 'scikit_model'
                pickle.dump(self.model, open(filename, 'wb'))
            else:
                onnx_model = onnxmltools.convert_keras(self.model)
                return onnx_model
