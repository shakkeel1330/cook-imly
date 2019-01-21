from keras.wrappers.scikit_learn import KerasRegressor
from optimizers.talos.talos import get_best_model
import pickle, onnxmltools


class SklearnKerasRegressor(KerasRegressor):
        def __init__(self, build_fn, **kwargs):
            super(KerasRegressor, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.val_metric = kwargs['val_metric']
            self.metric = kwargs['metric']

        def fit(self, x_train, y_train, **kwargs):
            primal_model = self.primal
            primal_model.fit(x_train, y_train)
            y_pred = primal_model.predict(x_train)
            primal_data = {
                'y_pred': y_pred,
                'model_name': primal_model.__class__.__name__
            }

            self.model, final_epoch, final_batch_size = get_best_model(x_train, y_train, 
                                                                       primal_data=primal_data,
                                                                       params=self.params, 
                                                                       val_metric=self.val_metric,
                                                                       metric=self.metric)
            # Epochs and batch_size passed in Talos as well
            self.model.fit(x_train, y_train, epochs=final_epoch,
                           batch_size=final_batch_size, verbose=0)
            return self.model

        def score(self, x, y, **kwargs):
            score = super(SklearnKerasRegressor, self).score(x, y, **kwargs)
            # keras_regressor treats all score values as loss and adds a '-ve' before passing
            return -score

        def save(self, using='dnn'):
            if using == 'sklearn':
                filename = 'scikit_model'
                pickle.dump(self.model, open(filename, 'wb'))
            else:
                onnx_model = onnxmltools.convert_keras(self.model)
                return onnx_model
