from sklearn.linear_model import LinearRegression
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Nadam
from keras.losses import mse
from keras.activations import linear
import talos as ta


class dope(LinearRegression):
    def __init__(self, model):
        self.model = model
        # if model == LinearRegression:
        #     k_model = build_keras_model(name='linear_regression')

    def keras_model(self, x_train, y_train, x_val, y_val, params):
        model = Sequential()
        model.add(Dense(params['first_neuron'],
                        input_dim=x_train.shape[1],
                        activation=params['activation']))

        model.compile(optimizer=params['optimizer'],
                      loss=params['losses'],
                      metrics=['mse'])

        out = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        validation_data=[x_val, y_val])

        return out, model

    def fit(self, x_train, y_train, using='dnn'):
        if using == 'sklearn':
            self.model.fit(x_train, y_train)
            return self
        else:
            self.fit_keras_model(x_train, y_train)

    def predict(self, x_test, using='dnn'):
        if using == 'sklearn':
            return self.model.predict(x_test)

    def score(self, x_test, y_test, using='dnn'):
        if using == 'sklearn':
            return self.model.score(x_test, y_test)

    def save(self, using='dnn'):
        if using == 'sklearn':
            filename = 'scikit_model'
            pickle.dump(self.model, open(filename, 'wb'))

        else: 
            self.save_as_onnx()

    def fit_keras_model(self, x_train, y_train):

        # Hard coding params for the time being. It should ideally be passed
        # as an argument

        p = {'lr': (2, 10, 30),
             'first_neuron': [1],
             'batch_size': [1, 2, 3, 4],
             'epochs': [10, 20, 40],
             'weight_regulizer': [None],
             'emb_output_dims': [None],
             'optimizer': ['SGD', 'nadam'],
             'losses': [mse],
             'activation': [linear]
             }

        h = ta.Scan(x_train, y_train,
                    params=p,
                    dataset_name='first_linear_regression',
                    experiment_no='a',
                    model=self.keras_model,
                    grid_downsample=0.5)

    def save_as_onnx():
        import os,json,zipfile,shutil
        from keras.models import model_from_json
        archive = zipfile.ZipFile('linear_regression_firstDataset.zip', 'r')
        model_file = archive.open('linear_regression_firstDataset_model.json')
        weight_file = archive.open('linear_regression_firstDataset_model.h5')

        with zipfile.ZipFile('linear_regression_firstDataset.zip', 'r') as zip_ref:
            zip_ref.extractall('./linear_regression_firstDataset_unzip')

        # json_file = open('model.json', 'r')
        loaded_model_json = model_file.read()

        # json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./linear_regression_firstDataset/linear_regression_firstDataset_model.h5")


        shutil.rmtree('./linear_regression_firstDataset_unzip')
        print("Loaded model from disk")

        import onnxmltools
        onnx_model = onnxmltools.convert_keras(loaded_model)
