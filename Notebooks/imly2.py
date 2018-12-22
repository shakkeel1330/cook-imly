''' Notes - 
1) Try plainly returning sklearn - DONE
2) Try returning plain keras - DONE
    + What is the purpose of kerasWrapper
        - Direct mapping/replication of scikit methods for a given Keras model
    + Why am I not able to directly alter the 'fit' method 
    in scikit? - FIXED
    + 1. Check the source code for keras wrapper - DONE
      2. Understand why you're unable to implement the keras fit manually/directly - DONE
      3. Move to step 3)

3) Merge and send these two model structures - DONE
    + Check the 'score' method in both tools
    + Possible ways to implement Talos
        + While defining the 'model' object. Right after keras wrapper
        + Passing it directly to a custom 'keras_fit' method. So, the Talos part
        would be triggered while the user calls m.fit()
4) Generalization
    + Will add details once I cover the first 3.
5) Testing performance with datasets

Qs
+ Are we still planning to use the pred by scikit as the target for Keras? - Yes.
'''

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Nadam
from keras.losses import mse
from keras.activations import linear
from keras.wrappers.scikit_learn import KerasRegressor
import talos as ta

import os
import json
import zipfile
import shutil
from keras.models import model_from_json
import onnxmltools
from talos import Deploy
import types


def dope(obj, **kwargs):
    ''' Pending -
    1) Argument generalization
    2) Extract model name before creating model -- DONE
    3) Getter and setter for params
    4) Duplicate file name issue in Deploy() -- FIXED
    5) compiling right after extractling Talos -- FIXED
        + How to extract the best model's opt and loss?
        + Try load_model by Predict()
    6) predict by scikit
    '''
    p = {   # use get_params
        'first_neuron': [1],
        'optimizer': ['SGD', 'nadam'],
        'losses': [mse],
        'activation': [linear]
    }
    kwargs.setdefault('using', 'dnn')
    kwargs.setdefault('params', p)
    if kwargs['using'] == 'dnn':
        p = kwargs['params']  # use set_params()
        name = obj.__class__.__name__
        create_model = get_model_design(name)
        model = KerasRegressor(build_fn=create_model,
                               epochs=10, batch_size=10, verbose=0)
        obj = model
        obj.fit = types.MethodType(fit_keras, obj)
        obj.save = types.MethodType(save, obj)
    return obj


def get_model_design(name):
    mapping = {
        'LinearRegression': f1,
        'KerasRegressor': f1
    }

    for key, value in mapping.items():
        if key == name:
            function_name = value

    return function_name


def f1(**kwargs):
    p = {
        'first_neuron': 1,
        'activation': 'linear',
        'optimizer': 'adam',
        'losses': 'mse'
    }
    kwargs.setdefault('params', p)
    kwargs.setdefault('x_train', np.array([[1], [2]]))

    model = Sequential()
    model.add(Dense(kwargs['params']['first_neuron'],
                    input_dim=kwargs['x_train'].shape[1],
                    activation=kwargs['params']['activation']))

    model.compile(optimizer=kwargs['params']['optimizer'],
                  loss=kwargs['params']['losses'],
                  metrics=['acc'])
    return model


def talos_model(x_train, y_train, x_val, y_val, params):
    build_model = get_model_design(params['model_name'])
    model = build_model(x_train=x_train, params=params)
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model


def fit_keras(self, x_train, y_train, **kwargs):

    # Hard coding params for the time being. It should ideally be passed
    # as an argument

    p = {'lr': (2, 10, 30),
         'first_neuron': [1],
         'batch_size': [10],
         'epochs': [10],
         'weight_regulizer': [None],
         'emb_output_dims': [None],
         'optimizer': ['nadam'],
         'losses': [mse],
         'activation': [linear]
         }
    kwargs.setdefault('params', p)
    kwargs['params']['model_name'] = [self.__class__.__name__]

    h = ta.Scan(x_train, y_train,
                params=kwargs['params'],
                dataset_name='first_linear_regression',
                experiment_no='a',
                model=talos_model,
                grid_downsample=0.5)

    best_model = extract_model(h)
    best_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    self.model = best_model
    # To delete the zip file that was created by Talos
    os.remove('./linear_regression_firstDataset.zip')
    return self.model.fit(x_train, y_train)

    # Avoid using weights
    # Re write the whole object as a new keras regressor object while the user calls m.fit()
    # KerasRegressor's build_fn required a function or an instance of a class as an argument. Hence the above
    # approach wasn't working(what we have is a keras model - the best_model object)


def extract_model(scan_object):
    Deploy(scan_object, 'linear_regression_firstDataset', metric='val_loss')
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
    loaded_model.load_weights(
        "./linear_regression_firstDataset_unzip/linear_regression_firstDataset_model.h5")

    shutil.rmtree('./linear_regression_firstDataset_unzip')
    print("Loaded model from disk")
    return loaded_model


def save(self, using='dnn'):
    if using == 'sklearn':
        filename = 'scikit_model'
        pickle.dump(self.model, open(filename, 'wb'))
    else:
        onnx_model = onnxmltools.convert_keras(self.model)
        return onnx_model
