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
+ Are we still planning to use the pred by scikit as the target for Keras?
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


def dope(obj, using='dnn'):
    if using == 'sklearn':
        return obj
    elif using == 'dnn':
        model = KerasRegressor(build_fn=create_model,
                               epochs=10, batch_size=10, verbose=0)
        # fit_keras(x_train,y_train,model)  
        obj.fit = fit_keras
        # obj.fit = model.fit
        obj.predict = model.predict
        obj.score = model.score
        return obj
    return obj


def create_model():
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# def select_fit(obj,using='dnn'):
#     if using == 'sklearn':
#         return 

#     elif using == 'dnn':
#           model = KerasRegressor(build_fn=create_model, epochs=10, batch_size=10, verbose=0)
#           obj.fit = model.fit
#           return obj


# def keras_model(self, x_train, y_train, x_val, y_val, params):
#     model = Sequential()
#     model.add(Dense(params['first_neuron'],
#                     input_dim=x_train.shape[1],
#                     activation=params['activation']))

#     model.compile(optimizer=params['optimizer'],
#                     loss=params['losses'],
#                     metrics=['mse'])

#     out = model.fit(x_train, y_train,
#                     batch_size=params['batch_size'],
#                     epochs=params['epochs'],
#                     verbose=0,
#                     validation_data=[x_val, y_val])

#     return out, model

# def fit(self, x_train, y_train, using='dnn'):
#     if using == 'sklearn':
#         self.model.fit(x_train, y_train)
#         return self
#     else:
#         self.fit_keras_model(x_train, y_train)

# def predict(self, x_test, using='dnn'):
#     if using == 'sklearn':
#         return self.model.predict(x_test)

# def score(self, x_test, y_test, using='dnn'):
#     if using == 'sklearn':
#         return self.model.score(x_test, y_test)

# def save(self, using='dnn'):
#     if using == 'sklearn':
#         filename = 'scikit_model'
#         pickle.dump(self.model, open(filename, 'wb'))

#     else:
#         self.save_as_onnx()

def fit_keras(self,x_train, y_train):

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
                model=self.model,
                grid_downsample=0.5)

# def save_as_onnx():
    # import os
    # import json
    # import zipfile
    # import shutil
    # from keras.models import model_from_json
    # archive = zipfile.ZipFile('linear_regression_firstDataset.zip', 'r')
    # model_file = archive.open('linear_regression_firstDataset_model.json')
    # weight_file = archive.open('linear_regression_firstDataset_model.h5')

    # with zipfile.ZipFile('linear_regression_firstDataset.zip', 'r') as zip_ref:
    #     zip_ref.extractall('./linear_regression_firstDataset_unzip')

    # # json_file = open('model.json', 'r')
    # loaded_model_json = model_file.read()

    # # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(
    #     "./linear_regression_firstDataset/linear_regression_firstDataset_model.h5")

    # shutil.rmtree('./linear_regression_firstDataset_unzip')
    # print("Loaded model from disk")

    # import onnxmltools
    # onnx_model = onnxmltools.convert_keras(loaded_model)


# Test section #


diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
sc = StandardScaler()
diabetes.data = sc.fit_transform(diabetes.data)

x = diabetes_X
y = diabetes.target


# Split the data into training/testing sets
x_train = diabetes_X[:-20]
x_test = diabetes_X[-20:]

# Split the targets into training/testing sets
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

model = LinearRegression()

m = dope(model)

m.fit(x_train=x_train,y_train=y_train)

# y_pred = m.predict(x_test)
# score = m.score(x_test,y_test)

# print(score, y_pred)
