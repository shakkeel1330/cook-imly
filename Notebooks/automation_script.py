import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression

def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
    '../data/client_secret.json', scope)

    gc = gspread.authorize(credentials)

    sh = gc.open('Dataset details')

    worksheet = sh.get_worksheet(0)

    return worksheet


def get_url(dataset_name):
    worksheet = load_sheet()

    dataset_list = worksheet.col_values(worksheet.find("Name").col)
    url_list = worksheet.col_values(worksheet.find("Link").col)
    params_list = worksheet.row_values(worksheet.find("use_bias").row)
    params_list = params_list[params_list.index('use_bias'):]
    params_dict = {x:i+7 for i,x in enumerate(params_list)}
    activation_col_nb = worksheet.find("Algorithm").col
    activation_list = worksheet.col_values(activation_col_nb)

    for dataset in dataset_list:
        if dataset_name == dataset:
            row_nb = dataset_list.index(dataset)
            data_url = url_list[row_nb]

    n_col_nb = worksheet.find("N").col
    p_col_nb = worksheet.find("P").col
    c_col_nb = worksheet.find("C").col
    activation_function = activation_list[row_nb]
    data_url = url_list[row_nb]

    dataset_info = {
        "url": data_url,
        'params_dict': params_dict,
        'activation_function': activation_function,
        'n_col': n_col_nb,
        'p_col': p_col_nb,
        'c_col': c_col_nb,
        'row_nb': row_nb
    }
    return dataset_info


def get_keras_params(X,Y,data_info,config):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    np.random.seed(7)

    # Defining model #
    if data_info['activation_function'] == "Logistic regression":
        activation = "sigmoid"
    input_dim = X.shape[1]
    epoch = config['epoch']
    batch_size = config['batch_size']


    model = Sequential()
    model.add(Dense(1,input_dim=input_dim,activation=activation))

    # Compile the model #

    model.compile(loss=config['model_info']['loss'], optimizer=config['model_info']['optimizer'],
     metrics=config['model_info']['metrics'])

    # Fit the model #

    model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size)

    # Evaluate the model #

    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Prepare Keras configuration #
    keras_params = model.get_config()
    keras_params = keras_params['layers'][0]['config']
    keras_params['kernel_initializer'] = keras_params['kernel_initializer']['class_name']
    keras_params['bias_initializer'] = keras_params['bias_initializer']['class_name']

    return scores[1]*100, keras_params


def get_scikit_params(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    # Import and create an instance of your model(Logistic regression)

    logisticRegr = LogisticRegression()

    # Train your model using the training dataset

    logisticRegr.fit(x_train,y_train)

    # Predict the output 

    predictions = logisticRegr.predict(x_test)

    score = logisticRegr.score(x_test,y_test)
    print(score)
    scikit_params = logisticRegr.get_params(deep=True)
    return score, scikit_params


def write_to_mastersheet(data,X,Y):

    worksheet = load_sheet()
    params_dict = data['params_dict']
    scikit_params = data['scikit_params']
    keras_params = data['keras_params']
    row_nb = data['row_nb']
    n_col_nb = data['n_col_nb']
    p_col_nb = data['p_col_nb']
    c_col_nb = data['c_col_nb']
    n = X.shape[0]
    p = X.shape[1]
    unique,count = np.unique(Y,return_counts=True)
    class1=count[0]/X.shape[0]*100
    class2=count[1]/X.shape[0]*100
    class_distribution = str(round(class1)) + " : " + str(round(class2))

    for param,col_nb in params_dict.items():
        for s_param,value in scikit_params.items():
            if param == s_param:
                if value == None:
                    value = 'None'
                worksheet.update_cell(row_nb+1, col_nb+1, value)
            

    for param,col_nb in params_dict.items():
        for k_param,value in keras_params.items():
            if param == k_param:
                if value == None:
                    value = 'None'
                worksheet.update_cell(row_nb+1, col_nb+1, value)

    worksheet.update_cell(row_nb+1, n_col_nb, n)
    worksheet.update_cell(row_nb+1, p_col_nb, p)
    worksheet.update_cell(row_nb+1, c_col_nb, class_distribution)