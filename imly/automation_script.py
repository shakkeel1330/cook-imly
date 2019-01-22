import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from imly import dope
import copy, json
from winmltools import convert_keras

model_mappings = {
    'linear_regression': 'LinearRegression',
    'logistic_regression': 'LogisticRegression'
}

classification_models = ['logistic_regression']


def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
    '../data/client_secret.json', scope)

    gc = gspread.authorize(credentials)

    sh = gc.open('Dataset details')

    worksheet = sh.get_worksheet(0)

    return worksheet


def get_dataset_info(dataset_name):
    worksheet = load_sheet()

    dataset_list = worksheet.col_values(worksheet.find("Name").col)
    url_list = worksheet.col_values(worksheet.find("Link").col)
    # params_list = worksheet.row_values(worksheet.find("use_bias").row)
    # params_list = params_list[params_list.index('use_bias'):]
    # params_dict = {x:i+7 for i,x in enumerate(params_list)}
    activation_col_nb = worksheet.find("Algorithm").col
    activation_list = worksheet.col_values(activation_col_nb)

    # TODO
    # Add error case. When no match is found.
    for dataset in dataset_list:
        if dataset_name == dataset:
            row_nb = dataset_list.index(dataset)
            data_url = url_list[row_nb]

    n_col_nb = worksheet.find("N").col
    p_col_nb = worksheet.find("P").col
    c_col_nb = worksheet.find("Class distribution").col
    activation_function = activation_list[row_nb]
    data_url = url_list[row_nb]

    dataset_info = {
        "url": data_url,
        # 'params_dict': params_dict,
        'activation_function': activation_function,
        'n_col': n_col_nb,
        'p_col': p_col_nb,
        'c_col': c_col_nb,
        'row_nb': row_nb
    }
    return dataset_info


def write_to_mastersheet(data, X, Y, accuracy_values):

    worksheet = load_sheet()
    # params_dict = data['params_dict']
    scikit_params = data['scikit_params']
    keras_params = data['keras_params']
    row_nb = data['row_nb']
    n_col_nb = data['n_col']
    p_col_nb = data['p_col']
    c_col_nb = data['c_col']
    n = X.shape[0]
    p = X.shape[1]
    if data['activation_function'] in classification_models:
        unique, count = np.unique(Y, return_counts=True)
        class1 = count[0]/X.shape[0]*100
        class_distribution = round(class1, 2)
    else:
        class_distribution = 'NA'

    col_nb = worksheet.find('scikit_json').col
    worksheet.update_cell(row_nb+1, col_nb, scikit_params)

    col_nb = worksheet.find('keras_json').col
    worksheet.update_cell(row_nb+1, col_nb, keras_params)

    worksheet.update_cell(row_nb+1, n_col_nb, n)
    worksheet.update_cell(row_nb+1, p_col_nb, p)
    worksheet.update_cell(row_nb+1, c_col_nb, class_distribution)
    worksheet.update_cell(row_nb+1, worksheet.find("Keras acc").col,
                          accuracy_values['keras'])
    worksheet.update_cell(row_nb+1, worksheet.find("Scikit acc").col,
                          accuracy_values['scikit'])
    worksheet.update_cell(row_nb+1, worksheet.find("Kfold").col,
                          accuracy_values['kfold'])
    worksheet.update_cell(row_nb+1, worksheet.find("Type").col, data['type'])


def run_imly(dataset_info, model_name, X, Y, test_size, **kwargs):
    # TODO 
    # Remove model_name from arguments. This data is available 
    # in dataset_info['activation_fn']
    kwargs.setdefault('params', {})
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

    for key, value in model_mappings.items():
        if key == model_name:
            name = value

    module = __import__('sklearn.linear_model', fromlist=[name])
    imported_module = getattr(module, name)
    model = imported_module
    model_instance = model()
    base_model = copy.deepcopy(model_instance)
    primal_model = copy.deepcopy(model_instance)

    # Primal
    primal_model.fit(x_train, y_train)
    y_pred = primal_model.predict(x_test)
    if (primal_model.__class__.__name__ == 'LogisticRegression'):
        primal_score = primal_model.score(x_test, y_test)
    else:
        # primal_score = primal_model.score(x_test, y_test)
        primal_score = mean_squared_error(y_test, y_pred)
    primal_params = primal_model.get_params(deep=True)

    # Keras 
    x_train = x_train.values  # Talos accepts only numpy arrays
    m = dope(base_model, params=kwargs['params'])
    m.fit(x_train, y_train)
    keras_score = m.score(x_test, y_test)

    # Prepare Keras configuration #
    keras_params = m.__dict__['model'].get_config()
    keras_params = keras_params['layers'][0]['config']
    keras_params['kernel_initializer'] = keras_params['kernel_initializer']['class_name']
    keras_params['bias_initializer'] = keras_params['bias_initializer']['class_name']

    dataset_info['scikit_params'] = json.dumps(primal_params)
    dataset_info['keras_params'] = json.dumps(keras_params)
    dataset_info['type'] = 'Binary'
    accuracy_values = {
        'keras': keras_score,
        'scikit': primal_score,
        'kfold': None
    }

    write_to_mastersheet(dataset_info, X, Y, accuracy_values)