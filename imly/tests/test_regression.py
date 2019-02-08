from . import automation_script
from os import path
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_diabetes():

    dataset_info = automation_script.get_dataset_info("diabetes")
    url = "../data/diabetes.csv"
    data = pd.read_csv(url, delimiter=",", header=None, index_col=False)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    data = pd.DataFrame(data)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # X = preprocessing.scale(X)
    # Y = preprocessing.normalize(Y)

    exp_results = automation_script.run_imly(dataset_info, 'linear_regression', X, Y, 0.60, return_exp_results=True)
    return exp_results


def load_iris():
    dataset_name = "uci_iris"
    dataset_info = automation_script.get_dataset_info(dataset_name)

    url = "../data/iris.csv"
    data = pd.read_csv(url, delimiter=",", header=None, index_col=False)
    class_name, index = np.unique(data.iloc[:,-1],return_inverse=True)
    data.iloc[:, -1] = index
    data = data.loc[data[4] != 2]
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    exp_results = automation_script.run_imly(dataset_info, 'logistic_regression', X, Y, 0.60, return_exp_results=True)
    return exp_results


def test_regression():
    iris_results = load_iris()
    assert 0 <= iris_results['keras'] <= 1
    assert 0 <= iris_results['scikit'] <= 1
    assert iris_results['correlation'] >= 0.5


def test_classification():
    diabetes_results = load_diabetes()
    assert 0 <= diabetes_results['keras'] <= 1
    assert 0 <= diabetes_results['scikit'] <= 1
    assert diabetes_results['correlation'] >= 0.5
