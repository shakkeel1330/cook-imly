import json
import numpy as np
import talos as ta
from ..talos.model import talos_model
from talos.utils.best_model import activate_model


def get_best_model(x_train, y_train, **kwargs):
    np.random.seed(7)
    y_pred = kwargs['primal_data']['y_pred']
    params = kwargs['params']
    
    params['model_name'] = [kwargs['primal_data']['model_name']]
    kwargs.setdefault('dataset_name', 'talos_readings')
    kwargs.setdefault('experiment_no', '1')
    dataset_name = kwargs['dataset_name']
    experiment_no = kwargs['experiment_no']
    performance_metric = kwargs['performance_metric']

    for name, value in params.items():
        if type(value) != list:
            params[name] = [value]

#     x_train = x_train.values
#     y_pred = y_pred.values  # Converting df to np.array
#     print("From talos.py --- ", params)
#     print("x_train_shape -- ", x_train.shape)
#     print("y_pred_shape -- ", y_pred.shape)

    h = ta.Scan(x_train, y_pred,
                params=params,
                dataset_name=dataset_name,
                experiment_no=experiment_no,
                model=talos_model,
                grid_downsample=0.5)

    report = h.data
    best_model = report.sort_values(performance_metric, ascending=True).iloc[0]
    best_model_id = best_model.name - 1
    dnn_model = activate_model(h, best_model_id)
    loss = best_model.losses
    optimizer = best_model.optimizer
    dnn_model.compile(optimizer=optimizer, loss=loss, metrics=[loss])
    return dnn_model
