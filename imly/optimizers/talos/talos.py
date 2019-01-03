import json
import numpy as np
import talos as ta
from ..talos.model import talos_model
from talos.utils.best_model import activate_model


def get_best_model(x_train, y_train, kwargs):
    np.random.seed(7)
    params_json = json.load(open('../imly/optimizers/talos/params.json'))
    params = params_json['params'][kwargs['model_name']]['config']
    y_pred = kwargs['y_pred']
    kwargs.setdefault('params', params)
    search_params = {**params, **kwargs['params']}
    
    search_params['model_name'] = [kwargs['model_name']]
    # kwargs['params']['model_name'] = [kwargs['model_name']] # Pull out the params to a separate var
    kwargs.setdefault('dataset_name', 'talos_readings')
    kwargs.setdefault('experiment_no', '1')
    dataset_name = kwargs['dataset_name']
    experiment_no = kwargs['experiment_no']
    # performance_metric = params_json['params'][''.join(kwargs['params']['model_name'])]['performance_metric']
    kwargs.setdefault('performance_metric', params_json['params'][''.join(search_params['model_name'])]['performance_metric'])
    performance_metric = kwargs['performance_metric']

    for param_name, param in search_params.items():
        if type(param) != list:
            search_params[param_name] = [param]

    print("From talos.py --- ", search_params)

    h = ta.Scan(x_train, y_pred,
                params=search_params,
                dataset_name=dataset_name,
                experiment_no=experiment_no,
                model=talos_model,
                grid_downsample=0.5)

    report = h.data
    best_model = report.sort_values(performance_metric, ascending=True).iloc[0]
    best_model_id = best_model.name - 1
    dnn_model = activate_model(h, best_model_id)
    loss = best_model.losses
    # print(loss)
    # try:
    #     loss = loss.split(" ")[1]  # Right way of implemetation?
    # except:
    #     pass
    optimizer = best_model.optimizer
    dnn_model.compile(optimizer=optimizer, loss=loss, metrics=[loss])
    return dnn_model
