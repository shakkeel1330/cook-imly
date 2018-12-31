import json
import numpy as np
import pandas as pd
import talos as ta
from architectures.talos.model import talos_model
from talos.utils.best_model import best_model, activate_model


def talos_optimization(x_train, y_train, kwargs):
    np.random.seed(7)
    params_json = json.load(open('../imly/architectures/talos/params.json'))
    params = params_json['params'][kwargs['model_name']]['config']
    losses = params['losses']
    params['losses'] = []
    for loss_name in losses:
        module = __import__('keras.losses', fromlist=[loss_name])
        params['losses'].append(getattr(module, loss_name))

    activation = params['activation']
    params['activation'] = []
    for activation_name in activation:
        module = __import__('keras.activations', fromlist=[activation_name])
        params['activation'].append(getattr(module, activation_name))

    kwargs.setdefault('params', params)
    kwargs['params']['model_name'] = [kwargs['model_name']]
    kwargs.setdefault('dataset_name', 'talos_readings')
    kwargs.setdefault('experiment_no', '1')
    performance_metric = params_json['params'][''.join(kwargs['params']['model_name'])]['performance_metric']
    kwargs.setdefault('performance_metric', performance_metric) 

    h = ta.Scan(x_train, kwargs['y_pred'],
                params=kwargs['params'],
                dataset_name=kwargs['dataset_name'],
                experiment_no=kwargs['experiment_no'],
                model=talos_model,
                grid_downsample=0.5)

    model_id = best_model(h, metric=kwargs['performance_metric'], asc=True)
    dnn_model = activate_model(h, model_id)
    report = h.data
    report = report.filter(items=[kwargs['performance_metric'], 'optimizer', 'losses'])
    loss = report.loc[:, 'losses'][0]
    loss = loss.split(" ")[1]
    optimizer = report.loc[:, 'optimizer'][0]

    dnn_model.compile(optimizer=optimizer, loss=loss, metrics=[loss])
    return dnn_model
