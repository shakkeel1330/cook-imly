import numpy as np
import pandas as pd
import talos as ta
from architectures.talos.model import f1_talos
from talos.utils.best_model import best_model, activate_model

loss_mapping = {
    'mean_squared_error': 'mse'
}


def talos_optimization(x_train, y_train, kwargs):
    np.random.seed(7)

    kwargs.setdefault('params', p)  # Add p from JSON
    kwargs.setdefault('dataset_name', 'talos_readings')
    kwargs.setdefault('experiment_no', '1')
    kwargs.setdefault('performance_metric', 'val_loss')  # Send from a mapping based on model_name

    h = ta.Scan(x_train, kwargs['y_pred'],
                params=kwargs['params'],
                dataset_name=kwargs['dataset_name'],
                experiment_no=kwargs['experiment_no'],
                model=f1_talos,
                grid_downsample=0.5)

    model_id = best_model(h, metric='val_loss', asc=True)
    dnn_model = activate_model(h, model_id)
    url = kwargs['dataset_name'] + '_' + kwargs['experiment_no'] + '.csv'
    report = pd.read_csv(url, delimiter=",", header=0, index_col=False)
    performance_metric = kwargs['performance_metric']

    report = report.filter(items=[performance_metric, 'optimizer', 'losses'])
    report = report.iloc[report[performance_metric].idxmax()]
    loss = report['losses']
    loss = loss.split(" ")[1]
    optimizer = report['optimizer']
    for key, value in loss_mapping.items():
        if loss == key:
            loss = value

    dnn_model.compile(optimizer=optimizer, loss=loss, metrics=[loss])