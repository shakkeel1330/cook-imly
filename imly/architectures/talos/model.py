from utils.model_mapping import get_model_design


def f1_talos(x_train, y_train, x_val, y_val, params):
    build_model = get_model_design(params['model_name'])
    model = build_model(x_train=x_train, params=params)
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model
    