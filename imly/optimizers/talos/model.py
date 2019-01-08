from utils.model_mapping import get_model_design


def talos_model(x_train, y_train, x_val, y_val, params):
    build_model, param_name = get_model_design(params['model_name'])
    model = build_model(x_train=x_train, params=params, param_name=param_name)
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model
    