from utils.model_mapping import get_model_design
from architectures.sklearn.model import create_model


def talos_model(x_train, y_train, x_val, y_val, params):
    fn_name, param_name = get_model_design(params['model_name'])

    mapping_instance = create_model(fn_name=fn_name, param_name=param_name)
    model = mapping_instance.__call__(x_train=x_train)

    out = model.fit(x_train, y_train)
                    # batch_size=params['batch_size'],
                    # epochs=params['epochs'],
                    # verbose=0)
                    # validation_data=[x_val, y_val])

    return out, model
