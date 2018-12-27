from architectures.sklearn.model import f1


def get_model_design(name):
    function_name = None
    mapping = {
        'LinearRegression': f1,
        'KerasRegressor': f1
    }

    for key, value in mapping.items():
        if key == name:
            function_name = value

    return function_name
