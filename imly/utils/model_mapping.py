import json


def get_model_design(name):
    # function_name = None
    mapping_data = json.load(open('../imly/utils/mapping.json'))

    try:
        fn_name = mapping_data[name]['model']
        param_name = mapping_data[name]['param']
        # module = __import__('architectures.sklearn.model',
        #                     fromlist=[function_name])
        # function = getattr(module, function_name)
    except KeyError:
        print('Invalid model name passed to mapping_data')

    # for key, value in mapping.items():
    #     if key == name:
    #         function_name = value

    return fn_name, param_name


# Pass params combination as well. f1 to glm
# Pass mappings as json. Also, Remove unnecessary mappings
# Remove mapping. Add try catch
