from utils.model_mapping import get_model_design
from architectures.sklearn.model import create_model
import os, json
# Initialize tune

import ray
from ray import tune

ray.init(ignore_reinit_error=True)


def get_best_model(x_train, y_train, **kwargs):

    y_pred = kwargs['primal_data']['y_pred']
    # params = kwargs['params']
    model_name = kwargs['primal_data']['model_name']
    fn_name, param_name = get_model_design(model_name)

    mapping_instance = create_model(fn_name=fn_name, param_name=param_name)

    def train_model(config, reporter):
        '''
        This functio is used by Tune to train the model with each iteration variations.

        Args:
            config(dict): A dictionary with the search params passed by Tune.
            Similar to the JSON we already have.
            reporter: A function used by Tune to keep a track of the metric by
            which the iterations should be optimized.
        '''
        # model = make_model(None)
        # print("---", os.getcwd(),"---")
        os.chdir('/home/shakkeel/Desktop/mlsquare/cook-imly/imly')
        params_json = json.load(open('../imly/optimizers/tune/params.json'))
        params = params_json['params'][model_name]['config']
        model = mapping_instance.__call__(x_train=x_train, params=params)
        model.fit(x_train, y_pred)
        accuracy = model.evaluate(x_train, y_pred)[1]
        reporter(mean_accuracy=accuracy)
    
    # define experiment config
    configuration = tune.Experiment(
        # TODO
        # load config from params.json
        # change search algo to hyperopt
        "experiment_name",
        run=train_model,
        resources_per_trial={"cpu": 4},
        stop={"mean_accuracy": 95},
        config={
            "optimizer": tune.grid_search(['adam', 'nadam']) # Edit to accept params.json
        }
    )

    trials = tune.run_experiments(configuration, verbose=2)
    metric = "mean_accuracy"

    """Restore a model from the best trial."""
    sorted_trials = get_sorted_trials(trials, metric)
    for best_trial in sorted_trials:
        try:
            print("Creating model...")
            # best_model = model_creator(best_trial.config)
            best_model = mapping_instance.__call__(x_train=x_train) #TODO Pass config as argument
            print(best_trial.logdir)
            print(best_trial.last_result["checkpoint"])
            weights = os.path.join(best_trial.logdir, best_trial.last_result["checkpoint"])
            print("Loading from", weights)
            best_model.load_weights(weights)
            break
        except Exception as e:
            print(e, "from tuner")
            print("Loading failed. Trying next model")

    return best_model


# Utils from Tune tutorials(Not a part of the Tune package)#

def get_sorted_trials(trial_list, metric):
    return sorted(trial_list, key=lambda trial: trial.last_result.get(metric, 0), reverse=True)




# def get_best_result(trial_list, metric):
#     """Retrieve the last result from the best trial."""
#     return {metric: get_best_trial(trial_list, metric).last_result[metric]} # The get_best_trial used here is different!


# def get_best_trial(model_creator, trial_list, metric): # Unnecessary! Shift it to get_best_model.
#     """Restore a model from the best trial."""
#     sorted_trials = get_sorted_trials(trial_list, metric)
#     for best_trial in sorted_trials:
#         try:
#             print("Creating model...")
#             model = model_creator(best_trial.config)
#             weights = os.path.join(best_trial.logdir, best_trial.last_result["checkpoint"])
#             print("Loading from", weights)
#             model.load_weights(weights)
#             break
#         except Exception as e:
#             print(e)
#             print("Loading failed. Trying next model")
#     return model


# Get best model #
# final_model = get_best_trial(make_model, trials, metric="mean_accuracy")

# TODO
# Currently using grid search. Use hyperopt as the search algorithm.