from enum import Enum

import pandas as pd
import sklearn.model_selection
from aa_data_prep import prepare_data
from aa_classifier import AuthorClassifier
import numpy as np
import kerastuner as kt
import tensorflow as tf
import datetime
import json
import pathlib


def log_run_inputs(rnn_inputs, dense_inputs, num_users, num_dense_inputs, num_rnn_inputs, num_rnn_inputs_dimension,
                   num_train_samples, search_title):
    input_info = {
        "rnn_inputs": list(rnn_inputs.keys()),
        "dense_inputs": list(dense_inputs.keys()),
        "input_dimensions": {
            "num_users": num_users,
            "num_dense_inputs": num_dense_inputs,
            "num_rnn_inputs": num_rnn_inputs,
            "num_rnn_inputs_dimension": num_rnn_inputs_dimension,
            "num_train_samples": num_train_samples
        }
    }
    j = json.dumps(input_info)
    with open("hyperparams/" + search_title + "/inputs.json", "w") as json_file:
        json_file.write(j)


def prepare_inputs(rnn_inputs, dense_inputs, targets):
    num_train_samples = targets.shape[0]
    assert len(rnn_inputs) > 0 or len(dense_inputs) > 0, "no inputs"

    inputs = []
    if len(rnn_inputs) == 0:
        rnn_network_inputs = np.zeros((num_train_samples, 0, 0))
    else:
        rnn_network_inputs = np.hstack(rnn_inputs.values())
        inputs.append(rnn_network_inputs)

    if len(dense_inputs) == 0:
        dense_network_inputs = np.zeros((num_train_samples, 0))
    else:
        dense_network_inputs = np.hstack(dense_inputs.values())
        if np.all(dense_network_inputs.astype('uint16') == dense_network_inputs):
            dense_network_inputs = dense_network_inputs.astype('uint16')
        inputs.append(dense_network_inputs)

    assert targets.shape[0] == dense_network_inputs.shape[0] == rnn_network_inputs.shape[0], \
        "non-matching number of samples for inputs and targets"
    return inputs, rnn_network_inputs, dense_network_inputs


def hyper_parameter_search(rnn_inputs, dense_inputs, targets: np.array,
                           validation_split=0.2, max_epochs=15,
                           selected_hyperparameters=None,
                           search_title=None,
                           model_name="AuthorAttributionModel"):
    """
    Perfoms a hyper-parameter search on the network, writes it to the file system and return resulting tuner
    :param rnn_inputs: dictionary with inputs to RNN (shapes Nx?, N = #posts)
    :param dense_inputs: dictionary with inputs to dense branch (shapes Nx?, N = #posts)
    :param targets: one-hot encoded target users as numpy array with (shape NxM, N = #posts, M = #users)
    :param validation_split: percentage of samples that should be used for validation
    :param max_epochs: maximal epochs run by the tuner
    :param selected_hyperparameters: Only tune for selected parameters,
        or fix all to just get the tuner for building the model
    :param search_title: title of the search, used to write logs to file system
    :param model_name: name of the model
    :return: Keras tuner object
    """

    if search_title is None:
        search_title = "search_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    num_train_samples = targets.shape[0]

    inputs, rnn_network_inputs, dense_network_inputs = prepare_inputs(rnn_inputs, dense_inputs, targets)
    pathlib.Path("hyperparams/" + search_title).mkdir(parents=True, exist_ok=True)

    num_users = targets.shape[1]
    num_dense_inputs = dense_network_inputs.shape[1]
    num_rnn_inputs = rnn_network_inputs.shape[1]
    num_rnn_inputs_dimension = rnn_network_inputs.shape[2]
    log_run_inputs(rnn_inputs, dense_inputs,
                   num_users, num_dense_inputs, num_rnn_inputs, num_rnn_inputs_dimension, num_train_samples,
                   search_title)

    classifier = AuthorClassifier(num_users, num_dense_inputs, num_rnn_inputs,
                                  num_rnn_inputs_dimension, num_train_samples, search_title, model_name)

    stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3)

    tensorboard_log_dir = "tensorboard_logs/" + search_title
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_images=True,
        # embeddings_freq=1,
        write_graph=True)

    if selected_hyperparameters is None:
        tuner = kt.Hyperband(classifier,
                             objective='val_accuracy',
                             max_epochs=max_epochs,
                             factor=3,
                             directory='hyperparams',
                             project_name=search_title)
    else:
        tunable_parameters = classifier.get_tunable_hyper_parameters()
        assert all(parameter in selected_hyperparameters.keys() for parameter in tunable_parameters.values.keys()), \
            "not all parameters needed where provided, need: " + str(tunable_parameters.values.keys())

        hp = kt.HyperParameters()
        for key in selected_hyperparameters.keys():
            if type(selected_hyperparameters[key]) != list:
                hp.Fixed(key, selected_hyperparameters[key])
            else:
                hp.Choice(key, selected_hyperparameters[key])

        tuner = kt.Hyperband(classifier,
                             hyperparameters=hp,
                             objective='val_accuracy',
                             max_epochs=max_epochs,
                             factor=3,
                             directory='hyperparams',
                             project_name=search_title)

    tuner.search(inputs, targets, validation_split=validation_split, callbacks=[hist_callback,
                                                                                stop_callback])
    print("Hyper-parameters search '" + search_title + "' completed. Top results:")
    tuner.results_summary(5)
    return tuner


class Configuration(Enum):
    ALL_FEATURES = "all_features"
    POST_CONTENT_FEATURES = "post_content_features"
    METADATA_FEATURES = "metadata_features"
    DENSE_FEATURES = "dense_features"


def load_inputs_for_configuration(config: Configuration):
    posts, data = prepare_data()
    if config == Configuration.ALL_FEATURES:
        rnn_in = {
            "embedded_posts": data["embedded_posts"]
        }
        dense_in = {
            "date_stats": data["date_stats"],
            "article_stats": data["article_stats"],
            "article_entities": data["article_entities"],
            "post_ratings": data["post_ratings"],
            "parent_posts": data["parent_posts"],
            "stylometric_features": data["stylometric"]
        }
    elif config == Configuration.POST_CONTENT_FEATURES:
        rnn_in = {
            "embedded_posts": data["embedded_posts"]
        }
        dense_in = {
            "stylometric_features": data["stylometric"]
        }
    elif config == Configuration.METADATA_FEATURES:
        rnn_in = {
        }
        dense_in = {
            "date_stats": data["date_stats"],
            "article_stats": data["article_stats"],
            "article_entities": data["article_entities"],
            "post_ratings": data["post_ratings"],
            "parent_posts": data["parent_posts"],
        }
    elif config == Configuration.DENSE_FEATURES:
        rnn_in = {
        }
        dense_in = {
            "date_stats": data["date_stats"],
            "article_stats": data["article_stats"],
            "article_entities": data["article_entities"],
            "post_ratings": data["post_ratings"],
            "parent_posts": data["parent_posts"],
            "stylometric_features": data["stylometric"]
        }
    else:
        raise Exception("Invalid Configuration")

    return data["targets"], rnn_in, dense_in


def final_training(model, rnn_inputs, dense_inputs, targets: np.array,
                   parameters: {}, validation_split=0.2, max_epochs=15,
                   search_title: str = None, model_name: str = "AuthorAttributionModel"):
    """
    Trains the model on the test set and runs validation on the test set
    :param model: the model to train
    :param rnn_inputs: dictionary with inputs to RNN (shapes Nx?, N = #posts)
    :param dense_inputs: dictionary with inputs to dense branch (shapes Nx?, N = #posts)
    :param targets: one-hot encoded target users as numpy array with (shape NxM, N = #posts, M = #users)
    :param validation_split: percentage of samples that should be used for validation
    :param max_epochs: number of maximal training epochs (might stop earlier due to Early Stopping)
    :param parameters: the hyper-parameters (just for logging them)
    :param search_title: title of the search, used to write logs to file system
    :param model_name: used when plotting the model
    """
    if search_title is None:
        search_title = "search_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = "final_training/" + search_title
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    tf.keras.utils.plot_model(model, log_dir + "/" + str(model_name) + ".png",
                              show_shapes=True, expand_nested=True)

    # Split Test and Training
    inputs, rnn_network_inputs, dense_network_inputs = prepare_inputs(rnn_inputs, dense_inputs, targets)
    train_size = 0.8
    if len(inputs) == 1:
        x_train, x_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(inputs[0], targets, train_size=train_size)
        x_train = [x_train]
        x_test = [x_test]
    else:
        x_train_0, x_test_0, x_train_1, x_test_1, y_train, y_test = \
            sklearn.model_selection.train_test_split(inputs[0], inputs[1], targets, train_size=train_size)
        x_train = [x_train_0, x_train_1]
        x_test = [x_test_0, x_test_1]

    # Final Training
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True)
    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_images=False,
        write_graph=True)

    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        validation_split=validation_split,
        callbacks=[early_stop, hist_callback],
        batch_size=32  # TODO maybe add as argument
    )
    result = model.evaluate(x_test, y_test)
    final_test_result = dict(zip(model.metrics_names, result))
    print("test_results: ", final_test_result)
    pd.DataFrame(history.history).to_csv(log_dir + "/history.csv")
    j = json.dumps(final_test_result)
    with open(log_dir + "/test_results.json", "w") as json_file:
        json_file.write(j)
    j = json.dumps(parameters)
    with open(log_dir + "/hyper_parameters.json", "w") as json_file:
        json_file.write(j)
    print("done")


if __name__ == '__main__':
    # You can now set parameters from here, if you provide a list the tuner will find the best one
    final_parameters = {'rnn_type': 'gru',
                        'rnn_dimension': 200,
                        'rnn_hidden_layer_1_units': 64,
                        'rnn_hidden_layer_2_units': 32,
                        'num_after_dense_input_layers': 0,
                        'num_after_dense_input_layer_neurons': 30,
                        'dropout': 0.3,
                        'dense_dropout': 0.3,
                        'neurons_final_dense_layers': 128,
                        'num_final_dense_layers': 1,
                        'learning_rate': [1e-04, 0.01, 0.1]}  # Try some learning rates
    configuration_type = Configuration.ALL_FEATURES
    model_name = "AuthorAttributionModel_" + str(configuration_type.value)
    search_title = "Test_final_training_" + str(configuration_type.value) + "_00"

    # Run the tuner
    targets, rnn_inputs, dense_inputs = load_inputs_for_configuration(configuration_type)
    tuner = hyper_parameter_search(rnn_inputs, dense_inputs, targets, selected_hyperparameters=final_parameters,
                                   search_title=search_title, model_name=model_name, max_epochs=50)

    # Run final training with best hyper-parameters on a fresh model and with Test/Train split
    parameters = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(parameters)
    final_training(model, rnn_inputs, dense_inputs, targets, parameters.values,
                   max_epochs=50,
                   search_title=search_title, model_name=model_name)
    print("done")
