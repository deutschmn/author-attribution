from aa_data_prep import prepare_data
from aa_classifier import AuthorClassifier
import numpy as np
import kerastuner as kt
import tensorflow as tf
import datetime
import json
import pathlib


def log_run_inputs(rnn_inputs, dense_inputs, num_users, num_dense_inputs, num_rnn_inputs, num_rnn_inputs_dimension, num_train_samples):
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


def hyper_parameter_search(rnn_inputs, dense_inputs, targets: np.array,
                           validation_split=0.2, search_title=None):
    """
    Perfoms a hyper-parameter search on the network, writes it to the file system and return resulting tuner
    :param rnn_inputs: dictionary with inputs to RNN (shapes Nx?, N = #posts)
    :param dense_inputs: dictionary with inputs to dense branch (shapes Nx?, N = #posts)
    :param targets: one-hot encoded target users as numpy array with (shape NxM, N = #posts, M = #users)
    :param validation_split: percentage of samples that should be used for validation
    :param search_title: title of the search, used to write logs to file system
    :return: Keras tuner object
    """

    if search_title is None:
        search_title = "search_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

    pathlib.Path("hyperparams/" + search_title).mkdir(parents=True, exist_ok=True)

    num_users = targets.shape[1]
    num_dense_inputs = dense_network_inputs.shape[1]
    num_rnn_inputs = rnn_network_inputs.shape[1]
    num_rnn_inputs_dimension = rnn_network_inputs.shape[2]

    log_run_inputs(rnn_inputs, dense_inputs,
                   num_users, num_dense_inputs, num_rnn_inputs, num_rnn_inputs_dimension, num_train_samples)

    classifier = AuthorClassifier(num_users, num_dense_inputs, num_rnn_inputs,
                                  num_rnn_inputs_dimension, num_train_samples, search_title)

    tensorboard_log_dir = "tensorboard_logs/" + search_title
    tuner = kt.Hyperband(classifier,
                         objective='val_accuracy',
                         max_epochs=15,
                         factor=3,
                         directory='tensorboard_logs')

    stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3)

    hist_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        embeddings_freq=1,
        write_graph=True)

    tuner.search(inputs, targets, validation_split=validation_split, callbacks=[hist_callback,
                                                                                stop_callback])

    print("Hyper-parameters search '" + search_title + "' completed. Top results:")
    tuner.results_summary(5)

    return tuner


if __name__ == '__main__':
    posts, data = prepare_data()

    rnn_inputs = {
        "embedded_posts": data["embedded_posts"]
    }

    dense_inputs = {
        "date_stats": data["date_stats"],
        "article_stats": data["article_stats"],
        "article_entities": data["article_entities"],
        "post_ratings": data["post_ratings"],
        "parent_posts": data["parent_posts"],
        "stylometric_features": data["stylometric"]
    }

    search_title = 'add_style'

    hyper_parameter_search(rnn_inputs, dense_inputs, data["targets"], search_title=search_title)

    print("done")
