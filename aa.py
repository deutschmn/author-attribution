from aa_data_prep import prepare_data
from aa_classifier import AuthorClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import kerastuner as kt


def split_train_test_val(X, test_split, val_split, rand_seed):
    X_train, X_test \
        = train_test_split(X, test_size=test_split, random_state=rand_seed)
    X_train, X_val \
        = train_test_split(X_train, test_size=val_split / (1 - test_split), random_state=rand_seed)
    return X_train, X_test, X_val


def prepare_input(data, num_article_category_1, num_article_category_2):
    rnn_inputs = data["embedded_posts"]

    # TODO: need to make sure all ID_Posts align, could also join frames together based on ID_Post to ensure this
    assert all((data[key].ID_Post.values == data["date_stats"].ID_Post.values).all()
               for key in ["article_stats", "post_ratings", "parent_posts", "article_entities"])

    date_inputs = np.asarray(data["date_stats"].drop("ID_Post", axis=1).drop("Timestamp", axis=1))
    article_inputs = np.hstack([tf.keras.utils.to_categorical(data["article_stats"]["ArticleCategory1"].cat.codes,
                                                              num_classes=num_article_category_1),
                                tf.keras.utils.to_categorical(data["article_stats"]["ArticleCategory2"].cat.codes,
                                                              num_classes=num_article_category_2)])

    # TODO: maybe dates should not contain floats -> easier to calculate and could prob. reduce size of
    #  array significantly
    dense_inputs = np.hstack([date_inputs,
                              article_inputs,
                              np.asarray(data["post_ratings"].drop("ID_Post", axis=1)),
                              np.asarray(data["article_entities"].drop("ID_Post", axis=1)),
                              np.asarray(data["parent_posts"].drop("ID_Post", axis=1))])
    return {"rnn": np.asarray(rnn_inputs), "dense": np.asarray(dense_inputs)}


if __name__ == '__main__':
    posts, data = prepare_data()

    # split data into train and validation
    rand_seed = np.random.randint(10000)
    train_split = 0.6
    test_split = 0.2
    val_split = 1 - train_split - test_split

    training_data = {}
    test_data = {}
    validation_data = {}
    for key in data.keys():
        training_data[key], test_data[key], validation_data[key] = split_train_test_val(data[key], test_split,
                                                                                        val_split,
                                                                                        rand_seed)

    num_article_category_1 = np.max(data["article_stats"]["ArticleCategory1"].cat.codes) + 1
    num_article_category_2 = np.max(data["article_stats"]["ArticleCategory2"].cat.codes) + 1

    X_train = prepare_input(training_data,
                            num_article_category_1, num_article_category_2)
    X_val = prepare_input(validation_data,
                          num_article_category_1, num_article_category_2)

    # build and train model
    num_users = training_data["targets"].shape[1]
    num_dense_inputs = X_train["dense"].shape[1]
    post_embedding_dimension = X_train["rnn"].shape[2]

    classifier = AuthorClassifier(num_users, post_embedding_dimension, num_dense_inputs)

    tuner = kt.Hyperband(classifier,
                         objective='val_accuracy',
                         max_epochs=15,
                         factor=3,
                         directory='hyperparams',
                         project_name='author_identification_4')

    tuner.search([X_train["rnn"], X_train["dense"]], training_data["targets"],
                 validation_data=([X_val["rnn"], X_val["dense"]], validation_data["targets"]))

    print("done")