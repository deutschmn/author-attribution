from aa_data_prep import prepare_data
from aa_classifier import AuthorClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


def split_train_test_val(X, test_split, val_split, rand_seed):
    X_train, X_test \
        = train_test_split(X, test_size=test_split, random_state=rand_seed)
    X_train, X_val \
        = train_test_split(X_train, test_size=val_split / (1 - test_split), random_state=rand_seed)
    return X_train, X_test, X_val


def prepare_input(embedded_posts, date_stats, article_stats, num_article_category_1, num_article_category_2):
    rnn_inputs = embedded_posts

    date_inputs = np.asarray(date_stats.drop("ID_Post", axis=1).drop("Timestamp", axis=1))
    article_inputs = np.hstack([tf.keras.utils.to_categorical(article_stats["ArticleCategory1"].cat.codes,
                                                              num_classes=num_article_category_1),
                                tf.keras.utils.to_categorical(article_stats["ArticleCategory2"].cat.codes,
                                                              num_classes=num_article_category_2)])
    dense_inputs = np.hstack([date_inputs, article_inputs])
    return {"rnn": np.asarray(rnn_inputs), "dense": np.asarray(dense_inputs)}


if __name__ == '__main__':
    posts, embedded_posts, date_stats, article_stats, targets = prepare_data()

    # split data into train and validation
    rand_seed = np.random.randint(10000)
    train_split = 0.6
    test_split = 0.2
    val_split = 1 - train_split - test_split

    embedded_posts_train, embedded_posts_test, embedded_posts_val \
        = split_train_test_val(embedded_posts, test_split, val_split, rand_seed)
    date_stats_train, date_stats_test, date_stats_val \
        = split_train_test_val(date_stats, test_split, val_split, rand_seed)
    article_stats_train, article_stats_test, article_stats_val \
        = split_train_test_val(article_stats, test_split, val_split, rand_seed)
    targets_train, targets_test, targets_val \
        = split_train_test_val(targets, test_split, val_split, rand_seed)

    # TODO try different hyper-parameters using sklearn.model_selection
    classifier = AuthorClassifier(epochs=100, dropout=0.15)

    num_article_category_1 = np.max(article_stats["ArticleCategory1"].cat.codes) + 1
    num_article_category_2 = np.max(article_stats["ArticleCategory2"].cat.codes) + 1

    X_train = prepare_input(embedded_posts_train, date_stats_train, article_stats_train,
                            num_article_category_1, num_article_category_2)
    X_val = prepare_input(embedded_posts_val, date_stats_val, article_stats_val,
                            num_article_category_1, num_article_category_2)
    classifier.fit(X_train, targets_train, X_val, targets_val)

    X_test = prepare_input(embedded_posts_test, date_stats_test, article_stats_test)
    test_score = classifier.score(X_test, targets_test)
    print("Test score = " + str(test_score))
