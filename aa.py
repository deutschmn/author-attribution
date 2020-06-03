from aa_data_prep import prepare_data
from aa_classifier import AuthorClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def split_train_test_val(X, test_split, val_split, rand_seed):
    X_train, X_test \
        = train_test_split(X, test_size=test_split, random_state=rand_seed)
    X_train, X_val \
        = train_test_split(X_train, test_size=val_split / (1 - test_split), random_state=rand_seed)
    return X_train, X_test, X_val


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
    classifier = AuthorClassifier(epochs=50)
    classifier.fit(embedded_posts_train, targets_train)

    test_score = classifier.score(embedded_posts_test, targets_test)
    print("Test score = " + str(test_score))
