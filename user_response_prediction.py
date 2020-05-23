import itertools

import pandas
import nltk
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import numpy as np

import dbase_helper
import data_analysis
import ner
import plt_helper


def prepare_data():
    entities = dbase_helper.generate_pkl("prepared_ner_articles.pkl", ner.generate_article_ner_frame)

    # Select named entities with minimal occurrence
    minimal_number_word_occurrences = 5
    minimal_number_words_per_article = 5
    word_occurrences = pandas.DataFrame(entities['Text'].value_counts())
    word_occurrences = word_occurrences[word_occurrences['Text'] >= minimal_number_word_occurrences]
    word_occurrences = word_occurrences.rename(columns={'Text': 'NumOccurrences'})
    interesting_words = word_occurrences.index.values
    occurrences, co_occurrences = ner.create_co_occurrence_matrix(interesting_words)

    article_ids = occurrences.index.values
    data = data_analysis.generate_joined_rating_articles_frame()
    data = data[data.ID_Article.isin(article_ids)]

    interesting_words_per_article = entities[entities['Text'].isin(interesting_words)].groupby(
        by='ID_Article', as_index=False
    ).agg(lambda x: len(list(x)))[['ID_Article', 'Text']]

    article_ids = interesting_words_per_article[
        interesting_words_per_article.Text > minimal_number_words_per_article].ID_Article
    data = data[data.ID_Article.isin(article_ids)]

    articles = data[['ID_Article', 'Title', 'MainCategory', 'SubCategory', 'RemainingPath']]
    ratings = data[['ID_Article', 'PositiveVotesCount', 'NegativeVotesCount']]

    # Plot the data we shall predict
    plt.hist(data.PositiveVotesCount, label="PositiveVotesCount")
    plt.hist(-data.NegativeVotesCount, label="NegativeVotesCount")
    ax = plt.gca()
    ax.set_yscale('log')
    plt.legend()
    plt_helper.save_and_show_plot("Logarithmic Vote Distribution over Articles")

    plt.hist(data.PositiveVotesCount, label="PositiveVotesCount")
    plt.hist(-data.NegativeVotesCount, label="NegativeVotesCount")
    plt.legend()
    plt_helper.save_and_show_plot("Vote Distribution over Articles")

    normalize = False
    if normalize:
        pos_mean = data.PositiveVotesCount.mean()
        pos_std = data.PositiveVotesCount.std()
        data.PositiveVotesCount = (data.PositiveVotesCount - pos_mean) / pos_std

        neg_mean = data.NegativeVotesCount.mean()
        neg_std = data.NegativeVotesCount.std()
        data.NegativeVotesCount = (data.NegativeVotesCount - neg_mean) / neg_std

        plt.hist(data.PositiveVotesCount, label="PositiveVotesCount")
        plt.hist(-data.NegativeVotesCount, label="NegativeVotesCount")
        ax = plt.gca()
        ax.set_yscale('log')
        plt.title("Normalized Data")
        plt.legend()
        plt.show()

    training_article_ids = np.random.choice(article_ids, round(len(article_ids) * 0.8))
    training_data = {
        "articles": articles[articles.ID_Article.isin(training_article_ids)],
        "ratings": ratings[ratings.ID_Article.isin(training_article_ids)],
        "occurrences": occurrences[occurrences.index.isin(training_article_ids)],
    }

    test_article_ids = np.setdiff1d(article_ids, training_article_ids)
    test_data = {
        "articles": articles[articles.ID_Article.isin(test_article_ids)],
        "ratings": ratings[ratings.ID_Article.isin(test_article_ids)],
        "occurrences": occurrences[occurrences.index.isin(test_article_ids)]
    }

    return training_data, test_data


def create_and_train_model(training_data):
    EPOCHS = 500

    Y = training_data['ratings'][['PositiveVotesCount', 'NegativeVotesCount']].values
    X = training_data['occurrences'].values
    assert X.shape[0] == Y.shape[0]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=[X.shape[1]]),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(Y.shape[1])
    ])

    optimizer = tf.keras.optimizers.Adam(0.00001)
    loss = 'mean_squared_logarithmic_error'
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mean_squared_logarithmic_error', 'mae'])

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(
        X,
        Y,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop],
        batch_size=10
    )

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt_helper.save_and_show_plot("User Response Prediction Training Loss (" + loss + ")")

    plt.plot(history.history['mean_absolute_error'], label='mean absolute error')
    plt.plot(history.history['val_mean_absolute_error'], label='mean validation error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt_helper.save_and_show_plot("User Response Prediction Training Absolute Error")

    print("done")
    return model


def test_model(test_data, model):
    y_test = test_data['ratings'][['PositiveVotesCount', 'NegativeVotesCount']].values
    x_test = test_data['occurrences'].values
    results = model.evaluate(x_test, y_test)

    print("\nTest Results:")
    for i in range(len(model.metrics_names)):
        print(model.metrics_names[i] + ": " + str(results[i]))


if __name__ == '__main__':
    training_data, test_data = prepare_data()
    model = create_and_train_model(training_data)
    test_model(test_data, model)
    print("done")
