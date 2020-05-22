import functools

import pandas
import flair  # https://github.com/flairNLP/flair
import nltk
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import dbase_helper
import plt_helper
import data_analysis


def ner_article_plots():
    entities = dbase_helper.generate_pkl("prepared_ner_articles.pkl", generate_article_ner_frame)
    pandas.DataFrame(entities['StemmedText'].value_counts().head(30)).plot.bar()
    plt_helper.save_and_show_plot("Entity Distribution")

    entities["Label"].value_counts().plot.bar()
    plt_helper.save_and_show_plot("Entity Label Distribution")

    joined_article_categories = data_analysis.generate_joined_category_articles_frame()
    articles_time = joined_article_categories[['Article_ID', 'PublishingDate']]

    for label in set(entities["Label"]):
        print("Doing plots for: " + label)
        label_entities = entities[entities['Label'] == label]
        label_series = label_entities["StemmedText"].value_counts().head(30)

        # For persons we check if name is subname and combine values (problematic with other lables e.g. österreich -
        # oberösterreich)
        if label == "PER":
            persons = label_series.index.values
            for person in persons:
                for compare_person in persons:
                    if compare_person in person and person != compare_person:
                        print(str(person) + " is not unique, subset of " + str(compare_person))
                        label_series[compare_person] += label_series[person]
                        label_series = label_series.drop(labels=[person])
                        break

        pandas.DataFrame(label_series.sort_values()).plot.barh()
        plt_helper.save_and_show_plot("Entities - " + label + " Distribution")
        top_entities = label_series.sort_values(ascending=False).head(6).index.values

        years = [2015, 2016]
        top_entity_entries = []
        for entity in top_entities:
            if label == "PER":
                entity_entries = label_entities[label_entities.StemmedText.str.contains(entity)]
                entity_entries = entity_entries.assign(StemmedText=entity)
            else:
                entity_entries = label_entities[label_entities.StemmedText == entity]
            top_entity_entries.append(entity_entries)
        top_entity_entries = pandas.concat(top_entity_entries)
        top_entity_entries = pandas.merge(top_entity_entries, articles_time)

        plt.style.use('seaborn-deep')
        year_entity_entries = top_entity_entries[top_entity_entries.PublishingDate.dt.year > 2014][
            ['PublishingDate', 'StemmedText']]
        year_entity_entries.PublishingDate = year_entity_entries.PublishingDate.dt.date
        plots = year_entity_entries['PublishingDate'].hist(by=year_entity_entries['StemmedText'], histtype='bar',
                                                           alpha=0.8, bins=12)
        fig = plt.gca().figure
        title = "Top Entities from " + label + "  over time"
        fig.suptitle(title, y=0.99)
        plt_helper.save_and_show_plot(title, False)

        values = []
        labels = []
        for entity in top_entities:
            values.append(year_entity_entries[year_entity_entries.StemmedText == entity]['PublishingDate'])
            labels.append(entity)
        plt.hist(values, label=labels)
        plt.legend()
        plt_helper.save_and_show_plot("Top Entities from " + label + " over time")

    print("done")


def generate_article_ner_frame():
    # NER-Data generated in colab see colab_notebooks/NER_Articles.ipynb
    predictions = pandas.read_pickle('pkl_saves/articles_ner.pkl')
    stemmer = nltk.stem.Cistem(case_insensitive=True)

    labels = []
    article_ids = []
    texts = []
    stemmed_texts = []
    for named_entities, article_id in zip(predictions.NEs, predictions.ID_Article):
        if article_id % 1000 == 0:
            print(str(article_id) + "/" + str(len(predictions.ID_Article)))
        for named_entity in named_entities['entities']:
            text = named_entity['text']
            for label_score in named_entity['labels']:
                text = re.sub('[\W]', '', text)
                stemmed_text = stemmer.stem(text)
                label = label_score.value
                labels.append(label)
                article_ids.append(article_id)
                texts.append(text)
                stemmed_texts.append(stemmed_text)

    entities = pandas.concat(
        [pandas.Series(article_ids), pandas.Series(labels), pandas.Series(texts), pandas.Series(stemmed_texts)],
        axis=1,
        keys=['Article_ID', 'Label', 'Text', 'StemmedText']
    )
    return entities


def create_co_occurrence_all():
    entities = dbase_helper.generate_pkl("prepared_ner_articles.pkl", generate_article_ner_frame)
    num_top_entities = 50
    pandas.DataFrame(entities['StemmedText'].value_counts().head(num_top_entities)).plot.bar()
    plt.title("Distribution of top " + str(num_top_entities) + " named entities over all "
              + str(entities['Article_ID'].size) + " Articles")
    plt.show()

    word_occurrences = pandas.DataFrame(entities['StemmedText'].value_counts())
    word_occurrences = word_occurrences[word_occurrences['StemmedText'] >= 10]
    word_occurrences = word_occurrences.rename(columns={'StemmedText': 'NumOccurrences'})

    interesting_words = word_occurrences.index.values
    create_co_occurrence_matrix(interesting_words, 'article_co_occurrences.csv')

    entities_without_locations = entities[entities.Label != 'LOC']
    word_occurrences = pandas.DataFrame(entities_without_locations['StemmedText'].value_counts())
    word_occurrences = word_occurrences[word_occurrences['StemmedText'] >= 10]
    word_occurrences = word_occurrences.rename(columns={'StemmedText': 'NumOccurrences'})
    interesting_words = word_occurrences.index.values
    create_co_occurrence_matrix(interesting_words, 'article_co_occurrences_without_locations.csv')
    print("done")


def create_co_occurrence_matrix(interesting_words: [str], filename: str = None):
    entities = dbase_helper.generate_pkl("prepared_ner_articles.pkl", generate_article_ner_frame)
    some = entities[entities['StemmedText'].isin(interesting_words)].groupby(by='Article_ID', as_index=False).agg(
        lambda x: ' '.join(list(x)))[['Article_ID', 'StemmedText']]
    interesting_articles = np.array(some['Article_ID'])

    percent_interesting_articles = (interesting_articles.size / np.unique(entities['Article_ID']).size) * 100
    print("We look at " + str(round(percent_interesting_articles, 2)) + "% of all articles")

    count_model = CountVectorizer(ngram_range=(1, 1))  # default unigram model
    X = count_model.fit_transform(np.array(some['StemmedText']))
    names = count_model.get_feature_names()
    # X[X > 0] = 1 # run this line if you don't want extra within-text cooccurence (see below)
    Xc = (X.T * X)  # this is co-occurrence matrix in sparse csr format
    Xc.setdiag(0)  # fill same word cooccurence to 0
    co_occurrences = pandas.DataFrame(data=Xc.toarray(), columns=names, index=names)
    co_occurrences.to_csv(dbase_helper.PKL_CACHE_FOLDER + '/' + filename, sep=',')


if __name__ == '__main__':
    ner_article_plots()
    create_co_occurrence_all()
