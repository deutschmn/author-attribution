import pandas
import flair  # https://github.com/flairNLP/flair
import nltk
import re

import dbase_helper
import plt_helper


def ne_article_plots():
    entities = dbase_helper.generate_pkl("prepared_ner_articles.pkl", generate_ner_frame)

    pandas.DataFrame(entities['StemmedText'].value_counts().head(30)).plot.bar()
    plt_helper.save_and_show_plot("Named Entities Distribution")

    entities["Label"].value_counts().plot.bar()
    plt_helper.save_and_show_plot("Named Entities Label Distribution")

    for label in set(entities["Label"]):
        label_entities = entities[entities['Label'] == label]
        pandas.DataFrame(label_entities["StemmedText"].value_counts().head(30)).plot.barh()
        plt_helper.save_and_show_plot("Named Entities - " + label + " Distribution")

    print("done")


def generate_ner_frame():
    # Data generated in colab see colab_notebooks/NER_Articles.ipynb
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
        keys=['ID_Article', 'Label', 'Text', 'StemmedText']
    )
    return entities


if __name__ == '__main__':
    ne_article_plots()
