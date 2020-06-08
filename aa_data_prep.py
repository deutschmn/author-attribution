import os.path

import gensim
import nltk.stem
import numpy as np
import pandas as pd
import tensorflow as tf

import dbase_helper
import embeddings.word2vec as word2vec
import ner

DEBUG = False


def compute_date_stats(posts):
    date_stats = pd.DataFrame()
    date_stats["ID_Post"] = posts["ID_Post"]

    date_stats["TimeOfDay"] = posts["CreatedAt"].apply(lambda x: x.hour + x.minute / 60)
    date_stats["Timestamp"] = posts["CreatedAt"].apply(lambda x: x.value)
    date_stats["DayOfWeek"] = posts["CreatedAt"].apply(lambda x: x.dayofweek)

    # TODO: maybe dates should not contain floats -> easier to calculate and could prob. reduce size of
    #  array significantly
    date_inputs = np.asarray(date_stats.drop("ID_Post", axis=1).drop("Timestamp", axis=1))

    return date_inputs


def compute_article_category_stats(posts):
    article_cat_stats = pd.DataFrame()
    article_cat_stats["ID_Post"] = posts["ID_Post"]

    article_cat_stats["ArticleCategory1"] = posts["Path"].apply(lambda x: x.split("/")[0]).astype('category')
    article_cat_stats["ArticleCategory2"] = posts["Path"].apply(lambda x: x.split("/")[int("/" in x)]).astype(
        'category')
    article_cat_stats["ArticleCategoryFull"] = posts["Path"].astype('category')

    article_inputs = tf.keras.utils.to_categorical(article_cat_stats["ArticleCategory1"].cat.codes)

    return article_inputs


def load_raw_posts():
    columns = ["ID_Post", "ID_User", "CreatedAt", "Status", "Headline", "p.Body", "ID_Article", "Path"]
    sql = "SELECT " + ", ".join(columns) + " FROM Posts p INNER JOIN Articles a USING (ID_Article)"
    # TODO filter deleted posts?

    # only treat posts from users with at least this many post
    min_user_posts = 500
    sql += ''' WHERE p.ID_User IN (
        SELECT ID_User
        FROM Posts q
        GROUP BY ID_User
        HAVING COUNT(*) > ''' + str(min_user_posts) + " )"

    if DEBUG:
        sql += " LIMIT 100"

    posts = dbase_helper.query_to_data_frame(sql, "posts-" + str(min_user_posts) + ".pkl")
    posts.columns = columns

    # drop posts with empty main text
    posts["p.Body"] = posts["p.Body"].replace("", np.nan)
    posts = posts.dropna()

    # parse date strings
    posts["CreatedAt"] = pd.to_datetime(posts["CreatedAt"])

    # convert to category data data to reencode IDs
    posts["ID_User"] = posts["ID_User"].astype('category')

    return posts


def load_or_create_post_embeddings(posts):
    embedding_dim = 50
    fn = "embeddings/post_embeddings_word2vec-" + str(embedding_dim)
    if os.path.isfile(fn):
        return gensim.models.KeyedVectors.load_word2vec_format(fn, binary=True)
    else:
        return word2vec.gensim_approach(posts["p.Body"], fn, embedding_dim)


def load_or_embed_posts(posts, post_embeddings):
    max_words = 100
    fn = "pkl_cache/embedded-posts-" + str(max_words) + ".npy"
    if os.path.isfile(fn):
        return np.load(fn)
    else:
        embedded_posts = embed_posts(posts, post_embeddings, max_words)
        np.save(fn, embedded_posts)
        return embedded_posts


def embed_posts(posts, post_embeddings, max_words):
    stemmer = nltk.stem.Cistem(case_insensitive=True)
    toktok = nltk.tokenize.ToktokTokenizer()

    stemd_tokend_posts = posts["p.Body"].apply(lambda x: toktok.tokenize(stemmer.stem(x)))

    embedding_column = np.empty_like(stemd_tokend_posts)
    embedding_column[:] = np.nan

    df = pd.DataFrame([stemd_tokend_posts, embedding_column], index=["words", "embeddings"]).T
    for index, row in df.iterrows():
        if int(index) % 100 == 0:
            print("index=" + str(index))
        row["embeddings"] = []
        for word in row["words"]:
            if word in post_embeddings:
                row["embeddings"].append(post_embeddings[word])
            else:
                # TODO is it okay to just skip words for which we don't have an embedding?
                pass

    return tf.keras.preprocessing.sequence.pad_sequences(df["embeddings"], padding='post', maxlen=max_words)


def encode_article_named_entities(posts):
    entities = dbase_helper.generate_pkl("prepared_ner_articles.pkl", ner.generate_article_ner_frame)

    # Select named entities with minimal occurrence
    minimal_number_word_occurrences = 20
    word_occurrences = pd.DataFrame(entities['Text'].value_counts())
    word_occurrences = word_occurrences[word_occurrences['Text'] >= minimal_number_word_occurrences]
    word_occurrences = word_occurrences.rename(columns={'Text': 'NumOccurrences'})
    entity_occurrences, co_occurrences = ner.create_co_occurrence_matrix(word_occurrences.index.values)
    num_articles = dbase_helper.query_to_data_frame("""
        SELECT MAX(Articles.ID_Article) FROM Articles;
        """, "number_articles.pkl")[0][0]
    entity_occurrences = entity_occurrences.reindex(index=range(num_articles), fill_value=0).astype('uint8')
    posts = posts[['ID_Post', 'ID_Article']]
    posts_entity_occurrences_in_article = posts.join(entity_occurrences, on='ID_Article').drop('ID_Article', axis=1)
    return posts_entity_occurrences_in_article.drop("ID_Post", axis=1)


def load_post_ratings(posts):
    post_ratings = dbase_helper.query_to_data_frame("""
            SELECT Posts.ID_Post, Posts.PositiveVotes, Posts.NegativeVotes FROM Posts;
            """, "post_votes.pkl")
    post_ratings.columns = ["ID_Post", "PositiveVotes", "NegativeVotes"]
    post_ratings[["PositiveVotes", "NegativeVotes"]] = post_ratings[["PositiveVotes", "NegativeVotes"]].astype('uint16')
    return post_ratings[post_ratings.ID_Post.isin(posts.ID_Post)].drop("ID_Post", axis=1)


def load_parent_posts(posts):
    parent_posts = dbase_helper.query_to_data_frame("""
                SELECT Posts.ID_Post, Posts.ID_Parent_Post FROM Posts;
                """, "post_parents.pkl")
    parent_posts.columns = ["ID_Post", "ID_Parent_Post"]

    # For now just encode if there exists a parent post
    parent_posts["Parent_Post"] = parent_posts.ID_Parent_Post >= 0
    return parent_posts[["ID_Post", "Parent_Post"]][parent_posts.ID_Post.isin(posts.ID_Post)].drop("ID_Post", axis=1)


def prepare_data():
    posts = load_raw_posts()

    post_embeddings = load_or_create_post_embeddings(posts)
    data = {
        "embedded_posts": load_or_embed_posts(posts, post_embeddings),
        "date_stats": compute_date_stats(posts),
        "article_stats": compute_article_category_stats(posts),
        "article_entities": encode_article_named_entities(posts),
        "post_ratings": load_post_ratings(posts),
        "parent_posts": load_parent_posts(posts),
        "targets": tf.keras.utils.to_categorical(posts["ID_User"].cat.codes)
    }

    # TODO: need to make sure all ID_Posts align, could also join frames together based on ID_Post to ensure this

    return posts, data
