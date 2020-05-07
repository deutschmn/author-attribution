import io
import logging

import tensorflow as tf
import pandas as pd
import numpy as np
from functools import reduce
import operator
import nltk.stem
import gensim


def tensorflow_approach():
    df = pd.read_csv('articles.csv', sep=';', header=None, quotechar="'", quoting=0)
    vocabulary_size = 1000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocabulary_size,
                                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                                                      split=' ', char_level=False, oov_token=None)
    data = df[1][0:400]
    stemmer = nltk.stem.Cistem(case_insensitive=True)
    data = [stemmer.stem(sentence) for sentence in list(data)]
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    sequence = reduce(operator.add, sequences)

    WINDOW_SIZE = 2

    # Table for generating negative samples in realistic manner
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocabulary_size)
    couples, lables = tf.keras.preprocessing.sequence.skipgrams(
        sequence=sequence,
        vocabulary_size=vocabulary_size,
        window_size=WINDOW_SIZE,
        sampling_table=sampling_table
    )
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")

    # Inputs
    input_target = tf.keras.Input((1,))
    input_context = tf.keras.Input((1,))

    # Create some layers
    embedding_dimension = 10

    # Main embedding layer that we want to train
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dimension,
        input_length=1,
        name='embedding_layer'
    )
    target_embedding = tf.keras.layers.Reshape((embedding_dimension,))(embedding_layer(input_target))
    context_embedding = tf.keras.layers.Reshape((embedding_dimension,))(embedding_layer(input_context))

    dot_product = tf.keras.layers.dot(inputs=[target_embedding, context_embedding], axes=1)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dot_product)

    model = tf.keras.Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    print(model.summary())

    model.fit(x=[word_context, word_target], y=np.array(lables), epochs=1)

    # Store vectors, stores in two files vectors + metadata, you can use this awesome tool to view the result:
    # https://projector.tensorflow.org/
    trained_embedding_weights = model.layers[2].get_weights()[0]
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    words = list(tokenizer.word_index.keys())[:vocabulary_size - 1]
    indices = list(tokenizer.word_index.values())[:vocabulary_size - 1]

    for num, word in zip(indices, words):
        vec = trained_embedding_weights[num]  # skip 0, it's padding.
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()

    print("done")


def gensim_approach():
    df = pd.read_csv('articles.csv', sep=';', header=None, quotechar="'", quoting=0)
    data = list(df[1])
    stemmer = nltk.stem.Cistem(case_insensitive=True)
    data = [stemmer.stem(sentence) for sentence in data]
    toktok = nltk.tokenize.ToktokTokenizer()
    data = [toktok.tokenize(sentence) for sentence in data]

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(data, iter=1000, min_count=10, size=300, workers=4)
    print("most common words: ", model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])
    model.wv.save_word2vec_format("word2vec", binary=True)
    print("done")


if __name__ == '__main__':
    gensim_approach()
    # tensorflow_approach()
