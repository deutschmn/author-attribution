import tensorflow as tf
import pandas as pd
import numpy as np
from functools import reduce
import operator


def main():
    df = pd.read_csv('articles.csv', sep=';', header=None, quotechar="'", quoting=0)
    vocabulary_size = 1000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocabulary_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                                       split=' ', char_level=False, oov_token=None)
    data = df[1][0:400]

    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    sequence = reduce(operator.add,sequences)

    WINDOW_SIZE = 2
    training_samples = []
    for current_word_idx, word in enumerate(sequence):
        for w in range(-WINDOW_SIZE, +WINDOW_SIZE):
            index = current_word_idx + w
            if 0 < index < len(sequence) - 1 and w != 0:
                training_samples.append([word, sequence[index]])

    training_samples = np.array(training_samples)
    x = tf.keras.utils.to_categorical(training_samples[:, 0])
    y = tf.keras.utils.to_categorical(training_samples[:, 1])

    embedding_dimension = 10
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(embedding_dimension, activation=tf.keras.activations.linear, input_shape=(vocabulary_size,)),
        tf.keras.layers.Dense(vocabulary_size, activation=tf.keras.activations.softmax),
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True)
    model.summary()

    model.fit(x, y, epochs=100, batch_size=512)
    print("done")



if __name__ == '__main__':
    main()