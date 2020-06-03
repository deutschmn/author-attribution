from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import numpy as np


class AuthorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 batch_size=32,
                 epochs=10,
                 dropout=0,
                 rnn_type='gru',
                 hidden_layer=[64, 32]):

        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.hidden_layer = hidden_layer

        self._model = None
        self._num_users = None
        self._num_posts = None
        self._post_max_length = None
        self._post_embedding_dimension = None

    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0], "number of posts in input and output must match"

        self._num_users = Y.shape[1]
        self._num_posts = X.shape[0]  # N
        self._post_max_length = X.shape[1]  # L
        self._post_embedding_dimension = X.shape[2]  # E

        rnn_type = None
        if self.rnn_type is 'gru':
            rnn_type = tf.keras.layers.GRU
        elif self.rnn_type is 'lstm':
            rnn_type = tf.keras.layers.LSTM
        elif self.rnn_type is 'simple':
            rnn_type = tf.keras.layers.SimpleRNN
        else:
            assert False, "Unknwon RNN type"

        inputs = tf.keras.Input(shape=(None, self._post_embedding_dimension))

        masked = tf.keras.layers.Masking()(inputs)
        rnn = rnn_type(self._num_users,
                       # input_shape=(X.shape[1], self._num_words),
                       return_sequences=(len(self.hidden_layer) > 0)
                       )(masked)

        # TODO maybe add dropouts

        for (i, hidden_neurons) in enumerate(self.hidden_layer):
            is_last_rnn_layer = i == len(self.hidden_layer) - 1
            rnn = rnn_type(hidden_neurons, return_sequences=not is_last_rnn_layer)(rnn)

        output = tf.keras.layers.Dense(self._num_users, activation='softmax')(rnn)

        self._model = tf.keras.Model(inputs, output)

        self._model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

        self._model.summary()
        tf.keras.utils.plot_model(self._model, "plots/aa-model.png", show_shapes=True, expand_nested=True)

        self._model.fit(X, Y,
                        epochs=self.epochs,
                        batch_size=self.batch_size)

        return self._model

    def score(self, X, y, **kwargs):
        return self._model.evaluate(X, y)[1]
