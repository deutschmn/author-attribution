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
        self._num_dense_inputs = None
        self._post_max_length = None
        self._post_embedding_dimension = None

    def fit(self, X, Y, X_val, Y_val):
        assert X["dense"].shape[0] == X["rnn"].shape[0] == Y.shape[0], "number of posts in inputs and output must match"

        self._num_users = Y.shape[1]
        self._num_posts = X["rnn"].shape[0]  # N
        self._num_dense_inputs = X["dense"].shape[1]
        self._post_max_length = X["rnn"].shape[1]  # L
        self._post_embedding_dimension = X["rnn"].shape[2]  # E

        # set up RNN
        rnn_type = None
        if self.rnn_type is 'gru':
            rnn_type = tf.keras.layers.GRU
        elif self.rnn_type is 'lstm':
            rnn_type = tf.keras.layers.LSTM
        elif self.rnn_type is 'simple':
            rnn_type = tf.keras.layers.SimpleRNN
        else:
            assert False, "Unknwon RNN type"

        rnn_input = tf.keras.Input(shape=(None, self._post_embedding_dimension))

        masked = tf.keras.layers.Masking()(rnn_input)
        rnn = rnn_type(self._num_users, return_sequences=(len(self.hidden_layer) > 0))(masked)

        if self.dropout > 0:
            rnn = tf.keras.layers.Dropout(self.dropout)(rnn)

        for (i, hidden_neurons) in enumerate(self.hidden_layer):
            is_last_rnn_layer = i == len(self.hidden_layer) - 1
            rnn = rnn_type(hidden_neurons, return_sequences=not is_last_rnn_layer)(rnn)

            if self.dropout > 0:
                rnn = tf.keras.layers.Dropout(self.dropout)(rnn)

        # dense inputs
        dense_input = tf.keras.Input(shape=(self._num_dense_inputs))
        # TODO probably need another layer here, but let's see
        # output and model creation
        concat = tf.keras.layers.concatenate([rnn, dense_input], axis=1)

        output = tf.keras.layers.Dense(self._num_users, activation='softmax')(concat)

        self._model = tf.keras.Model(inputs=[rnn_input, dense_input],
                                     outputs=[output])

        self._model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

        self._model.summary()
        tf.keras.utils.plot_model(self._model, "plots/aa-model.png", show_shapes=True, expand_nested=True)

        self._model.fit([X["rnn"], X["dense"]], Y,
                        validation_data=([X_val["rnn"], X_val["dense"]], Y_val),
                        epochs=self.epochs,
                        batch_size=self.batch_size)

        return self._model

    def score(self, X, y, **kwargs):
        return self._model.evaluate(X, y)[1]
