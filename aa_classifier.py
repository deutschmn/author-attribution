import tensorflow as tf


class AuthorClassifier:
    def __init__(self, num_users, post_embedding_dimension, num_dense_inputs):
        self.num_users = num_users
        self.post_embedding_dimension = post_embedding_dimension
        self.num_dense_inputs = num_dense_inputs

    def build_model(self, hp):
        rnn_types = {
            'gru': tf.keras.layers.GRU,
        #    'lstm': tf.keras.layers.LSTM,
        #    'simple': tf.keras.layers.SimpleRNN
        }

        rnn_type = hp.Choice('rnn_type', values=list(rnn_types.keys()))
        rnn_type = rnn_types[rnn_type]

        # TODO used to use the number of users here, maybe try different params
        rnn_dimension = hp.Int('rnn_dimension', min_value=150, max_value=250, step=50)

        hidden_layer = [64, 32]  # TODO make work with hp.Fixed('hidden_layers' or Choice

        rnn_input = tf.keras.Input(shape=(None, self.post_embedding_dimension))

        masked = tf.keras.layers.Masking()(rnn_input)
        rnn = rnn_type(rnn_dimension, return_sequences=(len(hidden_layer) > 0))(masked)

        dropout = hp.Choice('dropout', values=[0.15, 0.3])

        if dropout > 0:
            rnn = tf.keras.layers.Dropout(dropout)(rnn)

        for (i, hidden_neurons) in enumerate(hidden_layer):
            is_last_rnn_layer = i == len(hidden_layer) - 1
            rnn = rnn_type(hidden_neurons, return_sequences=not is_last_rnn_layer)(rnn)

            if dropout > 0:
                rnn = tf.keras.layers.Dropout(dropout)(rnn)

        # dense inputs
        dense_input = tf.keras.Input(shape=self.num_dense_inputs)

        # output and model creation
        concat = tf.keras.layers.concatenate([rnn, dense_input], axis=1)

        final_dense = concat
        final_dense_neurons = hp.Choice('neurons_final_dense_layers', values=[8, 16, 32, 64])
        for i in range(hp.Int('num_final_dense_layers', min_value=0, max_value=4, step=1)):
            final_dense = tf.keras.layers.Dense(final_dense_neurons, activation='relu')(final_dense)

        output = tf.keras.layers.Dense(self.num_users, activation='softmax')(final_dense)

        model = tf.keras.Model(inputs=[rnn_input, dense_input],
                               outputs=[output])

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()
        tf.keras.utils.plot_model(model, "plots/aa-model.png", show_shapes=True, expand_nested=True)

        return model
