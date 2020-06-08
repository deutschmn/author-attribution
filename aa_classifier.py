import tensorflow as tf
from kerastuner import HyperModel


class AuthorClassifier(HyperModel):
    def __init__(self, num_users, num_dense_inputs, num_rnn_inputs,
                 num_rnn_inputs_dimension, num_train_samples, search_title):
        super().__init__()
        self.num_users = num_users
        self.num_rnn_inputs = num_rnn_inputs
        self.num_rnn_inputs_dimension = num_rnn_inputs_dimension
        self.num_dense_inputs = num_dense_inputs
        self.search_title = search_title

    def build(self, hp):
        rnn_types = {
            'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM,
            'simple': tf.keras.layers.SimpleRNN
        }

        rnn_type = hp.Fixed('rnn_type', 'gru')  # hp.Choice('rnn_type', values=list(rnn_types.keys()))
        rnn_type = rnn_types[rnn_type]

        rnn_dimension = hp.Int('rnn_dimension', min_value=150, max_value=250, step=50)

        hidden_layer = [hp.Fixed("rnn_hidden_layer_1_units", 64),
                        hp.Fixed("rnn_hidden_layer_2_units", 32)]

        # layer "branches" that then merge in the concatenate
        concats = []
        inputs = []

        if self.num_rnn_inputs > 0:
            rnn_input = tf.keras.Input(shape=(None, self.num_rnn_inputs_dimension))
            inputs.append(rnn_input)

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

            concats.append(rnn)

        # dense inputs
        if self.num_dense_inputs > 0:
            dense_input = tf.keras.Input(shape=self.num_dense_inputs)
            inputs.append(dense_input)
            concats.append(dense_input)

        # output and model creation
        if len(concats) > 1:
            concat = tf.keras.layers.concatenate(concats, axis=1)
        else:
            concat = concats[0]

        final_dense = concat
        final_dense_neurons = hp.Choice('neurons_final_dense_layers', values=[32, 64, 128, 256])
        for i in range(hp.Int('num_final_dense_layers', min_value=0, max_value=3, step=1)):
            final_dense = tf.keras.layers.Dense(final_dense_neurons, activation='relu')(final_dense)

        output = tf.keras.layers.Dense(self.num_users, activation='softmax')(final_dense)

        model = tf.keras.Model(inputs=inputs,
                               outputs=[output])

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()
        tf.keras.utils.plot_model(model, "hyperparams/" + self.search_title + "/model.png",
                                  show_shapes=True, expand_nested=True)

        return model
