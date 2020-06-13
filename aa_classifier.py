import tensorflow as tf
from kerastuner import HyperModel
import kerastuner as kt


class AuthorClassifier(HyperModel):
    def __init__(self, num_users, num_dense_inputs, num_rnn_inputs,
                 num_rnn_inputs_dimension, num_train_samples, search_title, model_name="model"):
        super().__init__()
        self.num_users = num_users
        self.num_rnn_inputs = num_rnn_inputs
        self.num_rnn_inputs_dimension = num_rnn_inputs_dimension
        self.num_dense_inputs = num_dense_inputs
        self.search_title = search_title
        self.model_name = model_name

    def get_tunable_hyper_parameters(self) -> kt.HyperParameters:
        hp = kt.HyperParameters()
        self.build(hp)
        return hp

    def build(self, hp):
        rnn_types = {
            'gru': tf.keras.layers.GRU,
            'lstm': tf.keras.layers.LSTM,
            'simple': tf.keras.layers.SimpleRNN
        }

        rnn_type = hp.Fixed('rnn_type', 'gru')  # hp.Choice('rnn_type', values=list(rnn_types.keys()))
        rnn_type = rnn_types[rnn_type]

        rnn_dimension = hp.Fixed('rnn_dimension', 200)  # hp.Int('rnn_dimension', min_value=150, max_value=250, step=50)

        hidden_layer = [hp.Fixed("rnn_hidden_layer_1_units", 64),
                        hp.Fixed("rnn_hidden_layer_2_units", 32)]

        # layer "branches" that then merge in the concatenate
        concats = []
        inputs = []

        if self.num_rnn_inputs > 0:
            rnn_input = tf.keras.Input(shape=(None, self.num_rnn_inputs_dimension), name="rnn_input_layer")
            inputs.append(rnn_input)

            masked = tf.keras.layers.Masking(name="rnn_masking_layer")(rnn_input)
            rnn = rnn_type(rnn_dimension, return_sequences=(len(hidden_layer) > 0), name="rnn_layer")(masked)

            dropout = hp.Fixed('dropout', 0.3)  # hp.Choice('dropout', values=[0.15, 0.3])

            if dropout > 0:
                rnn = tf.keras.layers.Dropout(dropout, name="rnn_initial_dropout")(rnn)

            for (i, hidden_neurons) in enumerate(hidden_layer):
                is_last_rnn_layer = i == len(hidden_layer) - 1
                rnn = rnn_type(hidden_neurons, return_sequences=not is_last_rnn_layer, name="rnn_hidden_layer_" + str(i))(rnn)

                if dropout > 0:
                    rnn = tf.keras.layers.Dropout(dropout, name="rnn_dropout_" + str(i))(rnn)

            concats.append(rnn)

        # dense inputs
        if self.num_dense_inputs > 0:
            dense_input = tf.keras.Input(shape=self.num_dense_inputs, name="dense_input_layer")
            inputs.append(dense_input)

            num_after_dense_input_layers = \
                hp.Int('num_after_dense_input_layers', min_value=0, max_value=2, step=1)
            num_after_dense_input_layer_neurons = \
                hp.Int('num_after_dense_input_layer_neurons', min_value=30, max_value=200, step=30)
            dense_dropout = hp.Choice('dense_dropout', [0.3, 0.5, 0.8])

            dense_output = dense_input
            for i in range(num_after_dense_input_layers):
                dense_output = tf.keras.layers.Dense(num_after_dense_input_layer_neurons, activation='relu',
                                                     name="dense_hidden_layer_" + str(i))(dense_output)
                if dense_dropout != 0 and i != (num_after_dense_input_layers - 1):
                    dense_output = tf.keras.layers.Dropout(dense_dropout, name="dense_dropout_" + str(i))(dense_output)

            concats.append(dense_output)

        # output and model creation
        if len(concats) > 1:
            concat = tf.keras.layers.concatenate(concats, axis=1, name="concatenation_layer")
        else:
            concat = concats[0]

        final_dense = concat
        final_dense_neurons = hp.Choice('neurons_final_dense_layers', values=[128, 256, 512])
        num_final_dense_layers = hp.Fixed("num_final_dense_layers", 1)  # hp.Int('num_final_dense_layers', min_value=0, max_value=3, step=1)
        for i in range(num_final_dense_layers):
            final_dense = tf.keras.layers.Dense(final_dense_neurons, activation='relu',
                                                name="concatenated_dense_layer_" + str(i))(final_dense)

        output = tf.keras.layers.Dense(self.num_users, activation='softmax', name="final_layer")(final_dense)

        model = tf.keras.Model(inputs=inputs,
                               outputs=[output],
                               name=self.model_name)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        tf.keras.utils.plot_model(model, "hyperparams/" + self.search_title + "/" + str(self.model_name) + ".png",
                                  show_shapes=True, expand_nested=True)

        return model
