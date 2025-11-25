import tensorflow as tf

from src.models.lstm_model import create_lstm_model


class TestLSTMModel:
    def test_create_lstm_model_default(self):
        # We need a small input shape to quickly generate the model architecture
        input_shape = (10, 2)
        model = create_lstm_model(input_shape=input_shape)

        # Check if the returned object is a Keras Model
        assert isinstance(model, tf.keras.Model)

        # Default num_layers is 2
        # Note: the layers will be Input -> LSTM -> Dropout -> LSTM -> Dropout -> Dense
        # Number of layers in model.layers (Input is usually not counted in model.layers in Sequential)
        # Sequential adds layers. LSTM1, Drop1, LSTM2, Drop2, Dense => 5 layers

        # Check output shape
        # The output of dense is (batch_size, 1)
        assert model.output_shape == (None, 1)

        # Check optimizer
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
        assert model.loss == "mean_squared_error"

    def test_create_lstm_model_custom(self):
        input_shape = (5, 1)
        model = create_lstm_model(
            input_shape=input_shape, units=20, dropout_rate=0.5, learning_rate=0.01, num_layers=1
        )

        # layers: LSTM -> Dropout -> Dense (3 layers)
        assert model.output_shape == (None, 1)
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
