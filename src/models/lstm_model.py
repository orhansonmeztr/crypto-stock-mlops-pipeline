import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_lstm_model(input_shape, units=50, dropout_rate=0.2, learning_rate=0.001, num_layers=2):
    """
    Creates and compiles a multi-layer Keras LSTM model.

    Args:
        input_shape (tuple): (time_steps, features)
        units (int): Number of LSTM units per layer
        dropout_rate (float): Dropout rate
        learning_rate (float): Learning rate for Adam
        num_layers (int): Number of LSTM layers (default: 2)

    Returns:
        tf.keras.Model: Compiled model
    """
    layers = [Input(shape=input_shape)]

    for i in range(num_layers):
        # return_sequences=True for all layers except the last
        return_sequences = i < (num_layers - 1)
        layers.append(LSTM(units, return_sequences=return_sequences))
        layers.append(Dropout(dropout_rate))

    layers.append(Dense(1))

    model = Sequential(layers)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def prepare_lstm_data(df, look_back=60, target_col="close", feature_cols=None):
    """
    Prepares data for LSTM: Scales data and creates sequences.

    Args:
        df (pd.DataFrame): Input dataframe
        look_back (int): Number of past days to use
        target_col (str): Column to predict
        feature_cols (list): List of columns to use as input features

    Returns:
        X (np.array): 3D array [samples, look_back, features]
        y (np.array): Target array
        scaler (MinMaxScaler): Fitted scaler object (to inverse transform later)
    """
    if feature_cols is None:
        feature_cols = [target_col]

    # Scale data (LSTM works best with data between 0 and 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols])

    # Get target column index to separate it later
    target_idx = feature_cols.index(target_col)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back : i])
        y.append(scaled_data[i, target_idx])

    return np.array(X), np.array(y), scaler
