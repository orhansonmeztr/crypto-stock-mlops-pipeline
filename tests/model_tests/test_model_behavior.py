import numpy as np
import pandas as pd

from src.models.lstm_model import create_lstm_model, prepare_lstm_data


def test_prepare_lstm_data_shapes():
    """
    Verify that data preparation creates correct 3D sequences.
    """
    # Create dummy features data (e.g., 100 days, 5 features)
    data = pd.DataFrame(np.random.rand(100, 5), columns=["close", "rsi", "macd", "vol", "return"])

    look_back = 10
    target_col = "close"
    feature_cols = ["close", "rsi", "macd"]

    X, y, scaler = prepare_lstm_data(
        data, look_back=look_back, target_col=target_col, feature_cols=feature_cols
    )

    # Expected X shape: (Total - look_back, look_back, num_features)
    # 100 - 10 = 90 samples
    expected_samples = 90
    expected_features = len(feature_cols)

    assert X.shape == (expected_samples, look_back, expected_features)
    assert y.shape == (expected_samples,)

    # Check if scaling worked (min should be >= 0, max <= 1 approximately)
    assert X.min() >= 0.0
    assert X.max() <= 1.000001  # Float tolerance


def test_lstm_model_compilation():
    """
    Verify that the model compiles with correct input shape.
    """
    time_steps = 30
    features = 5
    input_shape = (time_steps, features)

    model = create_lstm_model(input_shape, units=10, dropout_rate=0.1)

    # Check model output shape (None, 1)
    assert model.output_shape == (None, 1)

    # Check if layer count is correct (Input -> LSTM -> Dropout -> Dense)
    # Note: Input layer is not always counted in model.layers depending on Keras version/style
    # But we expect at least LSTM, Dropout, Dense
    assert len(model.layers) >= 3


def test_model_prediction_flow():
    """
    Verify that the model can accept input and produce a prediction (Behavior Test).
    """
    # 1. Setup Data
    time_steps = 10
    features = 3
    input_shape = (time_steps, features)

    # 2. Create Model
    model = create_lstm_model(input_shape)

    # 3. Create Dummy Input (Batch Size = 1)
    # Shape: (1, 10, 3)
    dummy_input = np.random.random((1, time_steps, features)).astype(np.float32)

    # 4. Predict
    prediction = model.predict(dummy_input)

    # 5. Check Output
    assert prediction.shape == (1, 1)
    assert not np.isnan(prediction[0][0])
