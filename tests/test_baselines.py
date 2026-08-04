import numpy as np
import pandas as pd

from src.baselines import lag_calendar_features


def test_lag_calendar_feature_shape():
    inputs = np.zeros((3, 4, 2))
    dates = pd.date_range("2020-01-01", periods=20, freq="h")
    result = lag_calendar_features(inputs, dates, np.array([4, 5, 6]))
    assert result.shape == (3, 4 * 2 + 6)
