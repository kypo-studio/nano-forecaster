import numpy as np

from src.evaluate import metrics


def test_mape_excludes_zero_targets():
    result = metrics(np.array([[0.0, 2.0]]), np.array([[100.0, 1.0]]))
    assert result["MAPE"] == 50.0

