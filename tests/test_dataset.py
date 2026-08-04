import numpy as np

from data.dataset import Standardizer, temporal_boundaries, window_starts


def test_scaler_is_fit_on_train_only():
    train = np.array([[0.0], [2.0]])
    future = np.array([[1000.0]])
    scaler = Standardizer.fit(train)
    assert scaler.mean.item() == 1.0
    assert scaler.transform(future).item() == 999.0


def test_targets_do_not_cross_split_boundaries():
    n, input_len, horizon = 100, 10, 5
    train_end, val_end = temporal_boundaries(n, [0.7, 0.1, 0.2])
    starts = window_starts(n, input_len, horizon, train_end, val_end)
    assert np.all(starts["train"] + input_len + horizon <= train_end)
    assert np.all(starts["val"] + input_len >= train_end)
    assert np.all(starts["val"] + input_len + horizon <= val_end)
    assert np.all(starts["test"] + input_len >= val_end)

