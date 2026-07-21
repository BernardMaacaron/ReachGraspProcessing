import numpy as np
import pandas as pd
import pytest

from reachgrasp.preprocessing import (
    bandpass_filter,
    detect_movement_onset,
    normalize_signal,
    notch_filter,
)


def test_filters_preserve_dataframe_shape_and_labels():
    time = np.arange(4000, dtype=float) / 2000.0
    data = pd.DataFrame({
        "a": np.sin(2 * np.pi * 40 * time) + 0.1 * np.sin(2 * np.pi * 50 * time),
        "b": np.sin(2 * np.pi * 80 * time),
    })

    filtered = bandpass_filter(data, lowcut=20, highcut=500, fs=2000)
    filtered = notch_filter(filtered, freq=50, fs=2000)

    assert isinstance(filtered, pd.DataFrame)
    assert filtered.shape == data.shape
    assert filtered.columns.tolist() == data.columns.tolist()
    assert filtered.index.equals(data.index)


def test_constant_normalization_returns_zero_not_nan():
    normalized = normalize_signal(np.ones(20), method="zscore")
    assert np.all(normalized == 0)
    assert np.all(np.isfinite(normalized))


def test_no_detected_movement_raises():
    with pytest.raises(ValueError, match="No movement detected"):
        detect_movement_onset(np.zeros(100))
