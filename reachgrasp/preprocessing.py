"""Signal preprocessing utilities for the ReachGrasp dataset.

The functions in this module operate on either NumPy arrays or pandas
DataFrames. Time-series samples are expected along the first axis and channels
along the second axis. One-dimensional NumPy arrays are treated as single
signals.

Filtering uses zero-phase forward-backward filtering. Butterworth filters are
represented as second-order sections (SOS), which are generally more
numerically stable than transfer-function coefficients for moderate- and
high-order filters. Functions preserve DataFrame columns and indices whenever
the number of samples is unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d


SignalData = Union[np.ndarray, pd.DataFrame]
SmoothnessResult = Union[float, np.ndarray, pd.Series]


def _as_float_array(data: SignalData, name: str = "data") -> np.ndarray:
    """Convert supported signal data to a finite floating-point array.

    Parameters
    ----------
    data:
        One-dimensional signal data or a two-dimensional sample-by-channel
        matrix. pandas DataFrames are converted without modifying the input.
    name:
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        A floating-point array with one or two dimensions.

    Raises
    ------
    ValueError
        If the input is empty, has more than two dimensions, or contains NaN
        or infinite values.
    TypeError
        If the input cannot be converted to floating-point values.
    """

    try:
        values = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if values.ndim not in (1, 2):
        raise ValueError(f"{name} must be one- or two-dimensional, got shape {values.shape}.")
    if values.size == 0 or values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return values


def _validate_time(time: np.ndarray, expected_length: Optional[int] = None,
                   min_samples: int = 2, name: str = "time") -> np.ndarray:
    """Validate and return a strictly increasing one-dimensional time vector.

    Parameters
    ----------
    time:
        Sample times.
    expected_length:
        Required number of samples. No length check is performed when this is
        ``None``.
    min_samples:
        Minimum accepted number of time samples.
    name:
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        A one-dimensional floating-point time vector.

    Raises
    ------
    ValueError
        If the vector has the wrong shape or length, contains non-finite
        values, or is not strictly increasing.
    """

    time_values = np.asarray(time, dtype=float)

    if time_values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {time_values.shape}.")
    if time_values.size < min_samples:
        raise ValueError(f"{name} must contain at least {min_samples} samples.")
    if expected_length is not None and time_values.size != expected_length:
        raise ValueError(
            f"{name} and data must contain the same number of samples: "
            f"{time_values.size} != {expected_length}."
        )
    if not np.all(np.isfinite(time_values)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if time_values.size > 1 and np.any(np.diff(time_values) <= 0):
        raise ValueError(f"{name} must be strictly increasing with no duplicate samples.")

    return time_values


def _validate_filter_design(fs: float, order: int) -> float:
    """Validate common digital-filter parameters and return the Nyquist rate."""

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive finite sampling frequency, got {fs!r}.")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}.")

    return float(fs) / 2.0


def _apply_filter(data: SignalData, sos: np.ndarray) -> SignalData:
    """Apply a zero-phase SOS filter while preserving the input container type.

    The filter is applied once forward and once backward with
    :func:`scipy.signal.sosfiltfilt`. This removes phase delay and doubles the
    effective filter order. For two-dimensional inputs, filtering is performed
    independently down each column, with samples on axis 0.

    Parameters
    ----------
    data:
        A one-dimensional NumPy signal, a two-dimensional NumPy array shaped
        ``(n_samples, n_channels)``, or a DataFrame with the same orientation.
    sos:
        Second-order-section coefficients shaped ``(n_sections, 6)``, normally
        produced by ``scipy.signal.butter(..., output="sos")`` or
        :func:`scipy.signal.tf2sos`.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Filtered data with the same shape and container type as ``data``.
        DataFrame columns and index are preserved.

    Raises
    ------
    ValueError
        If the input or SOS matrix is invalid, or if the signal is too short
        for the padding required by ``sosfiltfilt``.
    """

    values = _as_float_array(data)
    sos_values = np.asarray(sos, dtype=float)

    if sos_values.ndim != 2 or sos_values.shape[1] != 6 or sos_values.shape[0] == 0:
        raise ValueError(
            "sos must have shape (n_sections, 6), as returned by a filter "
            f"design function with output='sos'; got {sos_values.shape}."
        )
    if not np.all(np.isfinite(sos_values)):
        raise ValueError("sos contains NaN or infinite values.")

    axis = 0 if values.ndim == 2 else -1
    try:
        filtered = signal.sosfiltfilt(sos_values, values, axis=axis)
    except ValueError as exc:
        raise ValueError(
            "Filtering failed. The signal may be too short for the filter's "
            "forward-backward padding requirement."
        ) from exc

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(filtered, columns=data.columns, index=data.index)
    return filtered


def bandpass_filter(data: SignalData, lowcut: float, highcut: float, fs: float,
                    order: int = 4) -> SignalData:
    """Apply a zero-phase Butterworth band-pass filter.

    The filter is designed directly in hertz and represented as second-order
    sections. It is then applied forward and backward, resulting in zero phase
    shift and an effective order of twice ``order``.

    Parameters
    ----------
    data:
        Signal data. Samples must occupy axis 0 and channels axis 1.
    lowcut:
        Lower pass-band edge in hertz. It must be greater than zero.
    highcut:
        Upper pass-band edge in hertz. It must be greater than ``lowcut`` and
        below the Nyquist frequency ``fs / 2``.
    fs:
        Sampling frequency in hertz.
    order:
        Butterworth design order. The forward-backward application doubles the
        effective order. The default is 4.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Filtered data with the same shape and container type as the input.

    Examples
    --------
    Filter surface EMG sampled at 2000 Hz:

    >>> filtered = bandpass_filter(emg_data, lowcut=20, highcut=500, fs=2000)
    """

    nyquist = _validate_filter_design(fs, order)
    if not np.isfinite(lowcut) or not np.isfinite(highcut):
        raise ValueError("lowcut and highcut must be finite.")
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            "Band-pass cutoffs must satisfy 0 < lowcut < highcut < fs / 2; "
            f"got lowcut={lowcut}, highcut={highcut}, fs={fs}."
        )

    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    return _apply_filter(data, sos)


def lowpass_filter(data: SignalData, cutoff: float, fs: float,
                   order: int = 4) -> SignalData:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    data:
        Signal data with samples on axis 0.
    cutoff:
        Cutoff frequency in hertz. It must lie strictly between zero and the
        Nyquist frequency ``fs / 2``.
    fs:
        Sampling frequency in hertz.
    order:
        Butterworth design order. Forward-backward filtering doubles the
        effective order. The default is 4.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Filtered data with the same shape and container type as the input.
    """

    nyquist = _validate_filter_design(fs, order)
    if not np.isfinite(cutoff) or not 0 < cutoff < nyquist:
        raise ValueError(f"cutoff must satisfy 0 < cutoff < fs / 2; got cutoff={cutoff}, fs={fs}.")

    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return _apply_filter(data, sos)


def highpass_filter(data: SignalData, cutoff: float, fs: float,
                    order: int = 4) -> SignalData:
    """Apply a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    data:
        Signal data with samples on axis 0.
    cutoff:
        Cutoff frequency in hertz. It must lie strictly between zero and the
        Nyquist frequency ``fs / 2``.
    fs:
        Sampling frequency in hertz.
    order:
        Butterworth design order. Forward-backward filtering doubles the
        effective order. The default is 4.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Filtered data with the same shape and container type as the input.
    """

    nyquist = _validate_filter_design(fs, order)
    if not np.isfinite(cutoff) or not 0 < cutoff < nyquist:
        raise ValueError(f"cutoff must satisfy 0 < cutoff < fs / 2; got cutoff={cutoff}, fs={fs}.")

    sos = signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return _apply_filter(data, sos)


def notch_filter(data: SignalData, freq: float, fs: float,
                 quality: float = 30.0) -> SignalData:
    """Apply a zero-phase IIR notch filter at a selected frequency.

    ``scipy.signal.iirnotch`` produces transfer-function coefficients ``(b, a)``.
    They are explicitly converted to second-order sections before filtering so
    that this function uses the same numerically stable SOS pipeline as the
    Butterworth filters.

    Parameters
    ----------
    data:
        Signal data with samples on axis 0.
    freq:
        Frequency to suppress in hertz, commonly 50 or 60 Hz for power-line
        interference. It must be below ``fs / 2``.
    fs:
        Sampling frequency in hertz.
    quality:
        Dimensionless quality factor ``Q``. Larger values produce a narrower
        rejection band. It must be positive. The default is 30.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Filtered data with the same shape and container type as the input.
    """

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive finite sampling frequency, got {fs!r}.")
    if not np.isfinite(freq) or not 0 < freq < fs / 2:
        raise ValueError(f"freq must satisfy 0 < freq < fs / 2; got freq={freq}, fs={fs}.")
    if not np.isfinite(quality) or quality <= 0:
        raise ValueError(f"quality must be positive and finite, got {quality!r}.")

    b, a = signal.iirnotch(freq, quality, fs=fs)
    sos = signal.tf2sos(b, a)
    return _apply_filter(data, sos)


def resample_signal(time: np.ndarray, data: SignalData, target_time: np.ndarray,
                    method: str = "linear") -> SignalData:
    """Interpolate a signal onto a new time base.

    Interpolation is applied independently to every channel. Target times
    outside the original interval are extrapolated, matching the behavior of
    the original ReachGrasp utility. Callers should avoid extrapolation when it
    does not have a physically meaningful interpretation.

    Parameters
    ----------
    time:
        Strictly increasing source timestamps shaped ``(n_samples,)``.
    data:
        Source signal values. Its first dimension must match ``time``.
    target_time:
        One-dimensional timestamps at which the signal should be evaluated.
        They do not need to use the same sampling frequency as ``time``.
    method:
        Interpolation method accepted by :class:`scipy.interpolate.interp1d`,
        such as ``"linear"``, ``"nearest"``, or ``"cubic"``. The default is
        ``"linear"``.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Resampled values with ``len(target_time)`` rows. DataFrame columns are
        preserved; the returned DataFrame receives a new RangeIndex because
        its sample count and time base have changed.

    Examples
    --------
    Downsample EMG to motion-capture timestamps:

    >>> emg_sync = resample_signal(emg_time, emg_data, motion_time)
    """

    values = _as_float_array(data)
    source_time = _validate_time(time, expected_length=values.shape[0])
    target_values = np.asarray(target_time, dtype=float)

    if target_values.ndim != 1:
        raise ValueError(f"target_time must be one-dimensional, got shape {target_values.shape}.")
    if target_values.size == 0:
        raise ValueError("target_time must contain at least one sample.")
    if not np.all(np.isfinite(target_values)):
        raise ValueError("target_time contains NaN or infinite values.")
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a non-empty interpolation method name.")

    try:
        interpolator = interp1d(
            source_time, values, kind=method, axis=0, bounds_error=False,
            fill_value="extrapolate", assume_sorted=True
        )
        resampled = np.asarray(interpolator(target_values), dtype=float)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"Could not resample data using interpolation method {method!r}.") from exc

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(resampled, columns=data.columns)
    return resampled


def calculate_rms(data: SignalData, window_size: int,
                  step_size: Optional[int] = None) -> SignalData:
    """Calculate root-mean-square amplitude in sliding windows.

    RMS is computed independently for every channel. Only complete windows are
    included; samples after the final complete window are ignored. This is
    commonly used to estimate the envelope or local amplitude of EMG signals.

    Parameters
    ----------
    data:
        One-dimensional signal data or a sample-by-channel matrix.
    window_size:
        Number of samples in each RMS window. It must be a positive integer no
        larger than the signal length.
    step_size:
        Number of samples between successive window starts. When omitted, it
        defaults to ``window_size``, producing non-overlapping windows.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        RMS values. The output has one row per complete window. DataFrame
        columns are preserved and the result uses a new RangeIndex.

    Examples
    --------
    Calculate 100 ms RMS windows with 50% overlap for EMG sampled at 2000 Hz:

    >>> emg_rms = calculate_rms(emg_data, window_size=200, step_size=100)
    """

    values = _as_float_array(data)

    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError(f"window_size must be a positive integer, got {window_size!r}.")
    if step_size is None:
        step_size = int(window_size)
    if isinstance(step_size, bool) or not isinstance(step_size, (int, np.integer)) or step_size < 1:
        raise ValueError(f"step_size must be a positive integer, got {step_size!r}.")
    if window_size > values.shape[0]:
        raise ValueError(
            f"window_size ({window_size}) cannot exceed the number of samples ({values.shape[0]})."
        )

    if values.ndim == 1:
        rms_values = _calculate_rms_1d(values, int(window_size), int(step_size))
    else:
        rms_values = np.column_stack([
            _calculate_rms_1d(values[:, index], int(window_size), int(step_size))
            for index in range(values.shape[1])
        ])

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(rms_values, columns=data.columns)
    return rms_values


def _calculate_rms_1d(signal_1d: np.ndarray, window_size: int,
                      step_size: int) -> np.ndarray:
    """Calculate sliding-window RMS for one validated one-dimensional signal."""

    n_windows = 1 + (signal_1d.size - window_size) // step_size
    rms = np.empty(n_windows, dtype=float)

    for index in range(n_windows):
        start = index * step_size
        window = signal_1d[start:start + window_size]
        rms[index] = np.sqrt(np.mean(np.square(window, dtype=float)))

    return rms


def calculate_velocity(time: np.ndarray, data: SignalData) -> SignalData:
    """Estimate the first time derivative of one or more signals.

    The derivative is computed with :func:`numpy.gradient`, which supports
    non-uniformly spaced timestamps. DataFrame results preserve the original
    index and append ``"_vel"`` to every column name.

    Parameters
    ----------
    time:
        Strictly increasing timestamps in seconds.
    data:
        Position, angle, or other time-varying values whose first dimension
        matches ``time``.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Estimated derivative with the same shape and container type as
        ``data``.
    """

    return _calculate_derivative(time, data, order=1, suffix="_vel")


def calculate_acceleration(time: np.ndarray, data: SignalData) -> SignalData:
    """Estimate the second time derivative of one or more signals.

    Acceleration is obtained by applying :func:`numpy.gradient` twice using the
    provided timestamps. DataFrame results preserve the original index and
    append ``"_acc"`` to every column name.

    Parameters
    ----------
    time:
        Strictly increasing timestamps in seconds.
    data:
        Position or angle values whose first dimension matches ``time``.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Estimated second derivative with the same shape and container type as
        ``data``.
    """

    return _calculate_derivative(time, data, order=2, suffix="_acc")


def _calculate_derivative(time: np.ndarray, data: SignalData, order: int = 1,
                          suffix: str = "") -> SignalData:
    """Calculate a numerical derivative while preserving the input type.

    Parameters
    ----------
    time:
        Strictly increasing timestamps.
    data:
        Signal values with samples along axis 0.
    order:
        Positive derivative order. At least ``order + 1`` samples are
        required.
    suffix:
        Suffix appended to DataFrame column labels.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Numerical derivative with the same shape as ``data``.
    """

    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}.")

    values = _as_float_array(data)
    time_values = _validate_time(
        time, expected_length=values.shape[0], min_samples=int(order) + 1
    )
    edge_order = 2 if time_values.size >= 3 else 1
    derivative = values.copy()

    for _ in range(int(order)):
        derivative = np.gradient(derivative, time_values, axis=0, edge_order=edge_order)

    if isinstance(data, pd.DataFrame):
        columns = [f"{column}{suffix}" if suffix else column for column in data.columns]
        return pd.DataFrame(derivative, columns=columns, index=data.index)
    return derivative


def normalize_signal(data: SignalData, method: str = "minmax") -> SignalData:
    """Normalize each signal independently using min-max or z-score scaling.

    For two-dimensional input, every column is normalized independently across
    samples. Constant channels have no defined scale; this implementation maps
    them to zero instead of returning NaN or infinite values.

    Parameters
    ----------
    data:
        One-dimensional signal data or a sample-by-channel matrix.
    method:
        ``"minmax"`` maps each signal to the interval ``[0, 1]``.
        ``"zscore"`` subtracts the mean and divides by the population standard
        deviation (``ddof=0``).

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Floating-point normalized data with the same shape and container type
        as the input. DataFrame index and columns are preserved.
    """

    values = _as_float_array(data)
    if not isinstance(method, str):
        raise ValueError("method must be either 'minmax' or 'zscore'.")

    normalized_method = method.lower()
    if normalized_method not in {"minmax", "zscore"}:
        raise ValueError(f"Unknown method {method!r}. Use 'minmax' or 'zscore'.")

    axis = None if values.ndim == 1 else 0
    if normalized_method == "minmax":
        minimum = np.min(values, axis=axis, keepdims=values.ndim == 2)
        scale = np.max(values, axis=axis, keepdims=values.ndim == 2) - minimum
        numerator = values - minimum
    else:
        mean = np.mean(values, axis=axis, keepdims=values.ndim == 2)
        scale = np.std(values, axis=axis, ddof=0, keepdims=values.ndim == 2)
        numerator = values - mean

    normalized = np.divide(
        numerator, scale, out=np.zeros_like(values, dtype=float), where=scale != 0
    )

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(normalized, columns=data.columns, index=data.index)
    return normalized


def detect_movement_onset(velocity: SignalData,
                          threshold_factor: float = 0.5) -> Tuple[int, int]:
    """Detect movement onset and offset from a velocity-magnitude threshold.

    For multichannel data, the sample-wise magnitude is defined as the maximum
    absolute velocity across channels. The adaptive threshold is
    ``threshold_factor * std(velocity_magnitude)``. The first and last samples
    above this threshold are returned.

    Parameters
    ----------
    velocity:
        One-dimensional velocity or a sample-by-channel velocity matrix.
    threshold_factor:
        Non-negative multiplier applied to the standard deviation of the
        velocity magnitude. Larger values require stronger activity to mark a
        sample as movement.

    Returns
    -------
    tuple of int
        ``(onset_index, offset_index)`` for the first and last samples whose
        velocity magnitude exceeds the adaptive threshold.

    Raises
    ------
    ValueError
        If ``threshold_factor`` is invalid or no sample exceeds the computed
        movement threshold.

    Notes
    -----
    This is a simple global-threshold detector. It does not enforce a minimum
    movement duration or suppress isolated threshold crossings.
    """

    values = _as_float_array(velocity, name="velocity")
    if not np.isfinite(threshold_factor) or threshold_factor < 0:
        raise ValueError(
            f"threshold_factor must be non-negative and finite, got {threshold_factor!r}."
        )

    magnitude = np.abs(values) if values.ndim == 1 else np.max(np.abs(values), axis=1)
    threshold = float(np.std(magnitude, ddof=0) * threshold_factor)
    active_indices = np.flatnonzero(magnitude > threshold)

    if active_indices.size == 0:
        raise ValueError(
            "No movement detected: velocity never exceeded the adaptive threshold "
            f"of {threshold:.6g}."
        )

    return int(active_indices[0]), int(active_indices[-1])


def segment_movement_phases(time: np.ndarray, velocity: SignalData,
                            threshold_factor: float = 0.5) -> dict[str, Tuple[int, int]]:
    """Describe rest and movement intervals using detected boundary indices.

    Parameters
    ----------
    time:
        Strictly increasing timestamps with one entry per velocity sample.
        The values are validated but the returned phases are expressed as
        indices.
    velocity:
        One-dimensional velocity or a sample-by-channel velocity matrix.
    threshold_factor:
        Threshold multiplier forwarded to :func:`detect_movement_onset`.

    Returns
    -------
    dict
        Dictionary containing ``"rest_before"``, ``"movement"``, and
        ``"rest_after"`` index pairs. The pairs preserve the historical
        ReachGrasp convention and therefore share their onset and offset
        boundary samples.
    """

    values = _as_float_array(velocity, name="velocity")
    _validate_time(time, expected_length=values.shape[0], min_samples=1)
    onset_idx, offset_idx = detect_movement_onset(values, threshold_factor)

    return {
        "rest_before": (0, onset_idx),
        "movement": (onset_idx, offset_idx),
        "rest_after": (offset_idx, values.shape[0] - 1),
    }


def calculate_movement_smoothness(time: np.ndarray,
                                  data: SignalData) -> SmoothnessResult:
    """Calculate a root-mean-square jerk measure for movement smoothness.

    Jerk is the third derivative of position or angle with respect to time.
    This function computes jerk numerically and returns its root-mean-square
    value over the complete trajectory. Lower values indicate less rapid change
    in acceleration and therefore generally smoother motion.

    Despite the historical function name and earlier documentation, this is
    **not** a dimensionless normalized-jerk metric. Its units depend on the
    input: for joint angles measured in degrees and time in seconds, the result
    is expressed in degrees per second cubed.

    Parameters
    ----------
    time:
        Strictly increasing timestamps. At least four samples are required.
    data:
        Position or joint-angle data with samples on axis 0.

    Returns
    -------
    float, numpy.ndarray, or pandas.Series
        A scalar for one-dimensional data, one value per column for a
        two-dimensional NumPy array, or a Series indexed by the original
        DataFrame columns.
    """

    values = _as_float_array(data)
    time_values = _validate_time(time, expected_length=values.shape[0], min_samples=4)
    edge_order = 2 if time_values.size >= 3 else 1
    jerk = values.copy()

    for _ in range(3):
        jerk = np.gradient(jerk, time_values, axis=0, edge_order=edge_order)

    if values.ndim == 1:
        return float(np.sqrt(np.mean(np.square(jerk, dtype=float))))

    smoothness = np.sqrt(np.mean(np.square(jerk, dtype=float), axis=0))
    if isinstance(data, pd.DataFrame):
        return pd.Series(smoothness, index=data.columns, name="rms_jerk")
    return smoothness
