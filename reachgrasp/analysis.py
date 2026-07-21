"""
Analysis utilities for ReachGrasp dataset.

Note: For basic correlation matrices, use pandas' built-in method directly:
    corr_matrix = joint_angles.corr()  # Pearson correlation
    corr_matrix = joint_angles.corr(method='spearman')  # Spearman correlation

This module provides higher-level analysis functions that go beyond simple correlations.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from typing import Tuple, Dict, List, Optional, Union


def find_highly_correlated_pairs(correlation_matrix: pd.DataFrame,
                                 threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """
    Find pairs of channels with high correlation.
    
    Parameters:
    - correlation_matrix: pd.DataFrame, correlation matrix
    - threshold: float, minimum absolute correlation value (default: 0.7)
    
    Returns:
    - list: [(channel1, channel2, correlation), ...]
    """
    pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_val
                ))
    
    # Sort by absolute correlation value
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return pairs


def calculate_cross_correlation(signal1: np.ndarray,
                               signal2: np.ndarray,
                               max_lag: int = 50) -> Tuple[np.ndarray, int, float]:
    """
    Calculate cross-correlation between two signals.
    
    Parameters:
    - signal1: np.ndarray, first signal
    - signal2: np.ndarray, second signal
    - max_lag: int, maximum lag to consider (in samples)
    
    Returns:
    - xcorr: np.ndarray, cross-correlation values
    - lag: int, lag at peak correlation (positive means signal1 leads)
    - peak_corr: float, peak correlation value
    """
    # Normalize signals
    sig1_norm = (signal1 - signal1.mean()) / signal1.std()
    sig2_norm = (signal2 - signal2.mean()) / signal2.std()
    
    # Calculate cross-correlation
    xcorr = signal.correlate(sig1_norm, sig2_norm, mode='full')
    xcorr = xcorr / len(signal1)
    
    # Find peak within max_lag
    center = len(xcorr) // 2
    start_idx = max(0, center - max_lag)
    end_idx = min(len(xcorr), center + max_lag + 1)
    
    search_region = xcorr[start_idx:end_idx]
    peak_idx = np.argmax(np.abs(search_region))
    
    lag = peak_idx + start_idx - center
    peak_corr = xcorr[peak_idx + start_idx]
    
    return xcorr, lag, peak_corr


def analyze_joint_coordination(joint_data: pd.DataFrame,
                              time: np.ndarray,
                              max_lag: int = 50,
                              min_correlation: float = 0.5) -> List[Dict]:
    """
    Analyze coordination between all joint pairs.
    
    Parameters:
    - joint_data: pd.DataFrame, joint angle data
    - time: np.ndarray, time values
    - max_lag: int, maximum time lag to consider
    - min_correlation: float, minimum correlation to report
    
    Returns:
    - list: [{
        'joint1': str,
        'joint2': str,
        'correlation': float,
        'lag': int,
        'lag_time': float
      }, ...]
    """
    results = []
    fs = 1.0 / np.mean(np.diff(time))  # Estimate sampling frequency
    
    for i in range(len(joint_data.columns)):
        for j in range(i + 1, len(joint_data.columns)):
            joint1 = joint_data.columns[i]
            joint2 = joint_data.columns[j]
            
            _, lag, peak_corr = calculate_cross_correlation(
                joint_data.iloc[:, i].values,
                joint_data.iloc[:, j].values,
                max_lag
            )
            
            if abs(peak_corr) >= min_correlation:
                results.append({
                    'joint1': joint1,
                    'joint2': joint2,
                    'correlation': peak_corr,
                    'lag': lag,
                    'lag_time': lag / fs
                })
    
    # Sort by correlation strength
    results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return results


def calculate_power_spectrum(data: np.ndarray,
                            fs: float,
                            nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate power spectral density using Welch's method.
    
    Parameters:
    - data: np.ndarray, input signal (1D or 2D)
    - fs: float, sampling frequency (Hz)
    - nperseg: int, segment length for Welch's method (default: fs)
    
    Returns:
    - frequencies: np.ndarray, frequency values
    - psd: np.ndarray, power spectral density
    """
    if nperseg is None:
        nperseg = int(fs)
    
    if data.ndim == 1:
        frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    else:
        # Calculate for first channel as example
        frequencies, psd = signal.welch(data[:, 0], fs=fs, nperseg=nperseg)
    
    return frequencies, psd


def calculate_median_frequency(data: np.ndarray, fs: float) -> float:
    """
    Calculate median frequency of power spectrum (useful for EMG fatigue).
    
    Parameters:
    - data: np.ndarray, input signal
    - fs: float, sampling frequency (Hz)
    
    Returns:
    - median_freq: float, median frequency (Hz)
    """
    frequencies, psd = calculate_power_spectrum(data, fs)
    
    # Calculate cumulative power
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    
    # Find frequency where cumulative power reaches 50%
    median_idx = np.where(cumulative_power >= 0.5 * total_power)[0][0]
    median_freq = frequencies[median_idx]
    
    return median_freq


def calculate_mean_frequency(data: np.ndarray, fs: float) -> float:
    """
    Calculate mean frequency of power spectrum.
    
    Parameters:
    - data: np.ndarray, input signal
    - fs: float, sampling frequency (Hz)
    
    Returns:
    - mean_freq: float, mean frequency (Hz)
    """
    frequencies, psd = calculate_power_spectrum(data, fs)
    mean_freq = np.sum(frequencies * psd) / np.sum(psd)
    return mean_freq


def calculate_sample_entropy(data: np.ndarray,
                            m: int = 2,
                            r: Optional[float] = None) -> float:
    """
    Calculate Sample Entropy as measure of signal complexity.
    
    Parameters:
    - data: np.ndarray, input signal (1D)
    - m: int, embedding dimension (default: 2)
    - r: float, tolerance (default: 0.2 * std)
    
    Returns:
    - entropy: float, sample entropy value
    """
    if r is None:
        r = 0.2 * np.std(data)
    
    N = len(data)
    
    def _max_dist(x_i, x_j, m):
        return max([abs(x_i[k] - x_j[k]) for k in range(m)])
    
    def _phi(m):
        patterns = np.array([[data[i + j] for j in range(m)] for i in range(N - m)])
        count = 0
        for i in range(len(patterns)):
            for j in range(len(patterns)):
                if i != j:
                    if _max_dist(patterns[i], patterns[j], m) < r:
                        count += 1
        return count / (len(patterns) * (len(patterns) - 1))
    
    phi_m = _phi(m)
    phi_m_plus_1 = _phi(m + 1)
    
    if phi_m == 0 or phi_m_plus_1 == 0:
        return np.inf
    
    return -np.log(phi_m_plus_1 / phi_m)


def calculate_channel_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for all channels.
    
    Parameters:
    - data: pd.DataFrame, multi-channel data
    
    Returns:
    - pd.DataFrame: Statistics (mean, std, min, max, range, RMS, etc.)
    """
    stats = pd.DataFrame({
        'Mean': data.mean(),
        'Std': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'Range': data.max() - data.min(),
        'RMS': np.sqrt((data ** 2).mean()),
        'Median': data.median(),
        'Q25': data.quantile(0.25),
        'Q75': data.quantile(0.75),
        'IQR': data.quantile(0.75) - data.quantile(0.25)
    })
    
    return stats


def detect_peaks(data: np.ndarray,
                height: Optional[float] = None,
                distance: Optional[int] = None,
                prominence: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """
    Detect peaks in signal.
    
    Parameters:
    - data: np.ndarray, input signal (1D)
    - height: float, minimum peak height
    - distance: int, minimum distance between peaks (samples)
    - prominence: float, minimum peak prominence
    
    Returns:
    - peak_indices: np.ndarray, indices of peaks
    - properties: dict, peak properties
    """
    peak_indices, properties = signal.find_peaks(
        data,
        height=height,
        distance=distance,
        prominence=prominence
    )
    
    return peak_indices, properties


def calculate_time_to_peak(data: np.ndarray, time: np.ndarray) -> Tuple[float, float]:
    """
    Calculate time to peak value and the peak value itself.
    
    Parameters:
    - data: np.ndarray, input signal (1D)
    - time: np.ndarray, time values
    
    Returns:
    - time_to_peak: float, time when peak occurs
    - peak_value: float, peak value
    """
    peak_idx = np.argmax(np.abs(data))
    time_to_peak = time[peak_idx]
    peak_value = data[peak_idx]
    
    return time_to_peak, peak_value


def segment_by_anatomical_region(joint_data: pd.DataFrame,
                                 joint_groups: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Segment joint data by anatomical regions.
    
    Parameters:
    - joint_data: pd.DataFrame, joint angle data with named columns
    - joint_groups: dict, anatomical grouping {region: [joint_names]}
    
    Returns:
    - dict: {region_name: pd.DataFrame}
    """
    segmented = {}
    
    for region_name, joint_list in joint_groups.items():
        # Get joints that exist in the data
        available_joints = [j for j in joint_list if j in joint_data.columns]
        if available_joints:
            segmented[region_name] = joint_data[available_joints]
    
    return segmented


def calculate_coordination_index(joint_data: pd.DataFrame) -> float:
    """
    Calculate overall coordination index based on average correlation.
    
    Parameters:
    - joint_data: pd.DataFrame, joint angle data
    
    Returns:
    - coordination_index: float, average absolute correlation
    """
    corr_matrix = joint_data.corr()  # Use pandas built-in
    
    # Get upper triangle (excluding diagonal)
    upper_triangle = np.triu(corr_matrix.values, k=1)
    
    # Calculate mean of absolute correlations
    non_zero = upper_triangle[upper_triangle != 0]
    coordination_index = np.mean(np.abs(non_zero))
    
    return coordination_index


def compare_tasks(data_dict: Dict[str, pd.DataFrame],
                 metric: str = 'range') -> pd.DataFrame:
    """
    Compare statistics across different tasks.
    
    Parameters:
    - data_dict: dict, {task_name: pd.DataFrame}
    - metric: str, metric to compare ('mean', 'std', 'range', 'rms')
    
    Returns:
    - pd.DataFrame: Comparison table (channels x tasks)
    """
    results = {}
    
    for task_name, data in data_dict.items():
        if metric == 'mean':
            results[task_name] = data.mean()
        elif metric == 'std':
            results[task_name] = data.std()
        elif metric == 'range':
            results[task_name] = data.max() - data.min()
        elif metric == 'rms':
            results[task_name] = np.sqrt((data ** 2).mean())
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return pd.DataFrame(results)


def extract_movement_window(time: np.ndarray,
                           data: pd.DataFrame,
                           start_time: float,
                           end_time: float) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract a specific time window from the data.
    
    Parameters:
    - time: np.ndarray, time values
    - data: pd.DataFrame, signal data
    - start_time: float, start time (seconds)
    - end_time: float, end time (seconds)
    
    Returns:
    - time_window: np.ndarray, time values in window
    - data_window: pd.DataFrame, data in window
    """
    mask = (time >= start_time) & (time <= end_time)
    time_window = time[mask]
    data_window = data.iloc[mask]
    
    return time_window, data_window
