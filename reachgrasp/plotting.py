"""
Utility functions for plotting ReachGrasp dataset data.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Optional, List, Union
import pandas as pd


def load_channel_mappings():
    """
    Load channel mappings from JSON file.
    
    Returns:
    - dict: Channel mappings dictionary
    """
    mapping_file = os.path.join(os.path.dirname(__file__), 'channel_mappings.json')
    with open(mapping_file, 'r') as f:
        return json.load(f)


def get_channel_labels(modality: str, acquisition: str, n_channels: int) -> List[str]:
    """
    Get channel labels for a specific modality and acquisition type.
    
    Parameters:
    - modality: str, 'emg', 'motion', or 'tactile'
    - acquisition: str, acquisition type (e.g., 'cometa', 'vicon', 'tactileglove')
    - n_channels: int, number of channels
    
    Returns:
    - list: Channel labels
    """
    mappings = load_channel_mappings()
    
    if modality not in mappings or acquisition not in mappings[modality]:
        return [f'Channel {i+1}' for i in range(n_channels)]
    
    mapping = mappings[modality][acquisition]
    labels = []
    
    for i in range(n_channels):
        channel_key = str(i + 1)
        if channel_key in mapping:
            labels.append(f'Ch{i+1}: {mapping[channel_key]}')
        else:
            # Handle range mappings (like for sessantaquattro)
            found = False
            for range_key, label in mapping.items():
                if '-' in range_key:
                    start, end = map(int, range_key.split('-'))
                    if start - 1 <= i <= end - 1:  # Convert to 0-based
                        labels.append(f'Ch{i+1}: {label}')
                        found = True
                        break
            if not found:
                labels.append(f'Channel {i+1}')
    
    return labels


def _plot_channels_generic(time: np.ndarray,
                          data: Union[pd.DataFrame, np.ndarray],
                          channel_names: Optional[List[str]] = None,
                          title: str = 'Channels',
                          ylabel: str = 'Amplitude',
                          figsize: tuple = (12, 8),
                          overlapping: bool = False,
                          show_cuts: bool = False,
                          cut_indices: Optional[List[int]] = None) -> None:
    """
    Internal helper: Generic multi-channel plotting function.
    
    Parameters:
    - time: time values
    - data: channel data (channels as columns)
    - channel_names: optional channel labels
    - title: plot title
    - ylabel: y-axis label
    - figsize: figure size
    - overlapping: if True, plot all channels on same axes
    - show_cuts: if True, add vertical lines at repetition boundaries
    - cut_indices: list of cut indices from timeCuts.json [start_idx1, end_idx1, ...]
                   Will be converted to time values using the time array
    """
    if isinstance(data, np.ndarray):
        n_channels = data.shape[1] if data.ndim > 1 else 1
    else:
        n_channels = data.shape[1]
    
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(n_channels)]
    
    # Convert cut indices to time values
    time_cuts = None
    if show_cuts and cut_indices is not None:
        time_cuts = [time[idx] for idx in cut_indices if idx < len(time)]
    
    if overlapping:
        plt.figure(figsize=figsize)
        for i in range(n_channels):
            label = channel_names[i] if i < len(channel_names) else f'Channel {i+1}'
            channel_data = data.iloc[:, i] if isinstance(data, pd.DataFrame) else data[:, i]
            plt.plot(time, channel_data, alpha=0.7, label=label)
        
        # Add cut markers if requested
        if time_cuts is not None:
            for i, cut in enumerate(time_cuts):
                color = 'green' if i % 2 == 0 else 'red'  # Green for start, red for end
                linestyle = '--' if i % 2 == 0 else ':'
                label = 'Rep Start' if i % 2 == 0 and i == 0 else ('Rep End' if i == 1 else None)
                plt.axvline(x=cut, color=color, linestyle=linestyle, alpha=0.6, linewidth=1.5, label=label)
        
        plt.title(f'{title} Overlapping')
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        # Grid layout
        n_cols = min(5, n_channels)
        n_rows = int(np.ceil(n_channels / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)
        
        for i in range(n_channels):
            row, col = divmod(i, n_cols)
            label = channel_names[i] if i < len(channel_names) else f'Channel {i+1}'
            channel_data = data.iloc[:, i] if isinstance(data, pd.DataFrame) else data[:, i]
            
            axes[row, col].plot(time, channel_data)
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].set_ylabel(ylabel)
            
            # Add cut markers if requested
            if time_cuts is not None:
                for j, cut in enumerate(time_cuts):
                    color = 'green' if j % 2 == 0 else 'red'
                    linestyle = '--' if j % 2 == 0 else ':'
                    axes[row, col].axvline(x=cut, color=color, linestyle=linestyle, 
                                          alpha=0.6, linewidth=1.5)
        
        # Hide empty subplots
        for i in range(n_channels, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
    
    plt.show()


def plot_emg_channels(time: np.ndarray,
                     emg_data: Union[pd.DataFrame, np.ndarray],
                     channel_names: Optional[List[str]] = None,
                     figsize: tuple = (12, 8),
                     show_labels: bool = False,
                     acquisition: str = 'cometa',
                     overlapping: bool = False,
                     show_cuts: bool = False,
                     cut_indices: Optional[List[int]] = None) -> None:
    """
    Plot EMG channels over time in a grid layout or overlapping.

    Parameters:
    - time: array-like, time values
    - emg_data: DataFrame or array, EMG channel data (channels as columns)
    - channel_names: list of strings, names for each channel (optional)
    - figsize: tuple, figure size (width, height)
    - show_labels: bool, whether to show anatomical labels for channels
    - acquisition: str, acquisition system ('cometa' or 'sessantaquattro')
    - overlapping: bool, whether to plot all channels on the same axes (default: False)
    - show_cuts: bool, whether to show repetition boundary markers (default: False)
    - cut_indices: list of cut indices from timeCuts.json [start_idx1, end_idx1, ...] (optional)
    """
    if show_labels and channel_names is None:
        n_channels = emg_data.shape[1]
        channel_names = get_channel_labels('emg', acquisition, n_channels)
    
    _plot_channels_generic(time, emg_data, channel_names, 'EMG Channels', 
                          'Amplitude (mV)', figsize, overlapping, show_cuts, cut_indices)


def plot_emg_channel(time: np.ndarray,
                    emg_channel: np.ndarray,
                    channel_name: str = "EMG Channel",
                    figsize: tuple = (10, 6)) -> None:
    """
    Plot a single EMG channel over time.

    Parameters:
    - time: array-like, time values
    - emg_channel: array-like, single EMG channel data
    - channel_name: string, name for the channel
    - figsize: tuple, figure size (width, height)
    """
    plt.figure(figsize=figsize)
    plt.plot(time, emg_channel)
    plt.title(channel_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_motion_channels(time: np.ndarray,
                        motion_data: Union[pd.DataFrame, np.ndarray],
                        channel_names: Optional[List[str]] = None,
                        figsize: tuple = (12, 8),
                        show_labels: bool = False,
                        acquisition: str = 'vicon',
                        overlapping: bool = False,
                        show_cuts: bool = False,
                        cut_indices: Optional[List[int]] = None) -> None:
    """
    Plot motion capture joint angles in a grid layout or overlapping.

    Parameters:
    - time: array-like, time values
    - motion_data: DataFrame or array, joint angle data (joints as columns)
    - channel_names: list of strings, names for each joint (optional)
    - figsize: tuple, figure size (width, height)
    - show_labels: bool, whether to show anatomical labels for joints
    - acquisition: str, acquisition system ('vicon' or 'cyberglove')
    - overlapping: bool, whether to plot all joints on the same axes (default: False)
    - show_cuts: bool, whether to show repetition boundary markers (default: False)
    - cut_indices: list of cut indices from timeCuts.json [start_idx1, end_idx1, ...] (optional)
    """
    if show_labels and channel_names is None:
        n_joints = motion_data.shape[1]
        channel_names = get_channel_labels('motion', acquisition, n_joints)
    
    _plot_channels_generic(time, motion_data, channel_names, 'Motion Channels',
                          'Angle (degrees)', figsize, overlapping, show_cuts, cut_indices)


def plot_tactile_channels(time: np.ndarray,
                         tactile_data: Union[pd.DataFrame, np.ndarray],
                         channel_names: Optional[List[str]] = None,
                         figsize: tuple = (12, 8),
                         show_labels: bool = False,
                         acquisition: str = 'tactileglove',
                         overlapping: bool = False,
                         show_cuts: bool = False,
                         cut_indices: Optional[List[int]] = None) -> None:
    """
    Plot tactile force measurements in a grid layout or overlapping.

    Parameters:
    - time: array-like, time values
    - tactile_data: DataFrame or array, force sensor data (sensors as columns)
    - channel_names: list of strings, names for each sensor (optional)
    - figsize: tuple, figure size (width, height)
    - show_labels: bool, whether to show anatomical labels for sensors
    - acquisition: str, acquisition system ('tactileglove')
    - overlapping: bool, whether to plot all sensors on the same axes (default: False)
    - show_cuts: bool, whether to show repetition boundary markers (default: False)
    - cut_indices: list of cut indices from timeCuts.json [start_idx1, end_idx1, ...] (optional)
    """
    if show_labels and channel_names is None:
        n_sensors = tactile_data.shape[1]
        channel_names = get_channel_labels('tactile', acquisition, n_sensors)
    
    _plot_channels_generic(time, tactile_data, channel_names, 'Tactile Channels',
                          'Force (N)', figsize, overlapping, show_cuts, cut_indices)


def plot_overlapping_modalities(data_dict, normalize=False, figsize=(12, 8), title="Overlapping Modalities"):
    """
    Plot multiple modalities overlapping on the same axes with different colors and scales.

    Parameters:
    - data_dict: dict, dictionary with modality data in format:
        {'modality_name': (time, data, channel_indices, color, label)}
        where:
        - time: array-like, time values
        - data: DataFrame, channel data
        - channel_indices: list of int, which channels to plot
        - color: str, matplotlib color
        - label: str, legend label
    - normalize: bool, whether to normalize each modality to [0,1] range
    - figsize: tuple, figure size (width, height)
    - title: str, plot title
    """
    plt.figure(figsize=figsize)

    for modality_name, (time, data, channel_indices, color, label) in data_dict.items():
        for ch_idx in channel_indices:
            if ch_idx < data.shape[1]:
                signal = data.iloc[:, ch_idx].values

                # Normalize if requested
                if normalize:
                    signal_min, signal_max = signal.min(), signal.max()
                    if signal_max > signal_min:
                        signal = (signal - signal_min) / (signal_max - signal_min)

                plt.plot(time, signal, color=color, alpha=0.7,
                        label=f'{label} Ch{ch_idx+1}' if len(channel_indices) > 1 else label)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude' if normalize else 'Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_cross_modal_timeline(emg_time, emg_data, motion_time, motion_data, tactile_time, tactile_data,
                             emg_channels=None, motion_channels=None, tactile_channels=None,
                             normalize=False, figsize=(15, 6), title="Cross-Modal Timeline"):
    """
    Create a timeline visualization showing all three modalities aligned.

    Parameters:
    - emg_time, motion_time, tactile_time: array-like, time values for each modality
    - emg_data, motion_data, tactile_data: DataFrame, channel data for each modality
    - emg_channels, motion_channels, tactile_channels: list of int, channel indices to plot (default: first channel)
    - normalize: bool, whether to normalize signals to [0,1] range
    - figsize: tuple, figure size (width, height)
    - title: str, plot title
    """
    # Default to first channel if not specified
    if emg_channels is None:
        emg_channels = [0]
    if motion_channels is None:
        motion_channels = [0]
    if tactile_channels is None:
        tactile_channels = [0]

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # EMG subplot
    for ch_idx in emg_channels:
        if ch_idx < emg_data.shape[1]:
            signal = emg_data.iloc[:, ch_idx].values
            if normalize:
                signal_min, signal_max = signal.min(), signal.max()
                if signal_max > signal_min:
                    signal = (signal - signal_min) / (signal_max - signal_min)
            axes[0].plot(emg_time, signal, 'b-', alpha=0.7, label=f'EMG Ch{ch_idx+1}')
    axes[0].set_title('Electromyography (EMG)')
    axes[0].set_ylabel('Amplitude (mV)' if not normalize else 'Normalized')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Motion subplot
    for ch_idx in motion_channels:
        if ch_idx < motion_data.shape[1]:
            signal = motion_data.iloc[:, ch_idx].values
            if normalize:
                signal_min, signal_max = signal.min(), signal.max()
                if signal_max > signal_min:
                    signal = (signal - signal_min) / (signal_max - signal_min)
            axes[1].plot(motion_time, signal, 'r-', alpha=0.7, label=f'Joint Ch{ch_idx+1}')
    axes[1].set_title('Motion Capture')
    axes[1].set_ylabel('Angle (°)' if not normalize else 'Normalized')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Tactile subplot
    for ch_idx in tactile_channels:
        if ch_idx < tactile_data.shape[1]:
            signal = tactile_data.iloc[:, ch_idx].values
            if normalize:
                signal_min, signal_max = signal.min(), signal.max()
                if signal_max > signal_min:
                    signal = (signal - signal_min) / (signal_max - signal_min)
            axes[2].plot(tactile_time, signal, 'g-', alpha=0.7, label=f'Force Ch{ch_idx+1}')
    axes[2].set_title('Tactile Sensing')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Force (N)' if not normalize else 'Normalized')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_emg_channels_overlapping(time, emg_data, channel_names=None, figsize=(12, 8), show_labels=False, acquisition='cometa', title="EMG Channels Overlapping"):
    """
    Plot all EMG channels overlapping on the same axes for direct comparison.

    Parameters:
    - time: array-like, time values
    - emg_data: DataFrame or array, EMG channel data (channels as columns)
    - channel_names: list of strings, names for each channel (optional)
    - figsize: tuple, figure size (width, height)
    - show_labels: bool, whether to show anatomical labels for channels
    - acquisition: str, acquisition system ('cometa' or 'sessantaquattro')
    - title: str, plot title
    """
    n_channels = emg_data.shape[1]

    plt.figure(figsize=figsize)

    # Get channel labels if requested
    if show_labels and channel_names is None:
        channel_names = get_channel_labels('emg', acquisition, n_channels)

    for i in range(n_channels):
        if channel_names and i < len(channel_names):
            label = channel_names[i]
        else:
            label = f'EMG Channel {i+1}'

        plt.plot(time, emg_data.iloc[:, i], alpha=0.7, label=label)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_motion_channels_overlapping(time, motion_data, channel_names=None, figsize=(12, 8), show_labels=False, acquisition='vicon', title="Motion Channels Overlapping"):
    """
    Plot all motion capture channels overlapping on the same axes for direct comparison.

    Parameters:
    - time: array-like, time values
    - motion_data: DataFrame or array, joint angle data (joints as columns)
    - channel_names: list of strings, names for each joint (optional)
    - figsize: tuple, figure size (width, height)
    - show_labels: bool, whether to show anatomical labels for joints
    - acquisition: str, acquisition system ('vicon' or 'cyberglove')
    - title: str, plot title
    """
    n_joints = motion_data.shape[1]

    plt.figure(figsize=figsize)

    # Get channel labels if requested
    if show_labels and channel_names is None:
        channel_names = get_channel_labels('motion', acquisition, n_joints)

    for i in range(n_joints):
        if channel_names and i < len(channel_names):
            label = channel_names[i]
        else:
            label = f'Joint {i+1}'

        plt.plot(time, motion_data.iloc[:, i], alpha=0.7, label=label)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tactile_channels_overlapping(time, tactile_data, channel_names=None, figsize=(12, 8), show_labels=False, acquisition='tactileglove', title="Tactile Channels Overlapping"):
    """
    Plot all tactile channels overlapping on the same axes for direct comparison.

    Parameters:
    - time: array-like, time values
    - tactile_data: DataFrame or array, force sensor data (sensors as columns)
    - channel_names: list of strings, names for each sensor (optional)
    - figsize: tuple, figure size (width, height)
    - show_labels: bool, whether to show anatomical labels for sensors
    - acquisition: str, acquisition system ('tactileglove')
    - title: str, plot title
    """
    n_sensors = tactile_data.shape[1]

    plt.figure(figsize=figsize)

    # Get channel labels if requested
    if show_labels and channel_names is None:
        channel_names = get_channel_labels('tactile', acquisition, n_sensors)

    for i in range(n_sensors):
        if channel_names and i < len(channel_names):
            label = channel_names[i]
        else:
            label = f'Sensor {i+1}'

        plt.plot(time, tactile_data.iloc[:, i], alpha=0.7, label=label)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()