"""Load and organize files from the ReachGrasp dataset.

The dataset itself remains outside the installed Python package. Every loader
accepts ``base_path``, which must point to the directory containing
``timeCuts.json`` and the ``sub-XX`` subject folders.

When ``split_repetitions`` is true, loading functions return a list of
repetitions or raise an exception. They never silently change the return type
to a continuous recording when repetition metadata is unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _construct_filepath(base_path: str, subject: str, modality: str, 
                       task: str, acquisition: str, extension: str) -> str:
    """
    Internal helper: Construct standardized file path for dataset files.
    
    Parameters:
    - base_path: base directory
    - subject: subject ID (e.g., 'sub-01')
    - modality: data modality ('emg', 'motion', 'tactile')
    - task: task name (e.g., 'Cyl')
    - acquisition: acquisition system (e.g., 'cometa')
    - extension: file extension ('csv', 'json', 'tsv')
    
    Returns:
    - str: full file path
    """
    filename = f'{subject}_task-{task}_acq-{acquisition}'
    if extension == 'tsv':
        filename += '_channels.tsv'
    else:
        filename += f'_{modality}.{extension}'
    
    return os.path.join(base_path, subject, modality, filename)


def load_time_cuts(base_path: str = '.') -> Dict:
    """
    Load time cuts information from timeCuts.json file.
    
    The timeCuts.json file contains start/end times for each repetition of tasks.
    Each recording typically contains 10 repetitions of the same action.
    
    Parameters:
    - base_path: str, base directory path (default: current directory)
    
    Returns:
    - dict: Time cuts data organized by subject and task
    
    Example:
    >>> time_cuts = load_time_cuts()
    >>> cuts = get_task_time_cuts(time_cuts, 'sub-01', 'Cyl')
    >>> # Returns: [start1, end1, start2, end2, ..., start10, end10]
    """
    filepath = os.path.join(base_path, 'timeCuts.json')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"timeCuts.json not found at: {filepath}\n"
            f"This file contains repetition boundaries for each task."
        )
    
    with open(filepath, 'r') as f:
        return json.load(f)


def get_task_time_cuts(time_cuts_data: Dict, subject: str, task: str) -> Optional[List[int]]:
    """Return repetition-boundary indices for one subject and task.

    Parameters
    ----------
    time_cuts_data:
        Parsed contents of ``timeCuts.json``.
    subject:
        Subject identifier such as ``"sub-01"``.
    task:
        Task identifier such as ``"Cyl"`` or ``"FroRea"``.

    Returns
    -------
    list of int or None
        Alternating start and end indices, or ``None`` when the subject or task
        does not exist in the metadata.

    Raises
    ------
    TypeError
        If the matching ``time2cut`` entry is not a list.
    ValueError
        If a cut value is not an integer index.
    """

    events = time_cuts_data.get("Events_ReachGrasp", {})
    subjects = events.get("subjects", [])
    subject_data = next(
        (entry for entry in subjects if entry.get("subject_name") == subject),
        None,
    )
    if subject_data is None:
        return None

    task_data = next(
        (entry for entry in subject_data.get("tasks", []) if entry.get("task_name") == task),
        None,
    )
    if task_data is None:
        return None

    cuts = task_data.get("time2cut", [])
    if not isinstance(cuts, list):
        raise TypeError(f"time2cut for {subject} {task} must be a list, got {type(cuts).__name__}.")

    indices: List[int] = []
    for value in cuts:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Cut indices for {subject} {task} must be integers, got {value!r}.")
        indices.append(int(value))
    return indices

def convert_cut_indices_to_times(time_array: np.ndarray, cut_indices: List[int]) -> List[float]:
    """Convert validated sample indices to timestamps.

    Parameters
    ----------
    time_array:
        One-dimensional timestamp array.
    cut_indices:
        Sample indices obtained from :func:`get_task_time_cuts`.

    Returns
    -------
    list of float
        Timestamp corresponding to each supplied index.

    Raises
    ------
    IndexError
        If any index lies outside ``time_array``.
    ValueError
        If ``time_array`` is not one-dimensional.
    """

    time_values = np.asarray(time_array, dtype=float)
    if time_values.ndim != 1:
        raise ValueError(f"time_array must be one-dimensional, got shape {time_values.shape}.")

    converted: List[float] = []
    for index in cut_indices:
        if index < 0 or index >= time_values.size:
            raise IndexError(f"Cut index {index} is outside a time array of length {time_values.size}.")
        converted.append(float(time_values[index]))
    return converted

def split_by_repetitions(time: np.ndarray, data: pd.DataFrame, time_cuts: List[int],
                         extract_movements: bool = True) -> List[Tuple[np.ndarray, pd.DataFrame]]:
    """Split a recording using alternating task start and end indices.

    With ``extract_movements=False``, each ``start_i:end_i`` task interval is
    returned. With ``extract_movements=True``, the movement between successive
    task intervals is returned as ``end_i:start_(i+1)``. Each output time vector
    starts at zero.

    Invalid or out-of-range cuts raise an exception. Segments are never skipped
    silently because that would change repetition numbering downstream.
    """

    time_values = np.asarray(time, dtype=float)
    if time_values.ndim != 1:
        raise ValueError(f"time must be one-dimensional, got shape {time_values.shape}.")
    if len(data) != time_values.size:
        raise ValueError(f"time and data lengths differ: {time_values.size} != {len(data)}.")
    if len(time_cuts) < 2 or len(time_cuts) % 2 != 0:
        raise ValueError(
            "time_cuts must contain at least one start/end pair and have even length; "
            f"got {len(time_cuts)} values."
        )

    cuts = np.asarray(time_cuts)
    if cuts.ndim != 1 or not np.issubdtype(cuts.dtype, np.integer):
        raise ValueError("time_cuts must be a one-dimensional sequence of integer indices.")
    cuts = cuts.astype(int, copy=False)
    if np.any(cuts < 0) or np.any(cuts > time_values.size):
        raise IndexError(
            f"time_cuts must lie between 0 and {time_values.size}, got {cuts.tolist()}."
        )
    if np.any(np.diff(cuts) < 0):
        raise ValueError("time_cuts must be ordered from earliest to latest.")

    if extract_movements:
        intervals = [(cuts[index], cuts[index + 1]) for index in range(1, len(cuts) - 1, 2)]
    else:
        intervals = [(cuts[index], cuts[index + 1]) for index in range(0, len(cuts), 2)]

    repetitions: List[Tuple[np.ndarray, pd.DataFrame]] = []
    for repetition_index, (start_idx, end_idx) in enumerate(intervals):
        if start_idx >= end_idx:
            raise ValueError(
                f"Repetition {repetition_index} has an empty or reversed interval: "
                f"{start_idx}:{end_idx}."
            )
        repetition_time = time_values[start_idx:end_idx] - time_values[start_idx]
        repetition_data = data.iloc[start_idx:end_idx].reset_index(drop=True)
        repetitions.append((repetition_time, repetition_data))

    if not repetitions:
        period_type = "movement" if extract_movements else "task"
        raise ValueError(f"The supplied cuts do not define any {period_type} repetitions.")
    return repetitions

def _clamp_near_zero(data: pd.DataFrame, threshold: float = 1e-10) -> pd.DataFrame:
    """Replace finite values whose absolute magnitude is below ``threshold`` with zero."""

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be non-negative and finite, got {threshold!r}.")
    cleaned = data.copy()
    cleaned[np.abs(cleaned) < threshold] = 0.0
    return cleaned

def load_timeseries_data(subject: str, task: str, acquisition: str, modality: str,
                         base_path: str = ".", split_repetitions: bool = True,
                         extract_movements: bool = True, clamp_near_zero: bool = True,
                         zero_threshold: float = 1e-10, verbose: bool = True) -> Union[
                             Tuple[np.ndarray, pd.DataFrame],
                             List[Tuple[np.ndarray, pd.DataFrame]],
                         ]:
    """Load one ReachGrasp time-series recording.

    Parameters
    ----------
    subject, task, acquisition, modality:
        Components of the ReachGrasp filename and directory convention.
    base_path:
        Dataset root containing ``timeCuts.json`` and subject directories.
    split_repetitions:
        When true, return a list of repetitions determined by ``timeCuts.json``.
        Missing or invalid repetition metadata raises an exception rather than
        returning a continuous recording with a different type.
    extract_movements:
        When splitting, select inter-task movement periods when true and task
        execution periods when false.
    clamp_near_zero:
        Replace numerical noise close to zero in channel values.
    zero_threshold:
        Absolute threshold used by ``clamp_near_zero``.
    verbose:
        Print the number and type of extracted repetitions.

    Returns
    -------
    tuple or list of tuple
        ``(time, data)`` for a continuous recording, or a list of such tuples
        when ``split_repetitions=True``.
    """

    filepath = _construct_filepath(base_path, subject, modality, task, acquisition, "csv")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    raw = pd.read_csv(filepath, header=None)
    if raw.shape[1] < 2:
        raise ValueError(f"Data file must contain time plus at least one channel: {filepath}")

    time = raw.iloc[:, 0].to_numpy(dtype=float)
    channels = raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"Time column contains NaN or infinite values: {filepath}")
    if time.size > 1 and np.any(np.diff(time) <= 0):
        raise ValueError(f"Time column must be strictly increasing: {filepath}")
    if not split_repetitions:
        if not np.all(np.isfinite(channels.to_numpy(dtype=float))):
            raise ValueError(f"Channel data contains NaN or infinite values: {filepath}")
        if clamp_near_zero:
            channels = _clamp_near_zero(channels, threshold=zero_threshold)
        return time, channels

    time_cuts = get_task_time_cuts(load_time_cuts(base_path), subject, task)
    if not time_cuts:
        raise KeyError(f"No time cuts found for {subject} {task} in {base_path}.")

    repetitions = split_by_repetitions(
        time, channels, time_cuts, extract_movements=extract_movements
    )
    for repetition_index, (repetition_time, repetition_data) in enumerate(repetitions):
        if not np.all(np.isfinite(repetition_data.to_numpy(dtype=float))):
            raise ValueError(
                f"Repetition {repetition_index} contains NaN or infinite values: {filepath}"
            )
        if clamp_near_zero:
            repetitions[repetition_index] = (
                repetition_time,
                _clamp_near_zero(repetition_data, threshold=zero_threshold),
            )
    if verbose:
        period_type = "movements" if extract_movements else "task execution periods"
        print(f"Split data into {len(repetitions)} {period_type}")
    return repetitions

def load_metadata(subject: str, task: str, acquisition: str, modality: str,
                  base_path: str = '.') -> Dict:
    """
    Load metadata JSON file for a specific recording.
    
    Parameters:
    - subject: str, subject ID (e.g., 'sub-01')
    - task: str, task name (e.g., 'Cyl')
    - acquisition: str, acquisition system (e.g., 'cometa')
    - modality: str, data modality ('emg', 'motion', 'tactile')
    - base_path: str, base directory path
    
    Returns:
    - dict: Metadata including device info, sampling rates, etc.
    """
    filepath = _construct_filepath(base_path, subject, modality, task, acquisition, 'json')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def load_channel_info(subject: str, task: str, acquisition: str, modality: str,
                     base_path: str = '.') -> pd.DataFrame:
    """
    Load channel information TSV file.
    
    Parameters:
    - subject: str, subject ID
    - task: str, task name
    - acquisition: str, acquisition system
    - modality: str, data modality
    - base_path: str, base directory path
    
    Returns:
    - pd.DataFrame: Channel descriptions with columns: name, type, units, sampling_frequency
    """
    filepath = _construct_filepath(base_path, subject, modality, task, acquisition, 'tsv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Channel info file not found: {filepath}")
    
    return pd.read_csv(filepath, sep='\t')


def load_complete_recording(subject: str, task: str, acquisition: str, modality: str,
                            base_path: str = ".") -> Dict:
    """Load one complete unsplit recording with metadata and channel information.

    Returns a dictionary with ``time``, ``data``, ``metadata``, and
    ``channel_info``. Repetition splitting is deliberately disabled because a
    function named ``load_complete_recording`` must have a stable continuous
    return shape.
    """

    time, data = load_timeseries_data(
        subject, task, acquisition, modality, base_path=base_path,
        split_repetitions=False
    )
    metadata = load_metadata(subject, task, acquisition, modality, base_path)
    channel_info = load_channel_info(subject, task, acquisition, modality, base_path)
    return {
        "time": time,
        "data": data,
        "metadata": metadata,
        "channel_info": channel_info,
    }

def load_multimodal_data(subject: str, task: str, base_path: str = '.') -> Dict:
    """
    Load all modalities for a specific subject and task.
    
    Parameters:
    - subject: str, subject ID (e.g., 'sub-01')
    - task: str, task name (e.g., 'Cyl')
    - base_path: str, base directory path
    
    Returns:
    - dict: {
        'emg_cometa': {...},
        'emg_sessantaquattro': {...},
        'motion_vicon': {...},
        'motion_cyberglove': {...},
        'tactile': {...}
      }
    """
    results = {}
    
    # Define available acquisition systems for each modality
    acquisitions = {
        'emg': ['cometa', 'sessantaquattro'],
        'motion': ['vicon', 'cyberglove'],
        'tactile': ['tactileglove']
    }
    
    for modality, acq_list in acquisitions.items():
        for acquisition in acq_list:
            try:
                key = f'{modality}_{acquisition}' if modality != 'tactile' else 'tactile'
                results[key] = load_complete_recording(subject, task, acquisition, modality, base_path)
            except FileNotFoundError:
                # Skip if file doesn't exist
                pass
    
    return results


def get_available_subjects(base_path: str = '.') -> List[str]:
    """
    Get list of available subject IDs in the dataset.
    
    Parameters:
    - base_path: str, base directory path
    
    Returns:
    - list: Subject IDs (e.g., ['sub-01', 'sub-02', ...])
    """
    subjects = []
    for item in os.listdir(base_path):
        if item.startswith('sub-') and os.path.isdir(os.path.join(base_path, item)):
            subjects.append(item)
    return sorted(subjects)


def get_available_tasks(subject: str, modality: str = 'emg', base_path: str = '.') -> List[str]:
    """
    Get list of available tasks for a specific subject.
    
    Parameters:
    - subject: str, subject ID
    - modality: str, modality to check (default: 'emg')
    - base_path: str, base directory path
    
    Returns:
    - list: Task names (e.g., ['Cyl', 'FroRea', ...])
    """
    tasks = set()
    modality_dir = os.path.join(base_path, subject, modality)
    
    if not os.path.exists(modality_dir):
        return []
    
    for filename in os.listdir(modality_dir):
        if filename.endswith('.csv'):
            # Extract task from filename: sub-01_task-Cyl_acq-cometa_emg.csv
            parts = filename.split('_')
            for part in parts:
                if part.startswith('task-'):
                    tasks.add(part.replace('task-', ''))
    
    return sorted(list(tasks))


def load_joint_angles_with_labels(subject: str, task: str, acquisition: str = "vicon",
                                  base_path: str = ".", split_repetitions: bool = True,
                                  extract_movements: bool = True,
                                  clamp_near_zero: bool = True,
                                  zero_threshold: float = 1e-10,
                                  verbose: bool = True) -> Union[
                                      Tuple[np.ndarray, pd.DataFrame],
                                      List[Tuple[np.ndarray, pd.DataFrame]],
                                  ]:
    """Load motion data and apply standardized Vicon joint labels.

    The return type is determined solely by ``split_repetitions``. Vicon files
    must contain exactly 31 channels; otherwise the function raises instead of
    returning an unexpectedly unlabeled DataFrame.
    """

    result = load_timeseries_data(
        subject, task, acquisition, "motion", base_path=base_path,
        split_repetitions=split_repetitions,
        extract_movements=extract_movements,
        clamp_near_zero=clamp_near_zero,
        zero_threshold=zero_threshold,
        verbose=verbose
    )
    joint_column_names = _get_joint_column_names()

    def apply_labels(data: pd.DataFrame) -> pd.DataFrame:
        if acquisition != "vicon":
            return data
        if data.shape[1] != len(joint_column_names):
            raise ValueError(
                f"Expected {len(joint_column_names)} Vicon channels, got {data.shape[1]}."
            )
        return pd.DataFrame(data.to_numpy(copy=True), columns=joint_column_names, index=data.index)

    if split_repetitions:
        if not isinstance(result, list):
            raise RuntimeError("split_repetitions=True did not return a repetition list.")
        return [(time, apply_labels(data)) for time, data in result]

    if isinstance(result, list):
        raise RuntimeError("split_repetitions=False unexpectedly returned repetitions.")
    time, data = result
    return time, apply_labels(data)

def _get_joint_column_names() -> List[str]:
    """
    Internal helper: Get standardized joint column names for Vicon data.
    
    Returns:
    - list: 31 joint angle column names with anatomical labels
    """
    return [
        # Thorax (Left) - indices 0-2
        'LThorax_X', 'LThorax_Y', 'LThorax_Z',
        # Elbow (Right) - indices 3-5
        'RElbow_X', 'RElbow_Y', 'RElbow_Z',
        # Shoulder (Right) - indices 6-8
        'RShoulder_X', 'RShoulder_Y', 'RShoulder_Z',
        # Thorax (Right) - indices 9-11
        'RThorax_X', 'RThorax_Y', 'RThorax_Z',
        # Wrist (Right) - indices 12-14
        'RWrist_X', 'RWrist_Y', 'RWrist_Z',
        # Index Finger MCP - indices 15-17
        'RIndex_X', 'RIndex_Y', 'RIndex_Z',
        # Pinky Finger MCP - indices 18-20
        'RPinky_X', 'RPinky_Y', 'RPinky_Z',
        # Ring Finger MCP - indices 21-23
        'RRing_X', 'RRing_Y', 'RRing_Z',
        # Middle Finger MCP - indices 24-26
        'RMiddle_X', 'RMiddle_Y', 'RMiddle_Z',
        # Thumb STT - indices 27-29
        'RThumb_STT_X', 'RThumb_STT_Y', 'RThumb_STT_Z',
        # Thumb MCP - index 30
        'RThumb_MCP_X'
    ]


def get_joint_groups() -> Dict[str, List[str]]:
    """
    Get anatomical grouping of joints for Vicon motion capture data.
    
    Returns:
    - dict: Joint groups organized by anatomical regions (11 regions)
    """
    return {
        'Thorax_Left': ['LThorax_X', 'LThorax_Y', 'LThorax_Z'],
        'Thorax_Right': ['RThorax_X', 'RThorax_Y', 'RThorax_Z'],
        'Shoulder': ['RShoulder_X', 'RShoulder_Y', 'RShoulder_Z'],
        'Elbow': ['RElbow_X', 'RElbow_Y', 'RElbow_Z'],
        'Wrist': ['RWrist_X', 'RWrist_Y', 'RWrist_Z'],
        'Index': ['RIndex_X', 'RIndex_Y', 'RIndex_Z'],
        'Middle': ['RMiddle_X', 'RMiddle_Y', 'RMiddle_Z'],
        'Ring': ['RRing_X', 'RRing_Y', 'RRing_Z'],
        'Pinky': ['RPinky_X', 'RPinky_Y', 'RPinky_Z'],
        'Thumb_STT': ['RThumb_STT_X', 'RThumb_STT_Y', 'RThumb_STT_Z'],
        'Thumb_MCP': ['RThumb_MCP_X']
    }


def get_arm_joints() -> List[str]:
    """
    Get list of arm joint names (proximal joints including wrist).
    
    Arm joints include: Thorax (Left/Right), Shoulder, Elbow, Wrist (15 joints)
    
    Returns:
    - list: Joint names for arm joints
    """
    groups = get_joint_groups()
    arm_regions = ['Thorax_Left', 'Thorax_Right', 'Shoulder', 'Elbow', 'Wrist']
    return sum([groups[r] for r in arm_regions], [])


def get_hand_joints() -> List[str]:
    """
    Get list of hand joint names (distal joints, fingers and thumb).
    
    Hand joints include: Index, Middle, Ring, Pinky, Thumb (STT and MCP) (16 joints)
    
    Returns:
    - list: Joint names for hand joints
    """
    groups = get_joint_groups()
    hand_regions = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb_STT', 'Thumb_MCP']
    return sum([groups[r] for r in hand_regions], [])


def get_region_colors() -> Dict[str, str]:
    """
    Get standard color mapping for anatomical regions.
    
    Returns:
    - dict: Color codes for each anatomical region
    """
    return {
        'Thorax_Left': '#FF6B6B',      # Red
        'Thorax_Right': '#FFA07A',     # Light Red
        'Shoulder': '#4ECDC4',         # Teal
        'Elbow': '#95E1D3',            # Light Teal
        'Wrist': '#FFD93D',            # Yellow
        'Index': '#6C5CE7',            # Purple
        'Middle': '#A29BFE',           # Light Purple
        'Ring': '#FD79A8',             # Pink
        'Pinky': '#FDCB6E',            # Orange
        'Thumb_STT': '#74B9FF',        # Light Blue
        'Thumb_MCP': '#0984E3'         # Blue
    }


def get_component_colors() -> Dict[str, str]:
    """
    Get standard color mapping for XYZ components.
    
    Returns:
    - dict: Color codes for movement components
    """
    return {
        'X': '#e74c3c',  # Red - Primary flexion/extension
        'Y': '#3498db',  # Blue - Secondary/abduction
        'Z': '#2ecc71'   # Green - Rotation
    }
