# ReachGrasp Python Utilities

This package contains Python code for loading, preprocessing, analysing, and plotting the ReachGrasp dataset. The code is installed as `reachgrasp`; the dataset files remain wherever they are stored on disk.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

Editable installation makes changes to the source files immediately available without reinstalling the package.

Verify the installation from outside the repository:

```bash
python -c "import reachgrasp; print(reachgrasp.__file__)"
```

## Dataset root

Loading functions accept `base_path`. It must point to the directory containing:

```text
ReachGrasp/
├── timeCuts.json
├── sub-01/
├── sub-02/
└── ...
```

The Python package does not contain or copy the dataset.

## Loading data

Use explicit submodule imports. The package initializer does not import plotting and analysis modules when only data loading is required.

```python
from reachgrasp.data_loading import load_timeseries_data

emg_time, emg = load_timeseries_data(
    "sub-01", "Cyl", "cometa", "emg",
    base_path="/path/to/ReachGrasp",
    split_repetitions=False
)
```

To load repetitions:

```python
from reachgrasp.data_loading import load_joint_angles_with_labels

repetitions = load_joint_angles_with_labels(
    "sub-01", "FroRea",
    base_path="/path/to/ReachGrasp",
    split_repetitions=True,
    extract_movements=True
)

time_values, joint_angles = repetitions[0]
```

When `split_repetitions=True`, missing or invalid cut metadata raises an exception. The loader does not silently return a continuous recording with a different type.

## Preprocessing

```python
from reachgrasp.preprocessing import bandpass_filter, calculate_rms, notch_filter

emg_filtered = bandpass_filter(emg, lowcut=20, highcut=500, fs=2000)
emg_filtered = notch_filter(emg_filtered, freq=50, fs=2000)
emg_rms = calculate_rms(emg_filtered, window_size=200, step_size=100)
```

Butterworth filters are represented as second-order sections and applied with `scipy.signal.sosfiltfilt`. The notch filter converts the transfer-function coefficients returned by `iirnotch` to second-order sections before filtering.

## Joint organisation

```python
from reachgrasp.data_loading import get_arm_joints, get_hand_joints, get_joint_groups

groups = get_joint_groups()
arm_joints = get_arm_joints()
hand_joints = get_hand_joints()
```

## Coordination analysis

Use pandas directly for basic correlation matrices:

```python
correlation_matrix = joint_angles.corr()
```

Use the higher-level analysis module for time-lagged coordination:

```python
from reachgrasp.analysis import analyze_joint_coordination

coordination = analyze_joint_coordination(
    joint_angles, time_values,
    max_lag=50,
    min_correlation=0.5
)
```

## Plotting

```python
from reachgrasp.plotting import plot_motion_channels

plot_motion_channels(
    time_values, joint_angles,
    show_labels=True,
    acquisition="vicon"
)
```

## Main modules

- `reachgrasp.data_loading`: dataset paths, metadata, repetitions, labels, subjects, tasks, and joint groups.
- `reachgrasp.preprocessing`: filters, interpolation, RMS, derivatives, normalization, movement detection, and RMS jerk.
- `reachgrasp.analysis`: coordination, spectral, statistical, and movement-window analyses.
- `reachgrasp.plotting`: EMG, motion, tactile, and cross-modal plots.

## Dataset file layout

```text
sub-XX/
├── emg/
│   ├── sub-XX_task-YYY_acq-cometa_emg.csv
│   ├── sub-XX_task-YYY_acq-cometa_emg.json
│   └── sub-XX_task-YYY_acq-cometa_channels.tsv
├── motion/
│   ├── sub-XX_task-YYY_acq-vicon_motion.csv
│   └── sub-XX_task-YYY_acq-cyberglove_motion.csv
└── tactile/
    └── sub-XX_task-YYY_acq-tactileglove_tactile.csv
```

CSV files contain time in the first column and channels in the remaining columns.
