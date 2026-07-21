# ReachGrasp Dataset - AI Coding Guidelines

## Environment Setup

**Virtual Environment:** This project requires the ARC virtual environment to be activated using virtualenvwrapper:
```bash
workon ARC
```

**CRITICAL:** Always activate the ARC environment before running Python code or installing packages.

## Project Overview
This is a multimodal neurophysiology dataset for reach-and-grasp tasks, containing EMG, motion capture, and tactile data from 10 subjects performing 16 different tasks. Data is organized in BIDS-like structure with subject/task/acquisition/modality hierarchy.

## Architecture & Analysis Patterns

### Notebook Structure
Two main analysis notebooks follow a consistent configuration-driven pattern:
- **IntroductoryAnalysis.ipynb**: Multimodal exploration (EMG + motion + tactile)
- **JointDataAnalysis.ipynb**: Specialized kinematic analysis with anatomical grouping

**Configuration Pattern:** Both notebooks use global configuration variables at the top:
```python
SELECTED_SUBJECT = 'sub-01'  # sub-01 through sub-10
SELECTED_ACTION = 'FroRea'   # 16 tasks available
```
When user changes these variables, re-run cells from "Data Loading" section onward.

**Joint Filtering (JointDataAnalysis.ipynb):** Use `JOINT_FILTER` variable for targeted analysis:
```python
JOINT_FILTER = 'all'   # All 31 joints (complete upper limb)
JOINT_FILTER = 'arm'   # 15 arm joints (Thorax, Shoulder, Elbow, Wrist)
JOINT_FILTER = 'hand'  # 16 hand joints (Fingers, Thumb)
JOINT_FILTER = ['RElbow_X', 'RWrist_X', 'RIndex_X']  # Custom joint list
```
- **'arm'** includes Wrist (proximal control + end-effector position)
- **'hand'** excludes Wrist (pure finger/thumb manipulation)
- **Custom list** allows biomechanical studies of specific joint combinations

### Data Structuring Convention
**Joint angles** in `JointDataAnalysis.ipynb` follow anatomical naming:
- Use `joint_column_names` list with descriptive names: `['LThorax_X', 'RElbow_Y', 'RWrist_Z', ...]`
- Group joints with `joint_groups` dictionary for anatomical regions (11 regions)
- XYZ components represent: X=flexion/extension, Y=abduction/adduction, Z=rotation
- Example from `JointDataAnalysis.ipynb` line 175-186:
  ```python
  joint_groups = {
      'Thorax_Left': ['LThorax_X', 'LThorax_Y', 'LThorax_Z'],
      'Shoulder': ['RShoulder_X', 'RShoulder_Y', 'RShoulder_Z'],
      # ... 11 total regions
  }
  ```

### Visualization Patterns
**Color-coding scheme for anatomical regions** (see `JointDataAnalysis.ipynb` line 583-595):
```python
region_colors = {
    'Thorax_Left': '#FF6B6B',    # Red
    'Shoulder': '#4ECDC4',       # Teal  
    'Wrist': '#FFD93D',          # Yellow
    # ... 11 unique colors for 11 regions
}
```
Use this when creating correlation heatmaps or multi-joint plots to maintain visual consistency.

**XYZ component colors** (standard across notebooks):
```python
component_colors = {'X': '#e74c3c', 'Y': '#3498db', 'Z': '#2ecc71'}  # Red, Blue, Green
```

## Data Organization
- **Subjects**: `sub-01` through `sub-10`
- **Tasks**: Cyl, EatFruit, FroRea, HC, HO, Pour, ReaCyl, ReaSph, Screw, Sph, Thumb, Trid, WE, WF, WP, WS
- **Acquisitions**: 
  - `cometa`/`sessantaquattro`: EMG recordings (10 channels, 2000 Hz)
  - `cyberglove`/`vicon`: Motion capture data (100 Hz)
  - `tactileglove`: Tactile sensor data
- **Modalities**: `emg`, `motion`, `tactile`
- **Repetitions**: Each recording contains 10 repetitions of the same task
  - Task boundaries defined in `timeCuts.json` (millisecond precision)
  - Loading functions split by repetitions by default (`split_repetitions=True`)
  - **Movement Extraction**: By default (`extract_movements=True`), functions extract movement periods between task boundaries (end_i to start_i+1), not task execution periods (start_i to end_i)
    - For reaching tasks (FroRea, ReaCyl, ReaSph), actual movements occur during return to start position
    - Results in 9 movement periods (vs 10 task execution periods)
    - Set `extract_movements=False` to get task execution periods instead
  - **Zero-Clamping**: By default (`clamp_near_zero=True`), near-zero values (< 1e-10) are set to exactly 0.0
    - Removes floating-point noise artifacts (e.g., 1e-12, 1e-17 in elbow/wrist joints)
    - Set `clamp_near_zero=False` to preserve raw values
    - Adjust threshold with `zero_threshold` parameter
  - Each repetition time-normalized relative to cut boundary (may have small offset < 1 sample period due to discrete sampling)
- **Time Format**: All data files store time in **seconds**
  - Vicon: 0.01s steps (100 Hz)
  - EMG: 0.0005s steps (2000 Hz)
  - `timeCuts.json`: Stored in milliseconds, automatically converted to seconds by loading functions

## File Structure Pattern
```
sub-XX/task-YYY_acq-ZZZ_modality.{csv,json,tsv}
```
- `.csv`: Time-series data (first column: time, subsequent: channels)
- `.json`: Metadata (device info, sampling rates, electrode placements)
- `.tsv`: Channel descriptions (name, type, units, sampling_frequency)

## Data Loading Patterns
Use pandas for CSV data with `header=None` since files lack headers:
```python
import pandas as pd
data = pd.read_csv('sub-01/emg/sub-01_task-Cyl_acq-cometa_emg.csv', header=None)
time = data.iloc[:, 0]  # ALWAYS first column
emg_channels = data.iloc[:, 1:]  # Remaining columns
```

**Channel label mapping:** Use `utils/channel_mappings.json` + `utils/plotting.py` helpers:
```python
from utils.plotting import get_channel_labels
labels = get_channel_labels('emg', 'cometa', n_channels=10)
# Returns: ['Ch1: Lat_Tric (Lateral Triceps)', 'Ch2: Med_Tric...', ...]
```

**Metadata files:** Every `.csv` has companion `.json` (metadata) and `.tsv` (channel info):
- `.json`: Device specs, sampling rates, electrode placements
- `.tsv`: Channel descriptions with name, type, units, sampling_frequency columns

## Common Analysis Workflows
1. **Load channel info**: Parse `.tsv` files to understand signal types
2. **Synchronize modalities**: Align time bases across EMG/motion/tactile for same task
3. **Filter preprocessing**: Apply bandpass filters (20-500Hz) to EMG data
4. **Feature extraction**: RMS, wavelet transforms for EMG; joint angles for motion

## Coding Conventions
- **Utility Functions**: Create dedicated utility functions for reusable operations and place them in independent Python files (e.g., `utils/plotting.py`, `utils/data_loading.py`)
- **Code Reuse**: Use internal helper functions to eliminate redundancy (e.g., `_apply_filter()`, `_calculate_derivative()`, `_plot_channels_generic()`)
- **Common Data Structures**: Maintain consistent patterns for data handling across all functions
  - All filter functions use the same type-preserving pattern (DataFrame → DataFrame, ndarray → ndarray)
  - Derivative calculations use unified internal logic for arbitrary orders
  - Plotting functions share a common generic implementation
  - File path construction uses centralized `_construct_filepath()` helper
- **Type Consistency**: Functions preserve input types (DataFrame in → DataFrame out, array in → array out)
- **Simple Plotting**: Start with basic matplotlib plots for data exploration before complex visualizations
- **Modular Code**: Break down analysis into functions that can be reused across different subjects/tasks/modalities
- **Error Handling**: Always check for file existence and handle missing data gracefully
- **Memory Efficiency**: Use chunked reading for large files and consider data types to minimize memory usage

**Import Pattern (all notebooks):**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmaps and statistical plots
from scipy import signal
from utils.plotting import plot_emg_channels, plot_motion_channels  # Custom helpers
```

## Utility Functions Library

The `utils/` folder provides comprehensive functions for all common workflows. The library is optimized for:
- **Code reuse**: Internal helpers eliminate redundancy across similar operations
- **Type preservation**: Functions maintain input types (DataFrame/ndarray) throughout processing
- **Consistent patterns**: Common data structures and interfaces across all modules

### Data Loading (`utils/data_loading.py`)
```python
from utils.data_loading import load_timeseries_data, load_joint_angles_with_labels
from utils.data_loading import get_joint_groups, get_arm_joints, get_hand_joints, get_region_colors
from utils.data_loading import load_time_cuts, get_task_time_cuts, split_by_repetitions

# Load any modality - split by repetitions (default extracts movement periods)
reps = load_timeseries_data('sub-01', 'FroRea', 'vicon', 'motion')  # Returns 9 movements
for i, (time, motion) in enumerate(reps):
    print(f"Movement {i+1}: {len(time)} samples")

# Load task execution periods instead of movements
task_reps = load_timeseries_data('sub-01', 'Cyl', 'cometa', 'emg', 
                                  extract_movements=False)  # Returns 10 tasks

# Load as continuous recording (no split)
time, emg = load_timeseries_data('sub-01', 'Cyl', 'cometa', 'emg', split_repetitions=False)

# Load without zero-clamping (preserve raw near-zero values)
reps_raw = load_timeseries_data('sub-01', 'FroRea', 'vicon', 'motion',
                                 clamp_near_zero=False)

# Load joint angles with anatomical labels (uses movement extraction + zero-clamping by default)
reps = load_joint_angles_with_labels('sub-01', 'FroRea')  # Returns 9 movements, cleaned
time, joint_angles = load_joint_angles_with_labels('sub-01', 'FroRea', split_repetitions=False)

# Manual repetition splitting
time_cuts = load_time_cuts()  # Load timeCuts.json
cuts = get_task_time_cuts(time_cuts, 'sub-01', 'Cyl')  # Get cuts for specific task
repetitions = split_by_repetitions(time, data, cuts)  # Split data manually

# Get predefined joint sets for filtering
joint_groups = get_joint_groups()  # Returns 11-region anatomical dictionary
region_colors = get_region_colors()  # Returns standard color palette
arm_joints = get_arm_joints()    # 15 joints: Thorax, Shoulder, Elbow, Wrist
hand_joints = get_hand_joints()  # 16 joints: Fingers, Thumb
```

### Preprocessing (`utils/preprocessing.py`)
```python
from utils.preprocessing import bandpass_filter, calculate_rms, resample_signal
from utils.preprocessing import calculate_velocity, calculate_acceleration, normalize_signal
from utils.preprocessing import detect_movement_onset

# All filter functions preserve input type (DataFrame → DataFrame, ndarray → ndarray)
emg_filtered = bandpass_filter(emg, lowcut=20, highcut=500, fs=2000)
emg_lowpass = lowpass_filter(emg, cutoff=50, fs=2000)
emg_highpass = highpass_filter(emg, cutoff=10, fs=2000)
emg_notch = notch_filter(emg, freq=60, fs=2000)  # Remove 60Hz powerline noise

# Calculate RMS with 100ms windows
emg_rms = calculate_rms(emg_filtered, window_size=200, step_size=100)

# Synchronize EMG with motion (resample to 100 Hz)
emg_resampled = resample_signal(emg_time, emg_data, motion_time)

# Kinematic analysis (derivatives use unified internal logic)
velocity = calculate_velocity(time, joint_angles)
acceleration = calculate_acceleration(time, joint_angles)

# Normalize signals
emg_norm = normalize_signal(emg, method='minmax')  # [0, 1] range
emg_zscore = normalize_signal(emg, method='zscore')  # z-score

# Movement detection
onset_idx, offset_idx = detect_movement_onset(velocity)
```

### Analysis (`utils/analysis.py`)
```python
from utils.analysis import analyze_joint_coordination, find_highly_correlated_pairs
from utils.analysis import calculate_channel_statistics, segment_by_anatomical_region

# Correlation analysis (use pandas built-in for basic correlations)
corr_matrix = joint_angles.corr()  # Pearson correlation
corr_matrix = joint_angles.corr(method='spearman')  # Spearman correlation

# Advanced coordination analysis with time lags
coord_results = analyze_joint_coordination(joint_angles, time, min_correlation=0.5)

# Find highly correlated pairs
highly_correlated = find_highly_correlated_pairs(corr_matrix, threshold=0.7)

# Statistics by region
stats = calculate_channel_statistics(joint_angles)
segmented_data = segment_by_anatomical_region(joint_angles, joint_groups)
```

### Plotting (`utils/plotting.py`)
```python
from utils.plotting import plot_emg_channels, plot_motion_channels, plot_tactile_channels
from utils.plotting import get_channel_labels

# All plotting functions use shared generic implementation (_plot_channels_generic)
# Auto-labeled plots
labels = get_channel_labels('emg', 'cometa', n_channels=10)
plot_emg_channels(time, emg, show_labels=True, acquisition='cometa')

# Overlapping multi-channel view
plot_emg_channels(time, emg, overlapping=True, show_labels=True)

# Motion and tactile plotting work identically
plot_motion_channels(time, motion, show_labels=True, acquisition='vicon')
plot_tactile_channels(time, tactile, show_labels=True, overlapping=True)
```

## Common Analysis Workflows
1. **Load data by repetitions**: Use `load_timeseries_data()` to get list of 9 movements (default behavior with `extract_movements=True`)
2. **Load task execution periods**: Set `extract_movements=False` to get 10 task execution periods instead of movements
3. **Load continuous recording**: Set `split_repetitions=False` for full recording without splitting
4. **Synchronize modalities**: Use `resample_signal()` to align EMG (2000Hz) with motion/tactile (100Hz)
5. **Filter preprocessing**: Use `bandpass_filter(emg, lowcut=20, highcut=500, fs=2000)` for EMG
6. **Feature extraction**: Use `calculate_rms()` for EMG amplitude, `calculate_velocity()` for kinematics
7. **Repetition-level analysis**: Process each movement separately for intra-task variability studies
8. **Clean numerical noise**: Default `clamp_near_zero=True` removes floating-point artifacts like 1e-12

## Data Synchronization
- **Sampling Rates**: EMG (2000Hz), motion/tactile (100Hz) - resample or interpolate for alignment
- **Time Format**: All data stored in seconds (Vicon: 0.01s steps, EMG: 0.0005s steps)
- **Time Cuts**: `timeCuts.json` stores boundaries in milliseconds, automatically converted to seconds
- **Time Alignment**: Use first column (time) as reference for cross-modal synchronization
- **Task Boundaries**: Recordings may include pre/post-task periods; identify active task segments
- **Repetition Timing**: When split by repetitions, time is normalized relative to cut boundary
  - Small offsets (< 1 sample period) may occur due to discrete sampling
  - Preserves accurate timing relationships within each repetition

## Task Descriptions
- **Hand Movements**: HO (extend the fingers), HC (flex the fingers)
- **Wrist Movements**: WP (palm facing down), WS (palm facing up), WF (bend hand so palm faces forearm), WE (lift hand backwards)
- **Object Grasps**: Cyl (grasp cylindrical glass), Sph (grasp tennis ball), Trid (grasp ping-pong ball with first three fingers), Thumb (thumb abduction)
- **Reaching Tasks**: FroRea (move hand toward table center), ReaCyl (approach and grasp cylindrical glass), ReaSph (approach and grasp tennis ball)
- **Functional Tasks**: Pour (approach, grasp and pour glass), Screw (approach, grasp with three fingers and screw bottle), EatFruit (approach, grasp and move tennis ball toward mouth)

## Dependencies
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib`/`plotly` for visualization
- `scipy` for signal processing

## Sensor Positioning and Channel Descriptions

The ReachGrasp dataset employs a comprehensive multimodal setup with specific sensor placements and channel definitions.

### EMG Channels

#### Bipolar EMG (Cometa Wave Plus - 10 channels, 2000 Hz)
Ten bipolar pre-gelled adhesive electrodes placed on upper limb muscles:

| Channel | Abbreviation | Muscle | Location |
| :---: | :--- | :--- | :--- |
| 1 | Lat_Tric | Lateral Head of Triceps | Triceps |
| 2 | Med_Tric | Medial Head of Triceps | Triceps |
| 3 | Long_Bic | Long Head of Biceps | Biceps |
| 4 | Short_Bic | Short Head of Biceps | Biceps |
| 5 | Ant_Delt | Anterior Deltoid | Shoulder |
| 6 | Midd_Delt | Middle Deltoid | Shoulder |
| 7 | Post_Delt | Posterior Deltoid | Shoulder |
| 8 | Upper_Trap | Upper Trapezius | Upper back |
| 9 | Brachiorad | Brachioradialis | Forearm |
| 10 | W_pronator | Pronator Teres | Forearm |

#### High-Density EMG (Sessantaquattro - 64 channels, 2000 Hz)
Two 32-electrode grids placed 5 cm below olecranon covering forearm circumference:

| Channels | Abbreviation Range | Muscle Group | Location |
| :---: | :--- | :--- | :--- |
| 1–32 | flexion_01 to flexion_32 | Flexor muscles | Forearm flexors (FDS, FCR, FCU) |
| 33–64 | extension_01 to extension_32 | Extensor muscles | Forearm extensors (EDC, ECR, ECU) |

### Kinematic Data (100 Hz)

#### Vicon Motion Capture (31 channels)
23 reflective markers using Plug-in Gait and RHand models:

| Joint | Abbreviation | Movement Components |
| :---: | :--- | :--- |
| LThorax | LThorax_X/Y/Z | Backward tilt, Right tilt, Right rotation |
| RElbow | RElbow_X/Y/Z | Flexion-Extension |
| RShoulder | RShoulder_X/Y/Z | Flexion-Extension, Abduction, Internal rotation |
| RThorax | RThorax_X/Y/Z | Backward tilt, Left tilt, Left rotation |
| RWrist | RWrist_X/Y/Z | Ulnar deviation, Flexion-Extension, Pronation-Supination |
| Finger joints | RIndexJ1, RPinkieJ1, etc. | MCP joint flexion-extension and abduction-adduction |

#### CyberGlove (18 channels)
Finger bending and wrist movements:

| Channel | Abbreviation | Description |
| :---: | :--- | :--- |
| 1-4 | ThumbRotate, ThumbMPJ, ThumbIJ, ThumbAb | Thumb movements |
| 5-6 | IndexMPJ, IndexPIJ | Index finger MCP/PIP |
| 7-9 | MiddleMPJ, MiddlePIJ, MiddleIndexAb | Middle finger + abduction |
| 10-12 | RingMPJ, RingPIJ, RingMiddleAb | Ring finger + abduction |
| 13-15 | PinkieMPJ, PinkiePIJ, PinkieRingAb | Pinkie finger + abduction |
| 16-18 | PalmArch, WristPitch, WristYaw | Palm arch, wrist flexion/extension, wrist deviation |

### Tactile Data (TactileGlove - 58 channels, 100 Hz)

Piezoresistive sensors measuring electrical resistance changes:
- **Palm**: 17 sensors
- **Fingers**: 9 (index/pinky), 8 (middle/ring), 7 (thumb)
- **Discarded sensors**: Channels 9, 11, 14 (ring finger tips) due to hardware issues

| Example Channels | Abbreviation | Location |
| :---: | :--- | :--- |
| rmo, mdo | Finger sensors | Various finger positions |
| pcim, pcip | Palm sensors | Palm and wrist areas |
| ptip, mp | Contact areas | Thumb palm, middle palm |

## Development Notes
- Data files are large (100k+ rows); use chunked reading for memory efficiency
- EMG data sampled at 2000Hz; motion/tactile at varying rates
- Electrode placement follows SENIAM guidelines for EMG
- Tasks involve object manipulation with different grasps and reaches
- **Performance**: Consider using `dask` for parallel processing of multiple subjects/tasks
- **Data Integrity**: Validate channel counts match metadata before analysis
- **Research Focus**: Dataset supports studies in motor control, prosthetics, and rehabilitation</content>
<parameter name="filePath">/home/bmaacaron-iit.local/Documents/Datasets/ReachGrasp/.github/copilot-instructions.md