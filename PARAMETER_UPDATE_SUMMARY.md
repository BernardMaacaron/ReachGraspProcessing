# Parameter Updates: Movement Extraction & Zero-Clamping

## Summary of Changes

This document tracks updates to data loading function parameters in `utils/data_loading.py`.

---

## Update 1: Movement Extraction Parameter Rename

The parameter `use_rest_periods` has been renamed to `extract_movements` across all data loading functions, with the default value changed from `False` to `True`.

### Rationale

1. **More intuitive naming**: "extract_movements" better describes what the parameter does - it extracts the actual movement periods from the recording
2. **Correct default behavior**: For reaching tasks like FroRea, the actual movements occur during the periods between task markers (when returning hand to start position)
3. **Better semantics**: The term "rest periods" was misleading since significant movement occurs during these segments

## What Changed

### Default Behavior (IMPORTANT)

**Before:**
- Default: `use_rest_periods=False` → extracted task execution periods (small movements)
- Need to set `use_rest_periods=True` to get actual reaching movements

**After:**
- Default: `extract_movements=True` → extracts movement periods (actual reaching movements)
- Set `extract_movements=False` if you want task execution periods

### Affected Functions

All in `utils/data_loading.py`:

1. **`split_by_repetitions()`**
   - Parameter: `use_rest_periods` → `extract_movements`
   - Default: `False` → `True`

2. **`load_timeseries_data()`**
   - Parameter: `use_rest_periods` → `extract_movements`
   - Default: `False` → `True`

3. **`load_joint_angles_with_labels()`**
   - Parameter: `use_rest_periods` → `extract_movements`
   - Default: `False` → `True`

### Affected Files

- ✅ `utils/data_loading.py` - Core functions updated
- ✅ `demo_rest_periods.py` - Demo updated with new parameter name

## Migration Guide

### If you were using default behavior (no parameter specified):

**Old code:**
```python
# This extracted task execution periods (small movements)
reps = load_joint_angles_with_labels('sub-01', 'FroRea')
```

**New behavior:**
```python
# This NOW extracts movement periods (large movements) - DEFAULT CHANGED
reps = load_joint_angles_with_labels('sub-01', 'FroRea')
```

**To maintain old behavior:**
```python
# Explicitly request task execution periods
reps = load_joint_angles_with_labels('sub-01', 'FroRea', extract_movements=False)
```

### If you were using `use_rest_periods=True`:

**Old code:**
```python
reps = load_joint_angles_with_labels('sub-01', 'FroRea', use_rest_periods=True)
```

**New code (parameter renamed):**
```python
reps = load_joint_angles_with_labels('sub-01', 'FroRea', extract_movements=True)
# OR just use default:
reps = load_joint_angles_with_labels('sub-01', 'FroRea')
```

### If you were using `use_rest_periods=False`:

**Old code:**
```python
reps = load_joint_angles_with_labels('sub-01', 'Cyl', use_rest_periods=False)
```

**New code (parameter renamed and default flipped):**
```python
reps = load_joint_angles_with_labels('sub-01', 'Cyl', extract_movements=False)
```

## Usage Examples

### For Reaching Tasks (FroRea, ReaCyl, ReaSph)

```python
# Use default - extracts movement periods (actual reaching movements)
reaching_movements = load_joint_angles_with_labels('sub-01', 'FroRea')

# Result: 9 movements with large joint angle changes (56-64° elbow span)
for i, (time, joints) in enumerate(reaching_movements):
    print(f"Movement {i+1}: {len(time)} samples")
```

### For Grasping Tasks (Cyl, Sph, Trid)

```python
# Use extract_movements=False to get task execution periods
grasping_tasks = load_joint_angles_with_labels('sub-01', 'Cyl', 
                                                extract_movements=False)

# Result: 10 task execution periods
for i, (time, joints) in enumerate(grasping_tasks):
    print(f"Task {i+1}: {len(time)} samples")
```

### Extracting Single Joint from Movement

```python
# Load movements
movements = load_joint_angles_with_labels('sub-01', 'FroRea')

# Extract elbow X from first movement
time_movement1, joints_movement1 = movements[0]
elbow_x = joints_movement1['RElbow_X'].values  # numpy array

print(f"Elbow X span: {elbow_x.max() - elbow_x.min():.2f}°")
# Output: ~56° (large movement)
```

## Key Differences

| Aspect | Task Execution (extract_movements=False) | Movement Periods (extract_movements=True) |
|--------|------------------------------------------|-------------------------------------------|
| **Extraction** | Between start/end markers | Between end/next-start markers |
| **Count** | 10 periods | 9 periods (one less) |
| **Duration** | 1-2 seconds | 5-7 seconds |
| **FroRea Elbow Span** | 0.5-11° (small) | 56-64° (large) |
| **Use Case** | Grasping tasks | Reaching tasks |

## Testing

Run these commands to verify the changes:

```bash
# Test default behavior
python3 -c "from reachgrasp.data_loading import load_joint_angles_with_labels; \
reps = load_joint_angles_with_labels('sub-01', 'FroRea'); \
print(f'Loaded {len(reps)} movements (default)')"

# Test explicit extract_movements=False
python3 -c "from reachgrasp.data_loading import load_joint_angles_with_labels; \
reps = load_joint_angles_with_labels('sub-01', 'FroRea', extract_movements=False); \
print(f'Loaded {len(reps)} task periods')"

# Run demo
python3 demo_rest_periods.py
```

---

## Update 2: Zero-Clamping Feature

Added functionality to clean up near-zero numerical noise in motion capture data.

### Rationale

Motion capture systems often produce floating-point artifacts with magnitudes like 1e-12 or 1e-17 that should be treated as exactly zero. This is particularly common in joint angles during reaching tasks where certain degrees of freedom are constrained.

### New Parameters

Added to `load_timeseries_data()` and `load_joint_angles_with_labels()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clamp_near_zero` | bool | `True` | If True, sets near-zero values to exactly 0 |
| `zero_threshold` | float | `1e-10` | Absolute values below this are set to 0 |

### Default Behavior

**With clamping (default: `clamp_near_zero=True`):**
- Values with absolute magnitude < 1e-10 are set to 0.0
- Removes floating-point noise like 1e-12, 1e-17
- Produces cleaner data for analysis and visualization

**Without clamping (`clamp_near_zero=False`):**
- Preserves original raw values from CSV files
- Useful when exact numerical precision is required

### Examples

**Example 1: Elbow Joint Noise Removal**
```python
# Load with clamping (default)
movements = load_joint_angles_with_labels('sub-01', 'FroRea')
time, joints = movements[0]

# RElbow_Y: all 595 samples have magnitude < 1e-15 → clamped to 0.0
# RElbow_Z: all 595 samples have magnitude < 1e-12 → clamped to 0.0
print(joints['RElbow_Y'].min())  # 0.0
print(joints['RElbow_Z'].max())  # 0.0
```

**Example 2: Preserve Raw Values**
```python
# Load without clamping
movements = load_joint_angles_with_labels('sub-01', 'FroRea', clamp_near_zero=False)
time, joints = movements[0]

# Original near-zero values preserved
print(joints['RElbow_Y'].values[:3])  # [6.39e-17, 5.63e-17, 3.74e-17]
print(joints['RElbow_Z'].values[:3])  # [5.73e-13, 7.30e-13, 6.99e-13]
```

**Example 3: Custom Threshold**
```python
# More aggressive clamping (< 1e-6 → 0)
movements = load_timeseries_data('sub-01', 'FroRea', 'vicon', 'motion',
                                  zero_threshold=1e-6)
```

### Verification

Test script: `verify_clamping.py`
- Shows before/after comparison
- Verifies 595 near-zero values in RElbow_Y/Z → all clamped to 0
- Confirms significant values (> 1e-10) are preserved

### Impact

**Cleaned joints in FroRea task:**
- RElbow_Y: 595/595 samples had noise < 1e-15
- RElbow_Z: 595/595 samples had noise < 1e-12
- Both now exactly 0.0 (representing locked elbow pronation/supination)

**Benefits:**
- Cleaner plots (no spurious 1e-12 oscillations)
- Better derivative calculations (no noise amplification)
- More intuitive data interpretation
- Faster comparisons (exact zeros vs near-zeros)
```

## Date

October 22, 2025
