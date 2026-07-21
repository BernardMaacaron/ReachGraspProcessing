"""
Example usage of ReachGrasp dataset utilities.

This script demonstrates common workflows for loading, processing,
analyzing, and visualizing the ReachGrasp dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import utilities
from reachgrasp import(
    load_timeseries_data,
    load_joint_angles_with_labels,
    get_joint_groups,
    get_region_colors,
    bandpass_filter,
    calculate_rms,
    resample_signal,
    analyze_joint_coordination,
    plot_emg_channels,
    plot_motion_channels,
    get_channel_labels
)


def example_emg_processing():
    """Example: Load and process EMG data."""
    print("=" * 70)
    print("EXAMPLE 1: EMG Processing")
    print("=" * 70)
    
    # Load EMG data
    print("\n1. Loading EMG data...")
    time, emg = load_timeseries_data('sub-01', 'Cyl', 'cometa', 'emg')
    print(f"   Loaded {emg.shape[1]} EMG channels, {len(time)} samples")
    print(f"   Duration: {time[-1]:.2f} seconds")
    
    # Filter EMG
    print("\n2. Filtering EMG (20-500 Hz bandpass)...")
    emg_filtered = bandpass_filter(emg, lowcut=20, highcut=500, fs=2000)
    print(f"   Filtered signal shape: {emg_filtered.shape}")
    
    # Calculate RMS
    print("\n3. Calculating RMS (100ms windows)...")
    emg_rms = calculate_rms(emg_filtered, window_size=200, step_size=100)
    print(f"   RMS shape: {emg_rms.shape}")
    
    # Get channel labels
    labels = get_channel_labels('emg', 'cometa', n_channels=10)
    print(f"\n4. Channel labels (first 3):")
    for i, label in enumerate(labels[:3]):
        print(f"   {label}")
    
    # Visualize
    print("\n5. Plotting EMG channels...")
    plot_emg_channels(time, emg_filtered, show_labels=True, acquisition='cometa')
    
    return time, emg, emg_filtered, emg_rms


def example_motion_analysis():
    """Example: Analyze joint kinematics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Motion Analysis")
    print("=" * 70)
    
    # Load motion data with labels
    print("\n1. Loading joint angle data with anatomical labels...")
    time, joint_angles = load_joint_angles_with_labels('sub-01', 'FroRea', 'vicon')
    print(f"   Loaded {joint_angles.shape[1]} joint angles")
    print(f"   Column names (first 5): {list(joint_angles.columns[:5])}")
    
    # Get anatomical grouping
    print("\n2. Getting anatomical regions...")
    joint_groups = get_joint_groups()
    print(f"   {len(joint_groups)} anatomical regions:")
    for region in list(joint_groups.keys())[:5]:
        print(f"   - {region}: {len(joint_groups[region])} joints")
    
    # Calculate statistics by region
    print("\n3. Calculating statistics by region...")
    for region_name, joint_list in list(joint_groups.items())[:3]:
        region_data = joint_angles[joint_list]
        mean_range = (region_data.max() - region_data.min()).mean()
        print(f"   {region_name}: Average range = {mean_range:.1f}°")
    
    # Correlation analysis
    print("\n4. Analyzing joint correlations...")
    corr_matrix = joint_angles.corr()  # Use pandas built-in
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    print(f"   Average correlation: {avg_corr:.3f}")
    
    # Visualize
    print("\n5. Plotting joint angles by anatomical region...")
    plot_motion_channels(time, joint_angles, show_labels=True, acquisition='vicon')
    
    return time, joint_angles, joint_groups


def example_multimodal_sync():
    """Example: Synchronize EMG and motion data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multi-Modal Synchronization")
    print("=" * 70)
    
    # Load EMG (2000 Hz)
    print("\n1. Loading EMG data (2000 Hz)...")
    emg_time, emg = load_timeseries_data('sub-01', 'Cyl', 'cometa', 'emg')
    print(f"   EMG: {len(emg_time)} samples, {emg_time[-1]:.2f} seconds")
    
    # Load motion (100 Hz)
    print("\n2. Loading motion data (100 Hz)...")
    motion_time, motion = load_timeseries_data('sub-01', 'Cyl', 'vicon', 'motion')
    print(f"   Motion: {len(motion_time)} samples, {motion_time[-1]:.2f} seconds")
    
    # Resample EMG to match motion
    print("\n3. Resampling EMG to match motion time base...")
    emg_resampled = resample_signal(emg_time, emg, motion_time)
    print(f"   Resampled EMG shape: {emg_resampled.shape}")
    print(f"   Now both signals have {len(motion_time)} samples")
    
    # Verify synchronization
    print("\n4. Verifying synchronization...")
    print(f"   EMG resampled: {emg_resampled.shape[0]} samples")
    print(f"   Motion: {motion.shape[0]} samples")
    print(f"   Time vectors match: {len(motion_time) == emg_resampled.shape[0]}")
    
    return emg_time, emg, motion_time, motion, emg_resampled


def example_coordination_analysis():
    """Example: Analyze joint coordination."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Joint Coordination Analysis")
    print("=" * 70)
    
    # Load joint angles
    print("\n1. Loading joint angle data...")
    time, joint_angles = load_joint_angles_with_labels('sub-01', 'FroRea', 'vicon')
    
    # Analyze coordination
    print("\n2. Analyzing joint coordination (this may take a moment)...")
    coord_results = analyze_joint_coordination(
        joint_angles, 
        time, 
        max_lag=50,
        min_correlation=0.5
    )
    
    print(f"\n3. Found {len(coord_results)} coordinated joint pairs")
    print("\nTop 5 coordinated pairs:")
    print(f"{'Joint 1':<20} {'Joint 2':<20} {'Correlation':>12} {'Time Lag':>12}")
    print("-" * 70)
    
    for result in coord_results[:5]:
        direction = "→" if result['lag'] > 0 else "←" if result['lag'] < 0 else "↔"
        print(f"{result['joint1']:<20} {direction:^3} {result['joint2']:<17} "
              f"{result['correlation']:>11.3f} {result['lag_time']:>11.3f}s")
    
    return coord_results


def example_visualization_comparison():
    """Example: Compare different visualization modes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Visualization Comparison")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading EMG data...")
    time, emg = load_timeseries_data('sub-01', 'Cyl', 'cometa', 'emg')
    emg_filtered = bandpass_filter(emg, lowcut=20, highcut=500, fs=2000)
    
    # Grid layout
    print("\n2. Plotting grid layout (each channel separate)...")
    plot_emg_channels(time, emg_filtered, show_labels=True, 
                     acquisition='cometa', overlapping=False)
    
    # Overlapping layout
    print("\n3. Plotting overlapping layout (all channels together)...")
    plot_emg_channels(time, emg_filtered, show_labels=True,
                     acquisition='cometa', overlapping=True)


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("ReachGrasp Dataset Utilities - Example Usage")
    print("=" * 70)
    
    try:
        # Run examples
        example_emg_processing()
        example_motion_analysis()
        example_multimodal_sync()
        example_coordination_analysis()
        example_visualization_comparison()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
