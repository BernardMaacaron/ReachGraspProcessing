import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reachgrasp.data_loading import (
    get_available_tasks,
    load_complete_recording,
    load_joint_angles_with_labels,
    load_timeseries_data,
)


def make_dataset(root: Path) -> Path:
    subject = "sub-01"
    task = "Cyl"
    motion_dir = root / subject / "motion"
    motion_dir.mkdir(parents=True)

    time = np.arange(100, dtype=float) / 100.0
    channels = np.column_stack([time + index for index in range(31)])
    recording = np.column_stack([time, channels])
    stem = f"{subject}_task-{task}_acq-vicon"
    pd.DataFrame(recording).to_csv(motion_dir / f"{stem}_motion.csv", header=False, index=False)
    (motion_dir / f"{stem}_motion.json").write_text(json.dumps({"SamplingFrequency": 100}))
    pd.DataFrame({"name": [f"channel-{index}" for index in range(31)]}).to_csv(
        motion_dir / f"{stem}_channels.tsv", sep="\t", index=False
    )

    cuts = {
        "Events_ReachGrasp": {
            "subjects": [{
                "subject_name": subject,
                "tasks": [{"task_name": task, "time2cut": [10, 20, 40, 50]}],
            }]
        }
    }
    (root / "timeCuts.json").write_text(json.dumps(cuts))
    return root


def test_continuous_and_split_loading_have_stable_types(tmp_path):
    dataset = make_dataset(tmp_path)

    time, data = load_timeseries_data(
        "sub-01", "Cyl", "vicon", "motion",
        base_path=str(dataset), split_repetitions=False
    )
    assert time.shape == (100,)
    assert data.shape == (100, 31)

    repetitions = load_timeseries_data(
        "sub-01", "Cyl", "vicon", "motion",
        base_path=str(dataset), split_repetitions=True,
        extract_movements=True, verbose=False
    )
    assert isinstance(repetitions, list)
    assert len(repetitions) == 1
    assert repetitions[0][0][0] == pytest.approx(0.0)
    assert repetitions[0][1].shape == (20, 31)


def test_split_loading_validates_only_selected_samples(tmp_path):
    dataset = make_dataset(tmp_path)
    path = dataset / "sub-01/motion/sub-01_task-Cyl_acq-vicon_motion.csv"
    recording = pd.read_csv(path, header=None)
    recording.iloc[[0, -1], 1:] = np.nan
    recording.to_csv(path, header=False, index=False)

    repetitions = load_timeseries_data(
        "sub-01", "Cyl", "vicon", "motion",
        base_path=str(dataset), split_repetitions=True,
        extract_movements=True, verbose=False
    )
    assert np.isfinite(repetitions[0][1]).all().all()

    with pytest.raises(ValueError, match="Channel data contains NaN"):
        load_timeseries_data(
            "sub-01", "Cyl", "vicon", "motion",
            base_path=str(dataset), split_repetitions=False
        )


def test_joint_loader_applies_vicon_labels(tmp_path):
    dataset = make_dataset(tmp_path)
    repetitions = load_joint_angles_with_labels(
        "sub-01", "Cyl", base_path=str(dataset),
        split_repetitions=True, verbose=False
    )

    _, joints = repetitions[0]
    assert joints.shape == (20, 31)
    assert "RShoulder_X" in joints.columns
    assert "RWrist_Z" in joints.columns


def test_complete_recording_is_unsplit(tmp_path):
    dataset = make_dataset(tmp_path)
    recording = load_complete_recording(
        "sub-01", "Cyl", "vicon", "motion", base_path=str(dataset)
    )

    assert recording["time"].shape == (100,)
    assert recording["data"].shape == (100, 31)
    assert recording["metadata"]["SamplingFrequency"] == 100


def test_missing_time_cuts_does_not_fall_back_to_continuous_data(tmp_path):
    dataset = make_dataset(tmp_path)
    (dataset / "timeCuts.json").unlink()

    with pytest.raises(FileNotFoundError):
        load_timeseries_data(
            "sub-01", "Cyl", "vicon", "motion",
            base_path=str(dataset), split_repetitions=True,
            verbose=False
        )


def test_motion_task_discovery(tmp_path):
    dataset = make_dataset(tmp_path)
    assert get_available_tasks("sub-01", modality="motion", base_path=str(dataset)) == ["Cyl"]
