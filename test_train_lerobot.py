"""Tests for camera wiring and norm-stat sampling that do not need JAX."""

from pathlib import Path

import numpy as np

from train_lerobot import GenericLeRobotInputs
from train_lerobot import resolve_keep_period
from train_lerobot import select_parquet_files


def test_missing_camera_slot_is_masked() -> None:
    transform = GenericLeRobotInputs(
        camera_slots=(
            ("base_0_rgb", "observation.images.top"),
            ("left_wrist_0_rgb", "observation.images.wrist"),
        ),
        state_key="observation.state",
        action_key="action",
    )
    top = np.zeros((4, 5, 3), dtype=np.uint8)
    wrist = np.ones((4, 5, 3), dtype=np.uint8)
    out = transform(
        {
            "observation.images.top": top,
            "observation.images.wrist": wrist,
            "observation.state": np.zeros(6, dtype=np.float32),
            "action": np.ones(6, dtype=np.float32),
        }
    )
    assert bool(out["image_mask"]["base_0_rgb"])
    assert bool(out["image_mask"]["left_wrist_0_rgb"])
    assert not bool(out["image_mask"]["right_wrist_0_rgb"])
    assert out["image"]["right_wrist_0_rgb"].shape == (4, 5, 3)
    assert out["image"]["right_wrist_0_rgb"].sum() == 0
    assert np.array_equal(out["actions"], np.ones(6, dtype=np.float32))


def test_norm_stats_frame_cap_defaults_to_every_file() -> None:
    files = [Path(f"file-{idx:03d}.parquet") for idx in range(10)]
    assert select_parquet_files(files, None, 100) == files
    assert select_parquet_files(files, 0, 100) == files
    first = select_parquet_files(files, 100, 100, seed=0)
    assert len(first) == 2
    assert select_parquet_files(files, 100, 100, seed=0) == first


def test_keep_period_follows_save_interval_by_default() -> None:
    # max_to_keep=1 prunes anything outside keep_period, so saves must be covered.
    assert resolve_keep_period(4000, None) == 4000
    assert resolve_keep_period(500, None) == 500
    assert resolve_keep_period(4000, 20000) == 20000
    assert resolve_keep_period(4000, 0) is None
    assert resolve_keep_period(0, None) is None
