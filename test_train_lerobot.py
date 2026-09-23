"""Tests for camera wiring and norm-stat sampling that do not need JAX."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from train_lerobot import GenericLeRobotInputs
from train_lerobot import _consume_in_order
from train_lerobot import delta_action_chunks
from train_lerobot import enable_line_buffered_stdout
from train_lerobot import install_resume_norm_stats
from train_lerobot import newest_checkpoint_norm_stats
from train_lerobot import relaxed_timestamp_tolerance
from train_lerobot import require_delta_for_absolute_dims
from train_lerobot import resolve_delta_mask
from train_lerobot import resolve_keep_period
from train_lerobot import select_parquet_files
from train_lerobot import stabilize_norm_stats


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


def test_enable_line_buffered_stdout_tolerates_unbuffered_streams() -> None:
    # pytest replaces sys.stdout with a stream that cannot be reconfigured.
    enable_line_buffered_stdout()


WIPE_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
    "left_gripper",
]


def test_timestamp_tolerance_covers_float32_but_not_a_dropped_frame() -> None:
    fps = 30.0
    tolerance = relaxed_timestamp_tolerance(fps)
    float32_error = 0.033447265625 - (1.0 / fps)
    assert float32_error < tolerance
    assert (1.0 / fps) > tolerance
    assert relaxed_timestamp_tolerance(fps, 0.05) == 0.05
    assert relaxed_timestamp_tolerance(None) == 1e-4


def test_delta_mask_keeps_only_the_last_dim_by_default() -> None:
    mask = resolve_delta_mask(14, None, WIPE_NAMES)
    assert mask == (True,) * 13 + (False,)
    assert resolve_delta_mask(6, None, None) == (True, True, True, True, True, False)


def test_absolute_action_dims_replace_the_last_dim_default() -> None:
    expected = (True,) * 12 + (False, False)
    assert resolve_delta_mask(14, "right_gripper,left_gripper", WIPE_NAMES) == expected
    assert resolve_delta_mask(14, "12,13", WIPE_NAMES) == expected


def test_absolute_action_dims_require_delta_and_known_names() -> None:
    with pytest.raises(ValueError, match="requires --delta_joint_actions"):
        require_delta_for_absolute_dims(False, "right_gripper")
    require_delta_for_absolute_dims(True, "right_gripper")
    require_delta_for_absolute_dims(False, None)
    with pytest.raises(ValueError, match="Unknown action dim"):
        resolve_delta_mask(14, "not_a_joint", WIPE_NAMES)


def test_delta_chunks_subtract_state_and_clamp_the_episode_end() -> None:
    state = np.array(
        [[0, 0, 10], [1, 0, 10], [2, 0, 10], [3, 0, 10], [4, 0, 10]],
        dtype=np.float32,
    )
    actions = np.array(
        [[0, 1, 10], [1, 1, 11], [2, 1, 12], [3, 1, 13], [4, 1, 14]],
        dtype=np.float32,
    )
    chunks = delta_action_chunks(state, actions, (True, True, False), 3)
    assert chunks.shape == (5, 3, 3)
    assert chunks[0, 0].tolist() == [0, 1, 10]
    assert chunks[0, 1].tolist() == [1, 1, 11]
    assert chunks[0, 2].tolist() == [2, 1, 12]
    assert chunks[4, 0].tolist() == [0, 1, 14]
    assert chunks[4, 2].tolist() == [0, 1, 14]
    raw_mean = actions.mean(axis=0)
    chunk_mean = chunks.reshape(-1, 3).mean(axis=0)
    assert not np.allclose(raw_mean, chunk_mean)


@dataclasses.dataclass
class _Stats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def test_quantile_guard_widens_only_constant_dimensions() -> None:
    constant = _Stats(
        mean=np.array([0.0, 5.0]),
        std=np.array([1.0, 2.0]),
        q01=np.array([0.0, 0.0]),
        q99=np.array([0.0, 4.0]),
    )
    updated = stabilize_norm_stats({"actions": constant})["actions"]
    assert updated.std is constant.std
    assert np.allclose(updated.mean, constant.mean)
    assert np.allclose(updated.q01, [-0.5, 0.0])
    assert np.allclose(updated.q99, [0.5, 4.0])

    varying = _Stats(
        mean=np.array([1.0, 2.0]),
        std=np.array([0.5, 0.5]),
        q01=np.array([0.0, 0.0]),
        q99=np.array([2.0, 4.0]),
    )
    assert stabilize_norm_stats({"state": varying})["state"] is varying


def test_resume_reuses_the_newest_checkpoint_norm_stats(tmp_path: Path) -> None:
    older = tmp_path / "1000" / "assets" / "training_dataset"
    newer = tmp_path / "2000" / "assets" / "training_dataset"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "norm_stats.json").write_text("old", encoding="utf-8")
    (newer / "norm_stats.json").write_text("new", encoding="utf-8")
    found = newest_checkpoint_norm_stats(tmp_path, "training_dataset")
    assert found is not None
    assert found.read_text(encoding="utf-8") == "new"
    assets = tmp_path / "assets" / "run"
    installed = install_resume_norm_stats(tmp_path, assets, "training_dataset")
    assert installed is not None
    assert installed.read_text(encoding="utf-8") == "new"
    assert newest_checkpoint_norm_stats(tmp_path / "missing", "training_dataset") is None


def test_fold_results_in_submission_order() -> None:
    """Norm stats are order-sensitive, so thread timing must not reorder them."""
    import concurrent.futures
    import time

    def finish_later(value: int, delay: float) -> int:
        time.sleep(delay)
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        # Earliest submissions finish last, so completion order is reversed.
        futures = [pool.submit(finish_later, value, 0.02 * (4 - value)) for value in range(5)]
        assert list(_consume_in_order(futures)) == [0, 1, 2, 3, 4]
