"""Tests for packed LeRobot v3.0 -> v2.1 conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import numpy as np

from lerobot_v3_compat import DatasetLayoutError
from lerobot_v3_compat import _numeric_feature_names
from lerobot_v3_compat import assert_v2_local_files
from lerobot_v3_compat import assert_v21_episode_stats_rows
from lerobot_v3_compat import convert_v3_to_v2
from lerobot_v3_compat import episodes_stats_compatible_with_v21
from lerobot_v3_compat import expected_v2_paths
from lerobot_v3_compat import generate_episodes_stats_from_parquet
from lerobot_v3_compat import load_sanitized_stats_json
from lerobot_v3_compat import load_tasks
from lerobot_v3_compat import sanitize_episode_stats
from lerobot_v3_compat import stats_from_episode_record
from lerobot_v3_compat import tasks_have_text
from lerobot_v3_compat import unflatten_dict
from lerobot_v3_compat import v2_chunk
from lerobot_v3_compat import v2_data_relpath
from lerobot_v3_compat import v2_video_relpath

CAM_A = "observation.images.camera_high"
CAM_B = "observation.images.camera_left_wrist"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _fake_extract(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(f"{src.name}:{start:.3f}:{end:.3f}".encode())


def _packed_v3_info() -> dict:
    video_feat = {
        "dtype": "video",
        "shape": [16, 16, 3],
        "names": ["height", "width", "channels"],
    }
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "robot_type": "test",
        "total_episodes": 3,
        "total_frames": 6,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "splits": {"train": "0:3"},
        "features": {
            "action": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
            "observation.state": {"dtype": "float32", "shape": [2], "names": ["a", "b"]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            CAM_A: video_feat,
            CAM_B: video_feat,
        },
    }


def _v3_stats_columns(*, include_numeric: bool = True, empty_video: bool = True) -> dict:
    cols: dict = {}
    if include_numeric:
        cols.update(
            {
                "stats/action/min": [[0.1, 0.2], [0.5, 0.6], [0.9, 1.0]],
                "stats/action/max": [[0.3, 0.4], [0.7, 0.8], [1.1, 1.2]],
                "stats/action/mean": [[0.2, 0.3], [0.6, 0.7], [1.0, 1.1]],
                "stats/action/std": [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
                "stats/action/count": [[2], [2], [2]],
                "stats/action/q01": [[0.1, 0.2], [0.5, 0.6], [0.9, 1.0]],
                "stats/observation.state/min": [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]],
                "stats/observation.state/max": [[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]],
                "stats/observation.state/mean": [[0.05, 0.05], [0.25, 0.25], [0.45, 0.45]],
                "stats/observation.state/std": [[0.05, 0.05], [0.05, 0.05], [0.05, 0.05]],
                "stats/observation.state/count": [[2], [2], [2]],
            }
        )
    if empty_video:
        cols.update(
            {
                f"stats/{CAM_A}/min": [[], [], []],
                f"stats/{CAM_A}/max": [[], [], []],
                f"stats/{CAM_A}/mean": [[], [], []],
                f"stats/{CAM_A}/std": [[], [], []],
                f"stats/{CAM_A}/count": [[], [], []],
            }
        )
    return cols


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_packed_v3(
    root: Path,
    *,
    missing_video: str | None = None,
    include_stats: bool = True,
) -> Path:
    """Build a 3-episode packed v3 tree that mirrors the production layout."""
    _write_json(root / "meta" / "info.json", _packed_v3_info())

    frames = {
        "episode_index": [0, 0, 1, 1, 2, 2],
        "frame_index": [0, 1, 0, 1, 0, 1],
        "index": [0, 1, 2, 3, 4, 5],
        "task_index": [0, 0, 0, 0, 0, 0],
        "timestamp": [0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
        "action": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
        "observation.state": [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
    }
    _write_table(root / "data" / "chunk-000" / "file-000.parquet", pa.table(frames))

    episodes = {
        "episode_index": [0, 1, 2],
        "length": [2, 2, 2],
        "task_index": [0, 0, 0],
        "tasks": [["pick"], ["pick"], ["pick"]],
        "dataset_from_index": [0, 2, 4],
        "dataset_to_index": [2, 4, 6],
        "data/chunk_index": [0, 0, 0],
        "data/file_index": [0, 0, 0],
        f"videos/{CAM_A}/chunk_index": [0, 0, 0],
        f"videos/{CAM_A}/file_index": [0, 0, 1],
        f"videos/{CAM_A}/from_timestamp": [0.0, 1.0, 0.0],
        f"videos/{CAM_A}/to_timestamp": [1.0, 2.0, 1.0],
        f"videos/{CAM_B}/chunk_index": [0, 0, 0],
        f"videos/{CAM_B}/file_index": [0, 1, 2],
        f"videos/{CAM_B}/from_timestamp": [0.0, 0.0, 0.0],
        f"videos/{CAM_B}/to_timestamp": [1.0, 1.0, 1.0],
    }
    if include_stats:
        episodes.update(_v3_stats_columns())
    _write_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", pa.table(episodes))

    _write_table(
        root / "meta" / "tasks.parquet",
        pa.table({"__index_level_0__": ["pick the block"], "task_index": [0]}),
    )

    video_files = {
        CAM_A: ["file-000.mp4", "file-001.mp4"],
        CAM_B: ["file-000.mp4", "file-001.mp4", "file-002.mp4", "file-003.mp4"],
    }
    for camera, names in video_files.items():
        for name in names:
            path = root / "videos" / camera / "chunk-000" / name
            if missing_video and path.as_posix().endswith(missing_video):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-mp4-" + name.encode())
    return root


def test_v2_chunk_formula() -> None:
    assert v2_chunk(0, 2) == 0
    assert v2_chunk(1, 2) == 0
    assert v2_chunk(2, 2) == 1
    assert v2_chunk(1058, 1000) == 1
    assert v2_data_relpath(1058, 1000) == "data/chunk-001/episode_001058.parquet"
    assert (
        v2_video_relpath(1058, CAM_A, 1000)
        == f"videos/chunk-001/{CAM_A}/episode_001058.mp4"
    )


def test_load_tasks_index_level_column(tmp_path: Path) -> None:
    _write_table(
        tmp_path / "meta" / "tasks.parquet",
        pa.table({"__index_level_0__": ["open the drawer"], "task_index": [3]}),
    )
    tasks = load_tasks(tmp_path)
    assert tasks == [{"task_index": 3, "task": "open the drawer"}]


def test_convert_packed_v3_writes_v2_chunks(tmp_path: Path) -> None:
    src = build_packed_v3(tmp_path / "v3")
    dest = convert_v3_to_v2(
        src,
        tmp_path / "v2",
        chunks_size=2,
        extract_video=_fake_extract,
    )

    info = json.loads((dest / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["chunks_size"] == 2
    assert "data_files_size_in_mb" not in info
    assert "video_files_size_in_mb" not in info
    assert info["data_path"].startswith("data/chunk-{episode_chunk:03d}/episode_")

    assert (dest / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (dest / "data" / "chunk-000" / "episode_000001.parquet").is_file()
    assert (dest / "data" / "chunk-001" / "episode_000002.parquet").is_file()
    assert not (dest / "data" / "chunk-000" / "episode_000002.parquet").exists()

    ep2 = pq.read_table(dest / "data" / "chunk-001" / "episode_000002.parquet")
    assert ep2.column("episode_index").to_pylist() == [2, 2]
    assert len(ep2) == 2

    assert (dest / "videos" / "chunk-000" / CAM_A / "episode_000000.mp4").is_file()
    assert (dest / "videos" / "chunk-000" / CAM_A / "episode_000001.mp4").is_file()
    assert (dest / "videos" / "chunk-001" / CAM_A / "episode_000002.mp4").is_file()
    assert (dest / "videos" / "chunk-001" / CAM_B / "episode_000002.mp4").is_file()

    # Shared cam_a file-000 must be split by timestamp, not reused as a whole file.
    cam_a_ep0 = (dest / "videos" / "chunk-000" / CAM_A / "episode_000000.mp4").read_bytes()
    cam_a_ep1 = (dest / "videos" / "chunk-000" / CAM_A / "episode_000001.mp4").read_bytes()
    assert cam_a_ep0 == b"file-000.mp4:0.000:1.000"
    assert cam_a_ep1 == b"file-000.mp4:1.000:2.000"
    cam_a_ep2 = dest / "videos" / "chunk-001" / CAM_A / "episode_000002.mp4"
    assert cam_a_ep2.is_symlink()
    assert cam_a_ep2.resolve().name == "file-001.mp4"

    # cam_b file_index is independent of the single data parquet (always file-000).
    assert (dest / "videos" / "chunk-000" / CAM_B / "episode_000001.mp4").is_symlink()
    assert (dest / "videos" / "chunk-000" / CAM_B / "episode_000001.mp4").resolve().name == "file-001.mp4"

    tasks = [
        json.loads(line)
        for line in (dest / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert tasks == [{"task_index": 0, "task": "pick the block"}]
    assert tasks_have_text(dest)

    episodes = [
        json.loads(line)
        for line in (dest / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["episode_index"] for row in episodes] == [0, 1, 2]
    assert episodes[0]["tasks"] == ["pick"]


def test_missing_video_file_fails_with_path(tmp_path: Path) -> None:
    src = build_packed_v3(tmp_path / "v3", missing_video=f"{CAM_A}/chunk-000/file-001.mp4")
    with pytest.raises(FileNotFoundError, match="file-001.mp4") as exc_info:
        convert_v3_to_v2(
            src,
            tmp_path / "v2",
            chunks_size=2,
            extract_video=_fake_extract,
        )
    assert CAM_A in str(exc_info.value)


def test_assert_v2_local_files_lists_missing_chunk(tmp_path: Path) -> None:
    dest = tmp_path / "partial"
    (dest / "data" / "chunk-000").mkdir(parents=True)
    (dest / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(b"x")
    info = {
        "chunks_size": 2,
        "features": {CAM_A: {"dtype": "video"}},
    }
    with pytest.raises(DatasetLayoutError, match="missing=") as exc_info:
        assert_v2_local_files(dest, info, [0, 2], chunks_size=2)
    message = str(exc_info.value)
    assert "chunk-001/episode_000002.parquet" in message
    assert f"videos/chunk-000/{CAM_A}/episode_000000.mp4" in message


def test_expected_v2_paths_include_videos() -> None:
    info = {"chunks_size": 1000, "features": {CAM_A: {"dtype": "video"}, "action": {"dtype": "float32"}}}
    paths = expected_v2_paths(info, [0, 1000], chunks_size=1000)
    assert "data/chunk-000/episode_000000.parquet" in paths
    assert "data/chunk-001/episode_001000.parquet" in paths
    assert f"videos/chunk-001/{CAM_A}/episode_001000.mp4" in paths
    assert all("action" not in path for path in paths)


def test_unflatten_keeps_dotted_feature_names() -> None:
    nested = unflatten_dict(
        {
            "stats/action/count": [2],
            "stats/observation.state/min": [0.0, 0.1],
            f"stats/{CAM_A}/count": [],
        }
    )
    assert nested["stats"]["action"]["count"] == [2]
    assert nested["stats"]["observation.state"]["min"] == [0.0, 0.1]
    assert nested["stats"][CAM_A]["count"] == []


def test_sanitize_drops_empty_image_stats_and_quantiles() -> None:
    raw = stats_from_episode_record(
        {
            "stats/action/min": [0.1, 0.2],
            "stats/action/max": [0.3, 0.4],
            "stats/action/mean": [0.2, 0.3],
            "stats/action/std": [0.1, 0.1],
            "stats/action/count": [],
            "stats/action/q01": [0.1, 0.2],
            f"stats/{CAM_A}/min": [],
            f"stats/{CAM_A}/max": [],
            f"stats/{CAM_A}/mean": [],
            f"stats/{CAM_A}/std": [],
            f"stats/{CAM_A}/count": [],
        }
    )
    cleaned = sanitize_episode_stats(raw, length=2)
    assert set(cleaned) == {"action"}
    assert cleaned["action"]["count"] == [2]
    assert "q01" not in cleaned["action"]
    assert CAM_A not in cleaned


def test_episodes_stats_compatible_rejects_empty_count() -> None:
    bad = [{"episode_index": 0, "stats": {CAM_A: {"count": [], "min": []}}}]
    with pytest.raises(DatasetLayoutError, match=r"count shape must be \(1,\)"):
        assert_v21_episode_stats_rows(bad)
    assert not episodes_stats_compatible_with_v21(bad)

    good = [
        {
            "episode_index": 0,
            "stats": {
                "action": {
                    "min": [0.0],
                    "max": [1.0],
                    "mean": [0.5],
                    "std": [0.1],
                    "count": [2],
                }
            },
        }
    ]
    assert_v21_episode_stats_rows(good)
    assert episodes_stats_compatible_with_v21(good)
    assert np.asarray(good[0]["stats"]["action"]["count"]).shape == (1,)


def test_convert_nests_stats_and_drops_empty_camera_count(tmp_path: Path) -> None:
    src = build_packed_v3(tmp_path / "v3")
    dest = convert_v3_to_v2(
        src,
        tmp_path / "v2",
        chunks_size=2,
        extract_video=_fake_extract,
    )
    rows = _load_jsonl(dest / "meta" / "episodes_stats.jsonl")
    assert [row["episode_index"] for row in rows] == [0, 1, 2]
    first = rows[0]["stats"]
    assert first["action"]["count"] == [2]
    assert "min" in first["action"]
    assert "q01" not in first["action"]
    assert CAM_A not in first
    assert CAM_B not in first
    assert_v21_episode_stats_rows(rows)
    assert episodes_stats_compatible_with_v21(dest / "meta" / "episodes_stats.jsonl")
    assert np.asarray(first["action"]["count"]).shape == (1,)


def test_convert_without_v3_stats_falls_back_to_parquet(tmp_path: Path) -> None:
    src = build_packed_v3(tmp_path / "v3", include_stats=False)
    dest = convert_v3_to_v2(
        src,
        tmp_path / "v2",
        chunks_size=2,
        extract_video=_fake_extract,
    )
    rows = _load_jsonl(dest / "meta" / "episodes_stats.jsonl")
    assert rows[0]["stats"]["action"]["count"] == [2]
    assert rows[0]["stats"]["observation.state"]["count"] == [2]
    assert CAM_A not in rows[0]["stats"]
    assert_v21_episode_stats_rows(rows)


def test_ensure_v21_episodes_stats_rewrites_incompatible_file(tmp_path: Path) -> None:
    from train_lerobot import ensure_v21_episodes_stats

    dest = tmp_path / "v2"
    _write_table(
        dest / "data" / "chunk-000" / "episode_000000.parquet",
        pa.table({"action": [[0.1, 0.2], [0.3, 0.4]]}),
    )
    episodes = dest / "meta" / "episodes.jsonl"
    episodes.parent.mkdir(parents=True, exist_ok=True)
    episodes.write_text(json.dumps({"episode_index": 0, "length": 2, "tasks": ["pick"]}) + "\n", encoding="utf-8")
    bad = dest / "meta" / "episodes_stats.jsonl"
    bad.write_text(
        json.dumps({"episode_index": 0, "stats": {CAM_A: {"count": []}}}) + "\n",
        encoding="utf-8",
    )
    ensure_v21_episodes_stats(dest, {"chunks_size": 1000, "total_episodes": 1})
    rows = _load_jsonl(bad)
    assert rows[0]["stats"]["action"]["count"] == [2]
    assert episodes_stats_compatible_with_v21(bad)


def test_generate_episodes_stats_overwrites_incompatible_file(tmp_path: Path) -> None:
    dest = tmp_path / "v2"
    table = pa.table(
        {
            "action": [[0.1, 0.2], [0.3, 0.4]],
            "observation.state": [[0.0, 0.0], [0.1, 0.1]],
        }
    )
    pq_path = dest / "data" / "chunk-000" / "episode_000000.parquet"
    _write_table(pq_path, table)
    bad = dest / "meta" / "episodes_stats.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        json.dumps({"episode_index": 0, "stats": {CAM_A: {"count": [], "min": []}}}) + "\n",
        encoding="utf-8",
    )
    assert not episodes_stats_compatible_with_v21(bad)
    generate_episodes_stats_from_parquet(dest, [0], chunks_size=1000)
    assert episodes_stats_compatible_with_v21(bad)
    rows = _load_jsonl(bad)
    assert rows[0]["stats"]["action"]["count"] == [2]


def _image_channel_stat(value: float) -> list:
    return [[[value]], [[value]], [[value]]]


def _real_v3_global_stats() -> dict:
    """Subset of the production v3 stats.json: quantiles + valid (3,1,1) cameras."""
    return {
        "action": {
            "count": [315225],
            "min": [-1.7] * 14,
            "max": [1.0] * 14,
            "mean": [-0.9] * 14,
            "std": [0.2] * 14,
            "q01": [-1.0] * 14,
            "q99": [-0.8] * 14,
        },
        "observation.state": {
            "count": [315225],
            "min": [-1.7] * 14,
            "max": [1.0] * 14,
            "mean": [-0.9] * 14,
            "std": [0.2] * 14,
            "q01": [-1.0] * 14,
            "q99": [-0.8] * 14,
        },
        "observation.base_move": {
            "count": [315225],
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
            "mean": [0.0, 0.0, 0.0],
            "std": [0.0, 0.0, 0.0],
        },
        CAM_A: {
            "count": [18549],
            "min": _image_channel_stat(0.0),
            "max": _image_channel_stat(1.0),
            "mean": _image_channel_stat(0.46),
            "std": _image_channel_stat(0.22),
        },
        CAM_B: {
            "count": [18549],
            "min": _image_channel_stat(0.0),
            "max": _image_channel_stat(1.0),
            "mean": _image_channel_stat(0.47),
            "std": _image_channel_stat(0.26),
        },
    }


def test_sanitize_global_stats_drops_quantiles_keeps_images() -> None:
    cleaned = sanitize_episode_stats(_real_v3_global_stats(), length=0)
    assert "q01" not in cleaned["action"]
    assert cleaned["action"]["count"] == [315225]
    assert cleaned[CAM_A]["count"] == [18549]
    assert np.asarray(cleaned[CAM_A]["mean"]).shape == (3, 1, 1)
    assert_v21_episode_stats_rows([{"episode_index": 0, "stats": cleaned}])


def test_numeric_feature_names_accepts_float64_fixed_size_list() -> None:
    table = pa.table(
        {
            "action": pa.array([[0.1] * 14, [0.2] * 14], type=pa.list_(pa.float64(), 14)),
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "note": pa.array(["a", "b"]),
        }
    )
    names = _numeric_feature_names(table)
    assert "action" in names
    assert "episode_index" in names
    assert "note" not in names


def test_convert_official_v21_from_real_v3_stats_layout(tmp_path: Path) -> None:
    src = build_packed_v3(tmp_path / "v3")
    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    info["features"]["action"]["dtype"] = "float64"
    info["features"]["action"]["fps"] = 30
    info["features"]["observation.base_move"] = {
        "dtype": "float64",
        "shape": [3],
        "names": ["x", "y", "theta"],
        "fps": 30,
    }
    _write_json(src / "meta" / "info.json", info)
    _write_json(src / "meta" / "stats.json", _real_v3_global_stats())

    dest = convert_v3_to_v2(
        src,
        tmp_path / "v2",
        chunks_size=2,
        extract_video=_fake_extract,
    )

    v2_info = json.loads((dest / "meta" / "info.json").read_text(encoding="utf-8"))
    assert v2_info["codebase_version"] == "v2.1"
    assert "fps" not in v2_info["features"]["action"]
    assert "fps" not in v2_info["features"]["observation.base_move"]
    assert v2_info["features"][CAM_A]["dtype"] == "video"

    global_stats = json.loads((dest / "meta" / "stats.json").read_text(encoding="utf-8"))
    assert "q01" not in global_stats["action"]
    assert set(global_stats[CAM_A]) == {"min", "max", "mean", "std", "count"}
    assert np.asarray(global_stats[CAM_A]["min"]).shape == (3, 1, 1)
    assert load_sanitized_stats_json(dest / "meta" / "stats.json") == global_stats

    rows = _load_jsonl(dest / "meta" / "episodes_stats.jsonl")
    first = rows[0]["stats"]
    assert first["action"]["count"] == [2]
    assert "q01" not in first["action"]
    assert first[CAM_A]["count"] == [2]
    assert np.asarray(first[CAM_A]["mean"]).shape == (3, 1, 1)
    assert first[CAM_B]["count"] == [2]
    assert "observation.base_move" in first
    assert_v21_episode_stats_rows(rows)
