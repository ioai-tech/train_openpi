"""Convert packed LeRobot v3.0 datasets to the v2.1 layout OpenPI can read.

Official OpenPI pins an old LeRobot that only understands v2.1 (one parquet/mp4
per episode). This module ports the NVIDIA GR00T convert_v3_to_v2 algorithm
without importing modern ``lerobot.datasets.utils``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("lerobot_v3_compat")

V2_CHUNKS_SIZE = 1000
V21 = "v2.1"
V30 = "v3.0"
MIN_VIDEO_DURATION = 1e-6
MISSING_PATH_LIMIT = 20

V2_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V2_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
CONVERT_STAMP_NAME = ".convert_complete.json"
VIEW_STAMP_NAME = ".view_complete.json"
MODEL_IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_SLOT_ALIASES = {
    "base": "base_0_rgb",
    "base_0_rgb": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "left_wrist_0_rgb": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
    "right_wrist_0_rgb": "right_wrist_0_rgb",
}

TASK_DESCRIPTION_COLUMNS = (
    "task",
    "__index_level_0__",
    "tasks",
    "task_name",
    "task_id",
    "name",
)

IMAGE_STAT_SHAPE = (3, 1, 1)
LEGACY_STAT_KEYS = ("min", "max", "mean", "std", "count")

ExtractVideoFn = Callable[[Path, Path, float, float], None]


class DatasetLayoutError(RuntimeError):
    """Raised when a converted v2.1 tree is missing files LeRobot 0.3 expects."""


def v2_chunk(episode_index: int, chunks_size: int = V2_CHUNKS_SIZE) -> int:
    return int(episode_index) // int(chunks_size)


def v2_data_relpath(episode_index: int, chunks_size: int = V2_CHUNKS_SIZE) -> str:
    return V2_DATA_PATH.format(
        episode_chunk=v2_chunk(episode_index, chunks_size),
        episode_index=int(episode_index),
    )


def v2_video_relpath(episode_index: int, video_key: str, chunks_size: int = V2_CHUNKS_SIZE) -> str:
    return V2_VIDEO_PATH.format(
        episode_chunk=v2_chunk(episode_index, chunks_size),
        video_key=video_key,
        episode_index=int(episode_index),
    )


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    keys: list[str] = []
    for name, feat in features.items():
        if isinstance(feat, dict) and feat.get("dtype") == "video":
            keys.append(name)
    return keys


def load_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    with info_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid info.json at {info_path}")
    return data


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (bytes, str, dict, list)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def _as_int(value: Any, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError("expected integer, got None")
        return default
    return int(value)


def load_episode_records(root: Path) -> list[dict[str, Any]]:
    episodes_dir = root / "meta" / "episodes"
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet")) if episodes_dir.is_dir() else []
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}")

    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        records.extend(pq.read_table(pq_path).to_pylist())
    records.sort(key=lambda rec: _as_int(rec.get("episode_index"), 0))
    if not records:
        raise ValueError(f"Episode metadata in {episodes_dir} is empty")
    return records


def _task_text_from_row(row: dict[str, Any], fallback_index: int) -> str:
    for column in TASK_DESCRIPTION_COLUMNS:
        if column not in row:
            continue
        raw = row[column]
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            parts = [str(item).strip() for item in raw if item is not None and str(item).strip()]
            if parts:
                return "; ".join(parts)
            continue
        text = str(raw).strip()
        if text:
            return text
    return f"task_{fallback_index}"


def load_tasks(root: Path) -> list[dict[str, Any]]:
    """Return ``[{task_index, task}, ...]`` from parquet or jsonl."""
    tasks_pq = root / "meta" / "tasks.parquet"
    tasks_jsonl = root / "meta" / "tasks.jsonl"
    rows: list[dict[str, Any]] = []

    if tasks_pq.is_file():
        table = pq.read_table(tasks_pq)
        for i, raw in enumerate(table.to_pylist()):
            task_index = _as_int(raw.get("task_index"), i)
            rows.append({"task_index": task_index, "task": _task_text_from_row(raw, task_index)})
        rows.sort(key=lambda item: item["task_index"])
        return rows

    if tasks_jsonl.is_file():
        with tasks_jsonl.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                task_index = _as_int(raw.get("task_index"), i)
                rows.append({"task_index": task_index, "task": _task_text_from_row(raw, task_index)})
        rows.sort(key=lambda item: item["task_index"])
        return rows

    return [{"task_index": 0, "task": "perform the task"}]


def tasks_have_text(root: Path) -> bool:
    """True when ``tasks.parquet`` or ``tasks.jsonl`` contains a non-empty task string.

    A missing task file is not treated as text. ``load_tasks`` invents
    ``perform the task`` in that case, which is not a dataset prompt.
    """
    root = Path(root)
    if not (root / "meta" / "tasks.parquet").is_file() and not (root / "meta" / "tasks.jsonl").is_file():
        return False
    for row in load_tasks(root):
        task = str(row.get("task") or "").strip()
        if task:
            return True
    return False


def _data_loc(record: dict[str, Any]) -> tuple[int, int]:
    chunk = record.get("data/chunk_index", record.get("chunk_index", 0))
    file_idx = record.get("data/file_index", record.get("file_index", 0))
    return _as_int(chunk, 0), _as_int(file_idx, 0)


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        grouped[_data_loc(record)].append(record)
    return grouped


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]],
    video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"
    for record in episode_records:
        if chunk_column not in record or file_column not in record:
            continue
        chunk_idx = record.get(chunk_column)
        file_idx = record.get(file_column)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(_as_int(chunk_idx), _as_int(file_idx))].append(record)
    return grouped


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_to_serializable(row), ensure_ascii=False) + "\n")


def unflatten_dict(flat: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    """Rebuild a nested dict from ``a/b/c`` keys. Dots in feature names are kept."""
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        parts = str(key).split(sep)
        cursor = nested
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = value
    return nested


def _is_empty_stat(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    size = getattr(value, "size", None)
    if size is not None:
        return int(size) == 0
    return False


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if not isinstance(value, (list, tuple)):
        return None
    dims: list[int] = []
    cursor: Any = value
    while isinstance(cursor, (list, tuple)):
        dims.append(len(cursor))
        if not cursor:
            break
        cursor = cursor[0]
    return tuple(dims)


def _is_visual_feature(name: str) -> bool:
    lower = name.lower()
    return "image" in lower or "video" in lower


def _as_count_list(value: Any, length: int) -> list[int] | None:
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            try:
                return [int(value[0])]
            except (TypeError, ValueError):
                pass
        if length > 0:
            return [int(length)]
        return None
    if hasattr(value, "tolist") and not isinstance(value, (bytes, str)):
        return _as_count_list(value.tolist(), length)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)]
    if length > 0:
        return [int(length)]
    return None


def stats_from_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    """Unflatten v3 ``stats/feature/metric`` columns, or accept a nested ``stats`` dict."""
    stats_flat = {key: record[key] for key in record if str(key).startswith("stats/")}
    if stats_flat:
        nested = unflatten_dict(stats_flat).get("stats")
        return nested if isinstance(nested, dict) else {}

    raw = record.get("stats")
    if not isinstance(raw, dict) or not raw:
        return {}
    first = next(iter(raw.values()), None)
    if isinstance(first, dict):
        return raw
    return unflatten_dict(raw)


def sanitize_episode_stats(raw_stats: dict[str, Any], length: int) -> dict[str, Any]:
    """Keep only official v2.1 / LeRobot 0.3 ``min/max/mean/std/count`` feature stats."""
    cleaned: dict[str, Any] = {}
    if not isinstance(raw_stats, dict):
        return cleaned
    for feat, feat_stats in raw_stats.items():
        if not isinstance(feat_stats, dict):
            continue
        min_v = feat_stats.get("min")
        max_v = feat_stats.get("max")
        mean_v = feat_stats.get("mean")
        std_v = feat_stats.get("std")
        if any(_is_empty_stat(item) for item in (min_v, max_v, mean_v, std_v)):
            continue
        count = _as_count_list(feat_stats.get("count"), length)
        if count is None:
            continue
        feat_name = str(feat)
        if _is_visual_feature(feat_name):
            if any(_shape_of(item) != IMAGE_STAT_SHAPE for item in (min_v, max_v, mean_v, std_v)):
                continue
        cleaned[feat_name] = {
            "min": _to_serializable(min_v),
            "max": _to_serializable(max_v),
            "mean": _to_serializable(mean_v),
            "std": _to_serializable(std_v),
            "count": count,
        }
    return cleaned


def load_sanitized_stats_json(path: Path) -> dict[str, Any]:
    """Load v3 ``meta/stats.json`` and keep only LeRobot 0.3-safe keys."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        return {}
    return sanitize_episode_stats(raw, length=0)


def write_v21_stats_json(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_serializable(stats), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _with_episode_count(feat_stats: dict[str, Any], length: int) -> dict[str, Any]:
    count = [int(length)] if length > 0 else list(feat_stats.get("count") or [])
    return {
        "min": feat_stats["min"],
        "max": feat_stats["max"],
        "mean": feat_stats["mean"],
        "std": feat_stats["std"],
        "count": count,
    }


def complete_episode_stats(
    stats: dict[str, Any],
    *,
    length: int,
    parquet_stats: dict[str, Any] | None = None,
    global_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill missing features from parquet, then official global ``stats.json``.

    Per-episode image stats are often empty placeholders in v3; global stats
    already store the LeRobot (3, 1, 1) image layout. Reusing them per episode
    matches ``backward_compatible_episodes_stats`` in LeRobot 0.3.
    """
    completed = dict(stats)
    for source in (parquet_stats, global_stats):
        if not source:
            continue
        for feat, feat_stats in source.items():
            if feat in completed or not isinstance(feat_stats, dict):
                continue
            if any(key not in feat_stats for key in LEGACY_STAT_KEYS):
                continue
            completed[feat] = (
                dict(feat_stats)
                if source is parquet_stats
                else _with_episode_count(feat_stats, length)
            )
    return completed


def _is_numeric_arrow_type(typ: pa.DataType) -> bool:
    if pa.types.is_floating(typ) or pa.types.is_integer(typ) or pa.types.is_decimal(typ):
        return True
    if pa.types.is_list(typ) or pa.types.is_large_list(typ) or pa.types.is_fixed_size_list(typ):
        return _is_numeric_arrow_type(typ.value_type)
    return False


def _numeric_feature_names(table: pa.Table) -> list[str]:
    names: list[str] = []
    for col in table.column_names:
        if _is_visual_feature(col):
            continue
        if _is_numeric_arrow_type(table.schema.field(col).type):
            names.append(col)
    return names


def episode_stats_from_parquet(
    dataset_dir: Path,
    episode_index: int,
    chunks_size: int = V2_CHUNKS_SIZE,
    *,
    float_features: list[str] | None = None,
) -> dict[str, Any]:
    """Compute per-episode min/max/mean/std/count from a v2.1 parquet file."""
    pq_file = Path(dataset_dir) / v2_data_relpath(episode_index, chunks_size)
    if not pq_file.is_file():
        return {}
    table = pq.read_table(pq_file)
    if float_features is None:
        float_features = _numeric_feature_names(table)
    ep_stats: dict[str, Any] = {}
    for col_name in float_features:
        if col_name not in table.column_names:
            continue
        col = table.column(col_name)
        try:
            arr = np.array(
                [row.as_py() if hasattr(row, "as_py") else row for row in col],
                dtype=np.float32,
            )
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            ep_stats[col_name] = {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "count": [int(len(arr))],
            }
        except (ValueError, TypeError):
            continue
    return ep_stats


def generate_episodes_stats_from_parquet(
    dataset_dir: Path,
    episode_indices: Iterable[int],
    chunks_size: int = V2_CHUNKS_SIZE,
) -> list[dict[str, Any]]:
    """Rewrite ``meta/episodes_stats.jsonl`` from per-episode parquet files."""
    indices = [int(idx) for idx in episode_indices]
    rows: list[dict[str, Any]] = []
    numeric_features: list[str] | None = None
    root = Path(dataset_dir)
    global_stats = load_sanitized_stats_json(root / "meta" / "stats.json")
    for ep_idx in indices:
        pq_file = root / v2_data_relpath(ep_idx, chunks_size)
        length = 0
        if pq_file.is_file():
            table = pq.read_table(pq_file)
            length = int(table.num_rows)
            if numeric_features is None:
                numeric_features = _numeric_feature_names(table)
        parquet_stats = episode_stats_from_parquet(
            root, ep_idx, chunks_size, float_features=numeric_features
        )
        stats = complete_episode_stats(
            {},
            length=length,
            parquet_stats=parquet_stats,
            global_stats=global_stats,
        )
        rows.append({"episode_index": ep_idx, "stats": stats})
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", rows)
    logger.info("Generated episodes_stats.jsonl for %s episodes", len(indices))
    return rows


def load_episodes_stats_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def assert_v21_episode_stats_rows(rows: Iterable[dict[str, Any]]) -> None:
    """Raise if stats would fail LeRobot 0.3 ``aggregate_stats``."""
    materialized = list(rows)
    if not materialized:
        raise DatasetLayoutError("episodes_stats.jsonl is empty")
    for row in materialized:
        stats = row.get("stats")
        ep_idx = row.get("episode_index")
        if not isinstance(stats, dict):
            raise DatasetLayoutError(f"episode {ep_idx}: stats must be a dict")
        for feat, feat_stats in stats.items():
            if not isinstance(feat_stats, dict):
                raise DatasetLayoutError(f"episode {ep_idx} feature {feat}: stats must be a dict")
            for key, value in feat_stats.items():
                arr = np.asarray(value)
                if arr.ndim == 0:
                    raise DatasetLayoutError(
                        f"episode {ep_idx} feature {feat}: '{key}' must have ndim>=1"
                    )
                if key == "count" and arr.shape != (1,):
                    raise DatasetLayoutError(
                        f"episode {ep_idx} feature {feat}: count shape must be (1,), got {arr.shape}"
                    )
                if "image" in str(feat) and key != "count" and tuple(arr.shape) != IMAGE_STAT_SHAPE:
                    raise DatasetLayoutError(
                        f"episode {ep_idx} feature {feat}: '{key}' shape must be (3,1,1), "
                        f"got {arr.shape}"
                    )


def assert_v21_episodes_stats(dest: Path) -> None:
    path = Path(dest) / "meta" / "episodes_stats.jsonl"
    if not path.is_file():
        raise DatasetLayoutError(f"Missing {path}")
    assert_v21_episode_stats_rows(load_episodes_stats_rows(path))
    stats_json = Path(dest) / "meta" / "stats.json"
    if stats_json.is_file():
        raw = json.loads(stats_json.read_text(encoding="utf-8"))
        assert_v21_episode_stats_rows([{"episode_index": "stats.json", "stats": raw}])


def episodes_stats_compatible_with_v21(stats_or_path: Path | Iterable[dict[str, Any]]) -> bool:
    if isinstance(stats_or_path, Path):
        if not stats_or_path.is_file():
            return False
        rows: Iterable[dict[str, Any]] = load_episodes_stats_rows(stats_or_path)
    else:
        rows = stats_or_path
    try:
        assert_v21_episode_stats_rows(rows)
    except (DatasetLayoutError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def convert_info(
    info: dict[str, Any],
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    chunks_size: int,
) -> dict[str, Any]:
    v2_info = dict(info)
    features: dict[str, Any] = {}
    selected_videos = set(video_keys)
    for key, feat in (info.get("features") or {}).items():
        if isinstance(feat, dict):
            copied = dict(feat)
            if copied.get("dtype") == "video" and key not in selected_videos:
                continue
            # Official v2.1 keeps fps only on video features (see GR00T convert_info).
            if copied.get("dtype") != "video":
                copied.pop("fps", None)
            features[key] = copied
        else:
            features[key] = feat
    v2_info["features"] = features
    total_episodes = int(info.get("total_episodes") or len(episode_records))
    v2_info["codebase_version"] = V21
    v2_info["chunks_size"] = chunks_size
    v2_info["data_path"] = V2_DATA_PATH
    v2_info["video_path"] = V2_VIDEO_PATH if video_keys else None
    v2_info.pop("data_files_size_in_mb", None)
    v2_info.pop("video_files_size_in_mb", None)
    v2_info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    v2_info["total_videos"] = total_episodes * len(video_keys)
    v2_info["total_episodes"] = total_episodes
    return v2_info


# LeRobotDataset rejects a step that misses 1/fps by more than this.
TIMESTAMP_TOLERANCE_S = 1e-4


def timestamps_within_tolerance(
    values: np.ndarray,
    fps: float,
    tolerance_s: float = TIMESTAMP_TOLERANCE_S,
) -> bool:
    """True when consecutive timestamps match ``1/fps`` inside ``tolerance_s``."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size < 2:
        return True
    if fps <= 0:
        return False
    return bool(np.all(np.abs(np.diff(series) - (1.0 / float(fps))) <= tolerance_s))


def rewrite_episode_timestamps(table: pa.Table, fps: float) -> pa.Table:
    """Replace timestamps that float32 can no longer space at ``1/fps``.

    A multi-hour episode stored as float32 drifts by more than LeRobot's
    ``1e-4`` s once the clock is large. ``i / fps`` in float64 stays exact.
    Episodes that already pass are returned unchanged.
    """
    if "timestamp" not in table.column_names or fps <= 0:
        return table
    current = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64).reshape(-1)
    if timestamps_within_tolerance(current, fps):
        return table
    fresh = np.arange(table.num_rows, dtype=np.float64) / float(fps)
    index = table.schema.get_field_index("timestamp")
    return table.set_column(index, "timestamp", pa.array(fresh, type=pa.float64()))


def repair_episode_timestamps(dataset_dir: Path, fps: float) -> int:
    """Rewrite episode parquet files whose timestamps miss LeRobot's tolerance.

    Returns the number of files changed. Safe to repeat.
    """
    data_root = Path(dataset_dir) / "data"
    if fps <= 0 or not data_root.is_dir():
        return 0
    rewritten = 0
    for path in sorted(data_root.glob("**/*.parquet")):
        table = pq.read_table(path)
        updated = rewrite_episode_timestamps(table, fps)
        if updated is table:
            continue
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(updated, tmp_path)
        os.replace(tmp_path, path)
        rewritten += 1
    if rewritten:
        logger.info(
            "Rewrote timestamps in %s episode file(s) at %s so steps stay within %.1e s of 1/fps",
            rewritten,
            dataset_dir,
            TIMESTAMP_TOLERANCE_S,
        )
    return rewritten


def convert_data(
    src: Path,
    dest: Path,
    episode_records: list[dict[str, Any]],
    chunks_size: int,
    fps: float | None = None,
) -> None:
    grouped = _group_episodes_by_data_file(episode_records)
    for (chunk_idx, file_idx), records in grouped.items():
        source_path = src / V3_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.is_file():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")

        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: _as_int(rec.get("dataset_from_index"), 0))
        has_range = all(
            rec.get("dataset_from_index") is not None and rec.get("dataset_to_index") is not None
            for rec in records
        )
        file_offset = _as_int(records[0].get("dataset_from_index"), 0) if has_range else 0

        for record in records:
            episode_index = _as_int(record["episode_index"])
            if has_range:
                start = _as_int(record["dataset_from_index"]) - file_offset
                stop = _as_int(record["dataset_to_index"]) - file_offset
                length = stop - start
                if length <= 0:
                    raise ValueError(
                        "Invalid episode length during data conversion: "
                        f"episode_index={episode_index}, length={length}"
                    )
                episode_table = table.slice(start, length)
            else:
                import pyarrow.compute as pc

                episode_table = table.filter(pc.equal(table.column("episode_index"), episode_index))
                if episode_table.num_rows <= 0:
                    raise ValueError(f"No rows for episode_index={episode_index} in {source_path}")

            if fps is not None and fps > 0:
                episode_table = rewrite_episode_timestamps(episode_table, fps)

            dest_path = dest / v2_data_relpath(episode_index, chunks_size)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dest_path)


def extract_video_segment(src: Path, dst: Path, start: float, end: float) -> None:
    """Cut ``[start, end)`` seconds from ``src`` into a per-episode mp4."""
    if start < 0 or end < 0:
        raise ValueError(f"Invalid video timestamps start={start} end={end}")
    if end <= start:
        raise ValueError(f"Start time {start} must be less than end time {end}")
    duration = max(end - start, MIN_VIDEO_DURATION)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # `-ss` before `-i` with `-c copy` leaves the first PTS about one frame
    # above 0 when combined with avoid_negative_ts. LeRobot's decoder then
    # rejects the clip (tolerance 1e-4). setts moves the copied timestamps to 0.
    copy_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-c",
        "copy",
        "-bsf:v",
        "setts=pts=PTS-STARTPTS:dts=DTS-STARTPTS",
        "-y",
        str(dst),
    ]
    try:
        subprocess.run(copy_cmd, check=True, timeout=300, capture_output=True, text=True)
        if dst.is_file() and dst.stat().st_size > 0:
            return
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found; it is required for v3 video conversion") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg stream-copy failed for %s -> %s (%s); retrying with re-encode", src, dst, exc)

    encode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-vf",
        "setpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-an",
        "-y",
        str(dst),
    ]
    try:
        subprocess.run(encode_cmd, check=True, timeout=300, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out while processing video '{src}' -> '{dst}'") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed while splitting video '{src}' into '{dst}'. {detail}") from exc
    if not dst.is_file() or dst.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced an empty video: {dst}")


def convert_videos(
    src: Path,
    dest: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    chunks_size: int,
    extract_video: ExtractVideoFn = extract_video_segment,
) -> None:
    if not video_keys:
        logger.info("No video features detected; skipping video conversion")
        return

    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        if not grouped:
            raise DatasetLayoutError(
                f"No video metadata for '{video_key}' in meta/episodes "
                f"(expected columns videos/{video_key}/chunk_index and file_index)"
            )

        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"

        for (chunk_idx, file_idx), records in grouped.items():
            src_path = src / V3_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not src_path.is_file():
                raise FileNotFoundError(
                    f"Expected MP4 file not found for {video_key}: {src_path}"
                )

            records = sorted(records, key=lambda rec: float(rec.get(from_col) or 0.0))
            unique_owner = len(records) == 1

            for record in records:
                episode_index = _as_int(record["episode_index"])
                if record.get(from_col) is None or record.get(to_col) is None:
                    raise DatasetLayoutError(
                        f"Missing timestamps for episode {episode_index} camera {video_key}"
                    )
                start = float(record[from_col])
                end = float(record[to_col])
                dest_path = dest / v2_video_relpath(episode_index, video_key, chunks_size)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if unique_owner:
                    if dest_path.exists() or dest_path.is_symlink():
                        dest_path.unlink()
                    _link_or_copy_whole_file(src_path, dest_path)
                    continue
                extract_video(src_path, dest_path, start, end)


def _relative_link_target(target: Path, link: Path) -> str:
    """Relative path from the directory holding ``link`` to ``target``.

    Relative links resolve identically on the host and inside a container as
    long as the two trees keep their relative layout. An absolute link records
    the container-side mount path, so it dangles everywhere else and ties the
    cache to one mount layout.
    """
    return os.path.relpath(Path(target).resolve(), Path(link).parent.resolve())


def _link_or_copy_whole_file(src_file: Path, dest_file: Path) -> str:
    """Publish a source video as an episode clip without re-encoding it.

    A hardlink shares the bytes, so the cache stays self-contained without
    duplicating storage. It fails with ``EXDEV`` across mount points (a source
    dataset and the cache are usually separate bind mounts), and a read-only
    source rejects it too, so fall back to a copy. A symlink is not used here:
    the two trees are mounted at unrelated paths, so no relative target exists
    and an absolute one would only work under the original mounts.
    """
    try:
        os.link(src_file, dest_file)
        return "hardlink"
    except OSError as exc:
        logger.debug("Hardlinking %s failed (%s); copying instead", dest_file, exc)
    shutil.copy2(src_file, dest_file)
    return "copy"


def _normalize_tasks_list(record: dict[str, Any], task_by_index: dict[int, str]) -> list[str]:
    raw = record.get("tasks")
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, (list, tuple)):
        parts = [str(item).strip() for item in raw if item is not None and str(item).strip()]
        if parts:
            return parts
    task_index = record.get("task_index")
    if task_index is not None:
        text = task_by_index.get(_as_int(task_index))
        if text:
            return [text]
    if task_by_index:
        return [next(iter(task_by_index.values()))]
    return ["perform the task"]


def convert_episodes_metadata(
    dest: Path,
    episode_records: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    chunks_size: int = V2_CHUNKS_SIZE,
    global_stats: dict[str, Any] | None = None,
) -> None:
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    numeric_features: list[str] | None = None

    for record in sorted(episode_records, key=lambda rec: _as_int(rec.get("episode_index"), 0)):
        episode_index = _as_int(record["episode_index"])
        length = record.get("length")
        if length is None and record.get("dataset_from_index") is not None:
            length = _as_int(record["dataset_to_index"]) - _as_int(record["dataset_from_index"])
        length_i = _as_int(length, 0)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "length": length_i,
                "tasks": _normalize_tasks_list(record, task_by_index),
            }
        )

        stats = sanitize_episode_stats(stats_from_episode_record(record), length_i)
        pq_file = dest / v2_data_relpath(episode_index, chunks_size)
        parquet_stats: dict[str, Any] | None = None
        if pq_file.is_file():
            if numeric_features is None:
                numeric_features = _numeric_feature_names(pq.read_table(pq_file))
            parquet_stats = episode_stats_from_parquet(
                dest, episode_index, chunks_size, float_features=numeric_features
            )
        stats = complete_episode_stats(
            stats,
            length=length_i,
            parquet_stats=parquet_stats,
            global_stats=global_stats,
        )
        stats_rows.append({"episode_index": episode_index, "stats": stats})

    _write_jsonl(dest / "meta" / "episodes.jsonl", episode_rows)
    _write_jsonl(dest / "meta" / "episodes_stats.jsonl", stats_rows)


def expected_v2_paths(
    info: dict[str, Any],
    episode_indices: Iterable[int],
    chunks_size: int | None = None,
) -> list[str]:
    size = int(chunks_size or info.get("chunks_size") or V2_CHUNKS_SIZE)
    video_keys = video_keys_from_info(info)
    paths: list[str] = []
    for episode_index in episode_indices:
        paths.append(v2_data_relpath(episode_index, size))
        for video_key in video_keys:
            paths.append(v2_video_relpath(episode_index, video_key, size))
    return paths


def assert_v2_local_files(
    dest: Path,
    info: dict[str, Any],
    episode_indices: Iterable[int],
    chunks_size: int | None = None,
    limit: int = MISSING_PATH_LIMIT,
) -> None:
    missing = [
        rel
        for rel in expected_v2_paths(info, episode_indices, chunks_size)
        if not (dest / rel).is_file()
    ]
    if not missing:
        return
    preview = missing[:limit]
    extra = f" (and {len(missing) - limit} more)" if len(missing) > limit else ""
    raise DatasetLayoutError(
        "Converted v2.1 dataset is missing files LeRobot 0.3 expects. "
        f"missing={len(missing)}{extra}: {preview}"
    )


def directory_is_writable(path: Path) -> bool:
    """Return whether ``path`` accepts a new file.

    Mode bits are not enough: a read-only bind mount still looks writable to
    root via ``os.access``, and the write then fails with ``EROFS``.
    """
    path = Path(path)
    if not path.is_dir():
        return False
    probe = path / f".write_probe_{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as fh:
            fh.write("")
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        logger.warning("Could not remove write probe %s", probe)
    return True


def conversion_stamp(src: Path, video_keys: Sequence[str], chunks_size: int) -> dict[str, Any]:
    """Identity of a v3 source plus the video keys written into a v2 tree."""
    info_path = Path(src) / "meta" / "info.json"
    stat = info_path.stat()
    info = load_info(Path(src))
    return {
        # 2: episode mp4s are stream-copied with PTS starting at 0.
        "stamp_version": 2,
        "info_size": stat.st_size,
        "info_mtime_ns": stat.st_mtime_ns,
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "video_keys": list(video_keys),
        "chunks_size": int(chunks_size),
    }


def _read_json_stamp(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_stamp(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_video_keys(info: dict[str, Any], video_keys: Sequence[str] | None) -> list[str]:
    available = video_keys_from_info(info)
    if video_keys is None:
        return available
    selected = list(video_keys)
    missing = [key for key in selected if key not in available]
    if missing:
        raise ValueError(f"Unknown video keys {missing}. Available: {available}")
    return selected


def _reset_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _symlink_children(src_dir: Path, dest_dir: Path) -> None:
    """Link each file under ``src_dir`` into ``dest_dir`` with relative targets.

    Replacing one entry in ``dest_dir`` then leaves the source untouched, and
    because the targets are relative the tree also resolves outside the
    container that created it.
    """
    if not src_dir.is_dir():
        return
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        rel = path.relative_to(src_dir)
        target = dest_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        os.symlink(_relative_link_target(path, target), target)


def _copy_children(src_dir: Path, dest_dir: Path) -> None:
    """Copy each file under ``src_dir`` into ``dest_dir``."""
    if not src_dir.is_dir():
        return
    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        target = dest_dir / path.relative_to(src_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _filter_stats(stats: dict[str, Any], drop_keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key not in drop_keys}


def stage_writable_dataset(src: Path, dest: Path) -> Path:
    """Return ``src`` when it can be edited, otherwise a writable copy to edit.

    Metadata and frame data are copied, so the staged tree is writable and
    self-contained. Only ``videos/`` is linked, because video data is large;
    that link is relative when the source sits inside the same tree as ``dest``
    and absolute otherwise, in which case the staged tree stays valid only
    while the source is mounted at the same path.
    """
    src = Path(src)
    meta = src / "meta"
    data = src / "data"
    writable = directory_is_writable(meta) and (not data.exists() or directory_is_writable(data))
    if writable:
        return src

    dest = Path(dest)
    marker = {"source": str(src.resolve())}
    existing = _read_json_stamp(dest / "meta" / ".staged.json")
    if existing == marker and (dest / "meta" / "info.json").is_file():
        logger.info("Reusing writable dataset view at %s", dest)
        return dest

    logger.info("Dataset at %s is read-only; staging a writable view at %s", src, dest)
    _reset_dir(dest)
    if meta.is_dir():
        shutil.copytree(meta, dest / "meta", symlinks=True, dirs_exist_ok=True)
        # copytree preserves a read-only source mode. The copy must accept metadata fixes.
        for copied in [dest / "meta", *(dest / "meta").rglob("*")]:
            if copied.is_symlink():
                continue
            copied.chmod(copied.stat().st_mode | (0o700 if copied.is_dir() else 0o600))
    videos = src / "videos"
    if videos.is_dir():
        link_target = _relative_link_target(videos, dest / "videos")
        # Climbing several levels means the source sits in a different tree,
        # which is normally a different mount.
        if link_target.count("..") > 1:
            logger.warning(
                "Staged dataset links videos with %r; that tree is only valid while the source "
                "dataset is mounted at %s",
                link_target,
                videos,
            )
        os.symlink(link_target, dest / "videos")
    _copy_children(data, dest / "data")
    _write_json_stamp(dest / "meta" / ".staged.json", marker)
    return dest


def make_camera_view(src: Path, dest: Path, keep_video_keys: Sequence[str]) -> Path:
    """Point ``dest`` at ``src`` while hiding video features that are not kept.

    LeRobot loads every video feature listed in ``info.json``. Dropping a camera
    from the training config is not enough; the view's metadata must omit it.
    """
    src = Path(src)
    dest = Path(dest)
    info = load_info(src)
    available = video_keys_from_info(info)
    selected = _resolve_video_keys(info, keep_video_keys)
    if set(selected) == set(available):
        return src

    stamp = {
        # 2: links are relative, so the view also resolves outside the container.
        "stamp_version": 2,
        "source": str(src.resolve()),
        "video_keys": selected,
    }
    existing = _read_json_stamp(dest / "meta" / VIEW_STAMP_NAME)
    if existing == stamp and (dest / "meta" / "info.json").is_file():
        logger.info("Reusing camera view at %s (%s)", dest, selected)
        return dest

    logger.info("Creating camera view at %s with videos %s", dest, selected)
    _reset_dir(dest)
    _symlink_children(src / "data", dest / "data")
    videos = src / "videos"
    if videos.is_dir():
        os.symlink(_relative_link_target(videos, dest / "videos"), dest / "videos")

    meta_src = src / "meta"
    meta_dest = dest / "meta"
    meta_dest.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.jsonl", "tasks.parquet", "episodes.jsonl"):
        source_file = meta_src / name
        if source_file.is_file() or source_file.is_symlink():
            os.symlink(_relative_link_target(source_file, meta_dest / name), meta_dest / name)

    drop_keys = set(available) - set(selected)
    viewed = convert_info(info, [], selected, int(info.get("chunks_size") or V2_CHUNKS_SIZE))
    viewed["total_episodes"] = info.get("total_episodes", viewed.get("total_episodes"))
    viewed["total_frames"] = info.get("total_frames")
    viewed["total_videos"] = int(viewed.get("total_episodes") or 0) * len(selected)
    with (meta_dest / "info.json").open("w", encoding="utf-8") as fh:
        json.dump(viewed, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    stats_path = meta_src / "stats.json"
    if stats_path.is_file():
        raw_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if isinstance(raw_stats, dict):
            write_v21_stats_json(meta_dest / "stats.json", _filter_stats(raw_stats, drop_keys))

    episodes_stats = meta_src / "episodes_stats.jsonl"
    if episodes_stats.is_file():
        filtered_rows = []
        for row in load_episodes_stats_rows(episodes_stats):
            stats = row.get("stats")
            if isinstance(stats, dict):
                row = dict(row)
                row["stats"] = _filter_stats(stats, drop_keys)
            filtered_rows.append(row)
        _write_jsonl(meta_dest / "episodes_stats.jsonl", filtered_rows)

    _write_json_stamp(meta_dest / VIEW_STAMP_NAME, stamp)
    return dest


def camera_view_dir(full_dir: Path, keep_video_keys: Sequence[str]) -> Path:
    """Stable sibling directory for a camera subset of ``full_dir``."""
    slug = "_".join(key.split(".")[-1] for key in keep_video_keys) or "novideo"
    digest = hashlib.sha256("\n".join(keep_video_keys).encode("utf-8")).hexdigest()[:8]
    full_dir = Path(full_dir)
    return full_dir.parent / f"{full_dir.name}__{slug}_{digest}"


def select_image_keys(
    image_keys: Sequence[str],
    *,
    cameras: Sequence[str] | None = None,
    drop_cameras: Sequence[str] | None = None,
) -> list[str]:
    """Keep an explicit camera list, then drop keys or substrings such as ``front``."""
    available = list(image_keys)
    if cameras:
        missing = [key for key in cameras if key not in available]
        if missing:
            raise ValueError(f"Unknown camera keys {missing}. Available: {available}")
        selected = [key for key in cameras if key in available]
    else:
        selected = list(available)

    def _dropped(key: str) -> bool:
        if not drop_cameras:
            return False
        return any(key == token or token in key for token in drop_cameras)

    selected = [key for key in selected if not _dropped(key)]
    if not selected:
        raise ValueError(f"No image keys left after camera selection. Available: {available}")
    if len(selected) > len(MODEL_IMAGE_SLOTS):
        unmapped = [key for key in selected if _preferred_slot(key) is None]
        detail = f" Keys without a base/wrist role: {unmapped}." if unmapped else ""
        raise ValueError(
            f"OpenPI has {len(MODEL_IMAGE_SLOTS)} image slots, got {selected}.{detail} "
            "Pass --cameras or --drop_cameras to choose at most 3."
        )
    return selected


def parse_camera_map(spec: str | None) -> dict[str, str] | None:
    """Parse ``base=key,left_wrist=key,right_wrist=key`` into slot names."""
    if spec is None or not spec.strip():
        return None
    mapping: dict[str, str] = {}
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"camera_map entry {item!r} must look like base=observation.images.front")
        slot_name, key = item.split("=", 1)
        slot = _SLOT_ALIASES.get(slot_name.strip())
        if slot is None:
            raise ValueError(
                f"Unknown camera slot {slot_name!r}. Expected base, left_wrist, or right_wrist."
            )
        if slot in mapping:
            raise ValueError(f"Camera slot {slot_name} was given twice")
        mapping[slot] = key.strip()
    return mapping


def _preferred_slot(key: str) -> str | None:
    lower = key.lower()
    if "wrist" in lower and "right" in lower:
        return "right_wrist_0_rgb"
    if "wrist" in lower:
        return "left_wrist_0_rgb"
    if any(token in lower for token in ("front", "base", "high", "exterior")):
        return "base_0_rgb"
    return None


def assign_camera_slots(
    image_keys: Sequence[str],
    camera_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map dataset image keys onto ``base_0_rgb`` / wrist slots.

    Keys that match a role fill that slot. Remaining keys fill empty slots in
    slot order. Unfilled slots are omitted so the caller can mask them.
    """
    keys = list(image_keys)
    if camera_map is not None:
        unknown_slots = [slot for slot in camera_map if slot not in MODEL_IMAGE_SLOTS]
        if unknown_slots:
            raise ValueError(f"Unknown camera slots {unknown_slots}")
        missing = [key for key in camera_map.values() if key not in keys]
        if missing:
            raise ValueError(f"camera_map keys {missing} are not in the selected cameras {keys}")
        return dict(camera_map)

    remaining = list(keys)
    mapping: dict[str, str] = {}
    for key in list(remaining):
        slot = _preferred_slot(key)
        if slot is None or slot in mapping:
            continue
        mapping[slot] = key
        remaining.remove(key)
    for slot in MODEL_IMAGE_SLOTS:
        if slot not in mapping and remaining:
            mapping[slot] = remaining.pop(0)
    return mapping


def convert_v3_to_v2(
    src_dir: Path,
    dest_dir: Path | None = None,
    *,
    chunks_size: int = V2_CHUNKS_SIZE,
    extract_video: ExtractVideoFn = extract_video_segment,
    video_keys: Sequence[str] | None = None,
) -> Path:
    """Convert a local v3.0 dataset to a v2.1 tree and validate required files.

    ``dest_dir`` is required. A previous default of ``/tmp`` is unsafe on hosts
    whose ``/tmp`` is a small tmpfs. When ``dest_dir`` already contains a
    matching ``meta/.convert_complete.json``, the tree is reused and
    ``extract_video`` is not called.
    """
    src = Path(src_dir)
    if dest_dir is None:
        raise ValueError(
            "dest_dir is required. Refusing to write a converted dataset to /tmp; "
            "pass an explicit directory on a real filesystem."
        )
    dest = Path(dest_dir)

    info = load_info(src)
    version = str(info.get("codebase_version", "")).lower()
    if not (version.startswith("v3") or version.startswith("3")):
        raise ValueError(f"Expected a v3 dataset, got codebase_version={info.get('codebase_version')!r}")

    selected_videos = _resolve_video_keys(info, video_keys)
    stamp = conversion_stamp(src, selected_videos, chunks_size)
    existing = _read_json_stamp(dest / "meta" / CONVERT_STAMP_NAME)
    if existing == stamp:
        logger.info("Reusing converted dataset at %s", dest)
        return dest
    if dest.exists():
        logger.info("Reconverting %s; existing tree at %s does not match this source", src, dest)
        _reset_dir(dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)

    episode_records = load_episode_records(src)
    tasks = load_tasks(src)
    v2_info = convert_info(info, episode_records, selected_videos, chunks_size)

    meta_dest = dest / "meta"
    meta_dest.mkdir(parents=True, exist_ok=True)
    with (meta_dest / "info.json").open("w", encoding="utf-8") as fh:
        json.dump(v2_info, fh, indent=2, ensure_ascii=False)

    global_stats = load_sanitized_stats_json(src / "meta" / "stats.json")
    dropped_videos = set(video_keys_from_info(info)) - set(selected_videos)
    if global_stats and dropped_videos:
        global_stats = _filter_stats(global_stats, dropped_videos)
    if global_stats:
        write_v21_stats_json(meta_dest / "stats.json", global_stats)

    _write_jsonl(meta_dest / "tasks.jsonl", tasks)
    convert_data(src, dest, episode_records, chunks_size, fps=float(info.get("fps") or 0))
    convert_videos(
        src,
        dest,
        episode_records,
        selected_videos,
        chunks_size,
        extract_video=extract_video,
    )
    convert_episodes_metadata(
        dest,
        episode_records,
        tasks,
        chunks_size=chunks_size,
        global_stats=global_stats,
    )
    if dropped_videos:
        stats_path = meta_dest / "episodes_stats.jsonl"
        filtered_rows = []
        for row in load_episodes_stats_rows(stats_path):
            stats = row.get("stats")
            if isinstance(stats, dict):
                row = dict(row)
                row["stats"] = _filter_stats(stats, dropped_videos)
            filtered_rows.append(row)
        _write_jsonl(stats_path, filtered_rows)

    episode_indices = [_as_int(rec["episode_index"]) for rec in episode_records]
    assert_v2_local_files(dest, v2_info, episode_indices, chunks_size=chunks_size)
    assert_v21_episodes_stats(dest)
    _write_json_stamp(meta_dest / CONVERT_STAMP_NAME, stamp)

    logger.info("Converted v3 dataset to v2 layout at %s", dest)
    logger.info(
        "  Episodes: %s, video keys: %s, v2 chunks_size: %s",
        len(episode_indices),
        selected_videos,
        chunks_size,
    )
    return dest
