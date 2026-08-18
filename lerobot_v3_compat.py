"""Convert packed LeRobot v3.0 datasets to the v2.1 layout OpenPI can read.

Official OpenPI pins an old LeRobot that only understands v2.1 (one parquet/mp4
per episode). This module ports the NVIDIA GR00T convert_v3_to_v2 algorithm
without importing modern ``lerobot.datasets.utils``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable
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
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        return False
    with tasks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
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
    for key, feat in (info.get("features") or {}).items():
        if isinstance(feat, dict):
            copied = dict(feat)
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


def convert_data(
    src: Path,
    dest: Path,
    episode_records: list[dict[str, Any]],
    chunks_size: int,
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
        "-avoid_negative_ts",
        "1",
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
                    os.symlink(str(src_path.resolve()), str(dest_path))
                    continue
                extract_video(src_path, dest_path, start, end)


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


def convert_v3_to_v2(
    src_dir: Path,
    dest_dir: Path | None = None,
    *,
    chunks_size: int = V2_CHUNKS_SIZE,
    extract_video: ExtractVideoFn = extract_video_segment,
) -> Path:
    """Convert a local v3.0 dataset to a v2.1 tree and validate required files."""
    src = Path(src_dir)
    dest = Path(dest_dir) if dest_dir is not None else Path("/tmp/lerobot_v2_compat")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    info = load_info(src)
    version = str(info.get("codebase_version", "")).lower()
    if not (version.startswith("v3") or version.startswith("3")):
        raise ValueError(f"Expected a v3 dataset, got codebase_version={info.get('codebase_version')!r}")

    episode_records = load_episode_records(src)
    video_keys = video_keys_from_info(info)
    tasks = load_tasks(src)
    v2_info = convert_info(info, episode_records, video_keys, chunks_size)

    meta_dest = dest / "meta"
    meta_dest.mkdir(parents=True, exist_ok=True)
    with (meta_dest / "info.json").open("w", encoding="utf-8") as fh:
        json.dump(v2_info, fh, indent=2, ensure_ascii=False)

    global_stats = load_sanitized_stats_json(src / "meta" / "stats.json")
    if global_stats:
        write_v21_stats_json(meta_dest / "stats.json", global_stats)

    _write_jsonl(meta_dest / "tasks.jsonl", tasks)
    convert_data(src, dest, episode_records, chunks_size)
    convert_videos(
        src,
        dest,
        episode_records,
        video_keys,
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

    episode_indices = [_as_int(rec["episode_index"]) for rec in episode_records]
    assert_v2_local_files(dest, v2_info, episode_indices, chunks_size=chunks_size)
    assert_v21_episodes_stats(dest)

    logger.info("Converted v3 dataset to v2 layout at %s", dest)
    logger.info(
        "  Episodes: %s, video keys: %s, v2 chunks_size: %s",
        len(episode_indices),
        video_keys,
        chunks_size,
    )
    return dest
