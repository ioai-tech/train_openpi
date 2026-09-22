#!/usr/bin/env python3
"""
Docker training wrapper for OpenPI with LeRobot datasets.

Auto-discovers dataset schema, computes normalization statistics,
constructs an OpenPI TrainConfig, and launches JAX-based training.

No modifications to OpenPI core code are required.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import random
import shutil
import sys

os.environ.setdefault("OPENPI_DATA_HOME", "/models")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Let JAX use 90% of GPU memory (vs default 75%) -- critical for A100 80GB training.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
# The pinned lerobot version uses HF_LEROBOT_HOME (not LEROBOT_HOME).
os.environ.pop("LEROBOT_HOME", None)
os.environ.setdefault("HF_LEROBOT_HOME", str(pathlib.Path.home() / ".cache" / "lerobot"))

sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/packages/openpi-client/src")
sys.path.insert(0, "/app")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

from lerobot_v3_compat import assign_camera_slots
from lerobot_v3_compat import camera_view_dir
from lerobot_v3_compat import conversion_stamp
from lerobot_v3_compat import convert_v3_to_v2
from lerobot_v3_compat import episodes_stats_compatible_with_v21
from lerobot_v3_compat import generate_episodes_stats_from_parquet
from lerobot_v3_compat import load_tasks
from lerobot_v3_compat import make_camera_view
from lerobot_v3_compat import parse_camera_map
from lerobot_v3_compat import select_image_keys
from lerobot_v3_compat import stage_writable_dataset
from lerobot_v3_compat import tasks_have_text
from lerobot_v3_compat import video_keys_from_info

DEFAULT_DATASET_DIR = pathlib.Path(os.environ.get("OPENPI_DATASET_DIR", "/data/input"))
DEFAULT_OUTPUT_DIR = pathlib.Path(os.environ.get("OPENPI_OUTPUT_DIR", "/data/output"))

logger = logging.getLogger("train_lerobot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenPI on a mounted LeRobot dataset")
    parser.add_argument("--dataset_dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR,
                        help="LeRobot dataset root (default: $OPENPI_DATASET_DIR or /data/input)")
    parser.add_argument("--output_dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR,
                        help="Checkpoint root (default: $OPENPI_OUTPUT_DIR or /data/output)")
    parser.add_argument("--run_name", type=str, default="docker_train",
                        help="TrainConfig name. Checkpoints land in <output_dir>/<run_name>/<exp_name>")
    parser.add_argument("--exp_name", type=str, default="train")
    parser.add_argument("--convert_dir", type=pathlib.Path, default=None,
                        help="Writable v3->v2 cache. Default: <output_dir>/.v21_cache/<fingerprint>. "
                             "Do not point this at a small tmpfs.")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Comma-separated image keys to keep, e.g. observation.images.top,observation.images.wrist")
    parser.add_argument("--drop_cameras", type=str, default=None,
                        help="Comma-separated image keys or substrings to drop, e.g. front")
    parser.add_argument("--camera_map", type=str, default=None,
                        help="Explicit slots: base=key,left_wrist=key,right_wrist=key")
    parser.add_argument("--delta_joint_actions", action="store_true",
                        help="Train joint dims as deltas and keep the last dim (gripper) absolute. "
                             "Default keeps the dataset action values unchanged.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gpus", type=str, default="all",
                        help="'all' or comma-separated GPU IDs (e.g. '0,1')")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Default language prompt when dataset has no tasks")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--fsdp_devices", type=str, default="auto",
                        help="FSDP device count: 'auto' (=GPU count when >=2), or integer")
    parser.add_argument("--lora", type=str, default="auto",
                        help="LoRA fine-tuning: 'auto' (=on for single GPU), 'true', or 'false'")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA decay rate (default: disabled to save VRAM, e.g. 0.99)")
    parser.add_argument("--action_horizon", type=int, default=50,
                        help="Action sequence length (default: 50)")
    _default_workers = min(os.cpu_count() or 8, 64)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader worker processes for training (default: 8)")
    parser.add_argument("--norm_stats_workers", type=int, default=_default_workers,
                        help=f"Parallel workers for fast norm-stats parquet reading "
                             f"(default: auto = min(cpu_count, 64), currently {_default_workers})")
    parser.add_argument("--norm_stats_max_frames", type=int, default=0,
                        help="Fast-path frame cap. 0 (default) reads every state/action row. "
                             "A positive value samples about that many frames from a seeded file subset. "
                             "Slow-path fallback still auto-caps very large datasets at 200,000 frames.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_dataset(dataset_dir: pathlib.Path) -> dict:
    """Read and return LeRobot dataset metadata."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"No LeRobot dataset at {dataset_dir}. "
            f"Expected {info_path}. Mount with -v /path/to/dataset:/data/input"
        )
    with open(info_path) as f:
        return json.load(f)


def setup_dataset_link(dataset_dir: pathlib.Path) -> str:
    """Symlink the mounted dataset into the LeRobot cache so the library can find it."""
    lerobot_home = pathlib.Path(
        os.environ.get("HF_LEROBOT_HOME", str(pathlib.Path.home() / ".cache" / "lerobot"))
    )
    repo_id = "docker/training_dataset"
    target = lerobot_home / repo_id
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    os.symlink(str(dataset_dir.resolve()), str(target))
    logger.info(f"Linked dataset: {dataset_dir} -> {target}")
    return repo_id


def _collect_episode_indices(dataset_dir: pathlib.Path, info: dict) -> list[int]:
    """Collect episode indices from metadata/files with safe fallbacks."""
    meta_dir = dataset_dir / "meta"
    episodes_jsonl = meta_dir / "episodes.jsonl"
    episode_indices: set[int] = set()

    if episodes_jsonl.exists():
        with open(episodes_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep_idx = row.get("episode_index")
                if isinstance(ep_idx, int):
                    episode_indices.add(ep_idx)

    if not episode_indices:
        data_root = dataset_dir / "data"
        if data_root.exists():
            for pq_file in data_root.glob("chunk-*/episode_*.parquet"):
                stem = pq_file.stem
                try:
                    episode_indices.add(int(stem.split("_")[-1]))
                except (TypeError, ValueError):
                    continue

    if not episode_indices:
        total_episodes = info.get("total_episodes")
        if isinstance(total_episodes, int) and total_episodes > 0:
            episode_indices.update(range(total_episodes))

    return sorted(episode_indices)


def ensure_v21_episodes_stats(dataset_dir: pathlib.Path, info: dict) -> None:
    """Ensure `meta/episodes_stats.jsonl` is readable by LeRobot 0.3."""
    meta_dir = dataset_dir / "meta"
    episodes_stats = meta_dir / "episodes_stats.jsonl"
    if episodes_stats_compatible_with_v21(episodes_stats):
        return

    episode_indices = _collect_episode_indices(dataset_dir, info)
    if not episode_indices:
        logger.warning("Unable to infer episode indices, skipping episodes_stats generation.")
        return

    reason = "not found" if not episodes_stats.exists() else "incompatible with LeRobot 0.3"
    logger.info(
        f"episodes_stats.jsonl {reason} for v2 dataset, generating from parquet "
        f"(episodes={len(episode_indices)}) ..."
    )
    chunks_size = info.get("chunks_size", 1000)
    try:
        chunks_size = int(chunks_size)
    except (TypeError, ValueError):
        chunks_size = 1000
    generate_episodes_stats_from_parquet(
        dataset_dir, episode_indices, chunks_size=max(1, chunks_size)
    )


def _replace_list_feature_type(obj):
    """
    Recursively normalize HF feature `_type: List` for datasets==3.6.0.

    Compatibility rule:
    - List + length -> Sequence
    - List without length -> LargeList
    """
    changed = False
    if isinstance(obj, dict):
        if obj.get("_type") == "List":
            if "length" in obj and obj.get("length") is not None:
                obj["_type"] = "Sequence"
            else:
                obj["_type"] = "LargeList"
                obj.pop("length", None)
            changed = True
        for value in obj.values():
            if isinstance(value, (dict, list)):
                child_changed = _replace_list_feature_type(value)
                changed = changed or child_changed
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                child_changed = _replace_list_feature_type(item)
                changed = changed or child_changed
    return changed


def normalize_parquet_hf_metadata(dataset_dir: pathlib.Path) -> None:
    """
    Best-effort fallback for datasets metadata incompatibility.

    Some parquet files carry HF metadata with `_type: "List"` which is not
    supported by datasets==3.6.0. We rewrite metadata to `LargeList`.
    """
    import pyarrow.parquet as pq

    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return

    parquet_files = sorted(p for p in data_dir.rglob("*.parquet") if p.is_file())
    if not parquet_files:
        return

    modified = 0
    for pq_file in parquet_files:
        schema = pq.read_schema(pq_file)
        metadata = schema.metadata or {}
        hf_raw = metadata.get(b"huggingface")
        if not hf_raw:
            continue
        if b"\"_type\":\"List\"" not in hf_raw and b"\"_type\": \"List\"" not in hf_raw:
            continue

        try:
            hf_meta = json.loads(hf_raw.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        if not _replace_list_feature_type(hf_meta):
            continue

        table = pq.read_table(pq_file)
        new_meta = dict(table.schema.metadata or {})
        new_meta[b"huggingface"] = json.dumps(hf_meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        patched = table.replace_schema_metadata(new_meta)
        tmp_path = pq_file.with_suffix(pq_file.suffix + ".tmp")
        pq.write_table(patched, tmp_path)
        os.replace(tmp_path, pq_file)
        modified += 1

    if modified > 0:
        logger.info(f"Normalized parquet HF metadata for {modified} file(s) (List -> LargeList).")


def analyze_features(info: dict) -> dict:
    """Extract image / state / action keys and dimensions from dataset metadata."""
    features = info.get("features", {})

    image_keys: list[str] = []
    state_key = None
    action_key = None
    state_dim = 0
    action_dim = 0
    has_tasks = "task_index" in features

    for key, feat in features.items():
        if not isinstance(feat, dict):
            continue
        dtype = str(feat.get("dtype", ""))
        shape = feat.get("shape", [])

        if dtype in ("image", "video"):
            image_keys.append(key)
        elif isinstance(shape, list) and len(shape) == 3 and (shape[-1] == 3 or shape[0] == 3):
            image_keys.append(key)
        elif "state" in key.lower() and state_key is None:
            state_key = key
            state_dim = shape[-1] if shape else 0
        elif key in ("action", "actions") and action_key is None:
            action_key = key
            action_dim = shape[-1] if shape else 0

    if not image_keys:
        raise ValueError("No image features found in the dataset")
    if state_key is None:
        raise ValueError("No state feature found in the dataset")
    if action_key is None:
        raise ValueError("No action feature found (expected key 'action' or 'actions')")

    return dict(
        image_keys=image_keys,
        state_key=state_key,
        action_key=action_key,
        state_dim=state_dim,
        action_dim=action_dim,
        has_tasks=has_tasks,
        fps=info.get("fps", 50),
    )


# ---------------------------------------------------------------------------
# Generic data transforms (no core-code changes needed)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class GenericLeRobotInputs:
    """Map dataset image keys onto OpenPI's three camera slots.

    ``camera_slots`` is ``(slot_name, dataset_key)`` pairs. Slots that are
    absent are zeros with ``image_mask=False``.
    """

    camera_slots: tuple
    state_key: str
    action_key: str = "action"

    @staticmethod
    def _parse_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        return img

    def __call__(self, data: dict) -> dict:
        model_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        slot_to_key = dict(self.camera_slots)
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}
        ref_shape = None

        for mkey in model_keys:
            dataset_key = slot_to_key.get(mkey)
            if dataset_key is not None and dataset_key in data:
                img = self._parse_image(data[dataset_key])
                images[mkey] = img
                image_masks[mkey] = np.True_
                if ref_shape is None:
                    ref_shape = img.shape
            else:
                shape = ref_shape or (224, 224, 3)
                images[mkey] = np.zeros(shape, dtype=np.uint8)
                image_masks[mkey] = np.False_

        result = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data[self.state_key]),
        }

        act = data.get(self.action_key)
        if act is None:
            act = data.get("actions")
        if act is not None:
            result["actions"] = np.asarray(act)

        if "prompt" in data:
            result["prompt"] = data["prompt"]

        return result


@dataclasses.dataclass(frozen=True)
class GenericLeRobotOutputs:
    """Trim padded actions back to the real action dimension (inference only)."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def _rows_per_parquet(path: pathlib.Path, column: str) -> int:
    import pyarrow.parquet as pq

    try:
        return int(pq.read_table(path, columns=[column]).num_rows)
    except Exception:
        return 0


def select_parquet_files(
    files: list[pathlib.Path],
    max_frames: int | None,
    rows_per_file: int,
    seed: int = 0,
) -> list[pathlib.Path]:
    """Return every file when ``max_frames`` is unset, else a seeded subset."""
    ordered = list(files)
    if max_frames is None or max_frames <= 0 or rows_per_file <= 0 or not ordered:
        return ordered
    n_files = max(1, max_frames // max(1, rows_per_file) + 1)
    if n_files >= len(ordered):
        return ordered
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:n_files]


def _compute_norm_stats_fast(
    config,
    dataset_dir: pathlib.Path,
    schema: dict,
    max_frames: int | None,
    num_workers: int,
) -> bool:
    """Compute norm-stats by reading state/action columns directly from parquet files.

    This completely bypasses LeRobot's video-decoding pipeline, making it orders of
    magnitude faster for large video datasets where only state/action statistics are needed.

    RunningStats.update() reshapes input to (-1, last_dim), so per-frame parquet data
    [N, feat_dim] produces statistically equivalent results to the full pipeline's
    [N, action_horizon, feat_dim] batches. ``max_frames is None`` reads every row.
    A positive cap samples a seeded subset of files.

    Returns True on success, False if the fast path cannot be used.
    """
    import concurrent.futures

    import pyarrow.parquet as pq
    import openpi.shared.normalize as normalize
    from tqdm import tqdm

    state_col = schema["state_key"]
    action_col = schema["action_key"]

    # Discover all parquet files across all chunk directories
    data_root = dataset_dir / "data"
    parquet_files = sorted(data_root.glob("**/*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found; skipping fast norm-stats path.")
        return False

    # Validate that required columns exist in the first file
    try:
        sample_schema = pq.read_schema(parquet_files[0])
        available = sample_schema.names
        if state_col not in available or action_col not in available:
            logger.warning(
                f"Parquet columns '{state_col}' or '{action_col}' not found "
                f"(available: {available}); skipping fast norm-stats path."
            )
            return False
    except Exception as e:
        logger.warning(f"Failed to read parquet schema: {e}; skipping fast norm-stats path.")
        return False

    files_to_process = list(parquet_files)
    if max_frames is not None and max_frames > 0:
        files_to_process = select_parquet_files(
            files_to_process,
            max_frames,
            _rows_per_parquet(parquet_files[0], state_col),
        )

    logger.info(
        f"Fast norm-stats: {len(files_to_process)}/{len(parquet_files)} parquet files, "
        f"workers={num_workers}"
        + (f", max_frames={max_frames}" if max_frames else "")
    )

    def _read_file(pq_path: pathlib.Path):
        try:
            table = pq.read_table(pq_path, columns=[state_col, action_col])
            state_arr = np.array(table.column(state_col).to_pylist(), dtype=np.float32)
            action_arr = np.array(table.column(action_col).to_pylist(), dtype=np.float32)
            return state_arr, action_arr
        except Exception as e:
            logger.warning(f"Skipping {pq_path.name}: {e}")
            return None, None

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    total_frames = 0

    if num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            future_list = [pool.submit(_read_file, p) for p in files_to_process]
            pbar = tqdm(
                concurrent.futures.as_completed(future_list),
                total=len(future_list),
                desc="norm-stats (fast)",
            )
            for fut in pbar:
                state_arr, action_arr = fut.result()
                if state_arr is None:
                    continue
                state_stats.update(state_arr)
                action_stats.update(action_arr)
                total_frames += len(state_arr)
                pbar.set_postfix(frames=total_frames)
    else:
        for pq_path in tqdm(files_to_process, desc="norm-stats (fast)"):
            state_arr, action_arr = _read_file(pq_path)
            if state_arr is None:
                continue
            state_stats.update(state_arr)
            action_stats.update(action_arr)
            total_frames += len(state_arr)

    if state_stats._count < 2 or action_stats._count < 2:
        logger.warning("Not enough frames for fast norm-stats; will fall back to slow path.")
        return False

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    data_config = config.data.create(config.assets_dirs, config.model)
    out = config.assets_dirs / data_config.asset_id
    logger.info(f"Saving norm stats → {out}  (frames={total_frames:,})")
    normalize.save(out, norm_stats)
    return True


def compute_norm_stats(
    config,
    dataset_dir: pathlib.Path,
    schema: dict,
    max_frames: int | None = None,
    num_workers: int = 0,
) -> None:
    """Compute and save normalization stats if they don't already exist.

    Strategy:
    1. Fast path: read state/action columns directly from parquet (skips video decoding).
       Expected speedup: from hours to minutes on large video datasets.
    2. Slow path (fallback): use the LeRobot DataLoader pipeline with configurable
       num_workers and optional max_frames sampling.
    """
    import openpi.shared.normalize as normalize
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as transforms

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is not None:
        logger.info("Normalization stats already present – skipping computation")
        return

    logger.info("Computing normalization statistics …")

    # --- Fast path: direct parquet reads (skips video decoding entirely) ---
    try:
        success = _compute_norm_stats_fast(
            config, dataset_dir, schema, max_frames, num_workers
        )
        if success:
            return
    except Exception as e:
        logger.warning(f"Fast norm-stats path failed ({e}); falling back to slow path.")

    # --- Slow path: full LeRobot DataLoader pipeline ---
    logger.info(
        f"Slow norm-stats: LeRobot DataLoader pipeline "
        f"(num_workers={num_workers}"
        + (f", max_frames={max_frames}" if max_frames else "")
        + ")"
    )

    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )

    class _RemoveStrings(transforms.DataTransformFn):
        def __call__(self, x):
            return {
                k: v for k, v in x.items()
                if not np.issubdtype(np.asarray(v).dtype, np.str_)
            }

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _RemoveStrings(),
        ],
    )

    total_samples = len(dataset)
    bs = min(config.batch_size, max(1, total_samples))

    # Auto-cap slow-path frames: when the dataset is large and the user hasn't set an
    # explicit limit, cap at 200,000 frames to keep runtime under ~10 minutes.
    # (The fast parquet path already reads all frames quickly; this cap only applies
    # when we fall back to the LeRobot DataLoader which does costly video decoding.)
    _SLOW_PATH_AUTO_CAP = 200_000
    _SLOW_PATH_CAP_THRESHOLD = 500_000
    effective_max_frames = max_frames
    if effective_max_frames is None and total_samples > _SLOW_PATH_CAP_THRESHOLD:
        effective_max_frames = _SLOW_PATH_AUTO_CAP
        logger.info(
            f"Slow-path norm-stats: dataset has {total_samples:,} frames; "
            f"auto-capping at {_SLOW_PATH_AUTO_CAP:,} randomly-sampled frames "
            f"for statistical accuracy. Pass --norm_stats_max_frames=0 to disable."
        )

    if effective_max_frames is not None and effective_max_frames > 0 and effective_max_frames < total_samples:
        n_batches = max(1, effective_max_frames // bs)
        shuffle = True
    else:
        n_batches = max(1, total_samples // bs)
        shuffle = False

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=bs,
        num_batches=n_batches,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    keys = ["state", "actions"]
    stats = {k: normalize.RunningStats() for k in keys}

    from tqdm import tqdm
    for batch in tqdm(loader, total=n_batches, desc="norm-stats"):
        for k in keys:
            if k in batch:
                stats[k].update(np.asarray(batch[k]))

    norm_stats = {
        k: s.get_statistics() for k, s in stats.items() if s._count >= 2
    }
    if not norm_stats:
        raise RuntimeError("Dataset too small to compute normalization statistics")

    out = config.assets_dirs / data_config.asset_id
    logger.info(f"Saving norm stats → {out}")
    normalize.save(out, norm_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cache_dir_for(args, dataset_dir: pathlib.Path, info: dict) -> pathlib.Path:
    if args.convert_dir is not None:
        return pathlib.Path(args.convert_dir)
    stamp = conversion_stamp(dataset_dir, video_keys_from_info(info), 1000)
    digest = hashlib.sha256(
        json.dumps(stamp, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return pathlib.Path(args.output_dir) / ".v21_cache" / digest


def prepare_dataset(args) -> dict:
    """Resolve the on-disk tree training should read, including camera filters."""
    dataset_dir = pathlib.Path(args.dataset_dir)
    output_dir = pathlib.Path(args.output_dir)
    if not (dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"No LeRobot dataset at {dataset_dir}. "
            "Expected meta/info.json. Mount with -v /path/to/dataset:/data/input"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    info = discover_dataset(dataset_dir)
    raw_images = analyze_features(info)["image_keys"]
    selected = select_image_keys(
        raw_images,
        cameras=_csv_list(args.cameras),
        drop_cameras=_csv_list(args.drop_cameras),
    )
    dropped = [key for key in raw_images if key not in selected]
    cache_dir = _cache_dir_for(args, dataset_dir, info)
    version = str(info.get("codebase_version", ""))
    logger.info(f"Dataset codebase_version : {version or 'unknown'}")
    logger.info(f"Image built for LeRobot  : {os.environ.get('LEROBOT_DATASET_VERSION') or 'auto'}")

    if version.startswith("v3") or version.startswith("3"):
        logger.info("Detected v3.0 dataset – converting to v2.1 layout for compatibility …")
        full_dir = convert_v3_to_v2(dataset_dir, cache_dir)
        logger.info(f"Converted dataset version: {discover_dataset(full_dir).get('codebase_version')}")
    else:
        full_dir = stage_writable_dataset(dataset_dir, cache_dir / "writable")

    full_videos = set(video_keys_from_info(discover_dataset(full_dir)))
    if set(selected) != full_videos:
        effective_dir = make_camera_view(full_dir, camera_view_dir(cache_dir, selected), selected)
    else:
        effective_dir = full_dir

    info = discover_dataset(effective_dir)
    if str(info.get("codebase_version", "")).lower().startswith("v2"):
        ensure_v21_episodes_stats(effective_dir, info)
    normalize_parquet_hf_metadata(effective_dir)

    schema = analyze_features(info)
    slots = assign_camera_slots(schema["image_keys"], parse_camera_map(args.camera_map))
    has_task_text = bool(schema["has_tasks"] and tasks_have_text(effective_dir))
    task_texts = [str(row["task"]) for row in load_tasks(effective_dir)] if has_task_text else []
    logger.info(f"  image keys : {schema['image_keys']}")
    logger.info(f"  dropped    : {dropped or 'none'}")
    logger.info(f"  camera slots: {slots}")
    logger.info(f"  state      : {schema['state_key']}  dim={schema['state_dim']}")
    logger.info(f"  action     : {schema['action_key']}  dim={schema['action_dim']}")
    logger.info(f"  has tasks  : {schema['has_tasks']}  task text: {task_texts or has_task_text}")
    return {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "effective_dir": effective_dir,
        "schema": schema,
        "slots": slots,
        "dropped": dropped,
        "has_task_text": has_task_text,
        "task_texts": task_texts,
    }


def write_run_manifest(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_manifest_into_steps(checkpoint_dir: pathlib.Path) -> None:
    manifest = checkpoint_dir / "run_manifest.json"
    if not manifest.is_file():
        return
    for child in checkpoint_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            shutil.copyfile(manifest, child / "run_manifest.json")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        prepared = prepare_dataset(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    output_dir = prepared["output_dir"]
    effective_dir = prepared["effective_dir"]
    schema = prepared["schema"]
    slots = prepared["slots"]
    has_task_text = prepared["has_task_text"]

    # ---- link dataset into LeRobot cache ----
    repo_id = setup_dataset_link(effective_dir)

    # ---- GPU ----
    if args.gpus != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"CUDA_VISIBLE_DEVICES = {args.gpus}")

    # ---- delayed heavy imports ----
    import jax
    import flax.nnx as nnx
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as _transforms
    from scripts.train import main as train_main

    # Auto-adjust batch_size: must be divisible by the number of JAX devices
    n_devices = jax.device_count()
    if args.batch_size < n_devices:
        logger.warning(
            f"batch_size ({args.batch_size}) < device_count ({n_devices}); "
            f"raising to {n_devices}"
        )
        args.batch_size = n_devices
    elif args.batch_size % n_devices != 0:
        adjusted = ((args.batch_size // n_devices) + 1) * n_devices
        logger.warning(
            f"batch_size ({args.batch_size}) not divisible by device_count ({n_devices}); "
            f"raising to {adjusted}"
        )
        args.batch_size = adjusted
    logger.info(f"JAX devices: {n_devices}  batch_size: {args.batch_size}")

    # ---- GPU memory optimization: auto-detect optimal strategy ----
    if args.fsdp_devices == "auto":
        fsdp_devices = n_devices if n_devices >= 2 else 1
    else:
        fsdp_devices = int(args.fsdp_devices)

    if args.lora == "auto":
        use_lora = (n_devices == 1)
    else:
        use_lora = args.lora.lower() in ("true", "1", "yes")

    ema_decay = args.ema_decay

    if use_lora:
        ema_decay = None

    logger.info("=" * 50)
    logger.info("[VRAM optimization] Auto-detected settings:")
    logger.info(f"  GPU count       : {n_devices}")
    logger.info(f"  FSDP devices    : {fsdp_devices}")
    logger.info(f"  LoRA fine-tune  : {use_lora}")
    logger.info(f"  EMA             : {'on (decay={})'.format(ema_decay) if ema_decay else 'off (saving ~5GB VRAM)'}")
    logger.info(f"  action_horizon  : {args.action_horizon}")
    logger.info(f"  num_workers     : {args.num_workers}")
    logger.info(f"  XLA mem fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'default')}")
    if n_devices == 1 and use_lora:
        logger.info("  Strategy: single GPU -> LoRA (trainable params ~50MB, optimizer ~200MB)")
    elif fsdp_devices > 1:
        logger.info(f"  Strategy: {n_devices} GPUs -> FSDP={fsdp_devices} (params sharded across devices)")
    logger.info("=" * 50)

    # ---- model type ----
    model_type = os.environ.get("MODEL_TYPE", "pi0")
    logger.info(f"Model type: {model_type}")

    if model_type == "pi05":
        weight_path = "/models/openpi-assets/checkpoints/pi05_base/params"
    else:
        weight_path = "/models/openpi-assets/checkpoints/pi0_base/params"

    # ---- model config (with LoRA / action_horizon) ----
    freeze_filter = nnx.Nothing()
    if use_lora:
        if model_type == "pi05":
            model_config = pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=args.action_horizon,
            )
        else:
            model_config = pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=args.action_horizon,
            )
        freeze_filter = model_config.get_freeze_filter()
        logger.info("LoRA enabled: only LoRA adapter weights are trainable")
    else:
        if model_type == "pi05":
            model_config = pi0_config.Pi0Config(
                pi05=True,
                action_horizon=args.action_horizon,
            )
        else:
            model_config = pi0_config.Pi0Config(
                action_horizon=args.action_horizon,
            )

    # ---- transforms ----
    generic_inputs = GenericLeRobotInputs(
        camera_slots=tuple(slots.items()),
        state_key=schema["state_key"],
        action_key=schema["action_key"],
    )
    generic_outputs = GenericLeRobotOutputs(action_dim=schema["action_dim"])
    input_transforms = [generic_inputs]
    output_transforms = [generic_outputs]
    action_mode = "absolute"
    delta_mask: tuple | None = None
    if args.delta_joint_actions:
        if schema["action_dim"] < 1:
            logger.error("delta_joint_actions requires a positive action dimension")
            sys.exit(1)
        delta_mask = _transforms.make_bool_mask(max(schema["action_dim"] - 1, 0), -1)
        input_transforms.append(_transforms.DeltaActions(delta_mask))
        output_transforms = [_transforms.AbsoluteActions(delta_mask), *output_transforms]
        action_mode = "delta_joints"
        logger.info(f"delta joint actions enabled, mask={delta_mask}")

    default_prompt = args.prompt or os.environ.get("DEFAULT_PROMPT", "perform the task")

    data_factory = _config.SimpleDataConfig(
        repo_id=repo_id,
        assets=_config.AssetsConfig(asset_id="training_dataset"),
        data_transforms=lambda _mc: _transforms.Group(
            inputs=input_transforms,
            outputs=output_transforms,
        ),
        model_transforms=_config.ModelTransformFactory(
            default_prompt=None if has_task_text else default_prompt,
        ),
        base_config=_config.DataConfig(
            prompt_from_task=has_task_text,
            action_sequence_keys=(schema["action_key"],),
        ),
    )

    # ---- optimizer / lr ----
    lr_schedule = _optimizer.CosineDecaySchedule(
        warmup_steps=min(1000, args.steps // 10),
        peak_lr=args.learning_rate or 2.5e-5,
        decay_steps=args.steps,
        decay_lr=2.5e-6,
    )

    # ---- assemble TrainConfig ----
    peak_lr = args.learning_rate or 2.5e-5
    config = _config.TrainConfig(
        name=args.run_name,
        model=model_config,
        data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(weight_path),
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        checkpoint_base_dir=str(output_dir),
        assets_base_dir="/workspace/assets",
        exp_name=args.exp_name,
        overwrite=True,
        wandb_enabled=False,
        save_interval=args.save_interval,
        keep_period=5000,
        lr_schedule=lr_schedule,
        num_workers=args.num_workers,
        fsdp_devices=fsdp_devices,
        ema_decay=ema_decay,
        freeze_filter=freeze_filter,
    )

    logger.info(f"batch_size={args.batch_size}  steps={args.steps}")
    logger.info(f"checkpoint_dir = {config.checkpoint_dir}")
    logger.info(f"weight source  = {weight_path}")

    norm_max_frames = args.norm_stats_max_frames if args.norm_stats_max_frames > 0 else None
    manifest = {
        "model_type": model_type,
        "lora": use_lora,
        "weight_path": weight_path,
        "openpi_git_ref": os.environ.get("OPENPI_GIT_REF", ""),
        "run_name": args.run_name,
        "exp_name": args.exp_name,
        "cameras": list(schema["image_keys"]),
        "camera_slots": slots,
        "dropped_cameras": prepared["dropped"],
        "prompt_from_task": has_task_text,
        "task_texts": prepared["task_texts"],
        "default_prompt": None if has_task_text else default_prompt,
        "action_horizon": args.action_horizon,
        "action_mode": action_mode,
        "delta_mask": list(delta_mask) if delta_mask is not None else None,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": peak_lr,
        "save_interval": args.save_interval,
        "norm_stats_max_frames": args.norm_stats_max_frames,
        "dataset_dir": str(prepared["dataset_dir"]),
        "effective_dataset_dir": str(effective_dir),
    }
    # Keep the manifest beside the checkpoint directory. Training with
    # overwrite=True deletes checkpoint_dir itself before the first step.
    manifest_sidecar = config.checkpoint_dir.parent / f"{config.checkpoint_dir.name}.run_manifest.json"
    write_run_manifest(manifest_sidecar, manifest)

    # ---- step 1: normalization statistics ----
    compute_norm_stats(
        config,
        dataset_dir=effective_dir,
        schema=schema,
        max_frames=norm_max_frames,
        num_workers=args.norm_stats_workers,
    )

    # ---- step 2: train ----
    logger.info("Starting training …")
    train_main(config)
    write_run_manifest(config.checkpoint_dir / "run_manifest.json", manifest)
    copy_manifest_into_steps(config.checkpoint_dir)
    logger.info(f"Training complete.  Checkpoints → {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
