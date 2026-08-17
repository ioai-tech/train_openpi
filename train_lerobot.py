#!/usr/bin/env python3
"""
Docker training wrapper for OpenPI with LeRobot datasets.

Auto-discovers dataset schema, computes normalization statistics,
constructs an OpenPI TrainConfig, and launches JAX-based training.

No modifications to OpenPI core code are required.
"""

import argparse
import dataclasses
import json
import logging
import os
import pathlib
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

from lerobot_v3_compat import convert_v3_to_v2
from lerobot_v3_compat import tasks_have_text
from lerobot_v3_compat import v2_data_relpath

DATASET_DIR = pathlib.Path("/data/input")
OUTPUT_DIR = pathlib.Path("/data/output")

logger = logging.getLogger("train_lerobot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenPI on a mounted LeRobot dataset")
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
    parser.add_argument("--norm_stats_max_frames", type=int, default=10000,
                        help="Limit frames sampled for norm-stats slow-path fallback "
                             "(default: auto-cap at 200,000 when dataset > 500,000 frames). "
                             "The fast parquet path always reads all frames regardless.")
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


def _generate_episodes_stats(
    dataset_dir: pathlib.Path,
    episode_indices: list[int],
    chunks_size: int = 1000,
):
    """Generate episodes_stats.jsonl from parquet data (required by lerobot 0.1.x)."""
    import pyarrow.parquet as pq

    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    float_features = None
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for ep_idx in episode_indices:
            pq_file = dataset_dir / v2_data_relpath(ep_idx, chunks_size)
            if not pq_file.exists():
                f.write(json.dumps({"episode_index": ep_idx, "stats": {}}) + "\n")
                continue

            table = pq.read_table(pq_file)
            if float_features is None:
                float_features = [
                    col for col in table.column_names
                    if table.schema.field(col).type in (
                        __import__("pyarrow").float32(),
                        __import__("pyarrow").float64(),
                        __import__("pyarrow").list_(__import__("pyarrow").float32()),
                        __import__("pyarrow").list_(__import__("pyarrow").float64()),
                    )
                ]

            ep_stats: dict = {}
            for col_name in float_features:
                col = table.column(col_name)
                try:
                    arr = np.array([
                        row.as_py() if hasattr(row, "as_py") else row
                        for row in col
                    ], dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    ep_stats[col_name] = {
                        "min": arr.min(axis=0).tolist(),
                        "max": arr.max(axis=0).tolist(),
                        "mean": arr.mean(axis=0).tolist(),
                        "std": arr.std(axis=0).tolist(),
                        "count": [len(arr)],
                    }
                except (ValueError, TypeError):
                    continue

            f.write(json.dumps({"episode_index": ep_idx, "stats": ep_stats}) + "\n")

    logger.info(f"Generated episodes_stats.jsonl for {len(episode_indices)} episodes")


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
    """Ensure `meta/episodes_stats.jsonl` exists for LeRobot v2.1 datasets."""
    meta_dir = dataset_dir / "meta"
    episodes_stats = meta_dir / "episodes_stats.jsonl"
    if episodes_stats.exists():
        return

    episode_indices = _collect_episode_indices(dataset_dir, info)
    if not episode_indices:
        logger.warning("Unable to infer episode indices, skipping episodes_stats generation.")
        return

    logger.info(
        "episodes_stats.jsonl not found for v2 dataset, generating from parquet "
        f"(episodes={len(episode_indices)}) ..."
    )
    chunks_size = info.get("chunks_size", 1000)
    try:
        chunks_size = int(chunks_size)
    except (TypeError, ValueError):
        chunks_size = 1000
    _generate_episodes_stats(dataset_dir, episode_indices, chunks_size=max(1, chunks_size))


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
    """Map any LeRobot schema to the three-image format expected by OpenPI models."""

    image_keys: tuple
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
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}
        ref_shape = None

        for i, mkey in enumerate(model_keys):
            if i < len(self.image_keys) and self.image_keys[i] in data:
                img = self._parse_image(data[self.image_keys[i]])
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
    [N, action_horizon, feat_dim] batches.

    Returns True on success, False if the fast path cannot be used.
    """
    import concurrent.futures
    import random

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

    files_to_process: list[pathlib.Path] = list(parquet_files)
    if max_frames is not None:
        # Estimate average frames per file from first file, then take a random subset
        try:
            n_sample = pq.read_table(parquet_files[0], columns=[state_col]).num_rows
            n_files_needed = max(1, max_frames // max(1, n_sample) + 1)
        except Exception:
            n_files_needed = len(files_to_process)
        random.shuffle(files_to_process)
        files_to_process = files_to_process[:n_files_needed]

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

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- validate dataset mount ----
    if not DATASET_DIR.exists():
        logger.error(
            "Dataset not found at /data/input. "
            "Mount your dataset: docker run -v /path/to/dataset:/data/input …"
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- discover dataset ----
    info = discover_dataset(DATASET_DIR)
    dataset_version = info.get("codebase_version", "unknown")
    expected_version = os.environ.get("LEROBOT_DATASET_VERSION", "")
    logger.info(f"Dataset codebase_version : {dataset_version}")
    logger.info(f"Image built for LeRobot  : {expected_version or 'auto'}")

    # ---- v3 -> v2 conversion if needed ----
    effective_dir = DATASET_DIR
    if dataset_version.startswith("v3") or dataset_version.startswith("3"):
        logger.info("Detected v3.0 dataset – converting to v2.1 layout for compatibility …")
        effective_dir = convert_v3_to_v2(DATASET_DIR)
        info = discover_dataset(effective_dir)
        logger.info(f"Converted dataset version: {info.get('codebase_version', 'unknown')}")

    effective_version = str(info.get("codebase_version", "")).lower()
    if effective_version.startswith("v2"):
        ensure_v21_episodes_stats(effective_dir, info)
    normalize_parquet_hf_metadata(effective_dir)

    schema = analyze_features(info)
    has_task_text = bool(schema["has_tasks"] and tasks_have_text(effective_dir))
    logger.info(f"  image keys : {schema['image_keys']}")
    logger.info(f"  state      : {schema['state_key']}  dim={schema['state_dim']}")
    logger.info(f"  action     : {schema['action_key']}  dim={schema['action_dim']}")
    logger.info(f"  has tasks  : {schema['has_tasks']}  task text: {has_task_text}")

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
        image_keys=tuple(schema["image_keys"]),
        state_key=schema["state_key"],
        action_key=schema["action_key"],
    )
    generic_outputs = GenericLeRobotOutputs(action_dim=schema["action_dim"])

    default_prompt = args.prompt or os.environ.get("DEFAULT_PROMPT", "perform the task")

    data_factory = _config.SimpleDataConfig(
        repo_id=repo_id,
        assets=_config.AssetsConfig(asset_id="training_dataset"),
        data_transforms=lambda _mc: _transforms.Group(
            inputs=[generic_inputs],
            outputs=[generic_outputs],
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
    config = _config.TrainConfig(
        name="docker_train",
        model=model_config,
        data=data_factory,
        weight_loader=weight_loaders.CheckpointWeightLoader(weight_path),
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        checkpoint_base_dir=str(OUTPUT_DIR),
        assets_base_dir="/workspace/assets",
        exp_name="train",
        overwrite=True,
        wandb_enabled=False,
        save_interval=args.save_interval,
        lr_schedule=lr_schedule,
        num_workers=args.num_workers,
        fsdp_devices=fsdp_devices,
        ema_decay=ema_decay,
        freeze_filter=freeze_filter,
    )

    logger.info(f"batch_size={args.batch_size}  steps={args.steps}")
    logger.info(f"checkpoint_dir = {config.checkpoint_dir}")
    logger.info(f"weight source  = {weight_path}")

    # ---- step 1: normalization statistics ----
    compute_norm_stats(
        config,
        dataset_dir=effective_dir,
        schema=schema,
        max_frames=args.norm_stats_max_frames,
        num_workers=args.norm_stats_workers,
    )

    # ---- step 2: train ----
    logger.info("Starting training …")
    train_main(config)
    logger.info(f"Training complete.  Checkpoints → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
