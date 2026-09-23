# IOAI OpenPI Trainer

[简体中文](README.zh-CN.md)

Train [OpenPI](https://github.com/Physical-Intelligence/openpi) Pi0 / Pi0.5 from
LeRobot v2/v3 datasets via Docker.
Images: [Docker Hub `ioaitech/train_openpi`](https://hub.docker.com/r/ioaitech/train_openpi).
Source: [ioai-tech/train_openpi](https://github.com/ioai-tech/train_openpi).

Requires Linux, an NVIDIA GPU, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Images

| Tag | Model | CUDA |
| --- | --- | --- |
| `pi0` / `latest` / `pi0-cuda126` | Pi0 | 12.6 |
| `pi05` / `pi05-cuda126` | Pi0.5 | 12.6 |

Platform: `linux/amd64`. NVIDIA driver >= 525. Base weights and the PaliGemma
tokenizer are baked in, so training can run offline.

## Quick start

```bash
docker pull ioaitech/train_openpi:pi0-cuda126

mkdir -p ./openpi-output
docker run --rm --gpus all \
  -v /path/to/lerobot_dataset:/data/input:ro \
  -v "$(pwd)/openpi-output":/data/output \
  ioaitech/train_openpi:pi0-cuda126 \
  --batch_size 1 \
  --steps 1000 \
  --save_interval 200
```

Mount a LeRobot dataset at `/data/input` (`meta/info.json` required). v3 is
converted to v2 automatically. Checkpoints go to
`/data/output/docker_train/train/`.

## More examples

```bash
# Single GPU (LoRA on)
docker run --rm --gpus '"device=0"' --shm-size=8g \
  -v /data/my_dataset:/data/input:ro \
  -v /data/my_output:/data/output \
  ioaitech/train_openpi:pi0-cuda126 \
  --batch_size 4 \
  --steps 20000 \
  --save_interval 1000

# Multi-GPU (FSDP on, LoRA off)
docker run --rm --gpus all --ipc=host \
  -v /data/my_dataset:/data/input:ro \
  -v /data/my_output:/data/output \
  ioaitech/train_openpi:pi05-cuda126 \
  --gpus 0,1 \
  --batch_size 8 \
  --steps 30000
```

## Flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--batch_size` | `1` | |
| `--steps` | `1000` | |
| `--gpus` | `all` | or `0,1` |
| `--prompt` | | used only when the dataset has no task text |
| `--save_interval` | `500` | |
| `--keep_period` | `--save_interval` | steps divisible by this are never pruned; `0` keeps only the newest |
| `--resume` | off | continue from the newest checkpoint in the run directory |
| `--learning_rate` | `2.5e-5` | |
| `--fsdp_devices` | `auto` | GPU count when >= 2 |
| `--lora` | `auto` | `true` / `false` |
| `--ema_decay` | off | e.g. `0.99` |
| `--action_horizon` | `50` | |
| `--num_workers` | `8` | |
| `--dataset_dir` | `/data/input` | or `$OPENPI_DATASET_DIR` |
| `--output_dir` | `/data/output` | or `$OPENPI_OUTPUT_DIR` |
| `--run_name` | `docker_train` | checkpoint parent directory |
| `--exp_name` | `train` | |
| `--convert_dir` | under `--output_dir` | v3→v2 cache; do not use a small tmpfs |
| `--cameras` | all image keys | comma-separated keys to keep |
| `--drop_cameras` | | key or substring, e.g. `front` |
| `--camera_map` | role-based | `base=key,left_wrist=key,right_wrist=key` |
| `--delta_joint_actions` | off | joint deltas; only the last dim stays absolute |
| `--absolute_action_dims` | | with `--delta_joint_actions`, indices or action names that stay absolute. Replaces the last-dim default, e.g. `right_gripper,left_gripper` |
| `--norm_stats_workers` | `min(cpu, 64)` | |
| `--norm_stats_max_frames` | `0` | `0` reads every state/action row |

## LoRA

[OpenPI](https://github.com/Physical-Intelligence/openpi): LoRA needs >22.5GB
VRAM and runs on a 24GB GPU (e.g. RTX 4090). Full fine-tuning needs >70GB.

Single-GPU default is on. Disable with `--lora false`.

## Cameras

Pi0 / Pi0.5 always have three slots: `base_0_rgb`, `left_wrist_0_rgb`,
`right_wrist_0_rgb`. Keys are matched by name (`front` / `base` / `high` /
`exterior` → base, `wrist` → left wrist, `right` + `wrist` → right wrist).
Remaining keys fill empty slots. A missing slot is a zero image with
`image_mask=false`.

At most three cameras are used. Pass `--cameras` or `--drop_cameras` when a
dataset has more. Example, three cameras then the same cache without the
front camera:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v /path/to/lerobot_dataset:/data/input:ro \
  -v /path/to/output:/data/output \
  -v /path/to/cache:/data/cache \
  ioaitech/train_openpi:pi05-cuda126 \
  --run_name pi05_my_task_3cam \
  --steps 30000 \
  --save_interval 5000 \
  --convert_dir /data/cache/my_task

docker run --rm --gpus all --shm-size=16g \
  -v /path/to/lerobot_dataset:/data/input:ro \
  -v /path/to/output:/data/output \
  -v /path/to/cache:/data/cache \
  ioaitech/train_openpi:pi05-cuda126 \
  --run_name pi05_my_task_no_front \
  --drop_cameras front \
  --steps 30000 \
  --save_interval 5000 \
  --convert_dir /data/cache/my_task
```

v3 datasets are converted once into `--convert_dir`. A later run with the same
directory and the same videos reuses that tree. A camera subset is a second
directory of symlinks plus a rewritten `meta/info.json`, so LeRobot does not
decode dropped cameras. Put the cache on a real disk. The converter will not
write to `/tmp`.

The cache and the camera view are self-contained: links between them are
relative and resolve outside the container that built them. Episodes that
cover a whole source video are hardlinked, or copied when the source and the
cache are separate mounts, so no file points back at the dataset mount.

Read-only dataset mounts stay read-only. A v2 dataset that cannot be edited is
staged into the cache before metadata fixes. Each run writes
`<run_name>/<exp_name>.run_manifest.json` with the camera map, prompt, and
whether actions were left absolute.

The image's entrypoint uses the host `libcuda` when the host driver is newer
than the image's CUDA compat library. That is the usual case for newer
datacenter and workstation GPUs, including compute capability 12.0.

## License

Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream
OpenPI.
