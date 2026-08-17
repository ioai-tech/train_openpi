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
| `--learning_rate` | `2.5e-5` | |
| `--fsdp_devices` | `auto` | GPU count when >= 2 |
| `--lora` | `auto` | `true` / `false` |
| `--ema_decay` | off | e.g. `0.99` |
| `--action_horizon` | `50` | |
| `--num_workers` | `8` | |
| `--norm_stats_workers` | `min(cpu, 64)` | |
| `--norm_stats_max_frames` | `10000` | |

## LoRA

[OpenPI](https://github.com/Physical-Intelligence/openpi): LoRA needs >22.5GB
VRAM and runs on a 24GB GPU (e.g. RTX 4090). Full fine-tuning needs >70GB.

Single-GPU default is on. Disable with `--lora false`.

## License

Apache-2.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream
OpenPI.
