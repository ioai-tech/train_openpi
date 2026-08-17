# IOAI OpenPI 训练镜像

[English](README.md)

通过 Docker 从 LeRobot v2/v3 数据集训练
[OpenPI](https://github.com/Physical-Intelligence/openpi) 的 Pi0 / Pi0.5。
镜像：[Docker Hub `ioaitech/train_openpi`](https://hub.docker.com/r/ioaitech/train_openpi)。
源码：[ioai-tech/train_openpi](https://github.com/ioai-tech/train_openpi)。

需要 Linux、NVIDIA GPU、Docker 以及
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

## 镜像

| 标签 | 模型 | CUDA |
| --- | --- | --- |
| `pi0` / `latest` / `pi0-cuda126` | Pi0 | 12.6 |
| `pi05` / `pi05-cuda126` | Pi0.5 | 12.6 |

平台：`linux/amd64`。NVIDIA 驱动 >= 525。基座权重和 PaliGemma tokenizer 已打进
镜像，可离线训练。

## 快速开始

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

将 LeRobot 数据集挂载到 `/data/input`（需包含 `meta/info.json`）。v3 会自动转成
v2。checkpoint 写入 `/data/output/docker_train/train/`。

## 更多示例

```bash
# 单卡（默认 LoRA）
docker run --rm --gpus '"device=0"' --shm-size=8g \
  -v /data/my_dataset:/data/input:ro \
  -v /data/my_output:/data/output \
  ioaitech/train_openpi:pi0-cuda126 \
  --batch_size 4 \
  --steps 20000 \
  --save_interval 1000

# 多卡（FSDP，LoRA 关闭）
docker run --rm --gpus all --ipc=host \
  -v /data/my_dataset:/data/input:ro \
  -v /data/my_output:/data/output \
  ioaitech/train_openpi:pi05-cuda126 \
  --gpus 0,1 \
  --batch_size 8 \
  --steps 30000
```

## 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch_size` | `1` | |
| `--steps` | `1000` | |
| `--gpus` | `all` | 或 `0,1` |
| `--prompt` | | 仅当数据集没有 task 文本时使用 |
| `--save_interval` | `500` | |
| `--learning_rate` | `2.5e-5` | |
| `--fsdp_devices` | `auto` | GPU >= 2 时等于卡数 |
| `--lora` | `auto` | `true` / `false` |
| `--ema_decay` | 关闭 | 例如 `0.99` |
| `--action_horizon` | `50` | |
| `--num_workers` | `8` | |
| `--norm_stats_workers` | `min(cpu, 64)` | |
| `--norm_stats_max_frames` | `10000` | |

## LoRA

[OpenPI](https://github.com/Physical-Intelligence/openpi) 官方：LoRA 微调约需
>22.5GB 显存，可在 24GB 机器（如 RTX 4090）上跑；全量微调约需 >70GB。

单卡默认开启。关闭：`--lora false`。

## 许可证

Apache-2.0。上游 OpenPI 说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
