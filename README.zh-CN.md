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
| `--dataset_dir` | `/data/input` | 或 `$OPENPI_DATASET_DIR` |
| `--output_dir` | `/data/output` | 或 `$OPENPI_OUTPUT_DIR` |
| `--run_name` | `docker_train` | checkpoint 的上一级目录 |
| `--exp_name` | `train` | |
| `--convert_dir` | 在 `--output_dir` 下 | v3→v2 缓存，不要放在小的 tmpfs 上 |
| `--cameras` | 全部图像键 | 逗号分隔，只保留这些键 |
| `--drop_cameras` | | 键名或子串，例如 `front` |
| `--camera_map` | 按名字角色 | `base=键,left_wrist=键,right_wrist=键` |
| `--delta_joint_actions` | 关闭 | 关节用增量，最后一维（夹爪）保持绝对 |
| `--norm_stats_workers` | `min(cpu, 64)` | |
| `--norm_stats_max_frames` | `0` | `0` 表示读完全部状态/动作 |

## LoRA

[OpenPI](https://github.com/Physical-Intelligence/openpi) 官方：LoRA 微调约需
>22.5GB 显存，可在 24GB 机器（如 RTX 4090）上跑；全量微调约需 >70GB。

单卡默认开启。关闭：`--lora false`。

## 摄像头

Pi0 / Pi0.5 固定三个槽位：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。
键名里的 `front` / `base` / `high` / `exterior` 进 base，`wrist` 进左手腕，
同时含 `right` 和 `wrist` 进右手腕。剩下的键按这个顺序填空槽。没有分到的槽位
是全零图像，并且 `image_mask=false`。

最多使用三个摄像头。数据集更多时用 `--cameras` 或 `--drop_cameras`。下面先训
三个摄像头，再用同一份缓存去掉 front：

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

v3 数据集只会在 `--convert_dir` 里转换一次。同一目录、同一组视频再次运行会直接
复用。去掉部分摄像头时，会在旁边做一个符号链接目录，并改写 `meta/info.json`，
LeRobot 就不会去解码被丢掉的摄像头。缓存要放在真实磁盘上，转换器不会再写到 `/tmp`。

只读挂载的数据不会被改写。无法写入的 v2 数据会先在缓存里做一个可写视图，再修补
元数据。每次运行还会在 `<run_name>/<exp_name>.run_manifest.json` 记下相机映射、
任务文本，以及动作是否保持绝对量。

镜像入口会在宿主机驱动比镜像里的 CUDA compat 库更新时改用宿主机的 `libcuda`。
较新的数据中心卡和工作站卡（包括 compute capability 12.0）属于这种情况。

## 许可证

Apache-2.0。上游 OpenPI 说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
