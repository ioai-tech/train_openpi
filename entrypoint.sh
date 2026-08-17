#!/usr/bin/env bash
set -euo pipefail

echo "============================================="
echo "  OpenPI Training Container"
echo "  Model : ${MODEL_TYPE:-pi0}"
echo "  LeRobot dataset: ${LEROBOT_DATASET_VERSION:-auto} (v2/v3 auto-detect)"
echo "============================================="

# Activate the virtual environment
# shellcheck disable=SC1091
source /.venv/bin/activate

# Ensure output directory exists
mkdir -p /data/output

# Validate dataset mount (skip for --help)
if [[ "${1:-}" != "--help" && "${1:-}" != "-h" ]]; then
    if [ ! -d /data/input ] || [ ! -f /data/input/meta/info.json ]; then
        echo "ERROR: No LeRobot dataset found at /data/input"
        echo ""
        echo "Mount your dataset:"
        echo "  docker run --rm --gpus all \\"
        echo "    -v /path/to/lerobot_dataset:/data/input:ro \\"
        echo "    -v /path/to/output:/data/output \\"
        echo "    ioaitech/train_openpi:pi0-cuda126"
        exit 1
    fi
    if [ ! -w /data/output ]; then
        echo "ERROR: /data/output is not writable. Check the bind-mount permissions."
        exit 1
    fi
    echo "Dataset found at /data/input"
    echo "Output directory: /data/output"
    echo ""
fi

# Force system NCCL (from CUDA runtime image) over jaxlib's bundled NCCL.
# jaxlib bundles NCCL compiled against CUDA 12.2 which triggers a known
# multi-GPU JIT deadlock on JAX 0.5.x (openpi issue #480).
# LD_LIBRARY_PATH alone is insufficient because jaxlib's dlopen resolves
# its own package directory first; LD_PRELOAD guarantees the system copy.
SYSTEM_NCCL=$(find /usr/lib -name 'libnccl.so.2' 2>/dev/null | head -1)
if [ -n "$SYSTEM_NCCL" ]; then
    export LD_PRELOAD="${SYSTEM_NCCL}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "Using system NCCL: $SYSTEM_NCCL"
fi

# Safety: strip CUDA_LAUNCH_BLOCKING if it leaked into the environment --
# synchronous kernel launches deadlock multi-GPU NCCL collectives.
unset CUDA_LAUNCH_BLOCKING

exec python /app/train_lerobot.py "$@"
