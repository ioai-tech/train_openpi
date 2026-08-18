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

# CUDA 12.6 images ship cuda-compat (libcuda 560.x) for older host drivers.
# On driver 570/580 that DSO overrides the host libcuda injected at
# /usr/local/nvidia/lib64 and JAX/XLA fails with
# CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (kernel 580.x vs DSO 560.x).
# Official table: cuda-compat-12-6 is incompatible with driver 570+ / 580+.
# Keep compat only when the host driver is older than the compat DSO.
_libcuda_version() {
    local so="$1"
    local resolved base
    resolved="$(readlink -f "$so" 2>/dev/null || true)"
    [ -n "$resolved" ] || resolved="$so"
    base="$(basename "$resolved")"
    echo "$base" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1 || true
}

_version_ge() {
    local left="$1" right="$2"
    [ "$(printf '%s\n%s\n' "$right" "$left" | sort -V | tail -n1)" = "$left" ]
}

_strip_ld_library_path() {
    local drop="$1"
    local new_path
    new_path="$(printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | awk -v d="$drop" 'NF && $0 != d && $0 != d "/"' | paste -sd: -)"
    export LD_LIBRARY_PATH="$new_path"
}

prefer_host_libcuda_over_stale_compat() {
    local compat_so="/usr/local/cuda/compat/libcuda.so.1"
    local compat_dir="/usr/local/cuda/compat"
    [ -e "$compat_so" ] || return 0

    local host_so="" d
    for d in /usr/local/nvidia/lib64 /usr/local/nvidia/lib /usr/lib/x86_64-linux-gnu; do
        if [ -e "$d/libcuda.so.1" ]; then
            host_so="$d/libcuda.so.1"
            break
        fi
    done

    local host_ver="" compat_ver=""
    compat_ver="$(_libcuda_version "$compat_so")"
    if [ -n "$host_so" ]; then
        host_ver="$(_libcuda_version "$host_so")"
    fi
    if [ -z "$host_ver" ] && command -v nvidia-smi >/dev/null 2>&1; then
        host_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
    fi

    if [ -z "$host_ver" ] || [ -z "$compat_ver" ]; then
        return 0
    fi

    if _version_ge "$host_ver" "$compat_ver"; then
        _strip_ld_library_path "$compat_dir"
        echo "Host driver ${host_ver} is newer than cuda-compat ${compat_ver}; using host libcuda"
    fi
}

prefer_host_libcuda_over_stale_compat

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
