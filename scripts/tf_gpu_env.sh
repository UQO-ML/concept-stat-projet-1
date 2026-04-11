#!/usr/bin/env sh
# Source this file after activating the virtual environment:
#   . ./scripts/tf_gpu_env.sh

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "VIRTUAL_ENV is not set. Activate your venv first."
  return 1 2>/dev/null || exit 1
fi

add_path_if_exists() {
  if [ -d "$1" ]; then
    if [ -z "${LD_LIBRARY_PATH:-}" ]; then
      LD_LIBRARY_PATH="$1"
    else
      LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH"
    fi
  fi
}

add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cublas/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cudnn/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cufft/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/curand/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusolver/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/nccl/lib"
add_path_if_exists "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/nvjitlink/lib"
add_path_if_exists "/usr/lib"
add_path_if_exists "/opt/cuda/lib64"

export LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH updated for TensorFlow GPU runtime."
