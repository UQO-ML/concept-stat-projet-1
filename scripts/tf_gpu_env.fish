#!/usr/bin/env fish
# Source this file after activating the virtual environment:
#   source ./scripts/tf_gpu_env.fish

if test -z "$VIRTUAL_ENV"
    echo "VIRTUAL_ENV is not set. Activate your venv first."
    return 1
end

set -l paths \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cublas/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cudnn/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cufft/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/curand/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusolver/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/cusparse/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/nccl/lib" \
    "$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia/nvjitlink/lib" \
    /usr/lib \
    /opt/cuda/lib64

set -l valid_paths
for p in $paths
    if test -d "$p"
        set valid_paths $valid_paths "$p"
    end
end

set -gx LD_LIBRARY_PATH $valid_paths $LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH updated for TensorFlow GPU runtime."
