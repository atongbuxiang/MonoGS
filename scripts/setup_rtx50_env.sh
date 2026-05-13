#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the MonoGS-rtx50 conda environment first."
  exit 1
fi

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc=$(command -v nvcc)"
echo "CC=$CC"
echo "CXX=$CXX"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
