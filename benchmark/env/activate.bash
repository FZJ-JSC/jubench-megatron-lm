#!/usr/bin/env bash

# Here we activate the python venv 

[ -x "$(command -v deactivate)" ] && deactivate

module purge
# Loading required modules
# CUDA>=11.6 ; PyTorch>=1.12 and NCCL>=NCCL/2.12.7-1 is required
# module load Stages/2023 
# module load StdEnv/2023 \
#             GCC/11.3.0 \
#             CMake/3.23.1 \
#             OpenMPI/4.1.4 \
#             Python/3.10.4 \
#             SciPy-bundle/2022.05 \
#             Ninja/1.10.2 \
#             CUDA/11.7 \
#             cuDNN/8.6.0.163-CUDA-11.7 \
#             NCCL/default-CUDA-11.7 \
#             PyTorch/1.12.0-CUDA-11.7 \
#             git/2.36.0-nodocs \
#             torchvision/0.13.1-CUDA-11.7 \
#             TensorFlow/2.11.0-CUDA-11.7\
#             pybind11/.2.9.2 \

module load Stages/2024 
module load StdEnv/2024 \
            GCC/12.3.0 \
            CMake/3.26.3 \
            OpenMPI/4.1.5 \
            Python/3.11.3 \
            SciPy-bundle/2023.07 \
            Ninja/1.11.1  \
            CUDA/12 \
            cuDNN/8.9.5.29-CUDA-12  \
            NCCL/default-CUDA-12 \
            PyTorch/2.1.2 \
            git/2.41.0-nodocs \
            torchvision/0.16.2 \
            jq/1.6 \
            Score-P/8.3 \
            mpi4py/3.1.4
module unload protobuf-python/.4.24.0

if ! [ -d "$VENV_DIR" ]; then
     echo 'Please execute `nice bash set_up.bash` to set venv before continuing.'
     return 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$(realpath "$VENV_DIR"/lib/python*/site-packages):$PYTHONPATH"
export PYTHONPATH+=:"$MEGATRON_LM_REPO"

# Set `TORCH_CUDA_ARCH_LIST` according to GPU's compute capability.
if [ "$SYSTEMNAME" = juwelsbooster ] || [ "$SYSTEMNAME" = jurecadc ]; then
    # A100
    export TORCH_CUDA_ARCH_LIST='8.0'
elif [ "$SYSTEMNAME" = juwels ]; then
    # V100
    export TORCH_CUDA_ARCH_LIST='7.0'
else
    echo "The machine \"$SYSTEMNAME\" is currently not supported."
    return
fi

# Also allow generating PTX instructions (just because, we could also
# not add these).
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"+PTX
export CXX=g++

