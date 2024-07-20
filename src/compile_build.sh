#!/usr/bin/env bash

set -euox pipefail

if [ "x$NLP_BENCH_ROOT" = "x" ]; then
    echo "NLP_BENCH_ROOT is not set. Please set it to the root directory of nlp-benchmark" >&2
    exit 1
fi

DONE_FILE=$NLP_BENCH_ROOT/src/build_done
PATCH_APPLIED=$NLP_BENCH_ROOT/src/patch_applied

if [ -f $DONE_FILE ]; then
    echo "$DONE_FILE exists, exiting"
    exit 0
fi

export PIP_CACHE_DIR=$NLP_BENCH_ROOT/pip_cache

source $NLP_BENCH_ROOT/src/variables.bash || exit 1

export CUDA_VISIBLE_DEVICES=0

[ -d "$VENV_DIR" ] || python -m venv --system-site-packages "$VENV_DIR"

source "$NLP_BENCH_ROOT"/benchmark/env/activate.bash || exit 1

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"

cd "$MEGATRON_LM_REPO"
ln -sf "$NLP_BENCH_ROOT"/src/prebuild_kernels.py ./prebuild_kernels.py
echo "Remove previous build if any."
rm -rf megatron/fused_kernels/build

#### apply add_tflops_logging.patch
if ! [ -f "$PATCH_APPLIED" ]; then
    git apply "$NLP_BENCH_ROOT"/benchmark/aux/add_tflops_logging.patch
    touch $PATCH_APPLIED
fi

### Install flash attention
python -m pip install nltk sentencepiece einops mpmath ninja setuptools==69.5.1
python -m pip install flash-attn --no-build-isolation 

# FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention.git
# FLASH_ATTN_COMMIT=${FLASH_ATTN_COMMIT:-v2.1.1}
# FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-build-isolation git+${FLASH_ATTN_URL}@${FLASH_ATTN_COMMIT}

cd "$NLP_BENCH_ROOT"/src/apex
python -m pip install \
       --config-settings="--build-option=--cpp_ext" \
       --config-settings="--build-option=--cuda_ext" \
       --no-build-isolation \
       --no-cache-dir \
       -v \
       --disable-pip-version-check \
       . 2>&1 \
    | tee apex_build.log

cd "$MEGATRON_LM_REPO"
echo "Building the fused kernels."
export PYTHONPATH+=:"$MEGATRON_LM_REPO"
python prebuild_kernels.py

cd ..

touch $DONE_FILE

echo "Done building and compiling Megatron-LM, apex and fused kernels!"
