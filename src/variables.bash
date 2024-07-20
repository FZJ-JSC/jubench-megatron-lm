#!/usr/bin/env bash

# Important directories

# The main directory you want to work in.
if [ "x$NLP_BENCH_ROOT" = x ]; then
    echo "\$NLP_BENCH_ROOT must be set to rootdir path of this folder to execute the benchmark"
    exit 1
fi

echo "NLP_BENCH_ROOT is set as $NLP_BENCH_ROOT"

if [ "x$SYSTEMNAME" = x ]; then
    SYSTEMNAME="unknown"
fi

# Where the Python virtual environment will be generated.
VENV_DIR="$NLP_BENCH_ROOT"/venv-"$SYSTEMNAME"
# src directory
SRC_DIR="$NLP_BENCH_ROOT"/src
# Where the Megatron-LM code will be stored
MEGATRON_LM_REPO="$SRC_DIR"/Megatron-LM

# Input data
VOCAB_FILE="$NLP_BENCH_ROOT"/benchmark/aux/tokenizers/gpt2-vocab.json
MERGE_FILE="$NLP_BENCH_ROOT"/benchmark/aux/tokenizers/gpt2-merges.txt

# Path to a singular, preprocessed dataset.
DATA_PATH="$SRC_DIR"/data/oscar_text_document

# Output data
# The main directory you want to store output in.
ROOT_OUTPUT_DIR="$NLP_BENCH_ROOT"/output


# Check whether variables were set.
[ "x$NLP_BENCH_ROOT" = x ] \
    && echo 'Please set `NLP_BENCH_ROOT` in `variables.bash.' && return 1
[ "x$VENV_DIR" = x ] \
    && echo 'Please set `VENV_DIR` in `variables.bash.' && return 1
[ "x$MEGATRON_LM_REPO" = x ] \
    && echo 'Please set `MEGATRON_LM_REPO` in `variables.bash.' \
    && return 1
[ "x$ROOT_OUTPUT_DIR" = x ] \
    && echo 'Please set `ROOT_OUTPUT_DIR` in `variables.bash.' && return 1
[ "x$VOCAB_FILE" = x ] \
    && echo 'Please set `VOCAB_FILE` in `variables.bash.' && return 1
[ "x$MERGE_FILE" = x ] \
    && echo 'Please set `MERGE_FILE` in `variables.bash.' && return 1
[ "x$DATA_PATH" = x ] \
    && echo 'Please set `DATA_PATH` in `variables.bash.' && return 1

:
