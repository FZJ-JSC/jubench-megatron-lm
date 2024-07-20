#!/usr/bin/env bash
#SBATCH --job-name=nlp-preprocessing-data
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=48            # number of cores per tasks
#SBATCH --hint=nomultithread        
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time=00:15:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=nlp-preprocessing-data.out             # output file name
#SBATCH --error=nlp-preprocessing-data.err              # error output file name


# srun doesnot inherit cpus-per-task from sbatch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

if [ "x$NLP_BENCH_ROOT" = "x" ]; then
    echo "NLP_BENCH_ROOT is not set. Please set it to the root directory of nlp-benchmark" >&2
    exit 1
fi


source $NLP_BENCH_ROOT/src/variables.bash
source "$NLP_BENCH_ROOT"/benchmark/env/activate.bash || exit 1

# Note this must end in `.jsonl`, not `.xz`. We automatically
# decompress the same name with `.xz` if that exists.
INPUT_PATH="$NLP_BENCH_ROOT"/benchmark/aux/oscar-10MB.jsonl
# Decompress if not already there.
if ! [ -f "$INPUT_PATH" ] && [ -f "$INPUT_PATH.xz" ]; then
    xz -dk "$INPUT_PATH.xz"
fi
OUTPUT_PREFIX="$NLP_BENCH_ROOT"/src/data/oscar

DONE_FILE1="$DATA_PATH".bin
DONE_FILE2="$DATA_PATH".idx

if [ -f $DONE_FILE1 ] && [ -f $DONE_FILE2 ]; then
    echo "$DONE_FILE1 and $DONE_FILE2 exists, exiting"
    exit 0
fi

export CUDA_VISIBLE_DEVICES=0
cd "$MEGATRON_LM_REPO" || exit 1

srun python ./tools/preprocess_data.py \
    --input "$INPUT_PATH" \
    --output-prefix "$OUTPUT_PREFIX" \
    --vocab-file "$VOCAB_FILE" \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file "$MERGE_FILE" \
    --append-eod \
    --workers 47

echo 'Done preprocessing data!'


