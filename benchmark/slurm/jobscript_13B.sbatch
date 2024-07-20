#!/bin/bash
#SBATCH --job-name=train_13Bmodel
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=48           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time=00:30:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account=exalab
#SBATCH --partition=dc-gpu

set -x -e
# srun doesnot inherit cpus-per-task from sbatch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "START TIME: $(date)"
echo "Submitted batch job $SLURM_JOBID"

if [ "x$NLP_BENCH_ROOT" = "x" ]; then
    echo "NLP_BENCH_ROOT is not set. Please set it to the root directory of nlp-benchmark" >&2
    exit 1
fi

source $NLP_BENCH_ROOT/src/variables.bash
source "$NLP_BENCH_ROOT"/benchmark/env/activate.bash || exit 1

# setting extra output paths
DATA_OUTPUT_PATH="$ROOT_OUTPUT_DIR"/13B_model/"$SLURM_JOB_ID"
CHECKPOINT_PATH=$DATA_OUTPUT_PATH/checkpoints
TENSORBOARD_PATH=$DATA_OUTPUT_PATH/tensorboard
LOGS_PATH=$DATA_OUTPUT_PATH/logs
mkdir -p $LOGS_PATH

cd "$MEGATRON_LM_REPO"
rm -f megatron/fused_kernels/build/lock
CLEAN_PREV_JIT_BUILD=0
((CLEAN_PREV_JIT_BUILD)) && rm -rf megatron/fused_kernels/{build,__pycache__}

# so processes know who to talk to
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
MASTER_PORT=6000

GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES  
TP_SIZE=2    # preferred <= to the size of a single node
PP_SIZE=4    # must be a divisor of nlayers
MICRO_BATCH_SIZE=1
GAS=256
#DP_SIZE=$NNODES*$GPUS_PER_NODE/($PP_SIZE*$TP_SIZE) # will get derived automatically by trainer
#GLOBAL_BATCH_SIZE has to be divisible by DP_SIZE*MICRO_BATCH_SIZE*GAS thereby calculating it 
GLOBAL_BATCH_SIZE=$(((NNODES * GPUS_PER_NODE / (PP_SIZE * TP_SIZE)) * MICRO_BATCH_SIZE * GAS))

NLAYERS=40    # NLAYERS must be a multiple of PP_SIZE here
NHIDDEN=5120
NHEADS=32
FFN_HIDDEN_SIZE=20480
SEQ_LEN=2048
SAVE_INTERVAL=1500

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 216_320 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --use-distributed-optimizer \
    "
  
GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples 300_000_000 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --tokenizer-type GPT2BPETokenizer \
    --loss-scale-window 500 \
    --hysteresis 2 \
    --min-loss-scale 1.0 \
    --initial-loss-scale 4096 \
    --fp16 \
    --seed 42 \
    --position-embedding-type rope \
    --use-flash-attn \
    --sequence-parallel \
    --recompute-activations \
    --recompute-granularity selective \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 1000 \
    --eval-iters 5 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    $(pwd)/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --split 7,3,1 \
    --distributed-backend nccl \
    "

if [ "$LOAD_CHECKPOINTS" = true ] ; then
    export CMD="$CMD\
        --load $CHECKPOINT_PATH \
        "
fi

echo $CMD

# Necessary for some Megatron-LM settings. We set it all the time just
# to be safe.
export CUDA_DEVICE_MAX_CONNECTIONS=1
# With CUDA_LAUNCH_BLOCKING=1, need to load NCCL/2.12.7-1-CUDA-11.5 
# since NCCL/2.14.3-1-CUDA-11.5 and later versions cause internal streams clashes
# export CUDA_LAUNCH_BLOCKING=1
# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# handle timeouts
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
# setting IB for out of band communication
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
# NCCL and Torch debug
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json


# for using pre-installed kernels
export DS_BUILD_OPS=1

clear; srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD' 2>&1 | tee -a "$LOGS_PATH"/main_log.txt

echo "END TIME: $(date)"


