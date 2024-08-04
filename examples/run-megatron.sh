#!/bin/bash
# [Note]: Commands in this script should be run under Megatron-LM folder
set -ex

GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

ATTN_TYPE="flash"
MBS=2
RANDOM_INIT=0
GC=0
GC_CNT=0
GBS=16
TORCH_PROFILE=0
TRAIN_ITERS=10000
SEQ_LEN=2048
TP_SIZE=1
PP_SIZE=1   

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    echo "$1" "$2"
    case $1 in
        --attn-type)
            ATTN_TYPE="$2"
            shift
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift
            ;;
        --mbs)
            MBS="$2"
            shift
            ;;
        --tp)
            TP_SIZE="$2"
            shift
            ;;
        --pp)
            PP_SIZE="$2"
            shift
            ;;
        --gbs)
            GBS="$2"
            shift
            ;;
        --random-init)
            RANDOM_INIT=1
            ;;
        --gc)
            GC=1
            ;;
        --gc-cnt)
            GC_CNT="$2"
            shift
            ;;
        --torch-profile)
            TORCH_PROFILE=1
            ;;
        --train-iters)
            TRAIN_ITERS="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

# llama3-8b

if [[ "$ATTN_TYPE" == "flash" ]]; then
    export NVTE_FLASH_ATTN=1
    export NVTE_FUSED_ATTN=0
    export NVTE_UNFUSED_ATTN=0
elif [[ "$ATTN_TYPE" == "fused" ]]; then
    export NVTE_FLASH_ATTN=0
    export NVTE_FUSED_ATTN=1
    export NVTE_UNFUSED_ATTN=0
elif [[ "$ATTN_TYPE" == "unfused" ]]; then
    export NVTE_FLASH_ATTN=0
    export NVTE_FUSED_ATTN=0
    export NVTE_UNFUSED_ATTN=1
else
    echo "Unknown attention type: $ATTN_TYPE"
    exit 1
fi

LLAMA3_MODEL_ARGS=()
# 设置随机初始化参数
if [[ "$RANDOM_INIT" -eq 1 ]]; then
    LLAMA3_MODEL_ARGS+=(
        --num-layers 32 
        --hidden-size 4096 
        --num-attention-heads 32 
        --seq-length ${SEQ_LEN} 
        --max-position-embeddings 8192
        --num-query-groups 8
        --group-query-attention
        --position-embedding-type rope
        --use-rotary-position-embeddings
        --disable-bias-linear
        --ffn-hidden-size 14336
        --swiglu
        --bf16
    )
else
    LLAMA3_MODEL_ARGS+=(
        --seq-length ${SEQ_LEN}
        --bf16
        --use-checkpoint-args
        --load ../models/llama3-8b-megatron-tp${TP_SIZE}-bf161
    )
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)
TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters 10000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 100000
    --train-iters ${TRAIN_ITERS}
    # --recompute-activations
    # --recompute-method uniform
    
    
)

if [[ "$GC" -eq 1 ]]; then
    TRAINING_ARGS+=(
        --recompute-granularity full
        --recompute-method block
        --recompute-num-layers ${GC_CNT}
    )
fi

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP_SIZE}
	--pipeline-model-parallel-size ${PP_SIZE}
)
EXTRA_ARGS=(
    --tokenizer-type Llama3Tokenizer
    --tokenizer-model ../models/llama3-8b/tokenizer.model
    --exit-on-missing-checkpoint
    --no-load-optim
    --no-load-rng
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --overlap-grad-reduce
    --overlap-param-gather
    --use-distributed-optimizer
    --normalization RMSNorm
    --transformer-impl transformer_engine
    --log-interval 10
)

if [[ "$TORCH_PROFILE" -eq 1 ]]; then
    EXTRA_ARGS+=(
        --torch-profile
    )
fi


DATA_ARGS=(
    --data-path ../data/_text_document
)

LOG_DIR="logs"

# 生成日志文件名称
log_filename="${LOG_DIR}/dev_attn-${ATTN_TYPE}_mbs-${MBS}_gbs-${GBS}_gc${GC}_gc_cnt-${GC_CNT}_random_init${RANDOM_INIT}_tp-${TP_SIZE}_pp-${PP_SIZE}.log"
mkdir -p ${LOG_DIR}

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${DATA_ARGS[@]} \
    ${LLAMA3_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EXTRA_ARGS[@]} 2>&1 | tee "$log_filename"
