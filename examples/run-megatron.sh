#!/bin/bash
# [Note]: Commands in this script should be run under Megatron-LM folder
set -ex

GPUS_PER_NODE=8
[ -z "$RANK" ] && RANK=0
[ -z "$WORLD_SIZE" ] && WORLD_SIZE=1
[ -z "$MASTER_ADDR" ] && MASTER_ADDR=127.0.0.1
[ -z "$MASTER_PORT" ] && MASTER_PORT=29500


ATTN_TYPE="flash"
MBS=2
RANDOM_INIT=0
GC=0 # 采用fullgc
SELECTIVE_GC=0 # 采用selective gc
GC_CNT=0
GBS=16
TORCH_PROFILE=0
TRAIN_ITERS=10000
SEQ_LEN=2048
TP_SIZE=1
PP_SIZE=1
VP_SIZE=1
SP=0 # 是否使用Sequence Parallelism，默认degree为TP_SIZE
LOG_INTERVAL=3
MODEL_NAME="llama-3"
MODEL_SIZE="8B"
TOKENIZER_CLASS="Llama3Tokenizer"
TOKENIZER_MODEL="tokenizer/llama-3/tokenizer.model"


# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    echo "$1" "$2"
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift
            ;;
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
        --vp)
            VP_SIZE="$2"
            shift
            ;;
        --sp)
            SP=1
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
        --selective-gc)
            SELECTIVE_GC=1
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
        --log-interval)
            LOG_INTERVAL="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# selective gc和gc不能同时为1
if [[ $GC -eq 1 && $SELECTIVE_GC -eq 1 ]]; then
    echo "Error: selective gc and full gc cannot be both 1"
    exit 1
fi

# 设置环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ $MODEL_NAME == "llama-3" ]]; then
    TOKENIZER_CLASS="Llama3Tokenizer"
    TOKENIZER_MODEL="tokenizer/${MODEL_NAME}/tokenizer.model"
else
    echo "Unknown model name: $MODEL_NAME"
    exit 1
fi

if [[ $MODEL_SIZE == "8B" ]]; then
    NUM_LAYERS=32
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=14336
    # could add more sizes here
elif [[ $MODEL_SIZE == "70B" ]]; then
    NUM_LAYERS=80
    HIDDEN_SIZE=8192
    NUM_ATTENTION_HEADS=64
    INTERMEDIATE_SIZE=28672
else
    echo "Unknown model size: $MODEL_SIZE"
    exit 1
fi


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

# 如果 num_layers不能整除 vp_size，则报错
if [[ $((NUM_LAYERS % VP_SIZE)) -ne 0 ]]; then
    echo "Error: num_layers should be divisible by vp_size"
    exit 1
fi

# num layers / pp size 必须整除 vp size
if [[ $((NUM_LAYERS / PP_SIZE % VP_SIZE)) -ne 0 ]]; then
    echo "Error: num_layers / pp_size should be divisible by vp_size"
    exit 1
fi




MODEL_ARGS=()
# 设置随机初始化参数
if [[ "$RANDOM_INIT" -eq 1 ]]; then
    MODEL_ARGS+=(
        --num-layers ${NUM_LAYERS}
        --hidden-size ${HIDDEN_SIZE}
        --num-attention-heads ${NUM_ATTENTION_HEADS}
        --seq-length ${SEQ_LEN} 
        --max-position-embeddings 8192
        --num-query-groups 8
        --group-query-attention
        --position-embedding-type rope
        --use-rotary-position-embeddings
        --disable-bias-linear
        --ffn-hidden-size ${INTERMEDIATE_SIZE}
        --swiglu
        --bf16
    )
else
    MODEL_ARGS+=(
        --seq-length ${SEQ_LEN}
        --bf16
        --use-checkpoint-args
        --load ../models/${MODEL_NAME}-${MODEL_SIZE}-megatron-tp${TP_SIZE}-bf161
    )
fi

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --node_rank $RANK
    --nnodes $WORLD_SIZE
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
    
    
)

if [[ "$GC" -eq 1 ]]; then
    TRAINING_ARGS+=(
        --recompute-granularity full
        --recompute-method block
        --recompute-num-layers ${GC_CNT}
    )
fi

if [[ "$SELECTIVE_GC" -eq 1 ]]; then
    TRAINING_ARGS+=(
        --recompute-activations
    )
fi


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP_SIZE}
	--pipeline-model-parallel-size ${PP_SIZE}
)
EXTRA_ARGS=(
    --tokenizer-type ${TOKENIZER_CLASS}
    --tokenizer-model ${TOKENIZER_MODEL}
    --exit-on-missing-checkpoint
    --no-load-optim
    --no-load-rng
    --untie-embeddings-and-output-weights
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --overlap-grad-reduce
    --overlap-param-gather
    # --tp-comm-overlap
    --use-distributed-optimizer
    --normalization RMSNorm
    --transformer-impl transformer_engine
    --log-interval ${LOG_INTERVAL}
)

# 如果vp >= 2，则在extra_args里加上vp_size
if [[ "$VP_SIZE" -ge 2 ]]; then
    EXTRA_ARGS+=(
        --num-layers-per-virtual-pipeline-stage ${VP_SIZE}
    )
fi

if [[ "$TORCH_PROFILE" -eq 1 ]]; then
    EXTRA_ARGS+=(
        --torch-profile
    )
fi

if [[ "$SP" -eq 1 ]]; then
    EXTRA_ARGS+=(
        --sequence-parallel
        # --tp-comm-overlap
    )
fi

DATA_ARGS=(
    --data-path ./data/_text_document
)

LOG_DIR="logs"

# 生成日志文件名称
log_filename="${LOG_DIR}/${MODEL_NAME}-${MODEL_SIZE}_dev_attn-${ATTN_TYPE}_mbs-${MBS}_gbs-${GBS}_gc${GC}_gc_cnt-${GC_CNT}_random_init${RANDOM_INIT}_tp-${TP_SIZE}_pp-${PP_SIZE}.log"
mkdir -p ${LOG_DIR}

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${DATA_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EXTRA_ARGS[@]} 2>&1 | tee "$log_filename"
