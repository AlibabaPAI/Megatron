
# =========seq_len 2048========
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --gc --gc-cnt 10

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type fused --gc --gc-cnt 10

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type unfused --gc --gc-cnt 26

bash examples/run-megatron.sh --random-init --mbs 1 --gbs 8 --attn-type unfused --gc --gc-cnt 16

# ==========profile==========
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --gc --gc-cnt 10 --torch-profile --train-iters 100


# =========seq_len 8192========
bash examples/run-megatron.sh --random-init --mbs 1 --gbs 8 --attn-type flash --seq-len 8192 --gc --gc-cnt 32 
# --torch-profile --train-iters 30

bash examples/run-megatron.sh --random-init --mbs 1 --gbs 8 --attn-type fused --seq-len 8192 --gc --gc-cnt 25

bash examples/run-megatron.sh --random-init --mbs 1 --gbs 8 --attn-type unfused --seq-len 8192 --gc --gc-cnt 32


bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type unfused --seq-len 8192 --gc --gc-cnt 32

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type fused --seq-len 8192 --gc --gc-cnt 32

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --seq-len 8192 --gc --gc-cnt 32 --torch-profile 


# TP = 2 FSDP = 4
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 8 --attn-type flash --seq-len 8192 --tp 2 --gc --gc-cnt 19

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 8 --attn-type fused --seq-len 8192 --tp 2 --gc --gc-cnt 19

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 8 --attn-type unfused --seq-len 8192 --tp 2 --gc --gc-cnt 31



# =========== Full GC + FA largest mbs ==============
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --seq-len 8192 --gc --gc-cnt 32



# PP = 2 FSDP = 4
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --seq-len 8192 --pp 2 --gc --gc-cnt 9

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type fused --seq-len 8192 --pp 2 --gc --gc-cnt 9

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type unfused --seq-len 8192 --pp 2 --gc --gc-cnt 32


## larger batch size
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 64 --attn-type flash --seq-len 8192 --pp 2 --gc --gc-cnt 9 --log-interval 5

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 128 --attn-type flash --seq-len 8192 --pp 2 --gc --gc-cnt 9 --log-interval 5

bash examples/run-megatron.sh --random-init --mbs 2 --gbs 256 --attn-type flash --seq-len 8192 --pp 2 --gc --gc-cnt 9 --log-interval 5


bash examples/run-megatron.sh --random-init --mbs 1 --gbs 64 --attn-type flash --seq-len 8192 --pp 2 --log-interval 5

bash examples/run-megatron.sh --random-init --mbs 1 --gbs 128 --attn-type flash --seq-len 8192 --pp 2 --log-interval 5

bash examples/run-megatron.sh --random-init --mbs 1 --gbs 256 --attn-type flash --seq-len 8192 --pp 2 --log-interval 5







