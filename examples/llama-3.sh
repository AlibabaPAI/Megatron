# ========= seq_len=2048 mbs=1,2 =========
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --gc --gc-cnt 10

# ========= seq_len=8192 mbs=2 profile =========
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --gc --gc-cnt 10 --torch-profile --train-iters 100

# ========= seq_len=8192 mbs=1,2 =========
bash examples/run-megatron.sh --random-init --mbs 1 --gbs 8 --attn-type flash --seq-len 8192 --gc --gc-cnt 25

# ========= seq_len=8192 mbs=2 tp=2 fsdp=4 =========
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 8 --attn-type flash --seq-len 8192 --tp 2 --gc --gc-cnt 19

# ========= seq_len=8192 mbs=2 pp=2 fsdp=4 =========
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 16 --attn-type flash --seq-len 8192 --pp 2 --gc --gc-cnt 9

# ========= seq_len=8192 mbs=2 GA =========
bash examples/run-megatron.sh --random-init --mbs 2 --gbs 256 --attn-type flash --seq-len 8192 --pp 2 --gc --gc-cnt 9 --log-interval 5

# ========= seq_len=8192 mbs=1 GA =========
bash examples/run-megatron.sh --random-init --mbs 1 --gbs 256 --attn-type flash --seq-len 8192 --pp 2 --log-interval 5 # OPTIMAL
