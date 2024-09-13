# ========= seq_len=8192 mbs=1 pp=4 tp=8 gc_cnt=80 GA =========
bash examples/run-megatron.sh --random-init --mbs 1 --gbs 256 --attn-type flash --model-size 70B --seq-len 8192 --tp 8 --pp 4 --vp 5 --gc --gc-cnt 80

# ========= seq_len=8192 mbs=1 pp=4 tp=8 gc_cnt=1 GA =========
bash examples/run-megatron.sh --random-init --mbs 1 --gbs 256 --attn-type flash --model-size 70B --seq-len 8192 --tp 8 --pp 4 --vp 5 --sp --gc --gc-cnt 1 # OPTIMAL

# ========= seq_len=8192 mbs=1 pp=4 tp=8 gc_cnt=1 GA =========
bash examples/run-megatron.sh --random-init --mbs 1 --gbs 64 --attn-type flash --model-size 70B --seq-len 8192 --tp 8 --pp 4 --vp 5 --sp --gc --gc-cnt 1

