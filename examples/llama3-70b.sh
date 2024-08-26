# fsdp=32
bash examples/run-megatron.sh --random-init --mbs 1 --gbs 32 --attn-type flash --model-size 70B --seq-len 2048 --gc --gc-cnt 80
