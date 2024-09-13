#!/bin/bash

# preprocess data for llama3-8b
python tools/preprocess_data.py \
    --input data/wikitext-2-raw-v1.json \
    --output-prefix data/ \
    --tokenizer-type Llama3Tokenizer \
    --tokenizer-model ./tokenizer/llama-3/tokenizer.model \
    --workers 4 \
    --append-eod

