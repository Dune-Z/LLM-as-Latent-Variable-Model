#!/bin/bash
set -e

OUTPUT_DIR="outputs/llama-3-8b-instruct-restem"
ROUND=1
NUM_ROUNDS=5

for ((i=1; i<=$NUM_ROUNDS; i++))
do
    if [ $i -eq 1 ]; then
        MODEL_NAME_OR_PATH="unsloth/llama-3-8b-instruct"
    else
        MODEL_NAME_OR_PATH="${OUTPUT_DIR}/round-$((i-1))/"
    fi
    ROUND_DIR="${OUTPUT_DIR}/round-${ROUND}"
    mkdir -p $ROUND_DIR

    CUDA_VISIBLE_DEVICES=0 python src/sample.py\
        output_file="${ROUND_DIR}/sample_output.jsonl"\
        model_name_or_path=$MODEL_NAME_OR_PATH
    
    CUDA_VISIBLE_DEVICES=0 python src/train.py\
        dataset_path="${ROUND_DIR}/sample_output.jsonl"\
        output_dir="${ROUND_DIR}/round-${i}"
done