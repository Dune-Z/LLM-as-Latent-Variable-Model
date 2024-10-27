#!/bin/bash
set -e

OUTPUT_DIR="outputs/gemma-2b-it-restem"
ROUND=1
NUM_ROUNDS=5

for ((i=1; i<=$NUM_ROUNDS; i++))
do
    if [ $i -eq 1 ]; then
        MODEL_NAME_OR_PATH="google/gemma-2b-it"
    else
        MODEL_NAME_OR_PATH="${OUTPUT_DIR}/round-$((i-1))/"
    fi
    ROUND_DIR="${OUTPUT_DIR}/round-${ROUND}"
    SAMPLE_FILE="${ROUND_DIR}/sample_output.jsonl"
    ACC_FILE="${ROUND_DIR}/accuracy.txt"
    mkdir -p $ROUND_DIR

    if [ ! -f "$SAMPLE_FILE" ]; then
        echo "Sampling data for round $i"
        CUDA_VISIBLE_DEVICES=0 python src/sample.py\
            output_file="$SAMPLE_FILE"\
            model_name_or_path=$MODEL_NAME_OR_PATH
    else
        echo "Sample file already exists for round $i. Skipping sampling."
    fi

    CUDA_VISIBLE_DEVICES=0 python src/evaluate.py\
        model_name_or_path=$MODEL_NAME_OR_PATH\
        output_file="$ACC_FILE"
   
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python src/train.py\
        dataset_path="$SAMPLE_FILE"\
        trainer.output_dir="$ROUND_DIR"
done