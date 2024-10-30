#!/bin/bash
set -e

OUTPUT_DIR="outputs/llama-3-8b-it-restem"
ROUND=1
NUM_ROUNDS=5
BASE_MODEL_PATH="unsloth/llama-3-8b-instruct"
CURRENT_SAMPLE_MODEL_PATH="unsloth/llama-3-8b-instruct"
CUDA_DEVICE=0

for ((i=1; i<=$NUM_ROUNDS; i++))
do
    ROUND_DIR="${OUTPUT_DIR}/round-${ROUND}"
    SAMPLE_FILE="${ROUND_DIR}/sample_output.jsonl"
    BEST_MODEL_PATH="${ROUND_DIR}/best_model"
    mkdir -p $ROUND_DIR

    if [ ! -f "$SAMPLE_FILE" ]; then
        echo "Sampling data for round $i"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/sample.py\
            output_file="$SAMPLE_FILE"\
            model_name_or_path=$CURRENT_SAMPLE_MODEL_PATH\
            sample_batch_size=32\
            dataset_kwargs.MATH.problem_size=-1
    else
        echo "Sample file already exists for round $i. Skip sampling."
    fi

    BEST_ACC=0.0
    INNER_ROUND=1
    CURRENT_MODEL_PATH=$BASE_MODEL_PATH
    TEMP_MODEL_PATH=$CURRENT_MODEL_PATH

    while true; do
        echo "Evaluating model for round $i"
        ACC_FILE="${ROUND_DIR}/accuracy-${INNER_ROUND}.txt"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/evaluate.py\
            model_name_or_path=$CURRENT_MODEL_PATH\
            output_file="$ACC_FILE"

        CURRENT_ACC=$(cat $ACC_FILE)
        echo "Curr accuracy: $CURRENT_ACC"
        echo "Best accuracy: $BEST_ACC"

        if (( $(echo "$CURRENT_ACC >= $BEST_ACC" | bc -l) )); then
            echo "Accuracy improved from $BEST_ACC to $CURRENT_ACC. Train new model."
            BEST_ACC=$CURRENT_ACC

            TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/train.py\
                model_name_or_path=$CURRENT_MODEL_PATH\
                dataset_path="$SAMPLE_FILE"\
                trainer.output_dir="$ROUND_DIR"\
                trainer.deepspeed="configs/llama-3-8b-it-deepspeed_config.json"

            TEMP_MODEL_PATH=$CURRENT_MODEL_PATH
            mv $ROUND_DIR/checkpoint-* "${ROUND_DIR}/checkpoint-inner-round-${INNER_ROUND}"
            CURRENT_MODEL_PATH="${ROUND_DIR}/checkpoint-inner-round-${INNER_ROUND}"

            python $CURRENT_MODEL_PATH/zero_to_fp32.py\
                $CURRENT_MODEL_PATH $CURRENT_MODEL_PATH

            INNER_ROUND=$((INNER_ROUND+1))
        else
            echo "Accuracy did not improve. Stop training."
            mv $TEMP_MODEL_PATH "$BEST_MODEL_PATH"
            break
        fi
    CURRENT_SAMPLE_MODEL_PATH=$BEST_MODEL_PATH
    done
done