#!/bin/bash
set -e
export MASTER_PORT=12355

OUTPUT_DIR="outputs/tinyllama-1.1b-math-restem"
ROUND=1
NUM_ROUNDS=5
BASE_MODEL_PATH="TinyLlama/TinyLlama_v1.1_math_code"
CURRENT_SAMPLE_MODEL_PATH="TinyLlama/TinyLlama_v1.1_math_code"
CUDA_DEVICE=3

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
            sample_batch_size=128\
            dataset_kwargs.GSM8K.problem_size=-1
    else
        echo "Sample file already exists for round $i. Skip sampling."
    fi

    BEST_ACC=0.0
    INNER_ROUND=1
    CURRENT_MODEL_PATH=$BASE_MODEL_PATH
    TEMP_MODEL_PATH=$CURRENT_MODEL_PATH

    while true; do
        echo "Evaluating model for round $i, inner round $INNER_ROUND"
        ACC_DIR="${ROUND_DIR}/accuracy-${INNER_ROUND}"
        # CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/evaluate.py\
        #     model_name_or_path=$CURRENT_MODEL_PATH\
        #     output_file="$ACC_FILE"
        # if the file $ACC_DIR/*/*.json exists, then skip the evaluation
        if [ ! -f $ACC_DIR/*/*.json ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE lm_eval \
                --model hf \
                --model_args pretrained=$CURRENT_MODEL_PATH \
                --task gsm8k \
                --batch_size auto \
                --output_path $ACC_DIR \
                --log_samples
        else
            echo "Accuracy file already exists for round $i. Skip evaluation."
        fi
        CURRENT_ACC=$(jq '.results.gsm8k["exact_match,flexible-extract"]' $ACC_DIR/*/*.json)
        echo "Curr accuracy: $CURRENT_ACC"
        echo "Best accuracy: $BEST_ACC"

        if (( $(echo "$CURRENT_ACC >= $BEST_ACC" | bc -l) )); then
            echo "Accuracy improved from $BEST_ACC to $CURRENT_ACC. Train new model."
            BEST_ACC=$CURRENT_ACC

            TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python src/train.py\
                model_name_or_path=$CURRENT_MODEL_PATH\
                dataset_path="$SAMPLE_FILE"\
                trainer.output_dir="$ROUND_DIR"\
                trainer.deepspeed="configs/gemma-2b-it-deepspeed_config.json"

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