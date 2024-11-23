CUDA_VISIBLE_DEVICES=0,1,2,3
ACCELERATE_LOG_LEVEL=info
MODEL_NAME=google/gemma-2-2b-it
OUTPUT_PATH=outputs/sft_gemma-2-2b-it
accelerate launch --num_processes 4 --main_process_port 29501 sft_src/baseline_sft.py \
    --deepspeed configs/sft_deepspeed.json\
    --model_name $MODEL_NAME\
    --output_dir $OUTPUT_PATH\
    --push_to_hub False\
    --attn_implementation eager\
    --train_set_path "${OUTPUT_PATH}/sample_output.jsonl"\
    --num_train_epochs 10
