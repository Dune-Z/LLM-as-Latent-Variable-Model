CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ACCELERATE_LOG_LEVEL=info
MODEL_NAME=google/gemma-2-2b-it
OUTPUT_PATH=outputs/sft_gemma-2-2b-it
HF_TOKEN=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

if [ ! -d "/mnt/hdfs/yifei/outputs/${OUTPUT_PATH}" ]; then
  mkdir -p "/mnt/hdfs/yifei/outputs/${OUTPUT_PATH}"
fi

python sft_src/sample.py\
    output_file="$OUTPUT_PATH/sample_output.jsonl"\
    model_name_or_path=$MODEL_PATH\
    sample_batch_size=1024\
    dataset_kwargs.METAMATH.problem_size=-1

accelerate launch --num_processes 8 --main_process_port 29501 sft_src/baseline_sft.py \
    --deepspeed configs/sft_deepspeed.json\
    --model_name $MODEL_NAME\
    --output_dir $OUTPUT_PATH\
    --push_to_hub False\
    --attn_implementation eager\
    --train_set_path "${OUTPUT_PATH}/sample_output.jsonl"\
    --num_train_epochs 10\
    --per_device_train_batch_size 16\
    --per_device_eval_batch_size 16\
    --gradient_accumulation_steps 1

# for each checkpoint, evaluate the model
for checkpoint in $(ls $OUTPUT_PATH/checkpoint* | sort -V); do
    echo "Evaluating checkpoint $checkpoint"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lm_eval \
        --model hf \
        --model_args pretrained=$checkpoint \
        --task gsm8k,hendrycks_math \
        --batch_size auto \
        --output_path "${checkpoint}-accuracy" \
        --log_samples
done