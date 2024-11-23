MODEL_PATH=outputs/sft_gemma-2-2b-it/checkpoint-4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH \
    --task gsm8k \
    --batch_size auto \
    --output_path "${MODEL_PATH}-accuracy" \
    --log_samples