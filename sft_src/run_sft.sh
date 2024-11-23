CUDA_VISIBLE_DEVICES=0,1,2,3
ACCELERATE_LOG_LEVEL=info
accelerate launch --num_processes 4 --main_process_port 29501 sft_src/baseline_sft.py \
    --deepspeed configs/sft_deepspeed.json\
    --model_name google/gemma-2-2b-it\
    --output_dir outputs/sft_gemma-2-2b-it/round-2\
    --push_to_hub False\
    --attn_implementation eager\
    --train_set_path outputs/sample_output.jsonl\
    --num_train_epochs 8