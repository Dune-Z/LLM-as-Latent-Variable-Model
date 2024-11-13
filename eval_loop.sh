for checkpoint in 642 538 431 323 215 107; do
    lm_eval --model hf \
    --model_args pretrained=/home/cyc2202/LLM-as-Latent-Variable-Model/outputs/restem_6epoch/checkpoint-${checkpoint} \
    --tasks gsm8k \
    --device cuda:4 \
    --batch_size 8 \
    --output_path ./Logs/restem_sample_6epoch/checkpoint_${checkpoint} \
    --log_samples
done