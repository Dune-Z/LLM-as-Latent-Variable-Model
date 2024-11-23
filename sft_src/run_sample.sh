MODEL_PATH=google/gemma-2-2b-it
# MODEL_PATH=outputs/sft_gemma-2-2b-it/checkpoint-27
OUTPUT_PATH=outputs/sft_gemma-2-2b-it/round-1

CUDA_VISIBLE_DEVICES=0,1,2,3 python sft_src/sample.py\
    output_file="$OUTPUT_PATH/sample_output.jsonl"\
    model_name_or_path=$MODEL_PATH\
    sample_batch_size=128\
    dataset_kwargs.METAMATH.problem_size=-1