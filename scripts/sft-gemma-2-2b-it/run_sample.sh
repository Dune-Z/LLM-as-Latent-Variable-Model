MODEL_PATH=google/gemma-2-2b-it
OUTPUT_PATH=outputs/sft_gemma-2-2b-it
CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun --nproc_per_node=4 sft_src/sample.py\
python sft_src/sample.py\
    output_file="$OUTPUT_PATH/sample_output.jsonl"\
    model_name_or_path=$MODEL_PATH\
    sample_batch_size=128\
    dataset_kwargs.METAMATH.problem_size=512\
    dataset_kwargs.METAMATH.sample_size=1\
    world_size=4
