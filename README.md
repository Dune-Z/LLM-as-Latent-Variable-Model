# LLM-as-Latent-Variable-Model

Only gemma-2-2b-it.sh and tinyllam-1.1b-math.sh are completely tested.
Run the following commands to launch the experiments:

```bash
bash scripts/gemma-2-2b-it.sh
```

The outputs will be saved in the `outputs` directory named after the script that was run.
It contains subdirectories for each RestEM round, in each round outputs, the checkpoint of each epoch is saved. You need to run the following command to convert the checkpoint from Zero format to float32:

```bash
python outputs/[SCRIPT_NAME]/round-[NUMBER]/checkpoint-[STEPS]/zero_to_fp32.py outputs/[SCRIPT_NAME]/round-[NUMBER]/checkpoint-[STEPS] outputs/[SCRIPT_NAME]/round-[NUMBER]/checkpoint-[STEPS]/
```

Then call evaluation by the following command:

```bash
CUDA_VISIBLE_DEVICES=3 lm_eval --model vllm --model_args pretrained=outputs/[SCRIPT_NAME]/round-[NUMBER]/checkpoint-[STEPS] --task gsm8k --batch_size auto --output_path outputs/[SCRIPT_NAME]/round-[NUMBER]/checkpoint-[STEPS] -accuracy --log_samples
```