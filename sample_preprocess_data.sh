#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1

datasets=("aime" "gpqa" "human_eval" "math" "bbh" "mmlu" "gsm8k" "mbpp")
work_dir="./work_dir"
mkdir -p $work_dir
model_name="allenai/Olmo-3-7B-Think"
revision="step_0200"

python preprocess_data.py --work_dir $work_dir \
    --datasets ${datasets[@]} \
    --model_name $model_name \
    --revision $revision \
    --tensor_parallel_size 8 \
    --batch_size 16 \
    --config "./configurations/sampling/olmo3_think_sample_params.json"
