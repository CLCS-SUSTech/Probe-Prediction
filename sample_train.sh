#!/bin/bash

model_name="allenai/Olmo-3-7B-Think"
datasets=("aime" "gpqa" "human_eval" "math" "bbh" "mmlu" "gsm8k" "mbpp")
zero_config="./configurations/accelerate/zero1.json"
probe_config="./configurations/probe/olmo3_submodel_probe.json"
revision="step_0200"
work_dir="./work_dir"

accelerate launch --config_file ${zero_config} train.py \
    --model_name ${model_name} \
    --revision ${revision} \
    --attn_implementation "flash_attention_2" \
    --component "olmo3" \
    --probe_config ${probe_config} \
    --datasets ${datasets[@]} \
    --work_dir ${work_dir} \
    --grad_accum_steps 8 \
    --run_name "olmo3-probe-think-step200" \
    --seed 42