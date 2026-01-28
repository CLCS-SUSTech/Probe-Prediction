#!/bin/bash

model_name="allenai/Olmo-3-7B-Think"
run_name="olmo3-probe-think-step200"
probe_path="./work_dir/experiments/olmo3-probe-think-step200/model/"
component="olmo3"
datasets=("aime" "gpqa" "human_eval" "math" "bbh" "mmlu" "gsm8k" "mbpp")
work_dir="./work_dir"
revision="step_0200"

python evaluate_probe.py \
    --run_name ${run_name} \
    --model_name ${model_name} \
    --revision ${revision} \
    --component ${component} \
    --datasets ${datasets[@]} \
    --work_dir ${work_dir} \
    --skip_evaluation \
    --batch_size 4

# --skip_evaluation: Skip the evaluation if the results are already computed and saved.