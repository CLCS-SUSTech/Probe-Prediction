from module.probe import ProbeConfig, MultiLayerProbe
from module.data import load_dataset_processor, DataProcessor
from module.component import get_component_cls
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, PreTrainedTokenizer
import logging
import argparse
from utils import compute_metrics_cls
import os
from typing import List
from datasets import concatenate_datasets

logger = logging.getLogger(__name__)

def tokenize_function(x, tokenizer: PreTrainedTokenizer):
    return tokenizer(x['prompt'], truncation=True, padding=True, max_length=4096)

def train(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        revision = args.revision,
        _attn_implementation=args.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    components = get_component_cls(args.component)()
    probe_cfg = ProbeConfig.from_json_file(args.probe_config)
    model = MultiLayerProbe(probe_cfg, base_model, components)

    os.makedirs(os.path.join(args.work_dir, "datasets"), exist_ok=True)

    trainsets = []
    testsets = []

    for dataset in args.datasets:
        logger.info(f"Loading dataset processor for {dataset}")
        stage3_dir = os.path.join(args.work_dir, "datasets", args.model_name.replace("/", "--"), args.revision, dataset, "stage3")
        dataset_processor = load_dataset_processor(
            dataset,
            stage3_dir,
        )
        dataset_processor.load()
        trainset = dataset_processor.get('train').shuffle(seed=42)
        testset = dataset_processor.get('test')
        if args.limit[0] > 0:
            trainset.select(DataProcessor.label_balance(
                trainset,
                label_column='label',
                positive_label_value=1.0,
                limit=min(args.limit[0], len(trainset)),
            ))
        if args.limit[1] > 0:
            testset.select(DataProcessor.label_balance(
                testset,
                label_column='label',
                positive_label_value=1.0,
                limit=min(args.limit[1], len(testset)),
            ))
        trainset = trainset.remove_columns([col for col in trainset.column_names if col not in ['prompt', 'label']])
        testset = testset.remove_columns([col for col in testset.column_names if col not in ['prompt', 'label']])
        trainsets.append(trainset)
        testsets.append(testset)
        
    if len(trainsets) == 1:
        trainset = trainsets[0]
        testset = testsets[0]
    else:
        trainset = concatenate_datasets(trainsets)
        testset = concatenate_datasets(testsets)

    trainset = trainset.shuffle(seed=42)

    trainset = trainset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    testset = testset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy='steps',
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate': 0.01},
        bf16=True,
        bf16_full_eval=True,
        logging_dir='./logs',
        logging_steps=1,
        report_to=['swanlab'],
        label_names=['label'],
        gradient_accumulation_steps=args.grad_accum_steps,
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_cls,
    )

    trainer.train()
    os.makedirs(os.path.join(args.work_dir, 'experiments', args.experiment_name, "model"), exist_ok=True)
    output_dir = os.path.join(args.work_dir, 'experiments', args.experiment_name, "model")

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Base model name or path')
    parser.add_argument('--revision', type=str, default='main', help='Model revision')
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", help='Attention implementation to use')
    parser.add_argument('--component', type=str, required=True, help='Component name')
    parser.add_argument('--probe_config', type=str, required=True, help='Path to probe\'s config file')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to train.')
    parser.add_argument('--work_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--grad_accum_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--run_name', type=str, default='probe-training', help='Run name for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--limit', type=List[int], default=[10000, -1], help='Limit number of samples for train and test sets')
    parser.add_argument('--experiment_name', type=str, default='probe_experiment', help='Experiment name for logging')
    args = parser.parse_args()
    train(args)
