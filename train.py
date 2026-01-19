from module.probe import ProbeConfig, MultiLayerProbe, ExtraLayerProbe, LoraProbe
from module.data import load_dataset_processor, DataProcessor
from module.component import get_component_cls
import contextlib
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, PreTrainedTokenizer, TrainerCallback, TrainerState, TrainerControl
import logging
import argparse
from utils import compute_metrics_cls, compute_metrics_reg
import os
from typing import List
from datasets import concatenate_datasets
import sys

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

logger = logging.getLogger(__name__)

def tokenize_function(x, tokenizer: PreTrainedTokenizer):
    return tokenizer(x['prompt'], truncation=True, padding=True, max_length=4096)

from transformers.modeling_utils import unwrap_model

class ProbeSaveCallback(TrainerCallback):
    def _save_probe(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        model_to_save = unwrap_model(model)

        params_to_gather = [p for p in model_to_save.parameters() if p.requires_grad]

        gather_context = contextlib.nullcontext()
        if HAS_DEEPSPEED and params_to_gather:
            gather_context = deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0)

        with gather_context:
            if args.process_index == 0:
                if hasattr(model_to_save, "save_pretrained_probe"):
                    logger.info(f"[ProbeSaveCallback] Saving merged probe weights to {checkpoint_folder}...")
                    model_to_save.save_pretrained_probe(checkpoint_folder)
                else:
                    logger.warning("[ProbeSaveCallback] Model missing 'save_pretrained_probe' method.")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if args.save_steps > 0 and state.global_step % args.save_steps == 0:
            self._save_probe(args, state, control, model)
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self._save_probe(args, state, control, model)
        return control

def train(args):

    accelerator = None
    is_main_process = True

    if HAS_ACCELERATE:
        accelerator = Accelerator()
        is_main_process = accelerator.is_main_process
    else:
        if torch.distributed.is_initialized():
            is_main_process = (torch.distributed.get_rank() == 0)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if HAS_ACCELERATE and accelerator is not None:
        logger.info(f"Accelerator active. Distributed Status: {accelerator.distributed_type}")
    else:
        logger.info("Accelerate not found. Using native PyTorch settings.")

    from transformers import set_seed
    set_seed(args.seed)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        revision = args.revision,
        _attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    base_model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None or base_model.config.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id
    components = get_component_cls(args.component)()
    probe_cfg = ProbeConfig.from_json_file(args.probe_config)
    probe_types = {
        "multi_layer": MultiLayerProbe,
        "extra_layer": ExtraLayerProbe,
        "lora": LoraProbe,
    }
    probe_clz = probe_types.get(probe_cfg.probe_type, MultiLayerProbe)
    model = probe_clz(probe_cfg, base_model, components)

    os.makedirs(os.path.join(args.work_dir, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, 'experiments', args.run_name), exist_ok=True)

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

    # Name 'label' will trigger unknown issues in Trainer
    if 'label' in trainset.column_names:
        trainset = trainset.rename_column("label", "labels")
    if 'label' in testset.column_names:
        testset = testset.rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.work_dir, 'experiments', args.run_name, "model"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy='steps',
        eval_steps=100,
        save_steps=200,
        save_strategy='no', # we use custom saving via callback
        save_total_limit=2,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate': 0.01},
        bf16=True,
        bf16_full_eval=True,
        logging_dir=os.path.join(args.work_dir, 'experiments', args.run_name, 'logs'),
        logging_steps=1,
        report_to=['swanlab'],
        label_names=['labels'],
        gradient_accumulation_steps=args.grad_accum_steps,
        run_name=args.run_name,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_reg,
        callbacks=[ProbeSaveCallback],
    )

    trainer.train()

    import torch.distributed as dist
    if torch.distributed.is_initialized():
        torch.distributed.barrier() # wait for all processes to finish training
    
    logger.info("Saving final model...")
    os.makedirs(os.path.join(args.work_dir, 'experiments', args.run_name, "model"), exist_ok=True)
    output_dir = os.path.join(args.work_dir, 'experiments', args.run_name, "model")
    
    params_to_gather = [p for p in model.parameters() if p.requires_grad]
    gather_context = contextlib.nullcontext()
    if HAS_DEEPSPEED and params_to_gather:
        logger.info("Using DeepSpeed to gather final probe parameters for saving.")
        gather_context = deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0)

    with gather_context:
        if is_main_process:
            logger.info(f"Saving merged probe weights to {output_dir}...")
            model.save_pretrained(output_dir)
    logger.info("Training complete.")

    if dist.is_initialized():
        dist.barrier()
        logger.info(f"[Rank {dist.get_rank()}] Destroying process group...")
        dist.destroy_process_group()

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
    parser.add_argument('--run_name', type=str, default='probe-training', help='Run name for experiment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per device')
    # parser.add_argument('--limit', type=List[int], default=[10000, -1], help='Limit number of samples for train and test sets')
    args = parser.parse_args()
    train(args)
