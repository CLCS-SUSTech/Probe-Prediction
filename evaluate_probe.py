import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from tqdm import tqdm
import argparse
import os
import json
import numpy as np
import sys
from typing import List
from scipy import stats
from sklearn.metrics import r2_score

from module.data import load_dataset_processor
from module.component import get_component_cls
from module.probe import ProbeWrapper

def tokenize_function(x, tokenizer, max_length=4096):
    return tokenizer(x['prompt'], truncation=True, padding="max_length", max_length=max_length)

def concordance_correlation_coefficient(y_true, y_pred):
    if len(y_true) <= 1:
        return np.nan

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    rho = np.corrcoef(y_true, y_pred)[0, 1]

    numerator = 2 * rho * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    if denominator == 0:
        return 0.0

    return numerator / denominator

def c_index_score(y_true, y_pred):
    n = len(y_true)
    if n <= 1:
        return 0.0
    
    concordant = 0
    permissible = 0
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    for i in range(n):
        diff_true = y_true[i+1:] - y_true[i]
        diff_pred = y_pred[i+1:] - y_pred[i]
        
        valid_mask = diff_true != 0
        if not np.any(valid_mask):
            continue
            
        permissible += np.sum(valid_mask)
        
        concordant += np.sum((diff_true[valid_mask] * diff_pred[valid_mask]) > 0)
        
        concordant += 0.5 * np.sum(diff_pred[valid_mask] == 0)
    
    if permissible == 0:
        return 0.0
    
    return concordant / permissible

from sklearn.metrics import roc_auc_score

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_pred - y_true) ** 2)
    
    if len(y_true) > 1 and np.var(y_true) > 0:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = 0.0

    spearman_corr, _ = stats.spearmanr(y_true, y_pred)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    ccc = concordance_correlation_coefficient(y_true, y_pred)

    c_index = c_index_score(y_true, y_pred)

    # compute AUROC, simply set threshold 0.5
    y_true_binary = (y_true >= 0.5).astype(int)
    
    auroc = roc_auc_score(y_true_binary, y_pred)
    
    return {
        "mse": float(mse),
        "r2": float(r2),
        "spearman": float(spearman_corr),
        "ccc": float(ccc),
        "c_index": float(c_index),
        "auroc": float(auroc),
    }

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading base model: {args.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        _attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    base_model.requires_grad_(False)
    base_model.to(device) 

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None or base_model.config.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading components for: {args.component}")
    components = get_component_cls(args.component)()

    probe_path = os.path.join(args.work_dir, "experiments", args.run_name, "model")

    print(f"Loading probe from: {probe_path}")

    probe_model = ProbeWrapper.from_pretrained(
        probe_path,
        base_model,
        components
    )
    probe_model.to(device)
    probe_model.requires_grad_(False)
    probe_model.eval()

    for dataset_name in args.datasets:

        output_dir = os.path.join(args.work_dir, "experiments", args.run_name, "eval_results", args.model_name.replace("/", "--"), args.revision, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        numpy_file = os.path.join(output_dir, f"all_preds_labels.npz")

        if os.path.exists(numpy_file) and args.skip_evaluation:
            print(f"Found existing predictions and labels at {numpy_file}, skipping evaluation for dataset {dataset_name}.")
            data = np.load(numpy_file)
            all_preds = data['all_preds']
            all_labels = data['all_labels']
        else:
            print(f"Loading dataset: {dataset_name}")
            print(f"Evaluating on dataset: {dataset_name}")
            stage3_dir = os.path.join(args.work_dir, "datasets", args.model_name.replace("/", "--"), args.revision, dataset_name, "stage3")
            dataset_processor = load_dataset_processor(
                dataset_name,
                stage3_dir,
            )
            dataset_processor.load()
            
            target_split = "test"
            print(f"Evaluating on split: {target_split}")
            dataset = dataset_processor.get(target_split)

            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['prompt', 'label']])
            
            dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

            if 'label' in dataset.column_names:
                dataset = dataset.rename_column("label", "labels")
            
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                collate_fn=default_data_collator
            )

            all_preds = []
            all_labels = []
            total_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    outputs = probe_model(**batch)
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    if loss is not None:
                        total_loss += loss.item() * batch['input_ids'].size(0)
                    
                    all_preds.extend(logits.float().cpu().numpy())
                    all_labels.extend(batch['labels'].float().cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            np.savez_compressed(numpy_file, all_preds=all_preds, all_labels=all_labels)
            print(f"Saved predictions and labels to {numpy_file}")

        results = compute_metrics(all_labels, all_preds)
        
        print("\n" + "="*30)
        print("Evaluation Results")
        print("="*30)
        print(json.dumps(results, indent=4))
        print("="*30)
        
        out_file = os.path.join(output_dir, f"eval_result.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True, help='Run name of the probe training experiment.')
    parser.add_argument('--model_name', type=str, required=True, help='Base model name or path')
    parser.add_argument('--revision', type=str, default='main', help='Model revision')
    parser.add_argument('--attn_implementation', type=str, default="flash_attention_2", help='Attention implementation')
    parser.add_argument('--component', type=str, required=True, help='Component name (e.g., olmo3)')
    parser.add_argument('--datasets', type=str, nargs='+', required=True, help='List of datasets to evaluate on')
    parser.add_argument('--work_dir', type=str, required=True, help='Root directory for datasets (same as work_dir in train.py)')    
    parser.add_argument('--batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation if predictions already exist')

    args = parser.parse_args()
    
    evaluate(args)