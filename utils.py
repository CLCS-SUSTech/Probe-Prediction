from sklearn.metrics import f1_score, roc_auc_score
from transformers.trainer_utils import EvalPrediction
from typing import List, Tuple, Dict, Optional
import numpy as np
from math import comb

def compute_metrics_cls(eval_pred: EvalPrediction | Tuple[np.ndarray, np.ndarray], compute_result: Optional[bool] = None) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions_raw = predictions
    predictions_sigmoid = 1 / (1 + np.exp(-np.clip(predictions, -500, 500)))
    pred_labels = predictions_sigmoid > 0.5
    labels = labels.astype(int)
    pred_labels = pred_labels.astype(int)
    accuracy = (pred_labels == labels).mean().item()
    f1 = f1_score(labels, pred_labels, average='weighted')
    try:
        auc_raw = roc_auc_score(labels, predictions_raw)
    except:
        auc_raw = 0.0
    try:
        auc_sigmoid = roc_auc_score(labels, predictions_sigmoid)
    except:
        auc_sigmoid = 0.0
    return {'acc': accuracy, 'f1': f1, 'auc_raw': auc_raw, 'auc_sigmoid': auc_sigmoid}

def compute_metrics_reg(eval_pred: EvalPrediction | Tuple[np.ndarray, np.ndarray], compute_result: Optional[bool] = None) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    rmse = np.sqrt(np.mean((predictions - labels) ** 2)).item()
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    r2 = r2.item()
    return {'rmse': rmse, 'r2': r2}
    
def pass_k(c: int, k: int, n: int) -> float:
    assert isinstance(c, int) and isinstance(k, int) and isinstance(n, int), "c, k, n must be integers"
    assert c >= 0 and k >= 0 and n >= 0, "c, k, n must be non-negative"
    assert c <= n and k <= n, "c, k cannot be greater than n"
    if n - c < k:
        return 1.0
    try:
        prob_no_success = comb(n - c, k) / comb(n, k)
    except ValueError:
        return 0.0
    return float(1.0 - prob_no_success)

if __name__ == "__main__":
    # test compute_metrics_reg, predictions are in (0, 1) range, labels are also float in (0, 1)
    preds = np.array([[0.8], [0.2], [0.6], [0.4]])
    labels = np.array([[1.0], [0.0], [1.0], [0.0]])
    eval_pred = (preds, labels)
    metrics = compute_metrics_reg(eval_pred)
    print("Metrics:", metrics)