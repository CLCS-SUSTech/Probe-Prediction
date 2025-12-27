from sklearn.metrics import f1_score, roc_auc_score
from transformers.trainer_utils import EvalPrediction
from typing import List, Tuple, Dict, Optional
import numpy as np

def compute_metrics_cls(eval_pred: EvalPrediction | Tuple[np.ndarray, np.ndarray], compute_result: Optional[bool] = None) -> Dict[str, float]:
    predictions, labels = eval_pred
    predictions_raw = predictions
    predictions_sigmoid = 1 / (1 + np.exp(-np.clip(predictions, -500, 500)))
    pred_labels = predictions_sigmoid > 0.5
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
    