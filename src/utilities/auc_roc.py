import numpy as np
from sklearn.metrics import roc_auc_score

def compute_auc_roc(all_labels: np.ndarray, all_probs: np.ndarray, classes: list) -> dict:
    """
    Compute per-class AUC-ROC and mean AUC-ROC.

    Skips classes with only one label  present in the batch (can't compute AUC for a class with no positives or no negatives).

    Args:
        all_labels : np.ndarray (N, 14) - ground truth binary labels
        all_probs : np.ndarray (N, 14) - predicted probabilities after sigmoid
        classes: list of class name strings

    Returns:
        dict with per-class AUC values and 'mean' key
    """

    aucs = {}
    for i, cls in enumerate(classes):
        if len(np.unique(all_labels[:, i])) < 2:
            # if Only one class present - skips (undefined AUC)
            aucs[cls] = float('nan')
            continue
        aucs[cls] = roc_auc_score(all_labels[:, i], all_probs[:, i])
    
    valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
    aucs['mean'] = float(np.mean(valid_aucs)) if valid_aucs else float('nan')
    
    return aucs