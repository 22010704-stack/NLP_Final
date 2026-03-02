"""
src/metrics.py
Evaluation metrics utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate all mandatory and optional metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Per-class metrics
    p_classes = precision_score(y_true, y_pred, average=None)
    r_classes = recall_score(y_true, y_pred, average=None)
    f_classes = f1_score(y_true, y_pred, average=None)
    
    for i, (p, r, f) in enumerate(zip(p_classes, r_classes, f_classes)):
        metrics[f"class_{i}_precision"] = p
        metrics[f"class_{i}_recall"] = r
        metrics[f"class_{i}_f1"] = f
        
    # ROC AUC (One-vs-Rest)
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics["roc_auc"] = 0.0
            
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """Plot normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()
    return cm


def get_model_size_mb(model):
    """Calculate model size in MegaBytes."""
    import torch
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
