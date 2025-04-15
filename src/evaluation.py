"""
Evaluation metrics for signature verification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report

def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate performance metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    # Add AUC if probability scores are provided
    if y_score is not None:
        metrics['auc'] = roc_auc_score(y_true, y_score)
    
    return metrics

def calculate_equal_error_rate(y_true, y_score):
    """Calculate the Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are closest
    abs_diff = np.abs(fpr - fnr)
    min_idx = np.argmin(abs_diff)
    
    eer = (fpr[min_idx] + fnr[min_idx]) / 2.0
    
    return eer, thresholds[min_idx]

def calculate_far_frr(y_true, y_score, threshold=0.5):
    """Calculate False Acceptance Rate and False Rejection Rate."""
    # Convert scores to binary predictions based on threshold
    y_pred = (y_score >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate FAR and FRR
    far = fp / (fp + tn)  # False Acceptance Rate
    frr = fn / (fn + tp)  # False Rejection Rate
    
    return far, frr

def plot_roc_curve(y_true, y_score, title="ROC Curve"):
    """Plot ROC curve and calculate AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()
    
    return roc_auc

def print_classification_report(y_true, y_pred, target_names=None):
    """Print classification report."""
    print(classification_report(y_true, y_pred, target_names=target_names))