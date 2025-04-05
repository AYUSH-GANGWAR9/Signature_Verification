import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plots a confusion matrix using a seaborn heatmap.
    """
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
    except Exception as e:
        print("Error plotting confusion matrix:", e)

def plot_roc_curve(y_test, y_scores, title="ROC Curve"):
    """
    Plots the ROC curve given true labels and predicted scores.
    """
    try:
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
    except Exception as e:
        print("Error plotting ROC curve:", e)

if __name__ == "__main__":
    import numpy as np
    cm_example = np.array([[10, 2], [3, 15]])
    plot_confusion_matrix(cm_example)
