import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import os
import numpy as np
from scripts.config import RESULTS_DIR
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(model, data, mask, threshold=0.5):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()
        pred = (out[mask] > threshold).float()

    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_scores = out[mask].detach().cpu().numpy()

    # Performance Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=1),
        "recall": recall_score(y_true, y_pred, zero_division=1),
        "f1": f1_score(y_true, y_pred, zero_division=1),
        "auc": roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.5,
        "conf_matrix": confusion_matrix(y_true, y_pred)
    }

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    metrics["pr_auc"] = auc(recall, precision)

    metrics["y_true"] = y_true
    metrics["y_scores"] = y_scores

    return metrics


def save_confusion_matrix(metrics, prefix="test"):

    # Confusion Matrix
    conf_matrix = np.array(metrics["conf_matrix"])  # Listeyi NumPy array'e dönüştür
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Confusion Matrix Görselleştirme
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=plt.gca())
    plt.title(f"Confusion Matrix ({prefix})")
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_confusion_matrix.png"))
    plt.close()


def plot_results(metrics, prefix="test"):
    if "y_scores" not in metrics:
        raise KeyError("'y_scores' key not found in metrics!")

    y_scores = np.array(metrics.get("y_scores", []))  # Eğer yoksa boş bir liste döndür
    y_true = np.array(metrics.get("y_true", []))  # Eğer yoksa boş bir liste döndür

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({prefix})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_pr_curve.png"))
    plt.close()
