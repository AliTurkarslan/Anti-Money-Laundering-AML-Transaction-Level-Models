import torch
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, confusion_matrix
from scripts.config import RESULTS_DIR, MODEL_TYPE, EPOCHS, LEARNING_RATE, DROPOUT_RATE

def evaluate_model(model, data, mask, threshold=0.5, mode="test"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index).squeeze()

    pred = (out[mask] > threshold).float()
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_scores = out[mask].cpu().detach().numpy()

    # Performance Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=1),
        "recall": recall_score(y_true, y_pred, zero_division=1),
        "f1": f1_score(y_true, y_pred, zero_division=1),
        "auc": roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.5,
        "conf_matrix": confusion_matrix(y_true, y_pred),
    }

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    metrics["pr_auc"] = auc(recall, precision)


    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    metrics["pr_auc"] = pr_auc

    metrics["y_true"] = y_true  # Eklendi
    metrics["y_scores"] = y_scores  # Eklendi

    # **Visualization and Saving**
    filename = f"{MODEL_TYPE}_{mode}_LR_{LEARNING_RATE}_Epochs_{EPOCHS}"
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Model: {MODEL_TYPE}, LR: {LEARNING_RATE}, Epochs: {EPOCHS}, Dropout: {DROPOUT_RATE} - {mode.capitalize()} Results")

    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall_curve, precision_curve, label=f"PR Curve (AUC: {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{mode.capitalize()} Precision-Recall Curve")
    plt.legend()

    # AUC Values
    plt.subplot(1, 2, 2)
    plt.bar(["AUC", "PR-AUC"], [metrics["auc"], metrics["pr_auc"]], color=['blue', 'orange'])
    plt.ylabel("Score")
    plt.title(f"{mode.capitalize()} AUC Scores")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}_results.png"))
    plt.show()

    return metrics
