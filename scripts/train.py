from sklearn.utils.class_weight import compute_class_weight
from torch.optim import Adam
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scripts.config import RESULTS_DIR, MODEL_TYPE, HIDDEN_DIM, OUTPUT_DIM, EPOCHS, LEARNING_RATE, DROPOUT_RATE

def train_model(model, data, epochs=50, lr=0.01, weight_decay=1e-4, mode="train"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # Calculate Class Weights
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=data.y.cpu().numpy())
    class_weights = torch.tensor(np.log1p(class_weights), dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.sum()
    print("Class Weights:", class_weights)

    # Loss Function and Optimization
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track Results
    results = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [], "pr_auc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Model Output
        out = model(data.x, data.edge_index)

        # Loss Calculation
        loss = criterion(out[data.train_mask].view(-1), data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        # Forecasts and Metrics
        pred = (out[data.train_mask] > 0).float()
        y_true = data.y[data.train_mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_scores = out[data.train_mask].detach().cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        auc_score = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0.5
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_curve, precision_curve)

        # Save Results
        results["loss"].append(loss.item())
        results["accuracy"].append(acc)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)
        results["auc"].append(auc_score)
        results["pr_auc"].append(pr_auc)

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")

    # Sonuçları JSON Dosyasına Kaydet
    filename = f"{MODEL_TYPE}_{mode}_LR_{lr}_Epochs_{epochs}"
    with open(os.path.join(RESULTS_DIR, f"{filename}_metrics.json"), "w") as f:
        json.dump(results, f)

    # Training Graphics
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Model: {MODEL_TYPE}, LR: {lr}, Epochs: {epochs}, Dropout: {DROPOUT_RATE} - {mode.capitalize()} Results")

    # Loss
    plt.subplot(3, 2, 1)
    plt.plot(results["loss"], label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{mode.capitalize()} Loss")
    plt.legend()

    # Accuracy
    plt.subplot(3, 2, 2)
    plt.plot(results["accuracy"], label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{mode.capitalize()} Accuracy")
    plt.legend()

    # Precision
    plt.subplot(3, 2, 3)
    plt.plot(results["precision"], label="Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.title(f"{mode.capitalize()} Precision")
    plt.legend()

    # Recall ve F1
    plt.subplot(3, 2, 4)
    plt.plot(results["recall"], label="Recall")
    plt.plot(results["f1"], label="F1")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title(f"{mode.capitalize()} Recall & F1")
    plt.legend()

    # AUC
    plt.subplot(3, 2, 5)
    plt.plot(results["auc"], label="AUC")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.title(f"{mode.capitalize()} AUC")
    plt.legend()

    # PR-AUC
    plt.subplot(3, 2, 6)
    plt.plot(results["pr_auc"], label="PR-AUC")
    plt.xlabel("Epochs")
    plt.ylabel("PR-AUC")
    plt.title(f"{mode.capitalize()} PR-AUC")
    plt.legend()

    # Grafik Kaydetme
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename}_metrics.png"))
    plt.show()
