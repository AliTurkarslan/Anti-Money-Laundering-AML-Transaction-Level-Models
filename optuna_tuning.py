import optuna
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from model import GCN, GAT, GraphNN  # Import model definitions

from main import data


# 1. Objective Function
def objective(trial, input_dim=None):
    # Define hyperparameters to be tuned by Optuna
    model_type = trial.suggest_categorical('model_type', ['GCN', 'GAT', 'GraphNN'])
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128, step=16)
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)

    # 2. Model Selection
    if model_type == 'GCN':
        model = GCN(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
    elif model_type == 'GAT':
        model = GAT(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
    elif model_type == 'GraphNN':
        model = GraphNN(input_dim, hidden_dim, output_dim, dropout_rate).to(device)

    # 3. Optimizer and Loss Function
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = BCEWithLogitsLoss()

    # 4. Training Process
    model.train()
    for epoch in range(20):  # Fixed number of epochs
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr)
        y_pred = torch.sigmoid(pred[data.test_mask]).cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()

    # Calculate AUC (Area Under Curve)
    auc_score = roc_auc_score(y_true, y_pred)

    return auc_score


# 6. Running Optuna Optimization
study = optuna.create_study(direction='maximize')  # Maximize AUC score
study.optimize(objective, n_trials=50)  # Perform 50 trials

# 7. Display Best Results
print("Best hyperparameters:", study.best_params)
print("Best AUC:", study.best_value)
