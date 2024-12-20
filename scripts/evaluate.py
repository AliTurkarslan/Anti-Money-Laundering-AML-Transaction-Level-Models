import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model using test data.
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = (pred > 0.5).float()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    print("Evaluation Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    from models.gnn_model import GAT
    from scripts.dataset import create_graph_dataset

    # Load graph dataset
    graph_data = create_graph_dataset("data/processed/balanced.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT(in_channels=graph_data.num_features, hidden_channels=16, out_channels=16, heads=8).to(device)

    # Load trained model
    model.load_state_dict(torch.load("models/trained_model.pth"))

    # Create test loader
    test_loader = NeighborLoader(graph_data, batch_size=128, input_nodes=graph_data.test_mask)

    # Evaluate model
    evaluate_model(model, test_loader, device)
