import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from scripts.dataset import create_graph_dataset
from models.gnn_model import GAT

def train_model(data, device, epochs=20):
    """
    Train GAT on unbalanced data.
    """
    model = GAT(in_channels=data.num_features, hidden_channels=16, out_channels=16, heads=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCELoss()

    # Split data for training, validation, and testing
    transform = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    data = transform(data)

    train_loader = NeighborLoader(data, batch_size=128, input_nodes=data.train_mask)
    val_loader = NeighborLoader(data, batch_size=128, input_nodes=data.val_mask)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(pred, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Validation step
        validate_model(model, val_loader, device)

    torch.save(model.state_dict(), "models/trained_model_unbalanced.pth")
    print("Model trained and saved!")

def validate_model(model, val_loader, device):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = (pred > 0.5).float()
            correct += (pred.squeeze() == batch.y).sum().item()
            total += batch.y.size(0)
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    graph_data = create_graph_dataset("data/processed/unbalanced.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(graph_data, device)
