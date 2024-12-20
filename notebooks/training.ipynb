# Import required libraries
import torch
from torch_geometric.loader import NeighborLoader
from scripts.dataset import create_graph_dataset
from models.gnn_model import GAT

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load graph dataset
data_path = "data/processed/unbalanced.csv"  # Can switch to balanced.csv for balanced training
data = create_graph_dataset(data_path)
data = data.to(device)

# Train-test split
from torch_geometric.transforms import RandomNodeSplit
split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
data = split(data)

# Initialize model
model = GAT(in_channels=data.num_features, hidden_channels=16, out_channels=1, heads=8).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Data loaders
train_loader = NeighborLoader(data, batch_size=128, input_nodes=data.train_mask)
val_loader = NeighborLoader(data, batch_size=128, input_nodes=data.val_mask)

# Training loop
epochs = 20
losses = []
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
    losses.append(total_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Plot loss curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), losses, marker="o")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Save model
torch.save(model.state_dict(), "models/trained_model_unbalanced.pth")
