import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        """
        Graph Attention Network (GAT) with 2 convolutional layers.
        """
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        self.lin = Linear(out_channels, 1)  # Final classification layer (1 output)
        self.sigmoid = torch.nn.Sigmoid()  # Binary classification

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the GAT model.
        """
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.lin(x)
        return self.sigmoid(x)
