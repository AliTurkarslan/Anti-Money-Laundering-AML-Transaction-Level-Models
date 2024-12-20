import pandas as pd
import torch
from torch_geometric.data import Data

def create_graph_dataset(csv_path):
    """
    Create PyTorch Geometric dataset from CSV.
    """
    df = pd.read_csv(csv_path)

    # Map accounts to node indices
    accounts = pd.concat([df['Account'], df['Account.1']]).unique()
    account_to_idx = {acc: idx for idx, acc in enumerate(accounts)}
    df['From'] = df['Account'].map(account_to_idx)
    df['To'] = df['Account.1'].map(account_to_idx)

    # Create edge index and edge attributes
    edge_index = torch.tensor(df[['From', 'To']].values, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(df[['Amount Paid', 'Amount Received']].values, dtype=torch.float)
    
    # Node features and labels
    x = torch.eye(len(accounts), dtype=torch.float)
    y = torch.tensor(df['Is Laundering'].values, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

if __name__ == "__main__":
    processed_path = "data/processed/balanced.csv"
    graph_data = create_graph_dataset(processed_path)
    print(graph_data)
