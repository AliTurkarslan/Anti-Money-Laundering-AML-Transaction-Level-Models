import torch

# Model Settings
MODEL_TYPE = "GCN"  # Alternatives: "GCN", "GAT", "GraphNN"
HIDDEN_DIM = 32  # Increased dimensions for more features
OUTPUT_DIM = 1
EPOCHS = 50  # Increased number of training epochs
LEARNING_RATE = 0.001# More stable learning rate
DROPOUT_RATE = 0.3  # Added dropout to prevent overfitting

# Data Settings
DATA_PATH = "data/HI-Small_Trans.csv"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # Added validation set
SEED = 42
BALANCE_DATA = True  # True: Use balanced dataset, False: Use original dataset
BALANCE_METHOD = "upsampling"  # Alternatives: 'upsampling', 'smote', or 'adasyn'

# GPU Usage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Performance Optimization
BATCH_SIZE = 128  # Batch processing for large datasets
WEIGHT_DECAY = 1e-4  # Regularization coefficient (L2 regularization)


import os
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
