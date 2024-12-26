from scripts.config import *  # Import configuration settings
from scripts.model import GCN, GAT, GraphNN  # Import model architectures
from scripts.preprocess import load_and_process_data  # Import data preprocessing functions
from scripts.train import train_model  # Import training function
from scripts.utils import evaluate_model, save_confusion_matrix, plot_results  # Import utility functions
import os

# Create a directory for saving results
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the Data
print("Loading data...")
data = load_and_process_data(DATA_PATH)
print("Data successfully loaded!")

# Check GPU/CPU Availability
data = data.to(DEVICE)
print(f"The model will run on {DEVICE}.")

# Check for Test Mask
if not hasattr(data, 'test_mask'):
    raise AttributeError("Test mask not found! Please check preprocess.py.")

# Model Selection
if MODEL_TYPE == "GCN":
    model = GCN(input_dim=data.num_features, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
elif MODEL_TYPE == "GAT":
    model = GAT(input_dim=data.num_features, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
elif MODEL_TYPE == "GraphNN":
    model = GraphNN(input_dim=data.num_features, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, dropout_rate=DROPOUT_RATE)
else:
    raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}. Please choose 'GCN', 'GAT', or 'GraphNN'.")

# Move Model to the Selected Device
model = model.to(DEVICE)

# Train the Model
print("Training the model...")
train_model(model, data, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, mode="train")
print("Model training completed!")

# Evaluate Test Results
print("Evaluating model on test data...")
metrics_test = evaluate_model(model, data, mask=data.test_mask)

# Generate Test Graphs
print("Generating test graphs...")
plot_results(metrics_test, prefix="test")
print("Test graphs saved!")

# Save Confusion Matrix for Test Data
save_confusion_matrix(metrics_test, prefix="test")
plot_results(metrics_test, prefix="test")
print("Test results saved.")

# Evaluate Training Results
print("Evaluating model on training data...")
metrics_train = evaluate_model(model, data, mask=data.train_mask)

# Save Confusion Matrix for Training Data
save_confusion_matrix(metrics_train, prefix="train")
plot_results(metrics_train, prefix="train")
print("Training results saved.")

print("All processes completed successfully!")

