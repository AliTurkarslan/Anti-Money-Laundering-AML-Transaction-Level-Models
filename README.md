# AMLwtGNN: Anti-Money Laundering with Graph Neural Networks

## Overview

This project implements **Graph Neural Networks (GNNs)** to detect suspicious financial transactions potentially related to **money laundering**. It processes transactional data, applies graph-based modeling, and uses advanced **machine learning techniques** to identify illicit patterns. This project focuses on developing transaction-level models for anti-money laundering (AML) in collaboration with Dmitry Pavlyuk, Igor Rodin, and Deloitte, Latvia.

This repository supports multiple GNN architectures (**GCN**, **GAT**, and **GraphNN**) and handles imbalanced datasets with techniques like **SMOTE**, **ADASYN**, and **upsampling**. Detailed performance metrics and visualizations are provided to analyze the results.

## Objectives
- To analyze and model the layering stage of money laundering.
- To apply machine learning techniques to detect suspicious activities.

## Team Members
- Dmitry Pavlyuk
- Igor Rodin
- Ali Turkarslan
- Niedre Vija
- Jolanta Krastina

## Directory Structure
📂 data 
├── 📄 README.md # Instructions for downloading the dataset 
├── (data files) # Not included due to size, instructions to download provided
📂 notebooks 
├── 📄 Initial_EDA.ipynb # Exploratory Data Analysis
📂 references 
├── 📄 Cheng et al., 2024 - Graph Neural Networks for Financial Fraud Detection 
├── 📄 Ikeda et al., 2020 - Feature Engineering for Fraud Detection 
├── 📄 Wan and Li, 2024 - Dynamic Graph Models for AML
📂 results 
├── 📂 GAT 
│   ├── 📄 train_confusion_matrix.png 
│   ├── 📄 train_pr_curve.png 
│   ├── 📄 test_confusion_matrix.png 
│   ├── 📄 test_pr_curve.png 
│   ├── 📄 GCN_train_metrics.png 
│   ├── 📄 README.md # Model-specific details 
├── 📂 GCN 
├── 📂 GraphNN
📂 scripts 
├── 📄 config.py # Configuration parameters 
├── 📄 evaluate.py # Model evaluation and metrics
├── 📄 model.py # GNN model architectures 
├── 📄 preprocess.py # Data preprocessing pipeline 
├── 📄 train.py # Model training script 
├── 📄 utils.py # Helper functions
📄 LICENSE.md 
📄 README.md 
📄 main.py # Main script to run the project 
📄 requirements.txt # Dependencies


- **data/**
  - `README.md` - Instructions for downloading the dataset
  - *(data files)* - Not included due to size, instructions to download provided

- **notebooks/**
  - `Initial_EDA.ipynb` - Exploratory Data Analysis

- **references/**
  - `Cheng et al., 2024` - Graph Neural Networks for Financial Fraud Detection
  - `Ikeda et al., 2020` - Feature Engineering for Fraud Detection
  - `Wan and Li, 2024` - Dynamic Graph Models for AML

- **results/**
  - **GAT/**
    - `train_confusion_matrix.png`
    - `train_pr_curve.png`
    - `test_confusion_matrix.png`
    - `test_pr_curve.png`
    - `GCN_train_metrics.png`
    - `README.md` - Model-specific details
  - **GCN/**
  - **GraphNN/**

- **scripts/**
  - `config.py` - Configuration parameters
  - `evaluate.py` - Model evaluation and metrics
  - `model.py` - GNN model architectures
  - `preprocess.py` - Data preprocessing pipeline
  - `train.py` - Model training script
  - `utils.py` - Helper functions

- `LICENSE.md`
- `README.md`
- `main.py` - Main script to run the project
- `requirements.txt` - Dependencies

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch and PyTorch Geometric
- Additional dependencies listed in requirements.txt

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AliTurkarslan/Anti-Money-Laundering-AML-Transaction-Level-Models.git
2. Install the required libraries.
   pip install -r requirements.txt
3. Check the environment setup:
   Ensure PyTorch and PyTorch Geometric are correctly installed by running:
    ```bash
    python -c "import torch; print(torch.__version__)"
    python -c "import torch_geometric; print(torch_geometric.__version__)"
4. Prepare dataset:
   The dataset is not included in this repository due to size constraints. Download it separately and place it in the data/ folder. See data/README.md for more details.
5. Train and evaluate the model:
   Run the main script to preprocess, train, and evaluate the model in one step:
   ```bash
   python main.py

  
## Dataset
**Not Included**: The dataset is too large to be included in this repository.  
**Download Instructions**: See `data/README.md` for detailed download instructions.

## Configuration
Modify settings in **scripts/config.py** for customization.

### Model Settings
MODEL_TYPE = "GAT"  # Options: "GCN" (Graph Convolutional Network), 
                    # "GAT" (Graph Attention Network), 
                    # "GraphNN" (General Graph Neural Network)
                    # Determines the type of model architecture used for training.

HIDDEN_DIM = 32     # Number of hidden units in each layer.
                    # Higher values may improve performance but increase computational cost.
                    # Suggested range: 16–128 based on dataset size.

OUTPUT_DIM = 1      # Number of output features.
                    # Typically set to 1 for binary classification problems.

EPOCHS = 20         # Number of times the model trains over the entire dataset.
                    # Increase for better learning but watch for overfitting.

LEARNING_RATE = 0.001  # Controls the step size for gradient descent.
                       # Lower values (e.g., 0.001) lead to slower but more stable training.
                       # Higher values (e.g., 0.01) speed up training but risk overshooting.

DROPOUT_RATE = 0.3  # Percentage of neurons dropped during training to prevent overfitting.
                    # Typical values: 0.2–0.5. Higher values increase regularization.

#### Data Balancing Settings
BALANCE_DATA = True  # Enables dataset balancing for handling class imbalance.
                     # Set to False if the dataset is already balanced.

BALANCE_METHOD = "upsampling"  # Options:
                               # 'upsampling' - Replicates minority class samples.
                               # 'smote' - Generates synthetic samples using SMOTE algorithm.
                               # 'adasyn' - Generates synthetic samples with ADASYN algorithm.
                               # Use 'smote' or 'adasyn' for non-linear data distributions.

## References
-Wei et al. - 2023 - A Dynamic Graph Convolutional Network for Anti-money Laundering
-Cheng et al. - 2024 - Graph Neural Networks for Financial Fraud Detection A Review
-Johannessen and Jullum - 2023 - Finding Money Launderers Using Heterogeneous Graph Neural Networks
-Wan and Li - 2024 - A Novel Money Laundering Prediction Model Based on a Dynamic Graph Convolutional Neural Network and
-Ikeda et al. - 2020 - A New Framework of Feature Engineering for Machine Learning in Financial Fraud Detection

## License
This project is licensed under the MIT License.









