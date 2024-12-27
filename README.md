# AMLwtGNN: Anti-Money Laundering Detection with Graph Neural Networks

## Overview

This project focuses on **Money Laundering Detection** using **Graph Neural Networks (GNNs)**. The aim is to identify suspicious financial transactions potentially related to money laundering it specifically targets the detection of suspicious transactions at the **layering stage** of money laundering activities. It processes transactional data, applies graph-based modeling, and uses advanced **machine learning techniques** to detect illicit patterns in collaboration with Dmitry Pavlyuk, Igor Rodin, and Deloitte, Latvia.  

This repository supports multiple GNN architectures (**GCN**, **GAT**, and **GraphNN**) and handles imbalanced datasets with techniques like **SMOTE**, **ADASYN**, and **upsampling**. Detailed performance metrics and visualizations are provided to analyze the results.

## Objectives
- To detect suspicious transactions potentially related to **money laundering** activities using Graph Neural Networks (GNNs).  
- To analyze and model the **layering stage** of money laundering to identify anomalies and patterns.  
- To evaluate machine learning techniques for improving detection accuracy and scalability.  

## Team Members
- Niedre Vija (Project Manager)
- Jolanta Krastina (Data Scientist)
- Ali Turkarslan (Data Engineer)
- Dmitry Pavlyuk (Academic Advisor - Transport and Telecommunication Institute)
- Igor Rodin (Industry Advisor - Deloitte Latvia)

## **Table of Contents**

| **Content**              | **Description**                                                                                       | **Link**                                                                                                                                              |
|--------------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **GitHub Repository**    | Code, scripts, and other files for model development, evaluation, and environment setup.              | [GitHub Repository](https://github.com/AliTurkarslan/Anti-Money-Laundering-AML-Transaction-Level-Models/tree/main)                                   |
| **Project Management**   | Project tracking with Kanban board.                                                                   | [Kanban Board](https://github.com/users/AliTurkarslan/projects/4)                                                                                    |
| **Meeting Minutes**      | Communication strategy description and meeting minutes.                                               | [Meeting Minutes](https://studentstsi-my.sharepoint.com/:f:/g/personal/st83072_students_tsi_lv/El7zoe8AJ7dPp8Kw7vcIi8gB_vv52ZQMQplCtwQ7h6ZXTQ?e=N5jxsf) |



## Directory Structure

- **📂data/**
  - `README.md` - Instructions for downloading the dataset
  - *(data files)* - Not included due to size, instructions to download provided

- **📂notebooks/**
  - `Initial_EDA.ipynb` - Exploratory Data Analysis

- **📂references/**
  - `Cheng et al., 2024` - Graph Neural Networks for Financial Fraud Detection
  - `Ikeda et al., 2020` - Feature Engineering for Fraud Detection
  - `...

- **📂results/**
  - **📂GAT/**
    - `train_confusion_matrix.png`
    - `train_pr_curve.png`
    - `test_confusion_matrix.png`
    - `test_pr_curve.png`
    - `GCN_train_metrics.png`
    - `README.md` - Model-specific details
  - **📂GCN/**
    - `...`
  - **📂GraphNN/**
    - `...`
   
- **📂scripts/**
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


## **Contribution of Team Members**

| **Category**              | **Member**                     | **Contribution**                                                                                                                                            |
|---------------------------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Preparation**       | **Jolanta Krastiņa (70%)**     | Led data exploration and selection, and discussions on dataset challenges, including handling imbalanced data and addressing time sensitivity in data splitting. |
|                           | **Ali Turkasan (30%)**         | Assisted with data exploration and selection, and ensured the dataset was properly formatted for further training and testing.                              |
| **Model Development**      | **Jolanta Krastiņa (30%)**     | Contributed to the initial implementation of the GNN model, and participated in discussions on key challenges, solutions, and configuration metrics.         |
|                           | **Ali Turkasan (70%)**         | Led the development of GNN models, including fine-tuning, optimization, and experimentation with different configurations.                                   |
| **Model Evaluation**       | **Jolanta Krastiņa (20%)**     | Contributed to the creation of evaluation metrics and the analysis of results.                                                                               |
|                           | **Ali Turkasan (80%)**         | Led the evaluation process, including generating performance plots, interpreting results, and conducting model comparisons.                                  |
| **Documentation**          | **Team (100%)**                | Documented technical aspects of the codebase, environment setup, implementation details, execution steps, and user instructions.                             |
| **Project Management**     | **Vija Niedre (100%)**         | Led project management, including planning, team coordination, and ensuring milestones were met.                                                              |




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
The dataset used in this project focuses on transactional data designed to simulate patterns indicative of **money laundering**.  
This dataset does not represent actual financial data but serves as a synthetic approximation for research purposes.
**Not Included**: The dataset is too large to be included in this repository.  
**Download Instructions**: See `data/README.md` for detailed download instructions.

## Configuration
Modify settings in **scripts/config.py** for customization.

### Model Settings
MODEL_TYPE = "GAT" # Options:  
                              # "GCN" (Graph Convolutional Network) - Focuses on general graph structures.  
                              # "GAT" (Graph Attention Network) - Adds attention mechanisms for improved feature learning.  
                              # "GraphNN" (General Graph Neural Network) - Suitable for more flexible node relationships.  
                              # These models are designed to **detect patterns and anomalies** in transaction data and are not intended for building full AML compliance systems.  

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

## Results
Sample Model:GAT
Visualizations:

  ![GAT_train_LR_0 001_Epochs_20_metrics](https://github.com/user-attachments/assets/12b6f320-579a-4594-8d09-4c6df42ba532)


## Hyperparameter Optimization (Optional)
This repository includes an example script (`optuna_tuning.py`) for hyperparameter optimization using Optuna.
#### Key Features
Supports tuning for model type, hidden dimensions, learning rate, and dropout rate.
Designed to maximize AUC scores by testing multiple configurations.
#### Note
Due to hardware limitations, this script has not been tested on large datasets but is included as a starting point for further development in GPU-supported environments.


## References
 - `Wei et al. - 2023 - A Dynamic Graph Convolutional Network for Anti-money Laundering
 - `Cheng et al. - 2024 - Graph Neural Networks for Financial Fraud Detection A Review
 - `Johannessen and Jullum - 2023 - Finding Money Launderers Using Heterogeneous Graph Neural Networks
 - `Wan and Li - 2024 - A Novel Money Laundering Prediction Model Based on a Dynamic Graph Convolutional Neural Network and
 - `Ikeda et al. - 2020 - A New Framework of Feature Engineering for Machine Learning in Financial Fraud Detection

## License
This project is licensed under the MIT License.









