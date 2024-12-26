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
├── data │ ├── README.md # Dataset download instructions │ └── (data files) # Not included due to size; download separately. ├── notebooks │ ├── Initial_EDA.ipynb # Exploratory Data Analysis ├── references │ ├── Cheng et al., 2024 - Graph Neural Networks for Financial Fraud Detection │ ├── Ikeda et al., 2020 - Feature Engineering for Fraud Detection │ ├── Wan and Li, 2024 - Dynamic Graph Models for AML ├── results │ ├── GCN │ │ ├── train_confusion_matrix.png │ │ ├── train_pr_curve.png │ │ ├── test_confusion_matrix.png │ │ ├── test_pr_curve.png │ │ ├── GCN_train_metrics.png │ │ └── README.md # Model-specific details │ ├── GAT │ ├── GraphNN ├── scripts │ ├── config.py # Configuration parameters │ ├── evaluate.py # Model evaluation and metrics │ ├── model.py # GNN model architectures │ ├── preprocess.py # Data preprocessing pipeline │ ├── train.py # Model training script │ ├── utils.py # Helper functions ├── LICENSE.md ├── README.md ├── main.py # Main script to run the project └── requirements.txt # Dependencies



- **data/**: Contains raw and processed datasets.
- **scripts/**: Includes data preprocessing, training, and evaluation scripts.
- **models/**: GNN model and saved weights.
- **notebooks/**: Jupyter notebooks for exploratory analysis and training.
- **results/**: Logs, metrics, and figures.
- **references/**: Research papers and external references.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/AliTurkarslan/Anti-Money-Laundering-AML-Transaction-Level-Models.git
2. Install the required libraries.
   pip install -r requirements.txt
4. Run the notebooks in the `notebooks/` directory.

## License
This project is licensed under the MIT License.
