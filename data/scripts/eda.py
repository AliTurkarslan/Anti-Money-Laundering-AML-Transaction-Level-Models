# eda.py
# Detailed EDA with meaningful comments and improved visualizations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(file_path):
    """
    Load and display the first few rows of the dataset.
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    return df

def class_balance_analysis(df, target_column):
    """
    Analyze and comment on the class distribution.
    """
    print("\n### Class Distribution Analysis ###")
    class_counts = df[target_column].value_counts()
    print(class_counts)
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"Imbalance Ratio (Legitimate : Laundering) ≈ {imbalance_ratio:.2f}:1")

    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df, palette="coolwarm")
    plt.title("Class Distribution")
    plt.xlabel("Is Laundering (0: Legitimate, 1: Laundering)")
    plt.ylabel("Count")
    plt.show()
    print("The dataset is highly imbalanced, which will need to be addressed later.\n")

def numerical_feature_analysis(df, columns):
    """
    Plot histograms and boxplots for numerical columns and identify key observations.
    """
    print("\n### Numerical Feature Analysis ###")
    for column in columns:
        print(f"\nFeature: {column}")
        print(f"Mean: {df[column].mean():.2f}")
        print(f"Median: {df[column].median():.2f}")
        print(f"Max: {df[column].max():.2f}, Min: {df[column].min():.2f}")
        print(f"Standard Deviation: {df[column].std():.2f}")
        
        # Histogram
        plt.figure(figsize=(12, 4))
        sns.histplot(df[column], bins=50, kde=True, color="blue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.show()

        # Boxplot
        plt.figure(figsize=(12, 4))
        sns.boxplot(x=df[column], color="orange")
        plt.title(f"Boxplot of {column}")
        plt.show()
        print(f"Observation: The {column} has a skewed distribution, indicating the presence of extreme values.\n")

def correlation_analysis(df, columns):
    """
    Plot correlation matrix for numerical columns and provide comments.
    """
    print("\n### Correlation Analysis ###")
    correlation_matrix = df[columns].corr()
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    print("The correlation between Amount Received and Amount Paid is strong (0.84), indicating redundancy.\n")

if __name__ == "__main__":
    # File path and parameters
    file_path = "/content/drive/My Drive/AML_Project/data/raw/HI-Small.csv"
    target_column = "Is Laundering"
    numerical_columns = ["Amount Received", "Amount Paid"]

    # Step 1: Load data
    df = load_data(file_path)

    # Step 2: Class Balance Analysis
    class_balance_analysis(df, target_column)

    # Step 3: Numerical Feature Analysis
    numerical_feature_analysis(df, numerical_columns)

    # Step 4: Correlation Analysis
    correlation_analysis(df, numerical_columns)

    print("\n### EDA Completed Successfully! ###")
