# eda.py
# Script to perform detailed Exploratory Data Analysis (EDA) on AML dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """
    Load the dataset from the specified path.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully!\n")
    print("First 5 rows of the dataset:")
    print(df.head())
    return df

def check_missing_values(df):
    """
    Check for missing values in the dataset.
    """
    print("\nChecking for missing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        print("No missing values detected.")

def class_distribution(df, target_column):
    """
    Plot class distribution for the target column.
    """
    print("\nClass distribution:")
    print(df[target_column].value_counts())
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df)
    plt.title("Class Distribution")
    plt.xlabel("Is Laundering (0: Legitimate, 1: Laundering)")
    plt.ylabel("Count")
    plt.show()

def plot_numerical_distributions(df, numerical_columns):
    """
    Plot the distributions of numerical features.
    """
    print("\nNumerical Feature Distributions:")
    for column in numerical_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[column], bins=50, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

def plot_correlation_matrix(df, numerical_columns):
    """
    Plot the correlation matrix for numerical features.
    """
    print("\nCorrelation Matrix:")
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

if __name__ == "__main__":
    # Define paths and columns
    file_path = "/content/drive/My Drive/AML_Project/data/raw/HI-Small.csv"  # Adjust path as needed
    target_column = "Is Laundering"  # Target variable
    numerical_columns = [
        "Amount Received", 
        "Amount Paid"
    ]

    # 1. Load data
    df = load_data(file_path)

    # 2. Check for missing values
    check_missing_values(df)

    # 3. Class distribution
    class_distribution(df, target_column)

    # 4. Numerical feature distributions
    plot_numerical_distributions(df, numerical_columns)

    # 5. Correlation matrix
    plot_correlation_matrix(df, numerical_columns)

    print("\nEDA completed successfully! 🎉")
