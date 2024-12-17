# data_analysis.py
# Script to perform EDA on Hi-small dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_dataset(file_path):
    """
    Loads the dataset from the given file path.
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    return df

def check_missing_values(df):
    """
    Checks for missing values in the dataset.
    """
    print("\nChecking for missing values:")
    print(df.isnull().sum())

def display_basic_statistics(df):
    """
    Displays basic statistics and data types.
    """
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nData types:")
    print(df.dtypes)

def visualize_class_distribution(df, target_column):
    """
    Visualizes the class distribution for the target column.
    """
    print("\nClass distribution:")
    print(df[target_column].value_counts())

    plt.figure(figsize=(8,6))
    sns.countplot(x=target_column, data=df)
    plt.title("Class Distribution")
    plt.xlabel("Label (0: Legitimate, 1: Laundering)")
    plt.ylabel("Count")
    plt.show()

def main():
    # Path to the dataset
    file_path = "/content/Hi-small.csv"  # Adjust this if stored elsewhere
    target_column = "Is Laundering"     # Correct column name for target variable
    
    # Load the dataset
    df = load_dataset(file_path)

    # Perform EDA
    check_missing_values(df)
    display_basic_statistics(df)
    visualize_class_distribution(df, target_column)

if __name__ == "__main__":
    main()
