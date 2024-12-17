import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    """
    Perform Exploratory Data Analysis on the AML dataset.
    """
    df = pd.read_csv(file_path)
    print("Dataset Overview:")
    print(df.info())

    print("\nClass Distribution:")
    print(df['Is Laundering'].value_counts())

    plt.figure(figsize=(8,6))
    sns.countplot(x='Is Laundering', data=df)
    plt.title("Class Distribution")
    plt.show()

    print("\nMissing Values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    file_path = "/content/drive/My Drive/AML_Project/data/raw/HI-Small.csv"
    perform_eda(file_path)
