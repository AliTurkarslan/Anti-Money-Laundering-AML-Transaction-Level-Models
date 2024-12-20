# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "data/raw/HI-Small_Trans.csv"
df = pd.read_csv(data_path)

# Display dataset info
print("Dataset Info:")
print(df.info())
print("\nSample Data:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])

# Distribution of target variable
plt.figure(figsize=(8, 6))
sns.countplot(x="Is Laundering", data=df)
plt.title("Distribution of Target Variable")
plt.show()

# Distribution of transaction amounts
plt.figure(figsize=(8, 6))
sns.histplot(df["Amount Received"], bins=30, kde=True, color="blue")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount Received")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Currency comparison
print("\nUnique Receiving Currencies:", df["Receiving Currency"].unique())
print("Unique Payment Currencies:", df["Payment Currency"].unique())

# Currency-based analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x="Receiving Currency", y="Amount Received", data=df)
plt.xticks(rotation=45)
plt.title("Transaction Amounts by Receiving Currency")
plt.show()
