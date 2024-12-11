# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/raw/HI-small.csv"
data = pd.read_csv(file_path)

# Perform basic EDA
print(data.info())
print(data.describe())

# Add further analysis steps below
# Example: Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()
