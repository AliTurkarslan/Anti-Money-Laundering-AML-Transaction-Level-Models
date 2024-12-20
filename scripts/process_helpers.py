import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_columns(df, columns):
    """
    Encode categorical columns using LabelEncoder.
    """
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])
    return df

def normalize_column(df, column):
    """
    Normalize a column using min-max scaling.
    """
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df
