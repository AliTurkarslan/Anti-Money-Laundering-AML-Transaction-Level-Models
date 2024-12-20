import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

def preprocess_data(df):
    """
    Preprocess dataset: normalize, encode, and feature engineer.
    """
    # Normalize timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].min()) / (df['Timestamp'].max() - df['Timestamp'].min())
    
    # Combine account IDs
    df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
    df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1']
    
    # Encode categorical columns
    encoder = LabelEncoder()
    for col in ['Payment Format', 'Receiving Currency', 'Payment Currency']:
        df[col] = encoder.fit_transform(df[col])
    
    return df

def balance_dataset(df):
    """
    Balance the dataset by oversampling the minority class.
    """
    majority = df[df['Is Laundering'] == 0]
    minority = df[df['Is Laundering'] == 1]
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    return pd.concat([majority, minority_upsampled])

if __name__ == "__main__":
    # Paths
    raw_path = "data/raw/HI-Small_Trans.csv"
    unbalanced_path = "data/processed/unbalanced.csv"
    balanced_path = "data/processed/balanced.csv"

    # Load raw data
    raw_df = pd.read_csv(raw_path)

    # Preprocess data
    preprocessed_df = preprocess_data(raw_df)
    preprocessed_df.to_csv(unbalanced_path, index=False)

    # Balance dataset
    balanced_df = balance_dataset(preprocessed_df)
    balanced_df.to_csv(balanced_path, index=False)
