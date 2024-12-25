import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scripts.config import TEST_SIZE, SEED, BALANCE_DATA, BALANCE_METHOD
from imblearn.over_sampling import SMOTE, ADASYN  # SMOTE eklenmeli
import numpy as np


# Exchange rates
exchange_rates = {
    'Australian Dollar': 0.65, 'Bitcoin': 90000.0, 'Brazil Real': 0.20,
    'Canadian Dollar': 0.75, 'Euro': 1.10, 'Mexican Peso': 0.05,
    'Ruble': 0.014, 'Rupee': 0.012, 'Saudi Riyal': 0.27,
    'Shekel': 0.28, 'Swiss Franc': 1.05, 'UK Pound': 1.25,
    'US Dollar': 1.0, 'Yen': 0.007, 'Yuan': 0.14, 'Dirham': 0.27,
    'Dinar': 3.25, 'Ringgit': 0.24
}


# Data balancing function with ADASYN
def balance_with_adasyn(df, target_col='Is Laundering', exclude_cols=None):
    """
    Balances the dataset using the ADASYN algorithm with dtype handling and excluded columns.

    Parameters:
        df (pd.DataFrame): Input dataset.
        target_col (str): The target column for balancing.
        exclude_cols (list): Columns to exclude from ADASYN processing (e.g., timestamps, IDs).

    Returns:
        pd.DataFrame: Balanced dataset.
    """

    # 1. Determine the columns to be excluded
    exclude_cols = exclude_cols if exclude_cols else []
    adasyn_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]

    # 2. Separate the features (X) and the target variable (y)
    X = df[adasyn_cols].copy()
    y = df[target_col]

    # 3. Convert non-numeric columns (label encoding)
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):  # Eğer sayısal değilse
            le = LabelEncoder()  # Label Encoding uygula
            X[col] = le.fit_transform(X[col].astype(str))  # Kategorik veriyi kodla

    # 4. ADASYN is applied
    adasyn = ADASYN(sampling_strategy='minority', n_neighbors=3, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # 5. Create a balanced data set
    balanced_df = pd.DataFrame(X_resampled, columns=adasyn_cols)
    balanced_df[target_col] = y_resampled  # Hedef değişken eklenir

    #6. Add excluded columns
    for col in exclude_cols:
        balanced_df[col] = df[col].values[:len(balanced_df)]  # Hariç tutulan sütunları koru

    # 7. Data type checking and correction is done
    balanced_df = balanced_df.astype(float)  # Tüm sütunlar float yapılır

    return balanced_df

# Upsampling Function
def create_balanced_dataset(df):
    majority = df[df['Is Laundering'] == 0]
    minority = df[df['Is Laundering'] == 1]

    minority_upsampled = resample(minority,
                                  replace=True,
                                  n_samples=len(majority),
                                  random_state=42)

    balanced_df = pd.concat([majority, minority_upsampled])
    return balanced_df


# Function for balancing dataset using SMOTE
def balance_dataset_with_smote(df, target_column='Is Laundering', exclude_columns=None):
    """
    Balances the dataset using SMOTE while excluding non-numeric columns.

    Parameters:
        df (pd.DataFrame): Input dataset.
        target_column (str): The target column for balancing.
        exclude_columns (list): Columns to exclude from SMOTE (e.g., timestamps).

    Returns:
        pd.DataFrame: Balanced dataset.
    """
    # Exclude specified columns (e.g., timestamps)
    exclude_columns = exclude_columns if exclude_columns else []
    smote_columns = [col for col in df.columns if col not in exclude_columns + [target_column]]

    # Separate features and target
    X = df[smote_columns].copy()
    y = df[target_column]

    # Convert non-numeric columns to numeric for SMOTE
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.factorize(X[col])[0]

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Reconstruct the balanced dataframe
    balanced_df = pd.DataFrame(X_resampled, columns=smote_columns)
    balanced_df[target_column] = y_resampled

    # Add back excluded columns
    for col in exclude_columns:
        balanced_df[col] = df[col].values[:len(balanced_df)]  # Keep excluded columns unchanged

    return balanced_df



# Data Loading and Processing Function
def load_and_process_data(file_path):
    # 1. Load Data
    df = pd.read_csv(file_path)

    #2. Normalize Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp_Normalized'] = (df['Timestamp'] - df['Timestamp'].min()) / (
        df['Timestamp'].max() - df['Timestamp'].min())

    #3. Create Unique Accounts
    df['From_Account'] = df['From Bank'].astype(str) + '_' + df['Account']
    df['To_Account'] = df['To Bank'].astype(str) + '_' + df['Account.1']

    #4. Convert Currencies to USD
    df['Receiving Rate'] = df['Receiving Currency'].map(exchange_rates)
    df['Payment Rate'] = df['Payment Currency'].map(exchange_rates)
    # Extra Features
    df['Amount Received USD'] = df['Amount Received'] * df['Receiving Rate']
    df['Amount Paid USD'] = df['Amount Paid'] * df['Payment Rate']
    df['Currency Mismatch'] = (df['Receiving Currency'] != df['Payment Currency']).astype(int)
    df['Self_Loop'] = (df['From_Account'] == df['To_Account']).astype(int)

    # Time dif. normalization
    df['Time Diff'] = df['Timestamp'].diff().dt.total_seconds().fillna(0)

    #6. Map Unique Accounts
    accounts = pd.concat([df['From_Account'], df['To_Account']]).unique()
    account_to_idx = {acc: idx for idx, acc in enumerate(accounts)}

    df['From'] = df['From_Account'].map(account_to_idx)
    df['To'] = df['To_Account'].map(account_to_idx)

    # 7. Tagging (Suspicious Actions)
    suspicious = df[df['Is Laundering'] == 1]
    s1 = suspicious[['From_Account', 'Is Laundering']]
    s2 = suspicious[['To_Account', 'Is Laundering']]
    s2 = s2.rename({'To_Account': 'From_Account'}, axis=1)

    suspicious = pd.concat([s1, s2], join='outer').drop_duplicates()

    accounts_df = pd.DataFrame({'Account': accounts, 'Is Laundering': 0})
    accounts_df.set_index('Account', inplace=True)

    accounts_df.update(suspicious.set_index('From_Account'))
    df['Is Laundering'] = accounts_df.loc[df['From_Account']].values  # Etiketleri eşleştir

    # 8. Balanced Dataset
    # Check Balanced Data Distribution
    print("Original data distribution:")
    print(df['Is Laundering'].value_counts())

    if BALANCE_DATA:
        if BALANCE_METHOD == "adasyn":
            print("Applying ADASYN to balance the dataset...")
            df = balance_with_adasyn(df, target_col='Is Laundering', exclude_cols=['Timestamp', 'ID'])
        elif BALANCE_METHOD == "upsampling":
            print("Applying upsampling to balance the dataset...")
            df = create_balanced_dataset(df)
        elif BALANCE_METHOD == "smote":
            print("Applying SMOTE to balance the dataset...")
            df = balance_dataset_with_smote(df)
        else:
            raise ValueError(f"Invalid BALANCE_METHOD: {BALANCE_METHOD}. Choose 'adasyn', 'upsampling', or 'smote'.")

    print("Balanced data distribution:")
    print(df['Is Laundering'].value_counts())

    # 9. Node Properties
    received = df.groupby('To_Account')['Amount Received USD'].mean()
    paid = df.groupby('From_Account')['Amount Paid USD'].mean()
    num_transactions = df['From_Account'].value_counts()

    accounts = pd.DataFrame(index=account_to_idx.keys())
    accounts['avg_received'] = received
    accounts['avg_paid'] = paid
    accounts['num_transactions'] = num_transactions
    accounts['currency_mismatch'] = df.groupby('From_Account')['Currency Mismatch'].mean()
    accounts['self_loop'] = df.groupby('From_Account')['Self_Loop'].mean()
    accounts['time_diff'] = df.groupby('From_Account')['Time Diff'].mean()
    accounts = accounts.fillna(0)

    # Create tensor
    x = torch.tensor(accounts.values, dtype=torch.float)
    edge_index = torch.tensor(df[['From', 'To']].values, dtype=torch.long).t().contiguous()

    y = torch.tensor(accounts_df['Is Laundering'].values, dtype=torch.float)

    #10. Masks
    num_nodes = len(accounts)
    train_idx, test_idx = train_test_split(range(num_nodes), test_size=TEST_SIZE, random_state=SEED)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    edge_features = torch.tensor(df[['Amount Paid USD', 'Amount Received USD', 'Currency Mismatch']].values,
                                 dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=y,
                train_mask=train_mask, test_mask=test_mask)


    return data
