import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_path: str) -> pd.DataFrame:
    """Load raw EEG data from CSV."""
    return pd.read_csv(data_path)

def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess data: normalize, split, reshape for LSTM.
    Returns: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=['y']).values
    y = pd.get_dummies(df['y']).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    X_train = X_train.reshape(-1, 178, 1)
    X_test = X_test.reshape(-1, 178, 1)
    
    return X_train, X_test, y_train, y_test