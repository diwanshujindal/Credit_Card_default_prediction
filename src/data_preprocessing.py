import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load the credit card default dataset"""
    df = pd.read_excel(path)
    df = df.drop('Unnamed: 0', axis=1)
    df.columns = df.iloc[0]
    df = df[1:]
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Convert all columns to numeric, coercing errors
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Handle EDUCATION column (combine categories 4,5,6,9 into 4)
    df_processed['EDUCATION'] = df_processed['EDUCATION'].replace([4,5,6,9], 4)
    
    # Handle MARRIAGE column (replace 0 with 3)
    df_processed['MARRIAGE'] = df_processed['MARRIAGE'].replace(0, 3)
    
    return df_processed

def prepare_features(df_processed):
    """Prepare features and target variable"""
    X = df_processed.drop('default payment next month', axis=1)
    y = df_processed['default payment next month']
    return X, y

def split_and_scale_data(X, y, test_size=0.2, val_size=0.5, random_state=42):
    """Split data into train, validation, test sets and scale features"""
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
