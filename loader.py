import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_features(file_path, meta_cols=None, label_col='label'):
    if meta_cols is None:
        meta_cols = []

    df = pd.read_csv(file_path)
    X = df.drop(columns=meta_cols + [label_col]).values

    # 1. Initialize LabelEncoder to handle the string labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[label_col].values)

    # Store the classes for the evaluation report
    target_names = le.classes_

    return X, y_encoded, target_names

def preprocess_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
