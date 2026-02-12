"""
Feature Engineer Agent - Feature Processing and Selection
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, LabelEncoder


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    """Get categorical column names."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


class FeatureAgent:
    """Encodes categorical features and scales numeric features."""
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
    
    def engineer(self, df: pd.DataFrame, target_col: str, task_type: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Engineer features.
        
        Args:
            df: Clean DataFrame
            target_col: Target column name
            task_type: 'classification' or 'regression'
            
        Returns:
            (X_engineered, y, feature_report)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical
        for col in get_categorical_cols(X):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Scale numeric
        numeric_cols = get_numeric_cols(X)
        if numeric_cols:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.scalers['numeric'] = scaler
        
        report = {
            'original_features': len(df.columns) - 1,
            'final_features': X.shape[1],
            'encoded_cols': list(self.encoders.keys()),
            'scaled_cols': numeric_cols,
        }
        
        return X, y, report
