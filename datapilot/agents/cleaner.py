"""
Cleaner Agent - Data Cleaning and Preprocessing
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


class CleanerAgent:
    """Cleans data by handling missing values, duplicates, and outliers."""
    
    def clean(self, df: pd.DataFrame, target_col: str, task_type: str, meta_features: np.ndarray = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean dataset intelligently based on meta-features.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            task_type: 'classification' or 'regression'
            meta_features: 32 meta-features to guide cleaning strategy
            
        Returns:
            (cleaned_dataframe, cleaning_report)
        """
        df_clean = df.copy()
        
        # Determine cleaning strategy based on meta-features
        if meta_features is not None:
            missing_ratio = meta_features[7]  # Index 7 is missing_ratio
            outlier_ratio = meta_features[12]  # Index 12 is outlier_ratio
            imbalance_ratio = meta_features[18]  # Index 18 is imbalance_ratio
        else:
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            outlier_ratio = 0.05
            imbalance_ratio = 0.0
        
        # Remove duplicates
        dup_count = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values (strategy depends on missing_ratio)
        if missing_ratio > 0.3:
            # High missing: drop columns with >30% missing
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() / len(df_clean) > 0.3:
                    df_clean = df_clean.drop(columns=[col])
        
        # Fill remaining missing values
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if col in get_numeric_cols(df_clean):
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col].fillna(mode_val[0], inplace=True)
        
        # Handle outliers (strategy depends on outlier_ratio)
        if outlier_ratio < 0.1:  # Only if <10% outliers
            for col in get_numeric_cols(df_clean):
                if col != target_col:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    df_clean[col] = df_clean[col].clip(lower, upper)
        
        # Handle class imbalance (if high imbalance, might need SMOTE later)
        if imbalance_ratio > 0.7 and task_type == 'classification':
            # Log for later handling
            pass
        
        report = {
            'original_shape': df.shape,
            'final_shape': df_clean.shape,
            'rows_removed': df.shape[0] - df_clean.shape[0],
            'duplicates_removed': dup_count,
            'missing_ratio': missing_ratio,
            'outlier_ratio': outlier_ratio,
            'imbalance_ratio': imbalance_ratio,
        }
        
        return df_clean, report
