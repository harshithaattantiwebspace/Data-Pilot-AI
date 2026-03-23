# agents/feature.py

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler,
    MinMaxScaler, RobustScaler
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import TargetEncoder
from agents.base import BaseAgent


class FeatureAgent(BaseAgent):
    """
    Agent responsible for feature engineering.
    
    This is the THIRD agent in the pipeline. It takes cleaned data and:
      1. Removes ID columns (not useful for modeling)
      2. Encodes categorical columns (binary→label, low-card→one-hot, high-card→target)
      3. Scales numerical columns (skewed→RobustScaler, normal→StandardScaler)
      4. Selects top features if dimensionality is too high (>50 features)
      5. Encodes the target variable (LabelEncoder for classification)
    
    Stores all encoders/scalers so they can be reused for new predictions.
    
    Owner: Bhavana
    """
    
    def __init__(self):
        super().__init__("FeatureAgent")
        self.encoders = {}   # Stores fitted encoders for each column
        self.scalers = {}    # Stores fitted scalers
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features for modeling"""
        self.log("Starting feature engineering...")
        
        df = state['current_data'].copy()
        target_col = state['target_column']
        task_type = state['task_type']
        column_types = state['profile_report']['column_types']
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        feature_report = {
            'encoding': {},
            'scaling': {},
            'feature_selection': {},
            'new_features': [],
            'dropped_columns': []
        }
        
        # =====================================================================
        # Step 1: Remove ID columns (not useful for ML)
        # =====================================================================
        id_cols = [c for c, t in column_types.items() if t == 'id' and c in X.columns]
        X = X.drop(columns=id_cols)
        feature_report['dropped_columns'] = id_cols
        self.log(f"Removed ID columns: {id_cols}")
        
        # =====================================================================
        # Step 2: Encode categorical columns
        # =====================================================================
        X, encoding_info = self._encode_categoricals(X, y, column_types, task_type)
        feature_report['encoding'] = encoding_info
        
        # =====================================================================
        # Step 3: Scale numerical columns
        # =====================================================================
        X, scaling_info = self._scale_numericals(X, column_types)
        feature_report['scaling'] = scaling_info
        
        # =====================================================================
        # Step 4: Feature selection (only if >50 features)
        # =====================================================================
        if len(X.columns) > 50:
            X, selection_info = self._select_features(X, y, task_type)
            feature_report['feature_selection'] = selection_info
        else:
            feature_report['feature_selection'] = {
                'method': 'none',
                'reason': f'Only {len(X.columns)} features — no selection needed'
            }
        
        # =====================================================================
        # Step 4b: VIF-based multicollinearity removal
        # =====================================================================
        X, vif_info = self._remove_high_vif_features(X)
        feature_report['vif_analysis'] = vif_info
        if vif_info.get('removed_features'):
            self.log(f"VIF: Removed {len(vif_info['removed_features'])} multicollinear features")
        
        # =====================================================================
        # Step 5: Encode target variable (classification only)
        # =====================================================================
        if task_type == 'classification':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
            self.encoders['target'] = le
            self.log(f"Encoded target: {list(le.classes_)}")
        
        # Update pipeline state
        state['X'] = X
        state['y'] = y
        state['feature_names'] = X.columns.tolist()
        state['feature_report'] = feature_report
        state['encoders'] = self.encoders
        state['scalers'] = self.scalers
        state['stage'] = 'featured'
        
        self.log(f"Feature engineering complete. {len(X.columns)} features ready for modeling.")
        return state
    
    # =========================================================================
    # STEP 2: Categorical Encoding
    # =========================================================================
    
    def _encode_categoricals(self, X: pd.DataFrame, y: pd.Series,
                            column_types: Dict[str, str],
                            task_type: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical columns using the best strategy per column.
        
        Strategy selection:
          - 2 unique values (binary)  → LabelEncoder (0/1)
          - ≤10 unique values         → One-Hot Encoding (drop_first to avoid multicollinearity)
          - >10 unique values         → Target Encoding (uses target mean per category)
        
        Target encoding is preferred for high-cardinality because:
          - One-hot would create too many columns
          - It captures the relationship between category and target
          - Smoothing prevents overfitting on rare categories
        """
        encoding_info = {}
        
        categorical_cols = [c for c in X.columns if column_types.get(c) == 'categorical']
        
        for col in categorical_cols:
            n_unique = X[col].nunique()
            
            if n_unique == 2:
                # Binary → Label encoding (0/1)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
                encoding_info[col] = {
                    'method': 'label',
                    'n_unique': n_unique,
                    'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
                }
                self.log(f"  {col}: Label encoded (binary, {n_unique} values)")
            
            elif n_unique <= 10:
                # Low cardinality → One-Hot encoding
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                encoding_info[col] = {
                    'method': 'onehot',
                    'n_unique': n_unique,
                    'new_cols': dummies.columns.tolist()
                }
                self.log(f"  {col}: One-hot encoded ({n_unique} values -> {len(dummies.columns)} columns)")
            
            else:
                # High cardinality → Target encoding
                te = TargetEncoder(cols=[col], smoothing=1.0)
                X[col] = te.fit_transform(X[col], y)
                self.encoders[col] = te
                encoding_info[col] = {
                    'method': 'target',
                    'n_unique': n_unique,
                    'reason': 'High cardinality — target encoding preserves info without explosion'
                }
                self.log(f"  {col}: Target encoded ({n_unique} values)")
        
        return X, encoding_info
    
    # =========================================================================
    # STEP 3: Numerical Scaling
    # =========================================================================
    
    def _scale_numericals(self, X: pd.DataFrame,
                         column_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale numerical columns to normalize their ranges.
        
        Strategy selection (based on average skewness):
          - Mean skewness > 1   → RobustScaler (resistant to outliers, uses median/IQR)
          - Mean skewness ≤ 1   → StandardScaler (zero mean, unit variance)
        
        Why scale?
          - Many ML algorithms (SVM, KNN, LogReg) are sensitive to feature magnitude
          - Tree-based models don't need it, but it doesn't hurt
          - Ensures fair comparison between features
        """
        scaling_info = {}
        
        numeric_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
        
        if len(numeric_cols) == 0:
            return X, scaling_info
        
        # Check distribution to pick scaler
        skewness = X[numeric_cols].skew().abs().mean()
        
        if skewness > 1:
            scaler = RobustScaler()
            method = 'robust'
            reason = f'Mean absolute skewness = {skewness:.2f} (>1) -> using RobustScaler'
        else:
            scaler = StandardScaler()
            method = 'standard'
            reason = f'Mean absolute skewness = {skewness:.2f} (<=1) -> using StandardScaler'
        
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        self.scalers['numeric'] = scaler
        
        scaling_info = {
            'method': method,
            'reason': reason,
            'columns_scaled': numeric_cols,
            'n_columns': len(numeric_cols)
        }
        
        self.log(f"Scaled {len(numeric_cols)} numeric columns with {method} scaler")
        return X, scaling_info
    
    # =========================================================================
    # STEP 4: Feature Selection (for high-dimensional data)
    # =========================================================================
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series,
                        task_type: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Select the most important features when dimensionality is too high (>50).
        
        Two-stage process:
          Stage 1: Remove highly correlated features (|correlation| > 0.95)
                   These are redundant — keeping both hurts more than helps
          
          Stage 2: Rank remaining by Mutual Information and keep top 50
                   MI measures general dependency (not just linear), works for both
                   classification and regression
        """
        self.log("Performing feature selection (>50 features detected)...")
        
        # Stage 1: Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X = X.drop(columns=to_drop)
        self.log(f"  Removed {len(to_drop)} highly correlated features")
        
        # Stage 2: Mutual Information ranking
        if task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        
        # Keep top 50 features
        top_features = mi_df.head(50)['feature'].tolist()
        X = X[top_features]
        
        selection_info = {
            'method': 'correlation_filter + mutual_information',
            'dropped_correlated': to_drop,
            'n_dropped_correlated': len(to_drop),
            'selected_features': top_features,
            'n_selected': len(top_features),
            'top_10_mi_scores': mi_df.head(10).to_dict('records')
        }
        
        self.log(f"  Selected top {len(top_features)} features by mutual information")
        return X, selection_info
    
    # =========================================================================
    # STEP 4b: VIF-Based Multicollinearity Removal
    # =========================================================================
    
    def _compute_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Variance Inflation Factor for each numeric feature.
        
        VIF measures how much a feature is explained by other features:
          VIF = 1 / (1 - R²)  where R² is from regressing feature j on all others
          
          VIF = 1    → no multicollinearity
          VIF = 1-5  → moderate (acceptable)
          VIF = 5-10 → high (concerning)
          VIF > 10   → severe (should remove one of the correlated features)
        
        Uses sklearn's LinearRegression instead of statsmodels to avoid
        an extra dependency.
        """
        from sklearn.linear_model import LinearRegression
        
        vif_data = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return vif_data
        
        X_numeric = X[numeric_cols].fillna(0)
        
        for col in numeric_cols:
            y_vif = X_numeric[col].values
            X_vif = X_numeric.drop(columns=[col]).values
            
            if X_vif.shape[1] == 0:
                continue
            
            try:
                lr = LinearRegression()
                lr.fit(X_vif, y_vif)
                r2 = lr.score(X_vif, y_vif)
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
                vif_data[col] = round(vif, 2)
            except Exception:
                vif_data[col] = 1.0
        
        return vif_data
    
    def _remove_high_vif_features(self, X: pd.DataFrame,
                                    threshold: float = 10.0) -> Tuple[pd.DataFrame, Dict]:
        """
        Iteratively remove the feature with highest VIF until all are below threshold.
        
        This is the standard data science approach to multicollinearity:
          1. Compute VIF for all features
          2. If max VIF > threshold, remove that feature
          3. Repeat until all VIF ≤ threshold
        
        Args:
            X: Feature DataFrame
            threshold: VIF threshold (default 10 — standard practice)
        
        Returns:
            Tuple of (cleaned DataFrame, VIF analysis report)
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Skip if too few numeric features
        if len(numeric_cols) < 3:
            return X, {
                'method': 'vif',
                'threshold': threshold,
                'removed_features': [],
                'final_vif_scores': {},
                'reason': 'Too few numeric features for VIF analysis'
            }
        
        removed = []
        max_iterations = min(len(numeric_cols), 20)  # Safety limit
        
        for _ in range(max_iterations):
            vif_scores = self._compute_vif(X)
            if not vif_scores:
                break
            
            max_vif_col = max(vif_scores, key=vif_scores.get)
            max_vif_val = vif_scores[max_vif_col]
            
            if max_vif_val > threshold:
                X = X.drop(columns=[max_vif_col])
                removed.append({'feature': max_vif_col, 'vif': max_vif_val})
                self.log(f"  VIF: Removed '{max_vif_col}' (VIF={max_vif_val:.1f} > {threshold})")
            else:
                break
        
        # Final VIF scores
        final_vif = self._compute_vif(X)
        
        vif_info = {
            'method': 'vif',
            'threshold': threshold,
            'removed_features': removed,
            'n_removed': len(removed),
            'final_vif_scores': final_vif
        }
        
        return X, vif_info
