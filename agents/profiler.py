# agents/profiler.py

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from agents.base import BaseAgent


class ProfilerAgent(BaseAgent):
    """
    Agent responsible for data profiling and meta-feature extraction.
    
    This is the FIRST agent in the pipeline. It analyzes the uploaded CSV and produces:
      1. Column type detection (numeric, categorical, datetime, text, id, binary)
      2. Auto-detection of target column (if user didn't specify)
      3. Task type determination (classification vs regression)
      4. Per-column statistics (mean, std, skewness, missing %, etc.)
      5. 32 meta-features for the RL Model Selector
      6. Data quality score (0-100)
      7. Warnings about potential data issues
    
    Owner: Harshitha
    """
    
    def __init__(self):
        super().__init__("ProfilerAgent")
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Profile the dataset and extract meta-features"""
        self.log("Starting data profiling...")
        
        df = state['raw_data']
        target_col = state.get('target_column')
        data_context = state.get('data_context', {})

        # Step 1: Detect column types
        column_types = self._detect_column_types(df)
        self.log(f"Detected column types: {column_types}")

        # Step 2: Auto-detect target if not specified
        # LLM context already set target_column in state during context phase,
        # so this fallback only runs if context phase failed or returned nothing.
        if not target_col:
            target_col = self._detect_target(df, column_types)
            self.log(f"Auto-detected target column: {target_col}")
        else:
            self.log(f"Using target column: {target_col}")

        # Step 3: Determine task type (classification or regression)
        task_type = self._determine_task_type(df, target_col, column_types)
        # Override with LLM-suggested task type if valid
        llm_task_type = data_context.get('task_type', '').lower()
        if llm_task_type in ('classification', 'regression') and llm_task_type != task_type:
            self.log(f"Task type overridden by LLM context: {task_type} → {llm_task_type}")
            task_type = llm_task_type
        self.log(f"Task type: {task_type}")
        
        # Step 4: Compute per-column statistics
        statistics = self._compute_statistics(df, column_types)
        
        # Step 5: Extract 32 meta-features for RL Model Selector
        meta_features = self._extract_meta_features(df, target_col, task_type, column_types)
        self.log(f"Extracted {len(meta_features)} meta-features")
        
        # Step 6: Compute data quality score
        quality_score = self._compute_quality_score(df)
        self.log(f"Data quality score: {quality_score}/100")
        
        # Step 7: Generate warnings about data issues
        warnings = self._generate_warnings(df, column_types)
        
        # Step 8: describe()-based anomaly detection (like a real DS)
        describe_anomalies = self._describe_anomaly_detection(df, column_types)
        if describe_anomalies:
            self.log(f"Found {len(describe_anomalies)} anomalies from describe() analysis")
        
        # Step 9: Check unique value uniformity (non-uniform categories)
        uniformity_issues = self._check_unique_value_uniformity(df, column_types)
        if uniformity_issues:
            self.log(f"Found non-uniform categories in {len(uniformity_issues)} columns")
        
        # Update pipeline state with all profiling results
        state['target_column'] = target_col
        state['task_type'] = task_type
        state['profile_report'] = {
            'column_types': column_types,
            'statistics': statistics,
            'quality_score': quality_score,
            'warnings': warnings,
            'describe_anomalies': describe_anomalies,
            'uniformity_issues': uniformity_issues,
            'n_rows': len(df),
            'n_cols': len(df.columns)
        }
        state['meta_features'] = meta_features
        state['stage'] = 'profiled'
        
        return state
    
    # =========================================================================
    # STEP 1: Column Type Detection
    # =========================================================================
    
    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect the semantic type of each column.
        
        Types detected:
          - 'numeric'     : continuous numbers (int/float with many unique values)
          - 'categorical' : discrete categories (strings or low-cardinality numbers)
          - 'binary'      : boolean columns
          - 'datetime'    : date/time columns
          - 'text'        : free-text columns (high avg word count)
          - 'id'          : identifier columns (nearly all unique values)
        """
        column_types = {}
        
        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            n_total = len(df[col].dropna())
            unique_ratio = n_unique / n_total if n_total > 0 else 0
            
            # Check if datetime
            if dtype == 'datetime64[ns]':
                column_types[col] = 'datetime'
            # Check if numeric
            elif dtype in ['int64', 'float64']:
                # Could be categorical if very few unique values relative to dataset
                if n_unique <= 20 and unique_ratio < 0.05:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numeric'
            # Check if boolean
            elif dtype == 'bool':
                column_types[col] = 'binary'
            # String/Object columns
            else:
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(100))
                    column_types[col] = 'datetime'
                except:
                    # Check if ID column (almost all values unique)
                    if unique_ratio > 0.95:
                        column_types[col] = 'id'
                    # Check if text (high average word count per cell)
                    elif df[col].dropna().astype(str).str.split().str.len().mean() > 5:
                        column_types[col] = 'text'
                    else:
                        column_types[col] = 'categorical'
        
        return column_types
    
    # =========================================================================
    # STEP 2: Auto-detect Target Column
    # =========================================================================
    
    def _detect_target(self, df: pd.DataFrame, column_types: Dict[str, str]) -> str:
        """
        Auto-detect the target column using heuristics:
          Priority 1: Column named 'target', 'label', 'y', 'class', 'outcome'
          Priority 2: Last column in the dataframe
          Priority 3: Column with lowest cardinality (for classification)
          Default:    Last numeric column
        """
        # Priority 1: Common target column names
        target_names = ['target', 'label', 'y', 'class', 'outcome']
        for name in target_names:
            if name in df.columns:
                return name
            # Case-insensitive match
            if name.lower() in [c.lower() for c in df.columns]:
                return [c for c in df.columns if c.lower() == name.lower()][0]
        
        # Priority 2: Last column (very common convention)
        last_col = df.columns[-1]
        if column_types.get(last_col) in ['categorical', 'binary', 'numeric']:
            return last_col
        
        # Priority 3: Column with lowest cardinality among categoricals
        categorical_cols = [c for c, t in column_types.items() if t in ['categorical', 'binary']]
        if categorical_cols:
            return min(categorical_cols, key=lambda c: df[c].nunique())
        
        # Default: last numeric column
        numeric_cols = [c for c, t in column_types.items() if t == 'numeric']
        return numeric_cols[-1] if numeric_cols else df.columns[-1]
    
    # =========================================================================
    # STEP 3: Determine Task Type
    # =========================================================================
    
    def _determine_task_type(self, df: pd.DataFrame, target_col: str,
                             column_types: Dict[str, str]) -> str:
        """
        Determine if the task is classification or regression.
        
        Rules:
          - If target is categorical/binary → classification
          - If target is numeric with ≤20 unique values → classification
          - If target is numeric with >20 unique values → regression
        """
        if column_types.get(target_col) in ['categorical', 'binary']:
            return 'classification'
        
        # Numeric target — check if discrete or continuous
        n_unique = df[target_col].nunique()
        if n_unique <= 20:
            return 'classification'
        return 'regression'
    
    # =========================================================================
    # STEP 4: Compute Per-Column Statistics
    # =========================================================================
    
    def _compute_statistics(self, df: pd.DataFrame,
                           column_types: Dict[str, str]) -> Dict[str, Dict]:
        """
        Compute descriptive statistics for every column.
        
        Numeric columns get: mean, std, min, max, median, skewness, kurtosis
        Categorical columns get: top values, cardinality
        All columns get: missing count, missing %, unique count
        """
        statistics = {}
        
        for col in df.columns:
            col_type = column_types.get(col, 'unknown')
            stats_dict = {
                'type': col_type,
                'missing_count': int(df[col].isna().sum()),
                'missing_pct': round(df[col].isna().mean() * 100, 2),
                'n_unique': int(df[col].nunique())
            }
            
            if col_type == 'numeric':
                stats_dict.update({
                    'mean': round(float(df[col].mean()), 4),
                    'std': round(float(df[col].std()), 4),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'skewness': round(float(df[col].skew()), 4),
                    'kurtosis': round(float(df[col].kurtosis()), 4)
                })
            elif col_type == 'categorical':
                value_counts = df[col].value_counts()
                stats_dict.update({
                    'top_values': value_counts.head(5).to_dict(),
                    'cardinality': int(df[col].nunique())
                })
            
            statistics[col] = stats_dict
        
        return statistics
    
    # =========================================================================
    # STEP 5: Extract 32 Meta-Features for RL Model Selector
    # =========================================================================
    
    def _extract_meta_features(self, df: pd.DataFrame, target_col: str,
                               task_type: str, column_types: Dict[str, str]) -> np.ndarray:
        """
        Extract exactly 32 normalized meta-features that describe the dataset.
        These are fed into the PPO RL agent to select the best ML model.
        
        Feature groups:
          [0-5]   Basic features (size, dimensionality)
          [6-8]   Missing value features
          [9-18]  Statistical features (skew, kurtosis, outliers, correlation)
          [19-21] Categorical features (cardinality)
          [22-24] Target features (imbalance, skew, kurtosis)
          [25-27] PCA features (intrinsic dimensionality)
          [28-31] Landmark features (quick model scores)
        """
        features = []
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Get numeric and categorical columns
        numeric_cols = [c for c in X.columns if column_types.get(c) == 'numeric']
        categorical_cols = [c for c in X.columns if column_types.get(c) == 'categorical']
        
        X_numeric = X[numeric_cols].fillna(X[numeric_cols].median()) if numeric_cols else pd.DataFrame()
        
        # === BASIC FEATURES (6) — indices [0-5] ===
        n_samples = len(df)
        n_features = len(X.columns)
        
        features.append(np.log10(n_samples + 1) / 6)           # [0] n_samples (log-normalized)
        features.append(np.log10(n_features + 1) / 4)          # [1] n_features (log-normalized)
        features.append(len(numeric_cols) / max(n_features, 1)) # [2] numeric ratio
        features.append(len(categorical_cols) / max(n_features, 1))  # [3] categorical ratio
        
        if task_type == 'classification':
            features.append(y.nunique() / 100)                  # [4] n_classes (normalized)
        else:
            features.append(0)
        
        features.append(n_features / max(n_samples, 1))        # [5] dimensionality ratio
        
        # === MISSING VALUE FEATURES (3) — indices [6-8] ===
        missing_ratio = df.isna().sum().sum() / (n_samples * len(df.columns))
        cols_with_missing = (df.isna().sum() > 0).sum() / len(df.columns)
        max_missing = df.isna().mean().max()
        
        features.append(missing_ratio)                          # [6] overall missing ratio
        features.append(cols_with_missing)                      # [7] fraction of cols with missing
        features.append(max_missing)                            # [8] worst column missing ratio
        
        # === STATISTICAL FEATURES (10) — indices [9-18] ===
        if len(X_numeric.columns) > 0:
            skewness = X_numeric.skew()
            kurtosis = X_numeric.kurtosis()
            
            features.append(np.clip(skewness.mean(), -10, 10) / 10)     # [9] mean skewness
            features.append(np.clip(kurtosis.mean(), -100, 100) / 100)  # [10] mean kurtosis
            
            # Outlier ratio (IQR method)
            Q1 = X_numeric.quantile(0.25)
            Q3 = X_numeric.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((X_numeric < Q1 - 1.5 * IQR) | (X_numeric > Q3 + 1.5 * IQR)).sum().sum()
            features.append(outliers / (n_samples * len(X_numeric.columns)))  # [11] outlier ratio
            
            # Mean absolute correlation
            if len(X_numeric.columns) > 1:
                corr = X_numeric.corr().abs()
                features.append(corr.values[np.triu_indices_from(corr.values, 1)].mean())  # [12]
            else:
                features.append(0)
            
            # Coefficient of variation
            cv = X_numeric.std() / (X_numeric.mean().abs() + 1e-10)
            features.append(np.clip(cv.mean(), 0, 10) / 10)    # [13] cv mean
            features.append(np.clip(cv.std(), 0, 10) / 10)     # [14] cv std
            
            features.append(np.clip(skewness.std(), 0, 10) / 10)       # [15] skew std
            features.append(np.clip(kurtosis.std(), 0, 100) / 100)     # [16] kurtosis std
            
            # Range ratio
            ranges = (X_numeric.max() - X_numeric.min()) / (X_numeric.std() + 1e-10)
            features.append(np.clip(ranges.mean(), 0, 100) / 100)      # [17] range ratio
            
            # Zero ratio
            features.append((X_numeric == 0).sum().sum() / (n_samples * len(X_numeric.columns)))  # [18]
        else:
            features.extend([0] * 10)
        
        # === CATEGORICAL FEATURES (3) — indices [19-21] ===
        if len(categorical_cols) > 0:
            cardinalities = [df[c].nunique() for c in categorical_cols]
            features.append(np.mean(cardinalities) / 100)               # [19] mean cardinality
            features.append(np.max(cardinalities) / 1000)               # [20] max cardinality
            features.append(sum(1 for c in cardinalities if c > 20) / len(categorical_cols))  # [21]
        else:
            features.extend([0, 0, 0])
        
        # === TARGET FEATURES (3) — indices [22-24] ===
        if task_type == 'classification':
            value_counts = y.value_counts(normalize=True)
            features.append(1 - value_counts.max())                     # [22] class imbalance
            features.append(0)                                          # [23] n/a for classification
            features.append(0)                                          # [24] n/a for classification
        else:
            y_numeric = pd.to_numeric(y, errors='coerce').dropna()
            if len(y_numeric) > 0:
                features.append(0)                                      # [22] n/a for regression
                features.append(np.clip(y_numeric.skew(), -10, 10) / 10)    # [23] target skewness
                features.append(np.clip(y_numeric.kurtosis(), -100, 100) / 100)  # [24] target kurtosis
            else:
                features.extend([0, 0, 0])
        
        # === PCA FEATURES (3) — indices [25-27] ===
        if len(X_numeric.columns) > 1:
            try:
                X_scaled = (X_numeric - X_numeric.mean()) / (X_numeric.std() + 1e-10)
                X_scaled = X_scaled.fillna(0)
                
                pca = PCA()
                pca.fit(X_scaled)
                
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                n_95 = np.argmax(cumvar >= 0.95) + 1
                var_50 = cumvar[min(len(cumvar) - 1, int(len(cumvar) * 0.5))]
                
                features.append(n_95 / len(X_numeric.columns))         # [25] components for 95% var
                features.append(var_50)                                 # [26] variance at 50% components
                features.append(n_95 / max(n_samples, 1))              # [27] intrinsic dimensionality
            except:
                features.extend([0.5, 0.5, 0.01])
        else:
            features.extend([1.0, 1.0, 0.01])
        
        # === LANDMARK FEATURES (4) — indices [28-31] ===
        landmarks = self._compute_landmarks(X_numeric, y, task_type)
        features.extend(landmarks)
        
        # Ensure exactly 32 features
        features = features[:32]
        while len(features) < 32:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_landmarks(self, X: pd.DataFrame, y: pd.Series,
                          task_type: str) -> List[float]:
        """
        Compute "landmark" scores — quick cross-validation of simple models.
        These tell the RL agent how the dataset responds to different model families:
          [0] Decision Tree score
          [1] Naive Bayes / Linear Regression score
          [2] Logistic Regression / Ridge score
          [3] KNN score
        """
        if len(X.columns) == 0 or len(X) < 50:
            return [0.5, 0.5, 0.5, 0.5]
        
        landmarks = []
        
        # Sample data for speed (max 1000 rows)
        sample_size = min(1000, len(X))
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[idx].values
        y_sample = y.iloc[idx]
        
        # Encode target if classification
        if task_type == 'classification':
            le = LabelEncoder()
            y_sample = le.fit_transform(y_sample.astype(str))
        else:
            y_sample = pd.to_numeric(y_sample, errors='coerce').fillna(0).values
        
        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        
        # Landmark 1: Decision Tree (max_depth=3 for speed)
        try:
            if task_type == 'classification':
                dt = DecisionTreeClassifier(max_depth=3, random_state=42)
            else:
                dt = DecisionTreeRegressor(max_depth=3, random_state=42)
            score = cross_val_score(dt, X_sample, y_sample, cv=3, scoring=scoring)
            landmarks.append(np.clip(score.mean(), 0, 1))
        except:
            landmarks.append(0.5)
        
        # Landmark 2: Naive Bayes (classification) / Linear Regression (regression)
        try:
            if task_type == 'classification':
                model = GaussianNB()
                score = cross_val_score(model, X_sample, y_sample, cv=3, scoring='accuracy')
            else:
                model = LinearRegression()
                score = cross_val_score(model, X_sample, y_sample, cv=3, scoring='r2')
            landmarks.append(np.clip(score.mean(), 0, 1))
        except:
            landmarks.append(0.5)
        
        # Landmark 3: Logistic Regression (classification) / Ridge (regression)
        try:
            if task_type == 'classification':
                model = LogisticRegression(max_iter=100, random_state=42)
                score = cross_val_score(model, X_sample, y_sample, cv=3, scoring='accuracy')
            else:
                model = Ridge(random_state=42)
                score = cross_val_score(model, X_sample, y_sample, cv=3, scoring='r2')
            landmarks.append(np.clip(score.mean(), 0, 1))
        except:
            landmarks.append(0.5)
        
        # Landmark 4: K-Nearest Neighbors
        try:
            if task_type == 'classification':
                model = KNeighborsClassifier(n_neighbors=3)
                score = cross_val_score(model, X_sample, y_sample, cv=3, scoring='accuracy')
            else:
                model = KNeighborsRegressor(n_neighbors=3)
                score = cross_val_score(model, X_sample, y_sample, cv=3, scoring='r2')
            landmarks.append(np.clip(score.mean(), 0, 1))
        except:
            landmarks.append(0.5)
        
        return landmarks
    
    # =========================================================================
    # STEP 6: Data Quality Score
    # =========================================================================
    
    def _compute_quality_score(self, df: pd.DataFrame) -> int:
        """
        Compute an overall data quality score from 0 to 100.
        
        Deductions:
          - Missing values:        up to -30 points
          - Duplicate rows:        up to -20 points
          - Constant columns:      up to -20 points
          - High cardinality cols: -5 points each
        """
        score = 100
        
        # Penalize missing values (up to -30)
        missing_ratio = df.isna().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 30
        
        # Penalize duplicate rows (up to -20)
        dup_ratio = df.duplicated().sum() / len(df)
        score -= dup_ratio * 20
        
        # Penalize constant columns (up to -20)
        constant_cols = sum(1 for c in df.columns if df[c].nunique() == 1)
        score -= (constant_cols / len(df.columns)) * 20
        
        # Penalize very high cardinality string columns (-5 each)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) > 0.9:
                score -= 5
        
        return max(0, min(100, int(score)))
    
    # =========================================================================
    # STEP 7: Generate Warnings
    # =========================================================================
    
    def _generate_warnings(self, df: pd.DataFrame,
                          column_types: Dict[str, str]) -> List[str]:
        """
        Generate human-readable warnings about potential data issues.
        These are displayed in the UI to help users understand their data.
        """
        warnings = []
        
        # Missing values warnings
        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            if missing_pct > 50:
                warnings.append(f"⚠️ Column '{col}' has {missing_pct:.1f}% missing values — consider dropping")
            elif missing_pct > 20:
                warnings.append(f"⚠️ Column '{col}' has {missing_pct:.1f}% missing values — will be imputed")
        
        # High cardinality warnings
        for col, ctype in column_types.items():
            if ctype == 'categorical' and df[col].nunique() > 100:
                warnings.append(f"⚠️ Column '{col}' has high cardinality ({df[col].nunique()} unique values)")
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                warnings.append(f"⚠️ Column '{col}' is constant (only 1 unique value) — will be dropped")
        
        # ID columns
        for col, ctype in column_types.items():
            if ctype == 'id':
                warnings.append(f"ℹ️ Column '{col}' appears to be an ID column — will be excluded from modeling")
        
        return warnings
    
    # =========================================================================
    # STEP 8: describe()-Based Anomaly Detection
    # =========================================================================
    
    def _describe_anomaly_detection(self, df: pd.DataFrame,
                                     column_types: Dict[str, str]) -> List[Dict]:
        """
        Like a real data scientist using df.describe() to spot unusual patterns.
        
        Checks:
          - max >> 75th percentile (extreme outliers)
          - min << 25th percentile (extreme low values)
          - mean far from median (heavy skew)
          - std >> mean (high coefficient of variation)
          - 50% of data = same value (near-constant)
        """
        anomalies = []
        
        numeric_cols = [c for c in df.columns if column_types.get(c) == 'numeric']
        if not numeric_cols:
            return anomalies
        
        desc = df[numeric_cols].describe()
        
        for col in desc.columns:
            max_val = desc.loc['max', col]
            min_val = desc.loc['min', col]
            q75 = desc.loc['75%', col]
            q25 = desc.loc['25%', col]
            mean_val = desc.loc['mean', col]
            std_val = desc.loc['std', col]
            median_val = desc.loc['50%', col]
            iqr = q75 - q25
            
            # Check: max >> 75th percentile (extreme outlier on the high end)
            if iqr > 0 and max_val > q75 + 3 * iqr:
                anomalies.append({
                    'column': col,
                    'issue': 'extreme_high_outlier',
                    'detail': f"max ({max_val:.2f}) is far above Q3 + 3×IQR ({q75 + 3*iqr:.2f})",
                    'severity': 'high'
                })
            
            # Check: min << 25th percentile (extreme outlier on the low end)
            if iqr > 0 and min_val < q25 - 3 * iqr:
                anomalies.append({
                    'column': col,
                    'issue': 'extreme_low_outlier',
                    'detail': f"min ({min_val:.2f}) is far below Q1 - 3×IQR ({q25 - 3*iqr:.2f})",
                    'severity': 'high'
                })
            
            # Check: mean very far from median (heavy skew)
            if std_val > 0 and abs(mean_val - median_val) > 2 * std_val:
                anomalies.append({
                    'column': col,
                    'issue': 'heavy_skew',
                    'detail': f"mean ({mean_val:.2f}) far from median ({median_val:.2f}) — data is heavily skewed",
                    'severity': 'medium'
                })
            
            # Check: coefficient of variation > 2 (very high variability)
            if mean_val != 0 and abs(std_val / mean_val) > 2:
                cv = abs(std_val / mean_val)
                anomalies.append({
                    'column': col,
                    'issue': 'high_variability',
                    'detail': f"coefficient of variation = {cv:.1f} — very dispersed data",
                    'severity': 'medium'
                })
            
            # Check: >50% of values are identical (near-constant)
            mode_count = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            if mode_count / len(df) > 0.5 and df[col].nunique() > 1:
                pct = mode_count / len(df) * 100
                anomalies.append({
                    'column': col,
                    'issue': 'dominant_value',
                    'detail': f"{pct:.0f}% of rows have the same value — near-constant column",
                    'severity': 'low'
                })
        
        return anomalies
    
    # =========================================================================
    # STEP 9: Unique Value Uniformity Check
    # =========================================================================
    
    def _check_unique_value_uniformity(self, df: pd.DataFrame,
                                        column_types: Dict[str, str]) -> Dict[str, Dict]:
        """
        Check for non-uniform category values that should be standardized.
        
        Detects issues like:
          - Case inconsistency: 'Male', 'male', 'MALE' → should all be 'Male'
          - Whitespace: ' Male', 'Male ', ' Male ' → should all be 'Male'
          - Abbreviations: 'M', 'F' mixed with 'Male', 'Female'
        
        Returns: {column_name: {canonical_lower: [variant1, variant2, ...]}}
        """
        uniformity_issues = {}
        
        for col in df.columns:
            if column_types.get(col) != 'categorical':
                continue
            
            values = df[col].dropna().unique()
            if len(values) > 50:  # Skip high-cardinality columns
                continue
            
            # Check for case/whitespace inconsistency
            lower_map = {}
            for v in values:
                v_str = str(v).strip().lower()
                if v_str not in lower_map:
                    lower_map[v_str] = []
                lower_map[v_str].append(str(v))
            
            issues = {k: v for k, v in lower_map.items() if len(v) > 1}
            if issues:
                uniformity_issues[col] = issues
        
        return uniformity_issues
