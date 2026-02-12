"""
Modeler Agent - Model Training and Ensemble Creation with RL Selection
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import pickle
import os


# ============================================================================
# ALL CLASSIFICATION MODELS (10)
# ============================================================================
CLASSIFICATION_MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'GaussianNB': GaussianNB(),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
    'SVC': SVC(kernel='rbf', probability=True, random_state=42),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
}

# ============================================================================
# ALL REGRESSION MODELS (11)
# ============================================================================
REGRESSION_MODELS = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0),
    'SVR': SVR(kernel='rbf'),
    'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=5),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
}


class ModelerAgent:
    """Trains all models and creates ensemble with optional RL selection."""
    
    def __init__(self):
        self.trained_models = {}
        self.ensemble = None
        self.rl_model = None
        self.model_scores = {}
        self.use_rl = False
        self._load_rl_model()
    
    def _load_rl_model(self):
        """Try to load pre-trained RL model."""
        try:
            # Try to load RL models
            for task_type in ['classification', 'regression']:
                model_path = f'rl_model_selector_{task_type}.pkl'
                if os.path.exists(model_path):
                    try:
                        from stable_baselines3 import PPO
                        self.rl_model = PPO.load(model_path)
                        self.use_rl = True
                        print(f"[OK] Loaded RL model from {model_path}")
                        break
                    except:
                        pass
        except:
            pass
    
    def _get_rl_recommendations(self, meta_features: np.ndarray, task_type: str) -> List[str]:
        """Get model recommendations from RL agent."""
        if not self.use_rl or self.rl_model is None:
            return self._get_default_recommendations(task_type)
        
        try:
            # Get RL action
            action, _ = self.rl_model.predict(meta_features, deterministic=True)
            
            if task_type == 'classification':
                models = list(CLASSIFICATION_MODELS.keys())
            else:
                models = list(REGRESSION_MODELS.keys())
            
            # Get top 3 models (action + neighbors)
            top_models = []
            for i in range(3):
                idx = (action + i) % len(models)
                top_models.append(models[idx])
            
            return top_models
        except:
            return self._get_default_recommendations(task_type)
    
    def _get_default_recommendations(self, task_type: str) -> List[str]:
        """Get default model recommendations."""
        if task_type == 'classification':
            return ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression']
        else:
            return ['RandomForestRegressor', 'GradientBoostingRegressor', 'Ridge']
    
    def train(self, X: pd.DataFrame, y: pd.Series, task_type: str, 
              meta_features: np.ndarray = None) -> Tuple[Any, np.ndarray, Dict]:
        """
        Train all models and create ensemble.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
            meta_features: 32 meta-features for RL selection
            
        Returns:
            (ensemble_model, y_pred, modeling_report)
        """
        
        # Select models based on RL or defaults
        if meta_features is not None:
            recommended = self._get_rl_recommendations(meta_features, task_type)
        else:
            recommended = self._get_default_recommendations(task_type)
        
        # Get all models
        if task_type == 'classification':
            all_models = CLASSIFICATION_MODELS
        else:
            all_models = REGRESSION_MODELS
        
        # Train all models
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        print(f"[*] Training {len(all_models)} {task_type} models...")
        
        for model_name, model in all_models.items():
            try:
                if task_type == 'classification':
                    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                
                self.model_scores[model_name] = scores.mean()
                model.fit(X, y)
                self.trained_models[model_name] = model
                print(f"[OK] {model_name}: CV Score = {scores.mean():.4f}")
            except Exception as e:
                print(f"[ERROR] {model_name}: {str(e)}")
                self.model_scores[model_name] = 0.0
        
        # Create ensemble from top 3 models
        top_3 = sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_models = {name: self.trained_models[name] for name, _ in top_3}
        
        print(f"\n[*] Creating ensemble from top 3 models:")
        for name, score in top_3:
            print(f"    • {name}: {score:.4f}")
        
        if task_type == 'classification':
            self.ensemble = VotingClassifier(
                estimators=list(top_3_models.items()),
                voting='soft'
            )
        else:
            self.ensemble = VotingRegressor(
                estimators=list(top_3_models.items())
            )
        
        self.ensemble.fit(X, y)
        y_pred = self.ensemble.predict(X)
        
        # Compute metrics
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred),
            }
        
        report = {
            'all_models_trained': list(self.trained_models.keys()),
            'model_scores': self.model_scores,
            'top_3_models': [name for name, _ in top_3],
            'ensemble_score': top_3[0][1],
            'metrics': metrics,
            'used_rl': self.use_rl,
        }
        
        return self.ensemble, y_pred, report
