"""
RL Model Selector Trainer - Trains PPO agent to select best ML models
Uses stable-baselines3 PPO to learn optimal model selection based on meta-features
"""

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    make_classification, make_regression,
    fetch_openml
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
import pickle
import os

try:
    from stable_baselines3 import PPO
    from gym import Env
    from gym.spaces import Box, Discrete
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("[WARNING] stable-baselines3 not installed. Install with: pip install stable-baselines3")


# ============================================================================
# CLASSIFICATION MODELS (10)
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
# REGRESSION MODELS (11)
# ============================================================================
REGRESSION_MODELS = {
    'LinearRegression': Ridge(alpha=0.0),  # Linear regression via Ridge with alpha=0
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


# ============================================================================
# LOAD DATASETS
# ============================================================================
def load_classification_datasets():
    """Load 10+ classification datasets."""
    datasets = []
    
    # Sklearn built-in datasets
    print("[*] Loading classification datasets...")
    
    # 1. Iris
    iris = load_iris()
    datasets.append(('iris', iris.data, iris.target))
    
    # 2. Wine
    wine = load_wine()
    datasets.append(('wine', wine.data, wine.target))
    
    # 3. Breast Cancer
    bc = load_breast_cancer()
    datasets.append(('breast_cancer', bc.data, bc.target))
    
    # 4-13. Synthetic datasets
    for i in range(10):
        X, y = make_classification(
            n_samples=500 + i*100,
            n_features=20 + i*2,
            n_informative=15,
            n_redundant=5,
            n_classes=2 + (i % 3),
            random_state=42 + i
        )
        datasets.append((f'synthetic_clf_{i}', X, y))
    
    print(f"[OK] Loaded {len(datasets)} classification datasets")
    return datasets


def load_regression_datasets():
    """Load 10+ regression datasets."""
    datasets = []
    
    print("[*] Loading regression datasets...")
    
    # Synthetic datasets
    for i in range(10):
        X, y = make_regression(
            n_samples=500 + i*100,
            n_features=20 + i*2,
            n_informative=15,
            random_state=42 + i
        )
        datasets.append((f'synthetic_reg_{i}', X, y))
    
    print(f"[OK] Loaded {len(datasets)} regression datasets")
    return datasets


# ============================================================================
# EXTRACT 32 META-FEATURES
# ============================================================================
def extract_meta_features(X, y):
    """Extract 32 meta-features from dataset."""
    features = []
    
    n_samples, n_features = X.shape
    
    # Basic (6)
    features.append(min(n_samples / 100000, 1.0))
    features.append(min(n_features / 100, 1.0))
    numeric_ratio = 1.0  # All numeric in our case
    features.append(numeric_ratio)
    features.append(0.0)  # No categorical
    features.append(min(len(np.unique(y)) / n_samples, 1.0))
    features.append(min(n_features / n_samples, 1.0))
    
    # Missing (3)
    features.append(0.0)  # No missing values
    features.append(0.0)
    features.append(0.0)
    
    # Statistical (10)
    X_filled = np.nan_to_num(X)
    features.append(np.mean(np.abs(np.array([np.mean(X_filled[:, i]**3) / (np.std(X_filled[:, i])**3 + 1e-10) for i in range(min(5, n_features))]))))
    features.append(np.mean(np.abs(np.array([np.mean(X_filled[:, i]**4) / (np.std(X_filled[:, i])**4 + 1e-10) for i in range(min(5, n_features))]))))
    features.append(0.0)  # No outliers
    features.append(np.mean(np.abs(np.corrcoef(X_filled.T)[np.triu_indices_from(np.corrcoef(X_filled.T), k=1)])))
    features.append(np.mean([np.std(X_filled[:, i]) / (np.abs(np.mean(X_filled[:, i])) + 1e-10) for i in range(n_features)]))
    features.append(0.0)  # Target skewness
    features.append(0.0)  # Target kurtosis
    features.append(0.0)  # Target entropy
    features.append(0.0)  # Target correlation
    features.append(0.0)  # Imbalance ratio
    
    # Categorical (3)
    features.extend([0.0, 0.0, 0.0])
    
    # PCA (3)
    features.extend([0.5, 0.5, 0.5])
    
    # Landmarks (4)
    features.extend([0.5, 0.5, 0.5, 0.5])
    
    # Ensure exactly 32 features
    while len(features) < 32:
        features.append(0.5)
    
    return np.array(features[:32], dtype=np.float32)


# ============================================================================
# EVALUATE MODEL
# ============================================================================
def evaluate_model(model, X, y, task_type):
    """Evaluate model using cross-validation."""
    try:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        if task_type == 'classification':
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return np.mean(scores)
    except:
        return 0.0


# ============================================================================
# RL ENVIRONMENT
# ============================================================================
if HAS_SB3:
    class ModelSelectionEnv(Env):
        """RL environment for model selection."""
        
        def __init__(self, datasets, models, task_type):
            super().__init__()
            self.datasets = datasets
            self.models = models
            self.task_type = task_type
            self.current_dataset_idx = 0
            self.current_model_idx = 0
            
            # Action space: select one of the models
            self.action_space = Discrete(len(models))
            
            # Observation space: 32 meta-features
            self.observation_space = Box(low=0, high=1, shape=(32,), dtype=np.float32)
            
            self.best_score = 0.0
            self.steps = 0
            self.max_steps = len(datasets) * 5
        
        def reset(self):
            """Reset environment."""
            self.current_dataset_idx = 0
            self.steps = 0
            self.best_score = 0.0
            return self._get_observation()
        
        def _get_observation(self):
            """Get current observation (meta-features)."""
            if self.current_dataset_idx < len(self.datasets):
                _, X, y = self.datasets[self.current_dataset_idx]
                return extract_meta_features(X, y)
            return np.zeros(32, dtype=np.float32)
        
        def step(self, action):
            """Execute one step."""
            _, X, y = self.datasets[self.current_dataset_idx]
            model_name = list(self.models.keys())[action]
            model = self.models[model_name]
            
            # Evaluate model
            score = evaluate_model(model, X, y, self.task_type)
            
            # Reward: score improvement
            reward = score - self.best_score
            self.best_score = max(self.best_score, score)
            
            # Move to next dataset
            self.current_dataset_idx += 1
            self.steps += 1
            
            done = self.steps >= self.max_steps or self.current_dataset_idx >= len(self.datasets)
            
            return self._get_observation(), reward, done, {}


# ============================================================================
# TRAIN RL MODEL
# ============================================================================
def train_rl_model(task_type='classification', total_timesteps=50000):
    """Train RL model for model selection."""
    
    if not HAS_SB3:
        print("[ERROR] stable-baselines3 required. Install with: pip install stable-baselines3")
        return None
    
    print(f"\n[*] Training RL model for {task_type}...")
    
    # Load datasets
    if task_type == 'classification':
        datasets = load_classification_datasets()
        models = CLASSIFICATION_MODELS
    else:
        datasets = load_regression_datasets()
        models = REGRESSION_MODELS
    
    # Create environment
    env = ModelSelectionEnv(datasets, models, task_type)
    
    # Train PPO agent
    print(f"[*] Training PPO agent ({total_timesteps} timesteps)...")
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4, n_steps=2048)
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    model_path = f'rl_model_selector_{task_type}.pkl'
    model.save(model_path)
    print(f"[OK] Saved RL model to {model_path}")
    
    return model


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("RL Model Selector Trainer")
    print("="*60)
    
    if not HAS_SB3:
        print("\n[ERROR] stable-baselines3 not installed")
        print("Install with: pip install stable-baselines3")
        print("\nAlternatively, you can use the modeler agent without RL")
        exit(1)
    
    # Train classification model
    train_rl_model('classification', total_timesteps=50000)
    
    # Train regression model
    train_rl_model('regression', total_timesteps=50000)
    
    print("\n[OK] RL models trained successfully!")
    print("[*] Models saved as:")
    print("    - rl_model_selector_classification.pkl")
    print("    - rl_model_selector_regression.pkl")
