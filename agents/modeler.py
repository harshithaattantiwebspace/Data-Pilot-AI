# agents/modeler.py

import pandas as pd
import numpy as np
from typing import Any, Dict, List
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from rl_selector.inference import RLModelSelector
from agents.base import BaseAgent


class ModelerAgent(BaseAgent):
    """
    Agent responsible for model training and selection.
    
    This is the FOURTH agent in the pipeline. It:
      1. Asks the RL Model Selector for top-3 model recommendations
      2. Trains each recommended model with 5-fold cross-validation
      3. Creates a Voting Ensemble combining all trained models
      4. Evaluates the ensemble with cross-validation
      5. Fits the final ensemble on all data
    
    The ensemble is mandatory because combining multiple models almost always
    outperforms any single model — it reduces variance and captures different
    patterns in the data.
    
    Owner: Manohar
    """
    
    def __init__(self):
        super().__init__("ModelerAgent")
        self.rl_selector = RLModelSelector()
        self.model_classes = self._get_model_classes()
    
    def _get_model_classes(self) -> Dict:
        """
        Map model name strings to their actual scikit-learn/xgboost/lightgbm classes.
        This allows us to instantiate any model by just knowing its name.
        """
        return {
            # === Classification Models ===
            'XGBClassifier': XGBClassifier,
            'LGBMClassifier': LGBMClassifier,
            'CatBoostClassifier': CatBoostClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC,
            'KNeighborsClassifier': KNeighborsClassifier,
            'GaussianNB': GaussianNB,
            # === Regression Models ===
            'XGBRegressor': XGBRegressor,
            'LGBMRegressor': LGBMRegressor,
            'CatBoostRegressor': CatBoostRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'ElasticNet': ElasticNet,
            'SVR': SVR,
            'KNeighborsRegressor': KNeighborsRegressor,
        }
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Train models and create ensemble"""
        self.log("Starting model training...")
        
        X = state['X']
        y = state['y']
        task_type = state['task_type']
        meta_features = state['meta_features']
        
        # =====================================================================
        # Step 1: Ask RL Model Selector for top-3 recommendations
        # =====================================================================
        recommendations = self.rl_selector.recommend(meta_features, task_type, top_k=3)
        self.log(f"RL recommendations:")
        for model_name, confidence in recommendations:
            self.log(f"  → {model_name} (confidence: {confidence:.1%})")
        
        # =====================================================================
        # Step 2: Train each recommended model with 5-fold CV
        # =====================================================================
        trained_models = {}
        cv_scores = {}
        
        for model_name, confidence in recommendations:
            self.log(f"Training {model_name}...")
            try:
                model, score = self._train_model(model_name, X, y, task_type)
                trained_models[model_name] = model
                cv_scores[model_name] = score
                self.log(f"  ✓ {model_name} CV score: {score:.4f}")
            except Exception as e:
                self.log(f"  ✗ {model_name} failed: {e}")
        
        # Safety check: need at least 1 trained model
        if len(trained_models) == 0:
            self.log("All recommended models failed! Falling back to RandomForest...")
            fallback_name = 'RandomForestClassifier' if task_type == 'classification' else 'RandomForestRegressor'
            model, score = self._train_model(fallback_name, X, y, task_type)
            trained_models[fallback_name] = model
            cv_scores[fallback_name] = score
        
        # =====================================================================
        # Step 3: Create Voting Ensemble (combines all trained models)
        # =====================================================================
        self.log("Creating ensemble...")
        ensemble_model = self._create_ensemble(trained_models, task_type)
        
        # =====================================================================
        # Step 4: Evaluate ensemble with cross-validation
        # =====================================================================
        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        try:
            ensemble_cv = cross_val_score(ensemble_model, X, y, cv=5, scoring=scoring)
            ensemble_score = ensemble_cv.mean()
            self.log(f"Ensemble CV score: {ensemble_score:.4f} (±{ensemble_cv.std():.4f})")
        except Exception as e:
            self.log(f"Ensemble CV failed ({e}), using mean of individual scores")
            ensemble_score = np.mean(list(cv_scores.values()))
        
        # =====================================================================
        # Step 5: Fit final ensemble on ALL data
        # =====================================================================
        self.log("Fitting final ensemble on full dataset...")
        ensemble_model.fit(X, y)
        
        # Determine best single model
        best_model_name = max(cv_scores, key=cv_scores.get)
        best_model_score = cv_scores[best_model_name]
        
        self.log(f"\n{'='*50}")
        self.log(f"RESULTS:")
        self.log(f"  Best single model: {best_model_name} ({best_model_score:.4f})")
        self.log(f"  Ensemble score:    {ensemble_score:.4f}")
        self.log(f"  Improvement:       {ensemble_score - best_model_score:+.4f}")
        self.log(f"{'='*50}")
        
        # Update pipeline state
        state['trained_models'] = trained_models
        state['cv_scores'] = cv_scores
        state['ensemble_model'] = ensemble_model
        state['ensemble_score'] = ensemble_score
        state['best_model_name'] = best_model_name
        state['model_recommendations'] = recommendations
        state['stage'] = 'modeled'
        
        return state
    
    # =========================================================================
    # STEP 2: Train a Single Model
    # =========================================================================
    
    def _train_model(self, model_name: str, X: pd.DataFrame,
                     y: pd.Series, task_type: str) -> tuple:
        """
        Train a single model and return it with its CV score.
        
        Process:
          1. Look up the model class by name
          2. Get sensible default hyperparameters for it
          3. Run 5-fold cross-validation to get the score
          4. Fit the model on full data
          5. Return (fitted_model, cv_score)
        
        Args:
            model_name: String name of the model (e.g., 'XGBClassifier')
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
        
        Returns:
            Tuple of (fitted model, mean CV score)
        """
        model_class = self.model_classes[model_name]
        
        # Get default hyperparameters
        params = self._get_default_params(model_name)
        model = model_class(**params)
        
        # 5-fold cross-validation
        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        
        # Fit on full data
        model.fit(X, y)
        
        return model, scores.mean()
    
    def _get_default_params(self, model_name: str) -> Dict:
        """
        Get sensible default hyperparameters for each model.
        
        These are NOT the optimal hyperparameters — they're safe defaults
        that work well in most cases. Optuna hyperparameter tuning could
        be added later for further improvement.
        
        Key decisions:
          - XGBoost/LightGBM/CatBoost: 100 trees, depth 6, auto GPU
          - RandomForest/ExtraTrees:    100 trees, depth 10, parallel
          - LogReg/SVC:                 high max_iter to ensure convergence
          - All models:                 random_state=42 for reproducibility
        """
        params = {
            # --- Boosting models (GPU-enabled) ---
            'XGBClassifier': {
                'n_estimators': 100, 'max_depth': 6,
                'tree_method': 'auto', 'random_state': 42,
                'eval_metric': 'logloss', 'verbosity': 0
            },
            'XGBRegressor': {
                'n_estimators': 100, 'max_depth': 6,
                'tree_method': 'auto', 'random_state': 42,
                'verbosity': 0
            },
            'LGBMClassifier': {
                'n_estimators': 100, 'max_depth': 6,
                'random_state': 42, 'verbose': -1
            },
            'LGBMRegressor': {
                'n_estimators': 100, 'max_depth': 6,
                'random_state': 42, 'verbose': -1
            },
            'CatBoostClassifier': {
                'iterations': 100, 'depth': 6,
                'random_state': 42, 'verbose': 0
            },
            'CatBoostRegressor': {
                'iterations': 100, 'depth': 6,
                'random_state': 42, 'verbose': 0
            },
            # --- Ensemble tree models ---
            'RandomForestClassifier': {
                'n_estimators': 100, 'max_depth': 10,
                'random_state': 42, 'n_jobs': -1
            },
            'RandomForestRegressor': {
                'n_estimators': 100, 'max_depth': 10,
                'random_state': 42, 'n_jobs': -1
            },
            'ExtraTreesClassifier': {
                'n_estimators': 100, 'max_depth': 10,
                'random_state': 42, 'n_jobs': -1
            },
            'ExtraTreesRegressor': {
                'n_estimators': 100, 'max_depth': 10,
                'random_state': 42, 'n_jobs': -1
            },
            'GradientBoostingClassifier': {
                'n_estimators': 100, 'max_depth': 6,
                'random_state': 42
            },
            'GradientBoostingRegressor': {
                'n_estimators': 100, 'max_depth': 6,
                'random_state': 42
            },
            # --- Linear models ---
            'LogisticRegression': {
                'max_iter': 1000, 'random_state': 42
            },
            'Ridge': {'random_state': 42},
            'Lasso': {'random_state': 42},
            'ElasticNet': {'random_state': 42},
            # --- Distance/kernel models ---
            'SVC': {'probability': True, 'random_state': 42},
            'SVR': {},
            'KNeighborsClassifier': {'n_neighbors': 5},
            'KNeighborsRegressor': {'n_neighbors': 5},
            # --- Probabilistic models ---
            'GaussianNB': {},
        }
        return params.get(model_name, {})
    
    # =========================================================================
    # STEP 3: Create Ensemble
    # =========================================================================
    
    def _create_ensemble(self, trained_models: Dict, task_type: str):
        """
        Create a Voting Ensemble from all trained models.
        
        How Voting Ensemble works:
          - Classification (voting='soft'):
              Each model predicts class probabilities → average them → pick highest
              Example: Model A says [0.3, 0.7], Model B says [0.4, 0.6], Model C says [0.2, 0.8]
              Average = [0.3, 0.7] → predict class 1
          
          - Regression:
              Each model predicts a number → average them
              Example: Model A says 50, Model B says 55, Model C says 48
              Average = 51
        
        Why ensemble?
          - Reduces overfitting (different models make different errors)
          - More stable predictions
          - Almost always beats any single model
        
        Args:
            trained_models: Dict of {model_name: fitted_model}
            task_type: 'classification' or 'regression'
        
        Returns:
            VotingClassifier or VotingRegressor (unfitted — will be fitted later)
        """
        estimators = [(name, model) for name, model in trained_models.items()]
        
        if task_type == 'classification':
            # 'soft' voting = average predicted probabilities (better than hard voting)
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
        else:
            ensemble = VotingRegressor(estimators=estimators)
        
        self.log(f"Created {'VotingClassifier' if task_type == 'classification' else 'VotingRegressor'} "
                f"with {len(estimators)} models: {[name for name, _ in estimators]}")
        
        return ensemble
