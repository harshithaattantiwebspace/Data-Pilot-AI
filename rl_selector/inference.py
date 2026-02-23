# rl_selector/inference.py

import numpy as np
from stable_baselines3 import PPO
from typing import List, Tuple
from utils.config import config


class RLModelSelector:
    """
    RL-based model selection at inference time.
    
    This class loads the pre-trained PPO models and uses them to recommend
    the best ML models for a new dataset, based on its 32 meta-features.
    
    How it works:
      1. Takes 32 meta-features from the ProfilerAgent
      2. Feeds them into the trained PPO policy network
      3. The policy outputs a probability distribution over all models
      4. Returns top-k models sorted by probability (confidence)
    
    Fallback: If no trained PPO model exists, returns default top-3 models
    (XGBoost, LightGBM, RandomForest — generally strong choices).
    
    Owner: Manohar
    """
    
    CLASSIFICATION_MODELS = [
        'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier',
        'RandomForestClassifier', 'ExtraTreesClassifier',
        'GradientBoostingClassifier', 'LogisticRegression',
        'SVC', 'KNeighborsClassifier', 'GaussianNB'
    ]
    
    REGRESSION_MODELS = [
        'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
        'RandomForestRegressor', 'ExtraTreesRegressor',
        'GradientBoostingRegressor', 'Ridge', 'Lasso',
        'ElasticNet', 'SVR', 'KNeighborsRegressor'
    ]
    
    def __init__(self):
        self.clf_model = None
        self.reg_model = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained PPO models from disk (if they exist)"""
        try:
            clf_path = f"{config.PPO_MODEL_PATH}/ppo_clf.zip"
            self.clf_model = PPO.load(clf_path, device="auto")
            print("[RLModelSelector] Loaded classification PPO model")
        except Exception as e:
            print(f"[RLModelSelector] Classification PPO model not found — will use defaults. ({e})")
        
        try:
            reg_path = f"{config.PPO_MODEL_PATH}/ppo_reg.zip"
            self.reg_model = PPO.load(reg_path, device="auto")
            print("[RLModelSelector] Loaded regression PPO model")
        except Exception as e:
            print(f"[RLModelSelector] Regression PPO model not found — will use defaults. ({e})")
    
    def recommend(self, meta_features: np.ndarray,
                  task_type: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k model recommendations with confidence scores.
        
        Args:
            meta_features: 32-element numpy array of dataset meta-features
                           (produced by ProfilerAgent._extract_meta_features)
            task_type:     'classification' or 'regression'
            top_k:         Number of model recommendations to return
        
        Returns:
            List of (model_name, confidence) tuples, sorted by confidence descending.
            Example: [('XGBClassifier', 0.42), ('LGBMClassifier', 0.28), ('CatBoost', 0.15)]
        
        The confidence scores are the PPO policy's action probabilities.
        Higher = the RL agent is more confident this model will perform well.
        """
        if task_type == 'classification':
            model = self.clf_model
            model_names = self.CLASSIFICATION_MODELS
        else:
            model = self.reg_model
            model_names = self.REGRESSION_MODELS
        
        if model is None:
            # Fallback: return safe defaults (boosting models generally win)
            print(f"[RLModelSelector] Using default recommendations (no trained model)")
            return [
                (model_names[0], 0.40),  # XGBoost
                (model_names[1], 0.35),  # LightGBM
                (model_names[3], 0.25),  # RandomForest
            ]
        
        # Get action probabilities from the PPO policy network
        obs = meta_features.reshape(1, -1).astype(np.float32)
        
        try:
            # Extract probability distribution over all models
            action_probs = model.policy.get_distribution(
                model.policy.obs_to_tensor(obs)[0]
            ).distribution.probs.cpu().detach().numpy()[0]
        except Exception as e:
            # Fallback if probability extraction fails
            print(f"[RLModelSelector] Warning: Could not get probabilities ({e})")
            action, _ = model.predict(obs, deterministic=True)
            # Create one-hot-ish distribution
            action_probs = np.ones(len(model_names)) * 0.05
            action_probs[action[0]] = 0.5
        
        # Sort models by probability (descending)
        sorted_indices = np.argsort(action_probs)[::-1]
        
        recommendations = []
        for i in range(min(top_k, len(model_names))):
            idx = sorted_indices[i]
            recommendations.append((model_names[idx], float(action_probs[idx])))
        
        return recommendations
