"""
Explainer Agent - Model Explainability and Reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, r2_score, precision_score, recall_score, f1_score


class ExplainerAgent:
    """Generates explanations and insights."""
    
    def explain(self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray,
                trained_models: Dict, task_type: str) -> Dict:
        """
        Generate comprehensive explanations.
        
        Args:
            X: Feature matrix
            y: Actual target values
            y_pred: Model predictions
            trained_models: Dictionary of trained models
            task_type: 'classification' or 'regression'
            
        Returns:
            explanation_report
        """
        
        # Feature importance
        importances = []
        for model in trained_models.values():
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        
        feature_importance = {}
        if importances:
            avg_imp = np.mean(importances, axis=0)
            for col, imp in zip(X.columns, avg_imp):
                feature_importance[col] = float(imp)
        
        # Sort by importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Compute metrics
        if task_type == 'classification':
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            metrics_dict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        else:
            r2 = r2_score(y, y_pred)
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))
            
            metrics_dict = {
                'r2': r2,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
            }
        
        # Insights
        insights = [
            f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features",
            f"🎯 Task: {task_type.capitalize()}",
            f"🔍 Models trained: {len(trained_models)}",
        ]
        
        if task_type == 'classification':
            insights.append(f"✅ Accuracy: {metrics_dict['accuracy']:.4f}")
            insights.append(f"📈 Precision: {metrics_dict['precision']:.4f}")
            insights.append(f"📊 Recall: {metrics_dict['recall']:.4f}")
            insights.append(f"🎯 F1-Score: {metrics_dict['f1']:.4f}")
        else:
            insights.append(f"✅ R² Score: {metrics_dict['r2']:.4f}")
            insights.append(f"📊 RMSE: {metrics_dict['rmse']:.4f}")
            insights.append(f"📈 MAE: {metrics_dict['mae']:.4f}")
        
        if top_features:
            insights.append(f"⭐ Top feature: {top_features[0][0]} (importance: {top_features[0][1]:.4f})")
        
        # Recommendations
        recommendations = []
        
        if task_type == 'classification':
            if metrics_dict['accuracy'] > 0.9:
                recommendations.append("✅ Excellent model performance - ready for production")
            elif metrics_dict['accuracy'] > 0.8:
                recommendations.append("⚠️ Good model performance - consider further tuning")
            else:
                recommendations.append("❌ Model performance needs improvement - review features and data")
            
            if metrics_dict['precision'] < metrics_dict['recall']:
                recommendations.append("💡 Recall is higher than precision - model is sensitive but may have false positives")
            elif metrics_dict['recall'] < metrics_dict['precision']:
                recommendations.append("💡 Precision is higher than recall - model is conservative but may miss positives")
        else:
            if metrics_dict['r2'] > 0.9:
                recommendations.append("✅ Excellent model fit - ready for production")
            elif metrics_dict['r2'] > 0.7:
                recommendations.append("⚠️ Good model fit - consider feature engineering")
            else:
                recommendations.append("❌ Model fit needs improvement - review features and data")
            
            recommendations.append(f"💡 Average prediction error: {metrics_dict['mae']:.4f}")
        
        recommendations.extend([
            "🔄 Use ensemble model in production",
            "📊 Monitor model performance regularly",
            "🔁 Retrain periodically with new data",
            "📈 Consider collecting more data for better generalization",
        ])
        
        report = {
            'top_features': top_features,
            'feature_importance': feature_importance,
            'metrics': metrics_dict,
            'insights': insights,
            'recommendations': recommendations,
            'model_count': len(trained_models),
            'task_type': task_type,
        }
        
        return report
