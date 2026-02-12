"""
Visualizer Agent - Visualization and Exploratory Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, mean_squared_error

sns.set_style("whitegrid")


def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """Get numeric column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


class VisualizerAgent:
    """Generates visualizations."""
    
    def __init__(self):
        self.figures = {}
    
    def visualize(self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray, 
                  task_type: str, trained_models: Dict = None) -> Dict:
        """
        Generate visualizations.
        
        Args:
            X: Feature matrix
            y: Target vector
            y_pred: Model predictions
            task_type: 'classification' or 'regression'
            trained_models: Dictionary of trained models for feature importance
            
        Returns:
            visualization_report
        """
        
        # Target distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        if task_type == 'classification':
            y.value_counts().plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Target Distribution (Classification)')
        else:
            ax.hist(y, bins=30, color='steelblue', edgecolor='black')
            ax.set_title('Target Distribution (Regression)')
        plt.tight_layout()
        self.figures['target_dist'] = fig
        
        # Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y, y_pred, alpha=0.6, color='steelblue')
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        ax.legend()
        plt.tight_layout()
        self.figures['actual_vs_pred'] = fig
        
        # Classification-specific visualizations
        if task_type == 'classification':
            # Confusion Matrix
            try:
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                plt.tight_layout()
                self.figures['confusion_matrix'] = fig
            except:
                pass
            
            # ROC Curve (for binary classification)
            try:
                if len(np.unique(y)) == 2:
                    fpr, tpr, _ = roc_curve(y, y_pred)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend()
                    plt.tight_layout()
                    self.figures['roc_curve'] = fig
            except:
                pass
        
        # Regression-specific visualizations
        if task_type == 'regression':
            # Residuals plot
            residuals = y - y_pred
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_pred, residuals, alpha=0.6, color='steelblue')
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title('Residual Plot')
            plt.tight_layout()
            self.figures['residuals'] = fig
            
            # Distribution of residuals
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(residuals, bins=30, color='steelblue', edgecolor='black')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Residuals')
            plt.tight_layout()
            self.figures['residuals_dist'] = fig
        
        # Feature distributions
        numeric_cols = get_numeric_cols(X)
        if len(numeric_cols) > 0:
            n_cols = min(len(numeric_cols), 4)
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, col in enumerate(numeric_cols):
                if idx < len(axes):
                    axes[idx].hist(X[col], bins=20, color='skyblue', edgecolor='black')
                    axes[idx].set_title(f'{col}')
            
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            self.figures['feature_dist'] = fig
        
        # Feature importance (if models provided)
        if trained_models:
            importances = []
            for model in trained_models.values():
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            if importances:
                avg_imp = np.mean(importances, axis=0)
                top_indices = np.argsort(avg_imp)[-10:][::-1]
                top_features = [X.columns[i] for i in top_indices]
                top_importances = avg_imp[top_indices]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(top_features, top_importances, color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Feature Importances')
                plt.tight_layout()
                self.figures['feature_importance'] = fig
        
        report = {
            'plots_generated': list(self.figures.keys()),
            'total_plots': len(self.figures),
            'figures': self.figures,
        }
        
        return report
