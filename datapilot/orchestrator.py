"""
Orchestrator - Pipeline Coordination
"""

import pandas as pd
from typing import Dict, Any
from .agents.profiler import ProfilerAgent
from .agents.cleaner import CleanerAgent
from .agents.feature_engineer import FeatureAgent
from .agents.modeler import ModelerAgent
from .agents.visualizer import VisualizerAgent
from .agents.explainer import ExplainerAgent


class DataPilot:
    """Main orchestrator - runs complete pipeline."""
    
    def __init__(self):
        self.profiler = ProfilerAgent()
        self.cleaner = CleanerAgent()
        self.feature_agent = FeatureAgent()
        self.modeler = ModelerAgent()
        self.visualizer = VisualizerAgent()
        self.explainer = ExplainerAgent()
        
        self.state = {}
    
    def run(self, df: pd.DataFrame, target_col: str) -> Dict:
        """
        Run complete pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Pipeline results dictionary
        """
        
        print("\n" + "="*60)
        print("DataPilot AI Pro - Autonomous Data Science Pipeline")
        print("="*60 + "\n")
        
        # Stage 1: Profile
        print("[1/6] Stage 1: Profiling Dataset...")
        profile_report, meta_features, target_col, task_type = self.profiler.analyze(df, target_col)
        self.state['profile'] = profile_report
        self.state['task_type'] = task_type
        print(f"   [OK] Task Type: {task_type}")
        print(f"   [OK] Shape: {profile_report['dataset_shape']}")
        
        # Stage 2: Clean
        print("\n[2/6] Stage 2: Cleaning Data...")
        df_clean, clean_report = self.cleaner.clean(df, target_col, task_type, meta_features)
        self.state['cleaning'] = clean_report
        print(f"   [OK] Rows removed: {clean_report['rows_removed']}")
        print(f"   [OK] Final shape: {clean_report['final_shape']}")
        
        # Stage 3: Feature Engineering
        print("\n[3/6] Stage 3: Feature Engineering...")
        X, y, feature_report = self.feature_agent.engineer(df_clean, target_col, task_type)
        self.state['features'] = feature_report
        print(f"   [OK] Features: {feature_report['final_features']}")
        
        # Stage 4: Modeling
        print("\n[4/6] Stage 4: Training Models...")
        ensemble, y_pred, model_report = self.modeler.train(X, y, task_type, meta_features)
        self.state['modeling'] = model_report
        print(f"   [OK] Ensemble created")
        print(f"   [OK] Metrics: {model_report['metrics']}")
        
        # Stage 5: Visualization
        print("\n[5/6] Stage 5: Generating Visualizations...")
        viz_report = self.visualizer.visualize(X, y, y_pred, task_type, self.modeler.trained_models)
        self.state['visualization'] = viz_report
        print(f"   [OK] Plots generated: {viz_report['total_plots']}")
        
        # Stage 6: Explanation
        print("\n[6/6] Stage 6: Generating Explanations...")
        explain_report = self.explainer.explain(X, y, y_pred, self.modeler.trained_models, task_type)
        self.state['explanation'] = explain_report
        print(f"   [OK] Top features identified")
        print(f"   [OK] Insights generated")
        
        print("\n" + "="*60)
        print("SUCCESS: Pipeline Complete!")
        print("="*60 + "\n")
        
        return self.state
    
    def get_summary(self) -> str:
        """Get pipeline summary."""
        summary = "\n[*] PIPELINE SUMMARY\n"
        summary += "="*60 + "\n"
        
        if 'profile' in self.state:
            summary += f"Dataset Shape: {self.state['profile']['dataset_shape']}\n"
            summary += f"Task Type: {self.state['task_type']}\n"
        
        if 'cleaning' in self.state:
            summary += f"Rows Removed: {self.state['cleaning']['rows_removed']}\n"
        
        if 'features' in self.state:
            summary += f"Final Features: {self.state['features']['final_features']}\n"
        
        if 'modeling' in self.state:
            summary += f"Models Trained: {len(self.state['modeling']['all_models_trained'])}\n"
            for metric, value in self.state['modeling']['metrics'].items():
                summary += f"  {metric}: {value:.4f}\n"
        
        if 'explanation' in self.state:
            summary += f"\nTop Features:\n"
            for feat, imp in self.state['explanation']['top_features'][:5]:
                summary += f"  • {feat}: {imp:.4f}\n"
        
        summary += "="*60 + "\n"
        return summary
