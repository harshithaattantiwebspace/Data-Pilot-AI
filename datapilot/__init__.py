"""
DataPilot AI Pro - Autonomous Data Science Platform
Simple, clean, modular implementation with 6 specialized agents.
"""

from .agents.profiler import ProfilerAgent
from .agents.cleaner import CleanerAgent
from .agents.feature_engineer import FeatureAgent
from .agents.modeler import ModelerAgent
from .agents.visualizer import VisualizerAgent
from .agents.explainer import ExplainerAgent
from .orchestrator import DataPilot

__version__ = "2.0"
__all__ = [
    "ProfilerAgent",
    "CleanerAgent",
    "FeatureAgent",
    "ModelerAgent",
    "VisualizerAgent",
    "ExplainerAgent",
    "DataPilot",
]
