"""
GBM Framework - A unified framework for Gradient Boosting Models with SHAP analysis
"""

__version__ = "0.2.4"

# Import core components for easier access
from .models import train_xgboost, train_lightgbm, train_catboost, train_random_forest
from .optimizer import SystemOptimizer
from .evaluation import evaluate_classification_model
from .shap_utils import generate_shap_values, visualize_shap