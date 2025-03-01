"""
GBM Framework - A unified framework for Gradient Boosting Models with integrated SHAP analysis and hyperparameter tuning.

The gbmframework package provides a consistent API for working with XGBoost, LightGBM, and CatBoost models,
along with integrated SHAP analysis, hyperparameter tuning, and utility functions.

Features
--------
* Unified Interface: Consistent API for XGBoost, LightGBM, and CatBoost models
* Integrated SHAP Analysis: Built-in feature importance visualization with SHAP
* Hyperparameter Tuning: Bayesian optimization with Hyperopt
* Automatic Dependency Management: Install packages as needed
* Model Comparison Tools: Easily compare performance and feature importance
* Ensemble Creation: Create ensemble predictions from multiple models

Installation
-----------
From PyPI (recommended):
    pip install gbmframework

For specific GBM frameworks:
    pip install gbmframework[xgboost]       # XGBoost support
    pip install gbmframework[lightgbm]      # LightGBM support
    pip install gbmframework[catboost]      # CatBoost support
    pip install gbmframework[hyperopt]      # Hyperparameter tuning
    pip install gbmframework[all]           # All dependencies

Quick Start
----------
Basic usage example:

    from gbmframework import GBMFactory, utils
    
    # Prepare data
    X_train, X_test, y_train, y_test = utils.prepare_data(df, target_column='target')
    
    # Create and train a model
    model = GBMFactory.create_model('xgboost')
    model.fit(X_train, X_test, y_train, y_test)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Explain a prediction
    explanation = model.explain_prediction(X_test.iloc[0])
    
    # Select important features
    top_features = model.select_features_by_importance(threshold=0.9)

Module Components
----------------
GBMBase: Abstract base class for all GBM models
XGBoostModel, LightGBMModel, CatBoostModel: Concrete implementations
GBMFactory: Factory class for creating model instances
GBMTuner: Hyperparameter tuning with Hyperopt
utils: Utility functions for data preparation, model comparison, etc.

Example Use Cases
----------------
1. Simple Classification (see examples/simple_classification.py):
   This example demonstrates a basic classification workflow using the breast cancer dataset.
   It shows how to load data, train multiple models, compare their performance, and visualize
   feature importance.
   
   Key concepts:
   - Using utils.prepare_data() for data preprocessing
   - Creating models with GBMFactory
   - Evaluating models with built-in metrics
   - Visualizing SHAP-based feature importance

2. Hyperparameter Tuning (see examples/hyperparameter_tuning.py):
   This example shows how to optimize model parameters using Bayesian optimization.
   The tuning process automatically searches the parameter space to find optimal settings.
   
   Key concepts:
   - Creating a GBMTuner instance
   - Running optimization with hyperopt
   - Visualizing optimization history
   - Training a final model with optimal parameters

3. Stock Prediction (see examples/stock_prediction.py):
   A real-world application that demonstrates using gradient boosting models for time series
   prediction of stock price movements.
   
   Key concepts:
   - Feature engineering for time series data
   - Time-based train/test splitting
   - Evaluating models in a trading simulation
   - Comparing multiple model types on the same task

Advanced Usage
-------------
Hyperparameter Tuning:
    from gbmframework.tuning import GBMTuner
    
    tuner = GBMTuner(
        model_class=model,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val
    )
    
    best_params = tuner.optimize(max_evals=50)
    tuner.plot_optimization_history()
    best_model = tuner.train_best_model()

Model Comparison:
    models = {
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'CatBoost': cat_model
    }
    
    comparison = utils.compare_models(models, X_test, y_test)
    agreement = utils.check_feature_importance_agreement(models)

Ensemble Creation:
    ensemble_pred, ensemble_proba = utils.create_ensemble(models, X_test)

SHAP Analysis
------------
The framework integrates SHAP (SHapley Additive exPlanations) for model interpretability:

    # SHAP values are automatically calculated during model fitting
    model.fit(X_train, X_test, y_train, y_test, create_shap_plots=True)
    
    # Explain an individual prediction
    explanation = model.explain_prediction(X_test.iloc[0])
    
    # Plot SHAP dependence for a specific feature
    model.plot_shap_dependence('feature_name', X_test)
    
    # Select features based on SHAP importance
    important_features = model.select_features_by_importance(threshold=0.9, use_shap=True)

Feature Importance Comparison
----------------------------
Compare feature importance rankings across different models:

    # Check agreement between models
    agreement = utils.check_feature_importance_agreement(models, n_features=10)
    
    # This identifies features that are consistently ranked as important
    # across different model types, providing more reliable feature selection.

Dependencies
-----------
Required:
    numpy, pandas, matplotlib, seaborn, scikit-learn, shap

Optional:
    xgboost, lightgbm, catboost, hyperopt

License
-------
This project is licensed under the MIT License.
"""

import subprocess
import importlib
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define required packages for each model type
DEPENDENCIES = {
    'xgboost': ['xgboost'],
    'lightgbm': ['lightgbm'],
    'catboost': ['catboost'],
    'hyperopt': ['hyperopt']
}

def check_install_dependencies(package_list, auto_install=True):
    """
    Check if dependencies are installed, and optionally install them.
    
    Parameters:
    -----------
    package_list : list
        List of package names to check
    auto_install : bool, optional (default=True)
        Whether to automatically install missing packages
        
    Returns:
    --------
    bool: Whether all packages are now available
    """
    missing_packages = []
    
    for package in package_list:
        try:
            importlib.import_module(package)
            logger.debug(f"Package '{package}' is already installed.")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        
        if auto_install:
            logger.info(f"Attempting to install missing packages...")
            
            for package in missing_packages:
                try:
                    logger.info(f"Installing {package}...")
                    subprocess.check_call(["pip", "install", package])
                    logger.info(f"Successfully installed {package}.")
                except Exception as e:
                    logger.error(f"Failed to install {package}: {str(e)}")
                    return False
            
            # Verify installations
            for package in missing_packages:
                try:
                    importlib.import_module(package)
                    logger.info(f"Successfully imported {package} after installation.")
                except ImportError:
                    logger.error(f"Still unable to import {package} after installation.")
                    return False
            
            return True
        else:
            logger.warning("Auto-install is disabled. Please install missing packages manually.")
            logger.info(f"You can install them with: pip install {' '.join(missing_packages)}")
            return False
    
    return True

# Import components based on available dependencies
from gbmframework.base import GBMBase
from gbmframework.models import GBMFactory

# Check if tuning dependencies are available
if check_install_dependencies(DEPENDENCIES['hyperopt'], auto_install=False):
    from gbmframework.tuning import GBMTuner
else:
    logger.warning("Hyperopt not available. Tuning capabilities will be limited.")
    GBMTuner = None

# Define version
__version__ = '0.1.6'
