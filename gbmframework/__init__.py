"""
GBM Framework - A unified framework for Gradient Boosting Models with SHAP analysis.

This package provides a common interface for XGBoost, LightGBM, and CatBoost models,
along with integrated SHAP analysis and hyperparameter tuning.
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
from ._version import __version__
