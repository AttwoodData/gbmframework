"""
Models subpackage for GBM Framework.

This module provides model implementations for different gradient boosting frameworks
and a factory class to create model instances.
"""

import logging

logger = logging.getLogger(__name__)

# Import model implementations if dependencies are available
try:
    from .xgboost_model import XGBoostModel
    logger.info("XGBoost support enabled")
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    from .lightgbm_model import LightGBMModel
    logger.info("LightGBM support enabled")
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM not available")
    LIGHTGBM_AVAILABLE = False

try:
    from .catboost_model import CatBoostModel
    logger.info("CatBoost support enabled")
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not available")
    CATBOOST_AVAILABLE = False

class GBMFactory:
    """Factory class for creating GBM model instances."""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create a model instance based on specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('xgboost', 'lightgbm', or 'catboost')
        **kwargs : dict
            Additional parameters to pass to the model constructor
            
        Returns:
        --------
        GBMBase: Instance of the requested model type
            
        Raises:
        -------
        ImportError: If the requested model type's dependencies are not available
        ValueError: If the model type is not recognized
        """
        model_type = model_type.lower()
        
        if model_type == 'xgboost':
            if XGBOOST_AVAILABLE:
                return XGBoostModel(**kwargs)
            else:
                raise ImportError("XGBoost is not available. Install with: pip install xgboost")
                
        elif model_type == 'lightgbm':
            if LIGHTGBM_AVAILABLE:
                return LightGBMModel(**kwargs)
            else:
                raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
                
        elif model_type == 'catboost':
            if CATBOOST_AVAILABLE:
                return CatBoostModel(**kwargs)
            else:
                raise ImportError("CatBoost is not available. Install with: pip install catboost")
                
        else:
            raise ValueError(f"Unknown model type: {model_type}. " 
                            f"Available types: 'xgboost', 'lightgbm', 'catboost'")
