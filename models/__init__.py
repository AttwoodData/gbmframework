"""
Model implementations and model factory.
"""

import logging
from gbmframework import check_install_dependencies, DEPENDENCIES

logger = logging.getLogger(__name__)

# Conditionally import models based on available dependencies
if check_install_dependencies(DEPENDENCIES['xgboost'], auto_install=False):
    from .xgboost_model import XGBoostModel
    logger.info("XGBoost support enabled")
else:
    logger.warning("XGBoost not available. XGBoostModel will not be accessible.")
    XGBoostModel = None

if check_install_dependencies(DEPENDENCIES['lightgbm'], auto_install=False):
    from .lightgbm_model import LightGBMModel
    logger.info("LightGBM support enabled")
else:
    logger.warning("LightGBM not available. LightGBMModel will not be accessible.")
    LightGBMModel = None

if check_install_dependencies(DEPENDENCIES['catboost'], auto_install=False):
    from .catboost_model import CatBoostModel
    logger.info("CatBoost support enabled")
else:
    logger.warning("CatBoost not available. CatBoostModel will not be accessible.")
    CatBoostModel = None


class GBMFactory:
    """Factory class to create appropriate GBM models."""
    
    @staticmethod
    def create_model(model_type, random_state=42, auto_install=True):
        """
        Create a model of the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('xgboost', 'lightgbm', or 'catboost')
        random_state : int, optional
            Random seed for reproducibility
        auto_install : bool, optional
            Whether to automatically install missing dependencies
            
        Returns:
        --------
        GBMBase: A model instance of the specified type
        """
        model_type = model_type.lower()
        
        if model_type == 'xgboost':
            if XGBoostModel is None:
                if auto_install:
                    if check_install_dependencies(DEPENDENCIES['xgboost']):
                        from .xgboost_model import XGBoostModel
                        return XGBoostModel(random_state=random_state)
                    else:
                        raise ImportError("Failed to install XGBoost")
                else:
                    raise ImportError("XGBoost is not installed. Set auto_install=True to attempt installation.")
            return XGBoostModel(random_state=random_state)
            
        elif model_type == 'lightgbm':
            if LightGBMModel is None:
                if auto_install:
                    if check_install_dependencies(DEPENDENCIES['lightgbm']):
                        from .lightgbm_model import LightGBMModel
                        return LightGBMModel(random_state=random_state)
                    else:
                        raise ImportError("Failed to install LightGBM")
                else:
                    raise ImportError("LightGBM is not installed. Set auto_install=True to attempt installation.")
            return LightGBMModel(random_state=random_state)
            
        elif model_type == 'catboost':
            if CatBoostModel is None:
                if auto_install:
                    if check_install_dependencies(DEPENDENCIES['catboost']):
                        from .catboost_model import CatBoostModel
                        return CatBoostModel(random_state=random_state)
                    else:
                        raise ImportError("Failed to install CatBoost")
                else:
                    raise ImportError("CatBoost is not installed. Set auto_install=True to attempt installation.")
            return CatBoostModel(random_state=random_state)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}. " 
                             f"Choose from 'xgboost', 'lightgbm', or 'catboost'.")
