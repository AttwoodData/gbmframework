"""
Hyperparameter tuning functionality using Hyperopt.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

# Import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope

logger = logging.getLogger(__name__)

class GBMTuner:
    """
    Class for hyperparameter tuning of GBM models using Hyperopt.
    """
    
    def __init__(self, model_class, X_train, X_val, y_train, y_val, 
                 metric='roc_auc', cv=5, random_state=42):
        """
        Initialize the tuner with data and model class.
        
        Parameters:
        -----------
        model_class : GBMBase
            The GBM model class to tune (XGBoostModel, LightGBMModel, or CatBoostModel)
        X_train : pandas.DataFrame
            Training features
        X_val : pandas.DataFrame
            Validation features
        y_train : pandas.Series
            Training target
        y_val : pandas.Series
            Validation target
        metric : str, optional (default='roc_auc')
            Metric to optimize ('accuracy', 'roc_auc', etc.)
        cv : int, optional (default=5)
            Number of cross-validation folds
        random_state : int, optional (default=42)
            Random seed for reproducibility
        """
        self.model_class = model_class
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.metric = metric
        self.cv = cv
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.trials = None
        
        # Determine model type from class name
        self.model_type = model_class.__class__.__name__.replace('Model', '').lower()
        
    def _get_search_space(self):
        """
        Define the hyperparameter search space based on model type.
        
        Returns:
        --------
        dict: Search space for Hyperopt
        """
        # Common parameters across all models
        common_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample': hp.uniform('colsample', 0.6, 1.0),
        }
        
        # Model-specific parameters
        if self.model_type == 'xgboost':
            model_space = {
                'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
                'min_child_weight': hp.loguniform('min_child_weight', np.log(1), np.log(10)),
                'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
                'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
                'reg_lambda': hp.loguniform('reg_lambda', np.log(1), np.log(100)),
                'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
            }
            
        elif self.model_type == 'lightgbm':
            model_space = {
                'num_leaves': scope.int(hp.quniform('num_leaves', 20, 150, 1)),
                'max_depth': scope.int(hp.quniform('max_depth', 3, 12, 1)),
                'min_child_samples': scope.int(hp.quniform('min_child_samples', 5, 100, 1)),
                'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
                'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
                'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
                'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
            }
            
            # Map colsample to feature_fraction
            common_space.pop('colsample')
            model_space['feature_fraction'] = common_space['subsample']
            
        elif self.model_type == 'catboost':
            model_space = {
                'depth': scope.int(hp.quniform('depth', 4, 10, 1)),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(100)),
                'random_strength': hp.loguniform('random_strength', np.log(1e-8), np.log(10)),
                'iterations': scope.int(hp.quniform('iterations', 50, 500, 10)),
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'border_count': scope.int(hp.quniform('border_count', 32, 255, 1)),
            }
            
            # Map subsample to bagging_fraction, colsample to rsm
            model_space['bagging_fraction'] = common_space.pop('subsample')
            model_space['rsm'] = common_space.pop('colsample')
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Combine common and model-specific parameters
        return {**common_space, **model_space}
    
    def _map_params(self, params):
        """
        Map Hyperopt parameters to model-specific parameters.
        
        Parameters:
        -----------
        params : dict
            Parameters from Hyperopt
            
        Returns:
        --------
        dict: Mapped parameters for the specific model
        """
        mapped_params = params.copy()
        
        # Handle parameter naming differences
        if self.model_type == 'xgboost':
            mapped_params['colsample_bytree'] = mapped_params.pop('colsample')
            
        elif self.model_type == 'lightgbm':
            # LightGBM parameter mapping handled in search space definition
            pass
            
        elif self.model_type == 'catboost':
            # CatBoost parameter mapping handled in search space definition
            pass
        
        return mapped_params
    
    def _objective(self, params):
        """
        Objective function to minimize.
        
        Parameters:
        -----------
        params : dict
            Hyperparameters to evaluate
            
        Returns:
        --------
        dict: Result dictionary for Hyperopt
        """
        # Map parameters to model-specific format
        model_params = self._map_params(params)
        
        # Create model instance
        model = self.model_class(random_state=self.random_state)
        
        # Use cross-validation if cv > 1, otherwise use validation set
        if self.cv > 1:
            # Disable visual outputs for cross-validation
            model.fit(
                X_train=self.X_train, 
                X_test=self.X_val, 
                y_train=self.y_train, 
                y_test=self.y_val,
                params=model_params,
                create_shap_plots=False,
                create_importance_plots=False
            )
            
            # Get predictions
            if self.metric == 'accuracy':
                score = accuracy_score(self.y_val, model.predict(self.X_val))
            elif self.metric == 'roc_auc':
                score = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
                
        else:
            # No cross-validation, use validation set
            model.fit(
                X_train=self.X_train, 
                X_test=self.X_val, 
                y_train=self.y_train, 
                y_test=self.y_val,
                params=model_params,
                create_shap_plots=False,
                create_importance_plots=False
            )
            
            # Get predictions
            if self.metric == 'accuracy':
                score = accuracy_score(self.y_val, model.predict(self.X_val))
            elif self.metric == 'roc_auc':
                score = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Return the result (negative score because Hyperopt minimizes)
        return {
            'loss': -score,  # Negative because Hyperopt minimizes
            'status': STATUS_OK,
            'model': model,
            'params': model_params,
            'score': score
        }
    
    def optimize(self, max_evals=50, verbose=True):
        """
        Run the hyperparameter optimization.
        
        Parameters:
        -----------
        max_evals : int, optional (default=50)
            Maximum number of evaluations
        verbose : bool, optional (default=True)
            Whether to print progress
            
        Returns:
        --------
        dict: Best parameters found
        """
        if verbose:
            print(f"Starting hyperparameter optimization for {self.model_type.upper()}")
            print(f"Metric: {self.metric}")
            print(f"Max evaluations: {max_evals}")
            print("-" * 50)
        
        # Define the search space
        space = self._get_search_space()
        
        # Initialize trials object to store results
        self.trials = Trials()
        
        # Run the optimization
        best = fmin(
            fn=self._objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=self.trials,
            verbose=verbose
        )
        
        # Get the best parameters
        self.best_params = self._map_params({k: float(v) for k, v in best.items()})
        
        # Convert integer parameters
        for param in ['max_depth', 'num_leaves', 'min_child_samples', 'n_estimators', 
                     'depth', 'border_count', 'iterations']:
            if param in self.best_params:
                self.best_params[param] = int(self.best_params[param])
        
        # Get the best score
        self.best_score = -min(self.trials.losses())
        
        if verbose:
            print("\nOptimization completed")
            print(f"Best {self.metric}: {self.best_score:.4f}")
            print("\nBest parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        
        return self.best_params
    
    def plot_optimization_history(self):
        """
        Plot the optimization history.
        """
        if self.trials is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        plt.figure(figsize=(12, 6))
        
        # Extract scores
        scores = [-loss for loss in self.trials.losses()]
        
        # Plot optimization history
        plt.plot(range(1, len(scores) + 1), scores, 'o-')
        plt.axhline(y=self.best_score, color='r', linestyle='--', 
                    label=f'Best score: {self.best_score:.4f}')
        
        plt.xlabel('Iteration')
        plt.ylabel(self.metric.upper())
        plt.title(f'{self.model_type.upper()} Hyperparameter Optimization History')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_parameter_importance(self, top_n=10):
        """
        Plot parameter importance based on correlation with score.
        
        Parameters:
        -----------
        top_n : int, optional (default=10)
            Number of top parameters to show
        """
        if self.trials is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        # Extract parameter values and scores
        param_values = {}
        for trial in self.trials.trials:
            if trial['result']['status'] == 'ok':
                # Extract parameters
                for param, value in trial['misc']['vals'].items():
                    if len(value) > 0:  # Ensure there's a value
                        if param not in param_values:
                            param_values[param] = []
                        param_values[param].append(float(value[0]))
        
        # Get scores
        scores = [-loss for loss in self.trials.losses()]
        
        # Calculate correlation with scores
        correlations = {}
        for param, values in param_values.items():
            if len(values) > 0 and len(values) == len(scores):
                correlations[param] = abs(np.corrcoef(values, scores)[0, 1])
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top N parameters
        plt.figure(figsize=(12, 6))
        params = [item[0] for item in sorted_correlations[:top_n]]
        corrs = [item[1] for item in sorted_correlations[:top_n]]
        
        plt.barh(params, corrs)
        plt.xlabel('Absolute Correlation with Score')
        plt.ylabel('Parameter')
        plt.title(f'Top {top_n} Parameter Importance')
        plt.tight_layout()
        plt.show()
    
    def train_best_model(self, X_train=None, y_train=None, X_test=None, y_test=None,
                        create_shap_plots=True, create_importance_plots=True):
        """
        Train a model with the best parameters.
        
        Parameters:
        -----------
        X_train, y_train, X_test, y_test : optional
            Data to use for training. If None, uses the data from initialization.
        create_shap_plots, create_importance_plots : bool, optional
            Whether to create SHAP and importance plots
            
        Returns:
        --------
        GBMBase: Trained model with best parameters
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        # Use provided data or fall back to initialization data
        X_train = X_train if X_train is not None else self.X_train
        y_train = y_train if y_train is not None else self.y_train
        X_test = X_test if X_test is not None else self.X_val
        y_test = y_test if y_test is not None else self.y_val
        
        # Create model with best parameters
        model = self.model_class(random_state=self.random_state)
        model.fit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=self.best_params,
            create_shap_plots=create_shap_plots,
            create_importance_plots=create_importance_plots
        )
        
        return model
