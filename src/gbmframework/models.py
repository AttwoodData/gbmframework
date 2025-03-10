"""
Models - Training functions for various gradient boosting models with hyperparameter optimization
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope

# Optional imports with error handling
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None


def train_random_forest(X_train, y_train, X_test, y_test, hyperopt_space=None, max_evals=50, 
                      handle_imbalance=False, class_weight=None, random_state=42, optimizer=None):
    """
    Train a Random Forest classifier with hyperparameter optimization using hyperopt.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : pandas.Series or numpy.ndarray
        Training labels
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        Test labels
    hyperopt_space : dict, optional
        Dictionary with hyperopt search space. If None, default space is used.
    max_evals : int, default=50
        Number of hyperopt evaluations
    handle_imbalance : bool, default=False
        If True, use class weights to handle imbalanced classes
    class_weight : dict or 'balanced', optional
        Custom class weights to use. If None and handle_imbalance is True, 'balanced' is used
    random_state : int, default=42
        Random seed for reproducibility
    optimizer : SystemOptimizer, optional
        System optimizer instance for multicore optimization
        
    Returns:
    --------
    dict
        Dictionary containing best model, best parameters, and training history
    """
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn is required for Random Forest training")
    
    # Default hyperopt search space if none provided
    if hyperopt_space is None:
        hyperopt_space = {
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 20, 1)),
            'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
            'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
            'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        }
    
    # Get optimized parameters if optimizer is provided
    n_jobs = -1  # Default to use all cores
    if optimizer is not None:
        opt_params = optimizer.get_optimized_parameters('randomforest')
        n_jobs = opt_params.get('n_jobs', -1)
    
    # Define the objective function for hyperopt
    def objective(params):
        # Convert parameters as needed
        for param_name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            if param_name in params:
                params[param_name] = int(params[param_name])
        
        # Set class weights if handling imbalance
        if handle_imbalance:
            params['class_weight'] = class_weight if class_weight is not None else 'balanced'
        
        # Create and train model
        model = RandomForestClassifier(
            **params,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5  # Default for failed calculations
            
        # Our objective is to maximize AUC, but hyperopt minimizes
        return {'loss': -auc, 'status': STATUS_OK, 'model': model}
    
    # Run hyperopt optimization
    if optimizer is not None:
        trials = optimizer.setup_parallel_hyperopt(exp_key=f'rf_{random_state}')
    else:
        trials = Trials()
    
    best = fmin(
        fn=objective,
        space=hyperopt_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=1,
        rstate=np.random.default_rng(random_state)
    )
    
    # Get best parameters (handling special parameter types)
    best_params = {}
    for key, value in best.items():
        if key == 'max_features':
            best_params[key] = ['sqrt', 'log2', None][value]
        else:
            best_params[key] = int(value) if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] else value
    
    # Add class weights if handling imbalance
    if handle_imbalance:
        best_params['class_weight'] = class_weight if class_weight is not None else 'balanced'
    
    # Train final model with best parameters
    best_model = RandomForestClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0
    )
    best_model.fit(X_train, y_train)
    
    # Evaluate best model
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Return results
    return {
        'model': best_model,
        'best_params': best_params,
        'best_score': auc,
        'trials': trials,
        'algorithm': 'RandomForest'
    }