def train_catboost(X_train, y_train, X_test, y_test, hyperopt_space=None, max_evals=50,
                 handle_imbalance=False, class_weights=None, random_state=42, optimizer=None):
    """
    Train a CatBoost classifier with hyperparameter optimization using hyperopt.
    
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
        If True, use class weights or auto_class_weights to handle imbalanced classes
    class_weights : list or dict or 'balanced', optional
        Custom class weights to use. If None and handle_imbalance is True, 'auto' is used
    random_state : int, default=42
        Random seed for reproducibility
    optimizer : SystemOptimizer, optional
        System optimizer instance for multicore optimization
        
    Returns:
    --------
    dict
        Dictionary containing best model, best parameters, and training history
    """
    if cb is None:
        raise ImportError("CatBoost is required for CatBoost training. Install with pip install catboost")
    
    # Default hyperopt search space if none provided
    if hyperopt_space is None:
        hyperopt_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'iterations': scope.int(hp.quniform('iterations', 50, 500, 10)),
            'depth': scope.int(hp.quniform('depth', 3, 10, 1)),
            'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(100)),
            'border_count': scope.int(hp.quniform('border_count', 32, 255, 1)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'random_strength': hp.loguniform('random_strength', np.log(1e-9), np.log(10)),
            'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)
        }
    
    # Get optimized parameters if optimizer is provided
    thread_count = -1  # Default to use all cores
    verbose = False
    if optimizer is not None:
        opt_params = optimizer.get_optimized_parameters('catboost')
        thread_count = opt_params.get('thread_count', -1)
        verbose = opt_params.get('verbose', False)
    
    # Define the objective function for hyperopt
    def objective(params):
        # Convert parameters as needed
        for param_name in ['iterations', 'depth', 'border_count']:
            if param_name in params:
                params[param_name] = int(params[param_name])
        
        # Handle class imbalance
        if handle_imbalance:
            if class_weights == 'balanced':
                params['auto_class_weights'] = 'Balanced'
            elif isinstance(class_weights, (list, dict)):
                params['class_weights'] = class_weights
            else:
                params['auto_class_weights'] = 'Balanced'
        
        # Create and train model
        model = cb.CatBoostClassifier(
            **params,
            random_seed=random_state,
            thread_count=thread_count,
            verbose=verbose
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
        trials = optimizer.setup_parallel_hyperopt(exp_key=f'cb_{random_state}')
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
        if key in ['iterations', 'depth', 'border_count']:
            best_params[key] = int(value)
        else:
            best_params[key] = value
    
    # Handle class imbalance for final model
    if handle_imbalance:
        if class_weights == 'balanced':
            best_params['auto_class_weights'] = 'Balanced'
        elif isinstance(class_weights, (list, dict)):
            best_params['class_weights'] = class_weights
        else:
            best_params['auto_class_weights'] = 'Balanced'
    
    # Train final model with best parameters
    best_model = cb.CatBoostClassifier(
        **best_params,
        random_seed=random_state,
        thread_count=thread_count,
        verbose=verbose
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
        'algorithm': 'CatBoost'
    }    
    # Train final model with best parameters
    best_model = lgb.LGBMClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
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
        'algorithm': 'LightGBM'
    }"""
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


def train_xgboost(X_train, y_train, X_test, y_test, hyperopt_space=None, max_evals=50,
                handle_imbalance=False, scale_pos_weight=None, random_state=42, optimizer=None):
    """
    Train an XGBoost classifier with hyperparameter optimization using hyperopt.
    
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
        If True, use scale_pos_weight to handle imbalanced classes
    scale_pos_weight : float, optional
        Custom scale_pos_weight. If None and handle_imbalance is True, 
        it will be calculated as negative_samples / positive_samples
    random_state : int, default=42
        Random seed for reproducibility
    optimizer : SystemOptimizer, optional
        System optimizer instance for multicore optimization
        
    Returns:
    --------
    dict
        Dictionary containing best model, best parameters, and training history
    """
    if xgb is None:
        raise ImportError("XGBoost is required for XGBoost training. Install with pip install xgboost")
    
    # Default hyperopt search space if none provided
    if hyperopt_space is None:
        hyperopt_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'gamma': hp.uniform('gamma', 0, 5),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-10), np.log(1)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-10), np.log(1))
        }
    
    # Get optimized parameters if optimizer is provided
    n_jobs = -1  # Default to use all cores
    verbosity = 0
    if optimizer is not None:
        opt_params = optimizer.get_optimized_parameters('xgboost')
        n_jobs = opt_params.get('n_jobs', -1)
        verbosity = opt_params.get('verbosity', 0)
    
    # Calculate scale_pos_weight if handling imbalance and not provided
    if handle_imbalance and scale_pos_weight is None:
        negative_samples = np.sum(y_train == 0)
        positive_samples = np.sum(y_train == 1)
        calculated_scale_pos_weight = negative_samples / positive_samples if positive_samples > 0 else 1.0
    else:
        calculated_scale_pos_weight = scale_pos_weight
    
    # Define the objective function for hyperopt
    def objective(params):
        # Convert parameters as needed
        for param_name in ['n_estimators', 'max_depth']:
            if param_name in params:
                params[param_name] = int(params[param_name])
        
        # Set scale_pos_weight if handling imbalance
        if handle_imbalance:
            params['scale_pos_weight'] = calculated_scale_pos_weight
        
        # Create and train model
        model = xgb.XGBClassifier(
            **params,
            random_state=random_state,
            n_jobs=n_jobs,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=verbosity
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
        trials = optimizer.setup_parallel_hyperopt(exp_key=f'xgb_{random_state}')
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
        if key in ['n_estimators', 'max_depth']:
            best_params[key] = int(value)
        else:
            best_params[key] = value
    
    # Add scale_pos_weight if handling imbalance
    if handle_imbalance:
        best_params['scale_pos_weight'] = calculated_scale_pos_weight
    
    # Train final model with best parameters
    best_model = xgb.XGBClassifier(
        **best_params,
        random_state=random_state,
        n_jobs=n_jobs,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=verbosity
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
        'algorithm': 'XGBoost'
    }


def train_lightgbm(X_train, y_train, X_test, y_test, hyperopt_space=None, max_evals=50,
                 handle_imbalance=False, class_weight=None, random_state=42, optimizer=None):
    """
    Train a LightGBM classifier with hyperparameter optimization using hyperopt.
    
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
        If True, use class weights or is_unbalance to handle imbalanced classes
    class_weight : dict or 'balanced', optional
        Custom class weights to use. If None and handle_imbalance is True, 'is_unbalance=True' is used
    random_state : int, default=42
        Random seed for reproducibility
    optimizer : SystemOptimizer, optional
        System optimizer instance for multicore optimization
        
    Returns:
    --------
    dict
        Dictionary containing best model, best parameters, and training history
    """
    if lgb is None:
        raise ImportError("LightGBM is required for LightGBM training. Install with pip install lightgbm")
    
    # Default hyperopt search space if none provided
    if hyperopt_space is None:
        hyperopt_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
            'num_leaves': scope.int(hp.quniform('num_leaves', 20, 150, 1)),
            'min_child_samples': scope.int(hp.quniform('min_child_samples', 1, 60, 1)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-10), np.log(1)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-10), np.log(1))
        }
    
    # Get optimized parameters if optimizer is provided
    n_jobs = -1  # Default to use all cores
    verbose = -1
    if optimizer is not None:
        opt_params = optimizer.get_optimized_parameters('lightgbm')
        n_jobs = opt_params.get('n_jobs', -1)
        verbose = opt_params.get('verbose', -1)
    
    # Define the objective function for hyperopt
    def objective(params):
        # Convert parameters as needed
        for param_name in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples']:
            if param_name in params:
                params[param_name] = int(params[param_name])
        
        # Handle class imbalance
        if handle_imbalance:
            if class_weight == 'balanced':
                # Calculate class weights
                negative_samples = np.sum(y_train == 0)
                positive_samples = np.sum(y_train == 1)
                params['class_weight'] = {
                    0: 1.0,
                    1: negative_samples / positive_samples if positive_samples > 0 else 1.0
                }
            elif isinstance(class_weight, dict):
                params['class_weight'] = class_weight
            else:
                params['is_unbalance'] = True
        
        # Create and train model
        model = lgb.LGBMClassifier(
            **params,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
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
        trials = optimizer.setup_parallel_hyperopt(exp_key=f'lgb_{random_state}')
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
        if key in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples']:
            best_params[key] = int(value)
        else:
            best_params[key] = value
    
    # Handle class imbalance for final model
    if handle_imbalance:
        if class_weight == 'balanced':
            # Calculate class weights
            negative_samples = np.sum(y_train == 0)
            positive_samples = np.sum(y_train == 1)
            best_params['class_weight'] = {
                0: 1.0,
                1: negative_samples / positive_samples if positive_samples > 0 else 1.0
            }
        elif isinstance(class_weight, dict):
            best_params['class_weight'] = class_weight
        else:
            best_params['is_unbalance'] = True