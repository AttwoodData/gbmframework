"""
Hyperparameter tuning example with the GBM Framework.

This script demonstrates hyperparameter optimization using Hyperopt with the GBM Framework.
It performs a Bayesian optimization search across the hyperparameter space and visualizes
the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the GBM Framework
try:
    from gbmframework import GBMFactory, utils
    from gbmframework.tuning import GBMTuner
except ImportError:
    import sys
    import os
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from gbmframework import GBMFactory, utils
    from gbmframework.tuning import GBMTuner

def load_and_prepare_data():
    """Load and prepare the breast cancer dataset."""
    # Load breast cancer dataset
    data = load_breast_cancer()
    
    # Create a dataframe
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Add target column (0 for malignant, 1 for benign)
    df['target'] = data.target
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {len(data.feature_names)}")
    logger.info(f"Target distribution: {df['target'].value_counts(normalize=True).to_dict()}")
    
    # Train-test-validation split
    X = df.drop(columns=['target'])
    y = df['target']
    
    # First split: training+validation vs test (80/20)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: training vs validation (75/25, which is 60/20 of the original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
    )
    
    # Scale the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    logger.info(f"Training data: {X_train_scaled.shape[0]} samples ({100*X_train_scaled.shape[0]/X.shape[0]:.1f}% of total)")
    logger.info(f"Validation data: {X_val_scaled.shape[0]} samples ({100*X_val_scaled.shape[0]/X.shape[0]:.1f}% of total)")
    logger.info(f"Test data: {X_test_scaled.shape[0]} samples ({100*X_test_scaled.shape[0]/X.shape[0]:.1f}% of total)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

def baseline_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train baseline models with default parameters."""
    # Define default parameters for each model
    default_params = {
        'xgboost': {},  # Use framework defaults
        'lightgbm': {},  # Use framework defaults
        'catboost': {}   # Use framework defaults
    }
    
    # Models to train
    model_types = ['xgboost', 'lightgbm', 'catboost']
    baseline_models = {}
    
    # Train each model type with default parameters
    for model_type in model_types:
        try:
            logger.info(f"Training baseline {model_type.upper()} model with default parameters")
            
            # Create model
            model = GBMFactory.create_model(model_type)
            
            # Fit model with default parameters
            model.fit(
                X_train=X_train, 
                X_test=X_val,  # Use validation set for evaluation during training
                y_train=y_train, 
                y_test=y_val,
                create_shap_plots=False,  # Disable plots for baseline models
                create_importance_plots=False
            )
            
            # Add to models dictionary
            baseline_models[model_type.upper()] = model
            
        except ImportError:
            logger.warning(f"{model_type.upper()} not available - skipping")
    
    # Evaluate baseline models on test set
    if baseline_models:
        logger.info("\nBaseline model performance on test set:")
        for name, model in baseline_models.items():
            # Make predictions
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            
            logger.info(f"{name}: Accuracy = {accuracy:.4f}, ROC AUC = {auc:.4f}")
    
    return baseline_models

def tune_model(model_type, X_train, X_val, y_train, y_val, metric='roc_auc', max_evals=50):
    """
    Tune a model using Hyperopt.
    
    Parameters:
    -----------
    model_type : str
        Type of model to tune ('xgboost', 'lightgbm', or 'catboost')
    X_train, X_val : pandas.DataFrame
        Training and validation features
    y_train, y_val : pandas.Series
        Training and validation targets
    metric : str, optional (default='roc_auc')
        Metric to optimize ('accuracy' or 'roc_auc')
    max_evals : int, optional (default=50)
        Maximum number of evaluations for Hyperopt
        
    Returns:
    --------
    tuple: (tuner, best_params)
    """
    try:
        import hyperopt
    except ImportError:
        logger.warning("Hyperopt not available. Cannot perform hyperparameter tuning.")
        logger.warning("Install with: pip install hyperopt")
        return None, None
    
    # Create model
    try:
        model = GBMFactory.create_model(model_type)
    except ImportError:
        logger.warning(f"{model_type.upper()} not available")
        return None, None
    
    logger.info(f"Creating {model_type.upper()} tuner with {metric} metric")
    
    # Create tuner
    tuner = GBMTuner(
        model_class=model,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        metric=metric
    )
    
    # Run optimization
    logger.info(f"Running hyperparameter optimization with {max_evals} evaluations")
    best_params = tuner.optimize(max_evals=max_evals)
    
    # Plot optimization history
    logger.info("Plotting optimization history")
    tuner.plot_optimization_history()
    
    # Plot parameter importance
    logger.info("Plotting parameter importance")
    tuner.plot_parameter_importance()
    
    return tuner, best_params

def evaluate_tuned_model(tuner, X_train, X_test, y_train, y_test):
    """
    Evaluate a tuned model on the test set.
    
    Parameters:
    -----------
    tuner : GBMTuner
        Tuner with best parameters
    X_train, X_test : pandas.DataFrame
        Training and test features
    y_train, y_test : pandas.Series
        Training and test targets
        
    Returns:
    --------
    tuple: (model, metrics)
    """
    if tuner is None:
        logger.warning("No tuner provided")
        return None, None
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    model = tuner.train_best_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        create_shap_plots=True,
        create_importance_plots=True
    )
    
    # Evaluate on test set
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info(f"Test set performance:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC AUC: {auc:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': auc,
        'report': classification_report(y_test, y_pred)
    }
    
    return model, metrics

def compare_with_baseline(baseline_models, tuned_model, X_test, y_test):
    """
    Compare tuned model with baseline models.
    
    Parameters:
    -----------
    baseline_models : dict
        Dictionary of baseline models
    tuned_model : GBMBase
        Tuned model
    X_test, y_test : pandas.DataFrame, pandas.Series
        Test data
    """
    if not baseline_models or tuned_model is None:
        logger.warning("Cannot compare: missing baseline models or tuned model")
        return
    
    # Evaluate all models
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    results = {'Tuned Model': {}}
    
    # Get tuned model predictions
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]
    
    results['Tuned Model']['accuracy'] = accuracy_score(y_test, y_pred)
    results['Tuned Model']['roc_auc'] = roc_auc_score(y_test, y_prob)
    
    # Get baseline predictions
    for name, model in baseline_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
    
    # Create DataFrame
    df_acc = pd.DataFrame({model: results[model]['accuracy'] for model in results}, index=['Accuracy'])
    df_auc = pd.DataFrame({model: results[model]['roc_auc'] for model in results}, index=['ROC AUC'])
    df = pd.concat([df_acc, df_auc])
    
    logger.info("\nModel Comparison:")
    logger.info(f"\n{df}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), [results[model]['accuracy'] for model in results])
    plt.title('Accuracy Comparison')
    plt.ylim(max(0.5, min([results[model]['accuracy'] for model in results]) - 0.05), 1.0)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot ROC AUC
    plt.subplot(1, 2, 2)
    plt.bar(results.keys(), [results[model]['roc_auc'] for model in results])
    plt.title('ROC AUC Comparison')
    plt.ylim(max(0.5, min([results[model]['roc_auc'] for model in results]) - 0.05), 1.0)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate improvement
    baseline_model = tuned_model.__class__.__name__.replace('Model', '').upper()
    if baseline_model in baseline_models:
        acc_improvement = results['Tuned Model']['accuracy'] - results[baseline_model]['accuracy']
        auc_improvement = results['Tuned Model']['roc_auc'] - results[baseline_model]['roc_auc']
        
        logger.info(f"\nImprovement from tuning {baseline_model}:")
        logger.info(f"Accuracy: +{acc_improvement:.4f} ({acc_improvement/results[baseline_model]['accuracy']*100:.2f}%)")
        logger.info(f"ROC AUC: +{auc_improvement:.4f} ({auc_improvement/results[baseline_model]['roc_auc']*100:.2f}%)")

def run_hyperparameter_tuning_example():
    """Run the hyperparameter tuning example."""
    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    # Train baseline models
    logger.info("\nTraining baseline models")
    baseline_models_dict = baseline_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    if not baseline_models_dict:
        logger.error("No baseline models were trained. Install at least one GBM framework.")
        return None, None, None
    
    # Choose a model to tune
    available_models = list(baseline_models_dict.keys())
    model_to_tune = available_models[0].lower()  # Use the first available model
    
    logger.info(f"\nTuning {model_to_tune.upper()} model")
    tuner, best_params = tune_model(
        model_type=model_to_tune,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        metric='roc_auc',
        max_evals=30  # Reduce for faster execution
    )
    
    if tuner is None:
        logger.error("Hyperparameter tuning failed. Please install hyperopt.")
        return baseline_models_dict, None, None
    
    # Evaluate tuned model
    logger.info("\nEvaluating tuned model")
    tuned_model, metrics = evaluate_tuned_model(tuner, X_train, X_test, y_train, y_test)
    
    # Compare with baseline
    logger.info("\nComparing tuned model with baseline models")
    compare_with_baseline(baseline_models_dict, tuned_model, X_test, y_test)
    
    return baseline_models_dict, tuned_model, metrics

if __name__ == "__main__":
    logger.info("Running hyperparameter tuning example")
    baseline_models, tuned_model, metrics = run_hyperparameter_tuning_example()
