"""
Simple classification example using the GBM Framework.

This example demonstrates how to use the GBM Framework to solve a basic classification
problem using the breast cancer dataset from scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
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
    
    logger.info(f"Dataset: Breast Cancer Wisconsin")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Features: {len(data.feature_names)}")
    logger.info(f"Target distribution: {df['target'].value_counts(normalize=True).to_dict()}")
    
    # Display feature information
    logger.info("\nFeature information:")
    feature_info = pd.DataFrame({
        'mean': df.iloc[:, :-1].mean(),
        'std': df.iloc[:, :-1].std(),
        'min': df.iloc[:, :-1].min(),
        'max': df.iloc[:, :-1].max()
    })
    logger.info(f"\n{feature_info.head()}")
    logger.info("...")
    
    # Prepare data using the utils function
    X_train, X_test, y_train, y_test = utils.prepare_data(
        data=df,
        target_column='target',
        test_size=0.25,
        random_state=42,
        scale=True,
        stratify=True
    )
    
    logger.info(f"\nTraining data: {X_train.shape[0]} samples")
    logger.info(f"Testing data: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def run_model_comparison():
    """Run a comparison of all three GBM models."""
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Define simple parameters for each model
    params = {
        'xgboost': {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100
        },
        'lightgbm': {
            'num_leaves': 31,
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100
        },
        'catboost': {
            'depth': 3,
            'learning_rate': 0.1,
            'iterations': 100
        }
    }
    
    # Create models
    models = {}
    model_types = ['xgboost', 'lightgbm', 'catboost']
    
    for model_type in model_types:
        try:
            logger.info(f"\nTraining {model_type.upper()} model")
            
            # Create model using factory
            model = GBMFactory.create_model(model_type)
            
            # Train the model
            model.fit(
                X_train=X_train, 
                X_test=X_test, 
                y_train=y_train, 
                y_test=y_test,
                params=params[model_type],
                create_shap_plots=True,
                create_importance_plots=True
            )
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            logger.info(f"\n{model_type.upper()} Model Test Performance:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"ROC AUC: {auc:.4f}")
            
            # Add to models dictionary
            models[model_type.upper()] = model
            
        except ImportError:
            logger.warning(f"{model_type.upper()} not available - skipping")
    
    # Compare models if we have more than one
    if len(models) > 1:
        # Compare models
        logger.info("\nComparing models")
        comparison = utils.compare_models(
            models_dict=models,
            X_test=X_test,
            y_test=y_test,
            metric='roc_auc',
            plot=True
        )
        logger.info(f"\nModel comparison:\n{comparison}")
        
        # Check feature importance agreement
        logger.info("\nChecking feature importance agreement")
        agreement = utils.check_feature_importance_agreement(models, n_features=10)
        logger.info(f"\nFeature importance agreement (top 10):\n{agreement.head(10)}")
        
        # Create ensemble prediction
        logger.info("\nCreating ensemble prediction")
        ensemble_pred, ensemble_proba = utils.create_ensemble(
            models_dict=models,
            X=X_test,
            method='average'
        )
        
        # Evaluate ensemble
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        logger.info(f"\nEnsemble model performance:")
        logger.info(f"Accuracy: {ensemble_accuracy:.4f}")
        logger.info(f"ROC AUC: {ensemble_auc:.4f}")
        
        # Display classification report for the ensemble
        logger.info(f"\nEnsemble Classification Report:")
        logger.info(f"\n{classification_report(y_test, ensemble_pred)}")
        
        # Compare ensemble with individual models
        individual_acc = [accuracy_score(y_test, model.predict(X_test)) for model in models.values()]
        individual_auc = [roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) for model in models.values()]
        
        best_model_acc = max(individual_acc)
        best_model_auc = max(individual_auc)
        
        logger.info(f"\nEnsemble vs. Best Individual Model:")
        logger.info(f"Accuracy: {ensemble_accuracy:.4f} vs {best_model_acc:.4f} " + 
                    f"({'better' if ensemble_accuracy > best_model_acc else 'worse'})")
        logger.info(f"ROC AUC: {ensemble_auc:.4f} vs {best_model_auc:.4f} " + 
                    f"({'better' if ensemble_auc > best_model_auc else 'worse'})")
    
    # Select a model to explain individual predictions
    if models:
        # Pick the first model
        model_name = list(models.keys())[0]
        model = models[model_name]
        
        # Select a sample to explain
        sample_idx = 0
        sample_X = X_test.iloc[[sample_idx]]
        sample_y = y_test.iloc[sample_idx]
        
        logger.info(f"\nExplaining a single prediction using {model_name} model")
        logger.info(f"True class: {sample_y} ({'Malignant' if sample_y == 0 else 'Benign'})")
        
        # Explain the prediction
        explanation = model.explain_prediction(sample_X)
        
        # Get the most influential features
        if 'shap_values' in explanation:
            shap_vals = explanation['shap_values']
            feature_names = sample_X.columns
            
            # Create a DataFrame of feature contributions
            if isinstance(shap_vals, np.ndarray):
                contributions = pd.DataFrame({
                    'Feature': feature_names,
                    'Contribution': shap_vals
                }).sort_values('Contribution', key=abs, ascending=False)
                
                logger.info(f"\nTop 5 influential features for this prediction:")
                logger.info(f"\n{contributions.head(5)}")
    
    return models

def run_simple_example():
    """Run the simple classification example."""
    logger.info("Running simple classification example with breast cancer dataset")
    
    # Run model comparison
    models = run_model_comparison()
    
    if not models:
        logger.error("No models were trained. Please install at least one GBM framework.")
        return None
    
    # Feature selection example
    if models:
        model_name = list(models.keys())[0]
        model = models[model_name]
        
        logger.info(f"\nPerforming feature selection with {model_name} model")
        
        # Select important features
        threshold = 0.90  # Select features that explain 90% of the variance
        selected_features = model.select_features_by_importance(threshold=threshold, use_shap=True)
        
        logger.info(f"Selected {len(selected_features)} features that explain {threshold:.0%} of the importance")
        logger.info(f"Selected features: {selected_features}")
    
    return models

if __name__ == "__main__":
    models = run_simple_example()
