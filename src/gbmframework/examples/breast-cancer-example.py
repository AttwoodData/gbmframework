"""
Breast Cancer Classification Example

This example demonstrates a complete workflow for model training,
hyperparameter optimization, and SHAP analysis using the
breast cancer dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Import from gbmframework
from gbmframework.models import train_random_forest, train_xgboost, train_lightgbm, train_catboost
from gbmframework.optimizer import SystemOptimizer
from gbmframework.evaluation import evaluate_classification_model
from gbmframework.shap_utils import generate_shap_values, visualize_shap


# Load and split breast cancer dataset
def load_and_split_breast_cancer():
    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = breast_cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def main():
    # Initialize the system optimizer
    optimizer = SystemOptimizer(enable_parallel=True, verbose=True)

    # Load data
    X_train, X_test, y_train, y_test = load_and_split_breast_cancer()
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Class distribution (train): {np.bincount(y_train)}")

    # Settings
    MAX_EVALS = 10  # Number of hyperopt evaluations
    SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP calculation

    # Dictionary to store results
    results = {}

    # 1. Train Random Forest
    print("\n1. Training Random Forest...")
    rf_result = train_random_forest(
        X_train, y_train, X_test, y_test,
        max_evals=MAX_EVALS,
        optimizer=optimizer  # Pass the optimizer
    )
    results['RandomForest'] = {
        'model': rf_result['model'],
        'score': rf_result['best_score'],
        'algorithm_type': 'randomforest'
    }
    print(f"RandomForest AUC: {rf_result['best_score']:.4f}")

    # 2. Train XGBoost
    print("\n2. Training XGBoost...")
    xgb_result = train_xgboost(
        X_train, y_train, X_test, y_test,
        max_evals=MAX_EVALS,
        optimizer=optimizer  # Pass the optimizer
    )
    results['XGBoost'] = {
        'model': xgb_result['model'],
        'score': xgb_result['best_score'],
        'algorithm_type': 'xgboost'
    }
    print(f"XGBoost AUC: {xgb_result['best_score']:.4f}")

    # 3. Train LightGBM
    print("\n3. Training LightGBM...")
    lgb_result = train_lightgbm(
        X_train, y_train, X_test, y_test,
        max_evals=MAX_EVALS,
        optimizer=optimizer  # Pass the optimizer
    )
    results['LightGBM'] = {
        'model': lgb_result['model'],
        'score': lgb_result['best_score'],
        'algorithm_type': 'lightgbm'
    }
    print(f"LightGBM AUC: {lgb_result['best_score']:.4f}")

    # 4. Train CatBoost
    print("\n4. Training CatBoost...")
    cb_result = train_catboost(
        X_train, y_train, X_test, y_test,
        max_evals=MAX_EVALS,
        optimizer=optimizer  # Pass the optimizer
    )
    results['CatBoost'] = {
        'model': cb_result['model'],
        'score': cb_result['best_score'],
        'algorithm_type': 'catboost'
    }
    print(f"CatBoost AUC: {cb_result['best_score']:.4f}")

    # Find the best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
    best_model = results[best_model_name]['model']
    best_score = results[best_model_name]['score']
    best_algo_type = results[best_model_name]['algorithm_type']

    print(f"\nBest model: {best_model_name} with AUC = {best_score:.4f}")

    # Evaluate the best model
    print("\nEvaluating best model...")
    eval_result = evaluate_classification_model(
        best_model, X_test, y_test,
        figsize=(12, 10)
    )

    # Generate SHAP values for the best model
    print("\nGenerating SHAP values for best model...")
    # If CatBoost is the best model, provide training data for SHAP calculation
    if best_algo_type.lower() == 'catboost':
        shap_result = generate_shap_values(
            model=best_model,
            X=X_test.iloc[:SHAP_SAMPLE_SIZE],
            algorithm_type=best_algo_type,
            X_train=X_train,  # Pass training data for CatBoost
            verbose=1,
            optimizer=optimizer  # Pass the optimizer
        )
    else:
        shap_result = generate_shap_values(
            model=best_model,
            X=X_test.iloc[:SHAP_SAMPLE_SIZE],
            algorithm_type=best_algo_type,
            verbose=1,
            optimizer=optimizer  # Pass the optimizer
        )

    # Create various SHAP visualizations
    print("\nCreating SHAP visualizations...")
    plt.figure(figsize=(12, 6))
    visualize_shap(
        shap_result=shap_result,
        plot_type='bar',
        plot_title=f'{best_model_name} Feature Importance',
        max_display=15,
        optimizer=optimizer  # Pass the optimizer
    )
    plt.tight_layout()
    plt.show()

    # Create summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    visualize_shap(
        shap_result=shap_result,
        plot_type='summary',
        plot_title=f'{best_model_name} SHAP Summary',
        max_display=15,
        optimizer=optimizer  # Pass the optimizer
    )
    plt.tight_layout()
    plt.show()

    # Clean up any resources
    optimizer.cleanup()

    print("\nWorkflow completed successfully!")


if __name__ == "__main__":
    main()