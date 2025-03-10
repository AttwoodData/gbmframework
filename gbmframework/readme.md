# GBM Framework

A unified framework for Gradient Boosting Models with SHAP analysis and system optimization.

## Features

- Support for multiple GBM implementations (XGBoost, LightGBM, CatBoost, Random Forest)
- Automated hyperparameter optimization with hyperopt
- Intelligent system resource detection and optimization
- Standardized evaluation metrics and visualization
- SHAP value integration for model explainability
- Simple, consistent API for model training and evaluation

## Installation

Basic installation:
```bash
pip install gbmframework
```

With specific boosting libraries:
```bash
pip install gbmframework[xgboost]    # With XGBoost
pip install gbmframework[lightgbm]   # With LightGBM
pip install gbmframework[catboost]   # With CatBoost
pip install gbmframework[hyperopt]   # With Hyperopt for optimization
pip install gbmframework[all]        # All dependencies
```

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Import gbmframework components
from gbmframework.models import train_xgboost, train_lightgbm, train_catboost, train_random_forest
from gbmframework.optimizer import SystemOptimizer
from gbmframework.evaluation import evaluate_classification_model
from gbmframework.shap_utils import generate_shap_values, visualize_shap

# Load and split data
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize system optimizer for efficient resource usage
optimizer = SystemOptimizer(enable_parallel=True, verbose=True)

# Train XGBoost model with hyperparameter optimization
result = train_xgboost(
    X_train, y_train, X_test, y_test,
    max_evals=10,  # Number of hyperopt trials
    optimizer=optimizer  # Pass the optimizer for resource management
)

# Access the best model and evaluate
model = result['model']
print(f"Best AUC: {result['best_score']:.4f}")

# Generate SHAP values for model interpretability
shap_result = generate_shap_values(
    model=model,
    X=X_test.iloc[:100],  # Use a subset for SHAP analysis
    algorithm_type='xgboost',
    optimizer=optimizer
)

# Visualize feature importance
visualize_shap(
    shap_result=shap_result,
    plot_type='bar',
    plot_title='XGBoost Feature Importance',
    optimizer=optimizer
)

# Clean up any resources
optimizer.cleanup()
```

## Documentation

For more information, see the [documentation](https://github.com/yourusername/gbmframework) or the examples directory.