# GBM Framework

A unified framework for Gradient Boosting Models with integrated SHAP analysis and hyperparameter tuning.

## Features

- **Unified Interface**: Consistent API for XGBoost, LightGBM, and CatBoost models
- **Integrated SHAP Analysis**: Built-in feature importance visualization with SHAP
- **Hyperparameter Tuning**: Bayesian optimization with Hyperopt
- **Automatic Dependency Management**: Install packages as needed
- **Model Comparison Tools**: Easily compare performance and feature importance
- **Ensemble Creation**: Create ensemble predictions from multiple models

## Installation

### From PyPI (recommended)

```bash
pip install gbmframework
```

### From Source

```bash
git clone https://github.com/yourusername/gbmframework.git
cd gbmframework
pip install -e .
```

By default, this installs only the core dependencies. To install specific GBM frameworks, you can use:

```bash
# Install with XGBoost support
pip install gbmframework[xgboost]

# Install with LightGBM support
pip install gbmframework[lightgbm]

# Install with CatBoost support
pip install gbmframework[catboost]

# Install with hyperparameter tuning support
pip install gbmframework[hyperopt]

# Install all dependencies
pip install gbmframework[all]
```

## Quick Start

```python
from gbmframework import GBMFactory, utils
from sklearn.datasets import load_breast_cancer

# Load and prepare data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Prepare data
X_train, X_test, y_train, y_test = utils.prepare_data(df, target_column='target')

# Create and train a model
model = GBMFactory.create_model('xgboost')
model.fit(X_train, X_test, y_train, y_test)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Explain a prediction
explanation = model.explain_prediction(X_test.iloc[0])

# Select important features
top_features = model.select_features_by_importance(threshold=0.9)
```

## Module Overview

The framework consists of the following components:

- **GBMBase**: Abstract base class for all GBM models
- **XGBoostModel, LightGBMModel, CatBoostModel**: Concrete implementations for each framework
- **GBMFactory**: Factory class for creating model instances
- **GBMTuner**: Hyperparameter tuning with Hyperopt
- **utils**: Utility functions for data preparation, model comparison, etc.

## Examples

See the `examples` directory for detailed examples:

- `simple_classification.py`: Basic classification example with all three GBM frameworks
- `hyperparameter_tuning.py`: Example of hyperparameter tuning with Hyperopt
- `stock_prediction.py`: Real-world example of predicting stock market movements

## Feature Importance Visualization

The framework provides rich visualization capabilities for feature importance:

- **SHAP Summary Plots**: Visualize global feature importance with SHAP
- **SHAP Dependence Plots**: Explore how individual features affect predictions
- **Feature Importance Comparison**: Compare native importance metrics with SHAP values
- **Importance Agreement**: Check consistency of feature rankings across different models

## Hyperparameter Tuning

The framework includes a powerful hyperparameter tuning module based on Hyperopt:

```python
from gbmframework.tuning import GBMTuner

# Create tuner
tuner = GBMTuner(
    model_class=model,
    X_train=X_train,
    X_val=X_val,
    y_train=y_train,
    y_val=y_val
)

# Run optimization
best_params = tuner.optimize(max_evals=50)

# Visualize results
tuner.plot_optimization_history()
tuner.plot_parameter_importance()

# Train final model with best parameters
best_model = tuner.train_best_model()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
