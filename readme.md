# GBM Framework

A unified framework for Gradient Boosting Models with SHAP analysis and system optimization.

## Gradient Boosting Algorithms

This framework supports four powerful tree-based ensemble methods, each with unique strengths:

### XGBoost
- **Developed by**: Tianqi Chen (2014)
- **Key innovation**: Regularized gradient boosting with system optimization
- **Performance profile**: Excellent on medium-sized datasets; scales reasonably to large datasets
- **Strengths**: Overall high performance, handles sparse data well, regularization controls overfitting
- **Limitations**: Memory-intensive for very large datasets, slower training than LightGBM
- **Best suited for**: Problems where model performance is critical, datasets that fit in memory

### LightGBM
- **Developed by**: Microsoft Research (Guolin Ke et al., 2017)
- **Key innovation**: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB)
- **Performance profile**: Very fast on wide datasets (many features), excellent scaling for large datasets
- **Strengths**: Fast training speed, low memory usage, high performance on categorical features
- **Limitations**: May overfit on small datasets without careful tuning
- **Best suited for**: Large datasets, especially those with many features, speed-critical applications

### CatBoost
- **Developed by**: Yandex (Anna Veronika Dorogush et al., 2018)
- **Key innovation**: Ordered boosting and native handling of categorical features
- **Performance profile**: Excellent on datasets with categorical features, competitive performance out-of-the-box
- **Strengths**: Superior handling of categorical features without preprocessing, robust against overfitting
- **Limitations**: Slower training than LightGBM for large datasets
- **Best suited for**: Datasets with many categorical features, use cases requiring minimal hyperparameter tuning

### Random Forest
- **Developed by**: Leo Breiman and Adele Cutler (2001)
- **Key innovation**: Bootstrap aggregation (bagging) with random feature selection
- **Performance profile**: Good baseline performance, highly parallelizable
- **Strengths**: Less prone to overfitting, fewer hyperparameters, good predictive uncertainty estimates
- **Limitations**: Generally lower predictive performance than boosting methods, larger model size
- **Best suited for**: Baseline models, applications requiring uncertainty estimates, highly imbalanced data

### Comparison on Dataset Characteristics

| Algorithm   | Very Wide Data (many features) | Very Tall Data (many rows) | Categorical Features | Training Speed | Default Performance |
|-------------|--------------------------------|----------------------------|----------------------|----------------|---------------------|
| XGBoost     | Good                           | Moderate                   | Requires encoding    | Moderate       | Very Good           |
| LightGBM    | Excellent                      | Excellent                  | Good                 | Very Fast      | Good                |
| CatBoost    | Good                           | Good                       | Excellent            | Moderate       | Excellent           |
| Random Forest| Moderate                      | Good                       | Requires encoding    | Fast           | Moderate            |

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
pip install gbmframework[shap]       # With SHAP for explainability
pip install gbmframework[all]        # All dependencies
```

## Key Functions and Parameters

The GBM Framework provides a consistent API across different gradient boosting implementations. Here's a reference guide to the main functions and their parameters:

### System Optimization

```python
from gbmframework.optimizer import SystemOptimizer

optimizer = SystemOptimizer(
    enable_parallel=True,  # Whether to enable parallel computation
    memory_safety=0.8,     # Memory safety factor (0.0-1.0)
    verbose=True           # Whether to print optimization information
)
```

The `SystemOptimizer` automatically detects system resources and configures optimal thread counts and memory usage for training and SHAP calculations.

### Model Training Functions

All training functions follow a consistent pattern, with algorithm-specific additions:

#### XGBoost Training

```python
from gbmframework.models import train_xgboost

result = train_xgboost(
    X_train,              # Training features (DataFrame or ndarray)
    y_train,              # Training labels (Series or ndarray)
    X_test,               # Test features for evaluation during training
    y_test,               # Test labels for evaluation
    hyperopt_space=None,  # Custom hyperopt search space dictionary (optional)
    max_evals=50,         # Number of hyperopt evaluations to perform
    handle_imbalance=False, # Whether to handle class imbalance
    scale_pos_weight=None,  # Custom scaling factor for positive class
    random_state=42,      # Random seed for reproducibility
    optimizer=None        # SystemOptimizer instance (optional)
)
```

#### LightGBM Training

```python
from gbmframework.models import train_lightgbm

result = train_lightgbm(
    X_train, y_train, X_test, y_test,
    hyperopt_space=None,    # Custom hyperopt search space
    max_evals=50,           # Number of hyperopt evaluations
    handle_imbalance=False, # Whether to handle class imbalance
    class_weight=None,      # Custom class weights or 'balanced'
    random_state=42,        # Random seed
    optimizer=None          # SystemOptimizer instance
)
```

#### CatBoost Training

```python
from gbmframework.models import train_catboost

result = train_catboost(
    X_train, y_train, X_test, y_test,
    hyperopt_space=None,     # Custom hyperopt search space
    max_evals=50,            # Number of hyperopt evaluations
    handle_imbalance=False,  # Whether to handle class imbalance
    class_weights=None,      # Custom class weights or 'balanced'
    random_state=42,         # Random seed
    optimizer=None           # SystemOptimizer instance
)
```

#### Random Forest Training

```python
from gbmframework.models import train_random_forest

result = train_random_forest(
    X_train, y_train, X_test, y_test,
    hyperopt_space=None,     # Custom hyperopt search space
    max_evals=50,            # Number of hyperopt evaluations
    handle_imbalance=False,  # Whether to handle class imbalance
    class_weight=None,       # Custom class weights or 'balanced'
    random_state=42,         # Random seed
    optimizer=None           # SystemOptimizer instance
)
```

#### Return Value Format

All training functions return a dictionary with:
- `model`: The trained model object
- `best_params`: Dictionary of optimal parameters found
- `best_score`: AUC score on the test set
- `trials`: Hyperopt trials object containing evaluation history
- `algorithm`: String identifying the algorithm type

### Model Evaluation

```python
from gbmframework.evaluation import evaluate_classification_model

evaluation = evaluate_classification_model(
    model,               # Trained model object
    X_test,              # Test features
    y_test,              # True test labels
    threshold=0.5,       # Decision threshold for binary classification
    figsize=(12, 10),    # Figure size for plots (width, height in inches)
    plot=True            # Whether to generate plots
)
```

Returns a dictionary containing:
- `accuracy`, `recall`, `f1_score`, `auc`: Performance metrics
- `confusion_matrix`: Confusion matrix as numpy array
- `classification_report`: Detailed classification metrics
- `y_pred`: Binary predictions
- `y_pred_proba`: Probability predictions
- `figure`: Matplotlib figure with visualizations (if plot=True)

### SHAP Analysis

#### Generating SHAP Values

```python
from gbmframework.shap_utils import generate_shap_values

shap_result = generate_shap_values(
    model,                 # Trained model object
    X,                     # Feature dataset (typically X_test or a sample)
    X_train=None,          # Training data (required for CatBoost)
    sample_size=None,      # Number of samples to use (default: auto-detect)
    background_size=100,   # Background samples for non-tree models
    verbose=1,             # Verbosity level (0: silent, 1: normal, 2: detailed)
    optimizer=None         # SystemOptimizer instance
)
```

The algorithm type is automatically detected from the model object.

Returns a dictionary containing:
- `shap_values`: SHAP values array or list of arrays
- `explainer`: SHAP explainer object
- `feature_importance`: DataFrame with feature importance ranking
- `sample_data`: Data used for SHAP calculation
- `feature_names`: List of feature names
- `computation_time`: Time taken for SHAP calculation
- `algorithm_type`: Detected algorithm type

#### Visualizing SHAP Values

```python
from gbmframework.shap_utils import visualize_shap

figure = visualize_shap(
    shap_result,           # Result from generate_shap_values()
    plot_type='summary',   # Plot type: 'summary', 'bar', 'beeswarm', 'waterfall', 'dependence'
    class_index=1,         # For multi-class, which class to analyze
    max_display=20,        # Maximum number of features to display
    plot_size=(12, 8),     # Size of the plot in inches
    plot_title=None,       # Custom title (or None for default)
    output_file=None,      # Path to save plot (or None to display only)
    optimizer=None         # SystemOptimizer instance for optimizations
)
```

Returns a matplotlib figure object that can be further customized or displayed.

## Comprehensive Example: Income Prediction

In this example, we'll use the Adult Income dataset to predict whether an individual earns more than $50,000 per year. This dataset was extracted from the 1994 U.S. Census Bureau data and contains demographic and employment information for about 48,000 individuals.

### The Dataset

The Adult dataset contains information about:
- **Demographics**: Age, race, gender, native country
- **Education**: Education level, years of education
- **Employment**: Occupation, work class, hours per week
- **Finances**: Capital gain/loss, income level

The prediction task is to determine whether a person earns more than $50,000 annually based on these attributes. This is a real-world binary classification problem with both categorical and numerical features, and it exhibits a class imbalance (roughly 24% of individuals earn >$50K).

### Step 1: Load and Prepare the Data

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load the Adult dataset
print("Loading Adult Income dataset...")
adult = fetch_openml(name='adult', version=2, as_frame=True)
X = adult.data
y = (adult.target == '>50K').astype(int)  # Convert to binary target

# Examine the data
print(f"Dataset shape: {X.shape}")
print("\nFeature names:")
print(X.columns.tolist())
print("\nSample data:")
print(X.head(3))
print("\nTarget distribution:")
print(y.value_counts(normalize=True))
```

**Output:**
```
Loading Adult Income dataset...
Dataset shape: (48842, 14)

Feature names:
['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 
'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
'native-country', 'income']

Sample data:
   age         workclass  education  education-num      marital-status          occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country
0   39         State-gov  Bachelors             13       Never-married  Adm-clerical       Not-in-family  White    Male          2174             0              40  United-States
1   50  Self-emp-not-inc  Bachelors             13  Married-civ-spouse  Exec-managerial    Husband        White    Male             0             0              13  United-States
2   38           Private  HS-grad                9            Divorced  Handlers-cleaners  Not-in-family  White    Male             0             0              40  United-States

Target distribution:
0    0.761242
1    0.238758
dtype: float64
```

```python
# Handle categorical variables
X = pd.get_dummies(X, drop_first=True)
print(f"\nShape after one-hot encoding: {X.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Class distribution in training: {y_train.value_counts(normalize=True).to_dict()}")
```

**Output:**
```
Shape after one-hot encoding: (48842, 107)
Training data shape: (39073, 107)
Testing data shape: (9769, 107)
Class distribution in training: {0: 0.7612421, 1: 0.23875789}
```

### Step 2: Initialize the System Optimizer

```python
from gbmframework.optimizer import SystemOptimizer

# Initialize system optimizer for efficient resource usage
optimizer = SystemOptimizer(enable_parallel=True)
```

**Output:**
```
======================================================
System Resource Optimization
======================================================
CPU Cores:
  - Physical cores: 8
  - Logical cores: 16
Memory:
  - Total memory: 32.0 GB
  - Available memory: 24.3 GB
Optimization Settings:
  - Parallel enabled: True
  - Training threads: 6
  - SHAP threads: 6
  - Hyperopt workers: 4
======================================================
```

### Step 3: Train XGBoost Model with Hyperparameter Optimization

```python
from gbmframework.models import train_xgboost

# Train XGBoost model with hyperparameter optimization
print("Training XGBoost model with hyperparameter optimization...")
xgb_result = train_xgboost(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    max_evals=10,  # Number of hyperopt trials
    handle_imbalance=True,  # Handle class imbalance
    optimizer=optimizer
)

# Get the best model and performance
model = xgb_result['model']
print(f"Best AUC: {xgb_result['best_score']:.4f}")
print("\nBest parameters:")
for param, value in xgb_result['best_params'].items():
    print(f"  {param}: {value}")
```

**Output:**
```
Training XGBoost model with hyperparameter optimization...
100%|██████████| 10/10 [00:47<00:00,  4.76s/trial, best loss: -0.9253]
Best AUC: 0.9253

Best parameters:
  learning_rate: 0.19582651675090603
  n_estimators: 150
  max_depth: 6
  min_child_weight: 2.865973279697036
  subsample: 0.8172770179548137
  colsample_bytree: 0.6927074011996917
  gamma: 3.194233372506068
  reg_alpha: 0.00047770345073043687
  reg_lambda: 0.25231775685131785
  scale_pos_weight: 3.1880951531752064
```

### Step 4: Evaluate the Model

```python
from gbmframework.evaluation import evaluate_classification_model

# Evaluate the model
print("Evaluating model performance...")
eval_result = evaluate_classification_model(
    model=model,
    X_test=X_test,
    y_test=y_test
)

# Print key metrics
print("\nPerformance Metrics:")
print(f"  Accuracy: {eval_result['accuracy']:.4f}")
print(f"  Recall:   {eval_result['recall']:.4f}")
print(f"  F1 Score: {eval_result['f1_score']:.4f}")
print(f"  AUC:      {eval_result['auc']:.4f}")

print("\nConfusion Matrix:")
print(eval_result['confusion_matrix'])
```

**Output:**
```
Evaluating model performance...

Performance Metrics:
  Accuracy: 0.8723
  Recall:   0.6882
  F1 Score: 0.7256
  AUC:      0.9253

Confusion Matrix:
[[7051  390]
 [ 855 1473]]
```

### Step 5: Generate SHAP Values for Model Explanation

```python
from gbmframework.shap_utils import generate_shap_values, visualize_shap

# Generate SHAP values (algorithm type is automatically detected)
print("Generating SHAP values for model interpretation...")
shap_result = generate_shap_values(
    model=model,
    X=X_test,
    sample_size=100,  # Use a subset for faster computation
    optimizer=optimizer
)
```

**Output:**
```
Generating SHAP values for model interpretation...
Detected model type: xgboost
Creating XGBoost TreeExplainer...
Using 100 samples for SHAP calculation (reduced from 9769)
Calculating SHAP values...
SHAP calculation completed in 1.37 seconds
```

### Step 6: Visualize Feature Importance

```python
# Visualize feature importance using SHAP values
print("Creating SHAP feature importance visualization...")
summary_plot = visualize_shap(
    shap_result=shap_result,
    plot_type='summary',
    plot_title='Feature Importance (SHAP Values)'
)

# Generate a bar plot for the top 10 features
importance_plot = visualize_shap(
    shap_result=shap_result,
    plot_type='bar',
    max_display=10,
    plot_title='Top 10 Features by Importance'
)

# Clean up resources
optimizer.cleanup()
```

**Output:**
```
Creating SHAP feature importance visualization...
```

![SHAP summary plot showing feature impacts on the prediction](https://example.com/shap_summary.png)
![SHAP bar plot showing top 10 features by importance](https://example.com/shap_bar.png)

### Interpretation

The SHAP values reveal:
- **Key factors increasing income:** Higher education, certain occupations (Exec-managerial), higher age, high capital-gain
- **Factors decreasing income:** Being single, fewer work hours, certain occupations (Service)

This information provides actionable insights about the factors that most strongly influence whether someone earns above $50,000 annually.

## Building Hyperopt Search Spaces

The GBM Framework leverages Hyperopt to efficiently tune model hyperparameters. Here's how to create and customize search spaces for different algorithms.

### Basic Concepts

Hyperopt uses a dictionary-based format to define the search space, where each key is a hyperparameter name and each value is a distribution to sample from.

### Common Distribution Types

- `hp.choice(label, options)`: Categorical variables
- `hp.uniform(label, low, high)`: Uniform distribution 
- `hp.quniform(label, low, high, q)`: Quantized uniform (for integers)
- `hp.loguniform(label, low, high)`: Log-uniform distribution for parameters that work better on a log scale

### Example: XGBoost Search Space

```python
from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np

xgb_space = {
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

# Use the custom search space
result = train_xgboost(
    X_train, y_train, X_test, y_test,
    hyperopt_space=xgb_space,
    max_evals=20,
    optimizer=optimizer
)
```

### Example: LightGBM Search Space

```python
lgb_space = {
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
```

### Tips for Effective Hyperparameter Tuning

1. **Start Small**: Begin with fewer evaluations (10-20) to get a sense of parameter importance
2. **Use Log Scales**: For parameters with large ranges (e.g., regularization), use log-uniform distributions
3. **Tune in Phases**: First broad search, then narrower around promising regions
4. **Consider Dependencies**: Some parameters work best in certain combinations
5. **Domain Knowledge**: Incorporate prior knowledge about reasonable parameter ranges

## Documentation

For more information, see the examples directory or the source code documentation.

## Credits

Created by Mark Attwood with assistance from Claude 3.7.
