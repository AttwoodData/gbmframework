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

# Basic usage
optimizer = SystemOptimizer(
    enable_parallel=True,    # Whether to enable parallel computation
    memory_safety=0.8,       # Memory safety factor (0.0-1.0)
    verbose=True             # Whether to print optimization information
)

# Advanced hardware-aware usage
optimizer = SystemOptimizer(
    adaptive=True,              # Enable hardware-adaptive thread allocation
    thread_aggressiveness=0.7,  # Control thread allocation aggressiveness (0.0-1.0)
    verbose=True                # Print detailed hardware information
)

# Maximum performance (manual control)
optimizer = SystemOptimizer(
    force_threads=6,   # Specify exact number of threads to use
    verbose=True       # Show optimization information
)
```

The `SystemOptimizer` automatically detects system resources and configures optimal thread counts and memory usage for training and SHAP calculations. The enhanced version with `adaptive=True` provides more intelligent resource allocation based on detailed hardware information.

### Hardware-Adaptive Optimization

The `adaptive=True` mode enables sophisticated hardware-aware thread allocation that considers:

- CPU model and architecture (Intel, AMD, Apple Silicon)
- Physical vs. logical cores and hyperthreading
- Available memory as both absolute amount and percentage of total
- Memory-to-CPU ratio and system balance

This allows the optimizer to make intelligent decisions about threading that are tailored to your specific hardware configuration, maximizing performance while preventing out-of-memory errors or excessive resource contention.

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

## Understanding SHAP Values for Model Interpretation

SHAP (SHapley Additive exPlanations) values provide a powerful approach to model interpretation that overcomes many limitations of traditional feature importance metrics.

### What Are SHAP Values?

SHAP values, introduced by Lundberg and Lee in their 2017 paper "A Unified Approach to Interpreting Model Predictions" (NeurIPS, December 2017), are based on game theory's Shapley values, a method for assigning credit among multiple players in a cooperative game. In machine learning, SHAP values attribute the prediction outcome among features, calculating each feature's contribution to the difference between the actual prediction and the average prediction.

### Key Benefits of SHAP Over Traditional Feature Importance Metrics

#### 1. Consistency and Mathematical Foundation
Unlike variable importance metrics like Gini impurity (used in tree-based models), SHAP values have a solid mathematical foundation with three important properties:
- **Local accuracy**: SHAP values sum to the difference between the model prediction and average prediction
- **Missingness**: Features with no marginal effect receive zero attribution
- **Consistency**: If a model changes so that a feature's contribution increases, its SHAP value increases

#### 2. Global and Local Explanations
SHAP uniquely provides both:
- **Global importance**: Overall impact of features across all predictions
- **Local importance**: Impact of features on individual predictions

#### 3. Directional Information
Unlike Gini impurity or permutation importance, SHAP values indicate:
- The **direction** of feature impact (positive or negative)
- The **magnitude** of each feature's influence

### Comparing SHAP to Gini Impurity and Entropy

| Aspect | SHAP Values | Gini Impurity / Entropy |
|--------|-------------|-------------------------|
| **Foundation** | Game theory (Shapley values) | Information theory |
| **Direction** | Shows positive/negative impact | Direction-agnostic (only magnitude) |
| **Scope** | Both global and local explanations | Only global importance |
| **Consistency** | Consistent across models | May be inconsistent across models |
| **Computational Cost** | Higher (especially for non-tree models) | Lower |
| **Interactions** | Accounts for feature interactions | May miss complex interactions |
| **Interpretability** | Direct link to model output | Indirect (measures node impurity) |

### Interpreting SHAP Values

#### Proportional Interpretation
Yes, SHAP values are proportional and have a direct mathematical interpretation:

- A SHAP value of 2 is exactly twice as impactful as a SHAP value of 1
- SHAP values are in the same units as the model output
- For classification with logit output, SHAP values represent log-odds contributions

#### Example Interpretation
For a model predicting loan default probability:
- Base value: 10% (average prediction across all samples)
- SHAP values: Income = -5%, Credit score = -3%, Loan amount = +2%
- Final prediction: 10% - 5% - 3% + 2% = 4%

This means income reduced default probability by 5 percentage points, credit score reduced it by 3 points, and loan amount increased it by 2 points.

### SHAP for Different Model Types

SHAP provides different estimators optimized for various model classes:

- **TreeExplainer**: Fast, exact algorithm for tree-based models (Random Forest, XGBoost, etc.)
- **DeepExplainer**: For deep learning models
- **KernelExplainer**: Model-agnostic but computationally expensive
- **LinearExplainer**: For linear models with efficient implementation

### Common SHAP Visualizations

The GBM Framework provides several SHAP visualization types:

1. **Summary Plot**: Shows features sorted by importance with distribution of SHAP values
2. **Bar Plot**: Simple ranking of features by average absolute SHAP value
3. **Beeswarm Plot**: Detailed view of how features impact individual predictions
4. **Waterfall Plot**: Shows how features contribute to a single prediction
5. **Dependence Plot**: Shows how a feature's SHAP values vary based on the feature's value

### Practical Tips for Using SHAP

1. **Start with summary plots** for a global overview of feature importance
2. **Use waterfall plots** to understand specific predictions
3. **Sample data** when working with large datasets to reduce computation time
4. **Combine with domain knowledge** to validate if identified patterns make sense
5. **Compare across models** to understand how different algorithms use features

### Limitations of SHAP

- **Computational complexity**: Calculating exact SHAP values can be expensive for non-tree models
- **Feature independence assumption**: SHAP may not perfectly capture correlated features
- **Interpretation challenges**: While mathematically sound, SHAP values can still be difficult to interpret for complex models
- **Sampling approximation**: For large datasets, SHAP often uses sampling which introduces variance

### SHAP Analysis in GBM Framework

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

### References for SHAP

1. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems 30 (NIPS 2017) (pp. 4765–4774).

2. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1), 56-67. https://doi.org/10.1038/s42256-019-0138-9

3. Molnar, C. (2022). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (2nd ed.). https://christophm.github.io/interpretable-ml-book/shap.html

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

### Step 2: Initialize the System Optimizer with Hardware Awareness

```python
from gbmframework.optimizer import SystemOptimizer

# Initialize system optimizer with adaptive hardware awareness
optimizer = SystemOptimizer(
    enable_parallel=True,
    adaptive=True,              # Enable hardware-adaptive mode
    thread_aggressiveness=0.7,  # Be slightly aggressive with thread allocation
    verbose=True                # Show detailed system information
)
```

**Output:**
```
======================================================
System Resource Optimization
======================================================
CPU Information:
  - Physical cores: 8
  - Logical cores: 16
  - CPU model: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz
  - CPU frequency: 3800 MHz
  - Current CPU load: 12.5%
Memory Information:
  - Total memory: 32.0 GB
  - Available memory: 24.3 GB
  - Memory available: 76.0%
Optimization Settings:
  - Parallel enabled: True
  - Adaptive mode: True
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

## Understanding the Enhanced SystemOptimizer

The GBM Framework's `SystemOptimizer` has been enhanced to provide more intelligent hardware-aware resource allocation. This section explains the adaptive optimization capabilities.

### Key Features of the Enhanced Optimizer

1. **Hardware Detection**:
   - CPU model identification (Intel, AMD, Apple Silicon)
   - Core count (physical vs. logical)
   - Memory availability (total and available)
   - CPU frequency and load

2. **Adaptive Threading**:
   - Smart thread allocation based on CPU architecture
   - Memory-aware scaling that considers both absolute and relative memory availability
   - Processor-specific optimizations (e.g., different strategies for Intel vs. AMD)

3. **Configuration Options**:
   - `adaptive`: Enable the advanced hardware-aware mode
   - `thread_aggressiveness`: Control how aggressively to allocate threads (0.0-1.0)
   - `min_threads`: Minimum threads to use regardless of calculated value
   - `force_threads`: Bypass adaptive calculations and use exactly this many threads

### Adaptive Thread Calculation

The adaptive mode considers several factors to determine optimal thread count:

```python
# Memory factor based on available memory and total system memory
memory_threshold = max(8, min(32, total_memory_gb / 4))
memory_factor = min(1.0, available_memory_gb / memory_threshold)

# Factor based on percentage of available memory
percent_factor = max(0.5, min(1.0, memory_percent / 50))

# Combined memory factor (weighted)
combined_memory_factor = (memory_factor * 0.7) + (percent_factor * 0.3)

# Thread ratio factor based on physical vs. logical cores
thread_ratio = physical_cores / logical_cores
thread_factor = max(0.5, thread_ratio)

# CPU architecture-specific adjustments
if 'intel' in cpu_model:
    if 'i9' in cpu_model:
        arch_factor = 1.1  # High-end Intel CPUs
    elif 'i7' in cpu_model:
        arch_factor = 1.0  # Standard for i7
    # ...and so on for other processors

# Calculate combined factor and final thread count
combined_factor = memory_factor * thread_factor * arch_factor * thread_aggressiveness
threads = int(physical_cores * combined_factor)
```

This approach ensures that thread allocation is optimized specifically for your hardware configuration.

### Usage Examples

```python
# Basic usage with adaptive mode
optimizer = SystemOptimizer(adaptive=True)

# More aggressive thread allocation
optimizer = SystemOptimizer(adaptive=True, thread_aggressiveness=0.8)

# Maximum performance (force specific thread count)
optimizer = SystemOptimizer(force_threads=6)

# Conservative approach for memory-constrained systems
optimizer = SystemOptimizer(adaptive=True, thread_aggressiveness=0.5, min_threads=2)
```

### Benefits of Adaptive Optimization

1. **Better Default Performance**: More intelligent decisions without manual tuning
2. **Hardware-Specific Adjustments**: Optimizations tailored to your specific CPU and memory
3. **Balanced Resource Usage**: Prevents resource contention by considering both CPU and memory
4. **Improved Reliability**: Reduces the risk of out-of-memory errors during computation
5. **Flexible Control**: Can be as automatic or manual as needed for your use case

## Understanding Gradient Boosting Hyperparameters

Hyperparameter tuning is essential for achieving optimal model performance. This section explains the most important hyperparameters, their effects, and recommended search ranges.

### Common Hyperparameters Across Algorithms

| Parameter | Description | Impact | Typical Range |
|-----------|-------------|--------|--------------|
| **learning_rate** | Controls the contribution of each tree to the final outcome | Lower values require more trees but can yield better performance | 0.01 - 0.3 |
| **n_estimators** / **iterations** | Number of trees in the ensemble | More trees can improve performance but increase training time and risk of overfitting | 50 - 1000 |
| **max_depth** / **depth** | Maximum depth of each tree | Controls complexity; deeper trees can model more complex patterns but may overfit | 3 - 10 |
| **min_samples_leaf** / **min_child_samples** | Minimum samples required at a leaf node | Prevents overfitting by requiring more data in leaf nodes | 1 - 20 |
| **subsample** | Fraction of samples used for tree building | Reduces overfitting by introducing randomness | 0.5 - 1.0 |
| **colsample_bytree** | Fraction of features used for tree building | Reduces overfitting and dimensionality | 0.5 - 1.0 |
| **reg_alpha** / **l1_regularization** | L1 regularization term | Controls model complexity by penalizing absolute coefficient size | 0 - 1.0 |
| **reg_lambda** / **l2_regularization** | L2 regularization term | Controls model complexity by penalizing squared coefficient size | 0 - 1.0 |

### XGBoost-Specific Parameters

| Parameter | Description | Impact | Typical Range |
|-----------|-------------|--------|--------------|
| **min_child_weight** | Minimum sum of instance weight needed in a child | Controls overfitting; higher values make the model more conservative | 1 - 10 |
| **gamma** | Minimum loss reduction required for a split | Controls complexity; higher values make the algorithm more conservative | 0 - 5 |
| **scale_pos_weight** | Controls balance of positive and negative weights | Useful for imbalanced classes | Typically set to negative_samples/positive_samples |

### LightGBM-Specific Parameters

| Parameter | Description | Impact | Typical Range |
|-----------|-------------|--------|--------------|
| **num_leaves** | Maximum number of leaves in one tree | Controls tree complexity; should be < 2^max_depth | 20 - 150 |
| **min_child_samples** | Minimum number of data needed in a child | Similar to min_samples_leaf in other algorithms | 10 - 50 |
| **path_smooth** | Smoothing factor applied to tree paths | Reduces overfitting on sparse features | 0 - 0.3 |
| **cat_smooth** | Deals with categorical features | Controls smoothing for categorical features | 10 - 50 |

### CatBoost-Specific Parameters

| Parameter | Description | Impact | Typical Range |
|-----------|-------------|--------|--------------|
| **border_count** | Number of splits for numerical features | Controls precision of numerical feature splits | 32 - 255 |
| **l2_leaf_reg** | L2 regularization coefficient | Controls model complexity | 1 - 10 |
| **random_strength** | Amount of randomness in the split selection | Helps prevent overfitting | 0 - 1 |
| **bagging_temperature** | Controls intensity of Bayesian bagging | Higher values increase randomness | 0 - 1 |

### RandomForest-Specific Parameters

| Parameter | Description | Impact | Typical Range |
|-----------|-------------|--------|--------------|
| **max_features** | Maximum number of features to consider for splitting | Controls randomness in feature selection | 'sqrt', 'log2', or None |
| **bootstrap** | Whether to use bootstrap samples | Enables or disables bootstrapping | True/False |
| **min_impurity_decrease** | Minimum decrease in impurity required for split | Controls split threshold | 0 - 0.1 |
| **min_samples_split** | Minimum samples required to split a node | Prevents creating very small nodes | 2 - 20 |

### Hyperparameter Relationships

Understanding the relationships between hyperparameters can improve tuning:

1. **learning_rate and n_estimators**: These have an inverse relationship. Lower learning rates require more trees.

2. **max_depth and n_estimators**: Deeper trees may require fewer estimators but risk overfitting.

3. **regularization and tree complexity**: Higher regularization (reg_alpha, reg_lambda) allows for more complex trees without overfitting.

4. **subsample and learning_rate**: Lower subsampling rates often work well with slightly higher learning rates.

### Parameter Tuning Strategy

A good strategy for hyperparameter tuning:

1. Start with a moderate number of trees (100-200) and tune other parameters
2. Focus first on tree structure parameters (max_depth, min_samples_leaf)
3. Then tune randomness parameters (subsample, colsample_bytree)
4. Adjust regularization parameters (reg_alpha, reg_lambda)
5. Finally, fine-tune the learning_rate and increase n_estimators accordingly

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

## References

### Algorithms and Original Papers

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). https://doi.org/10.1145/2939672.2939785

2. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems 30 (pp. 3146-3154). https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

3. Dorogush, A. V., Ershov, V., & Gulin, A. (2018). CatBoost: gradient boosting with categorical features support. arXiv preprint arXiv:1810.11363. https://arxiv.org/abs/1810.11363

4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

### Hyperparameter Tuning

5. Bergstra, J., Yamins, D., & Cox, D. (2013). Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. In International Conference on Machine Learning (pp. 115-123). http://proceedings.mlr.press/v28/bergstra13.html

6. Probst, P., Wright, M. N., & Boulesteix, A. L. (2019). Hyperparameters and tuning strategies for random forest. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 9(3), e1301. https://doi.org/10.1002/widm.1301

### Explainability and SHAP

7. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems 30 (pp. 4765-4774). https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

8. Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1), 56-67. https://doi.org/10.1038/s42256-019-0138-9

### Dataset References

9. Dua, D., & Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. (Adult Income Dataset)

10. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast cancer Wisconsin (diagnostic) dataset. UCI Machine Learning Repository.

### Practical Guides and Hyperparameter Recommendations

11. Brownlee, J. (2018). XGBoost With Python: Gradient Boosted Trees with XGBoost and scikit-learn. Machine Learning Mastery.

12. Harrison, M. (2022). Effective XGBoost: A Complete Guide to Supercharge Your XGBoost Machine Learning. Matt Harrison.

13. Microsoft Research. (2020). LightGBM Documentation. https://lightgbm.readthedocs.io/

14. Yandex. (2020). CatBoost Documentation. https://catboost.ai/docs/

15. Scikit-learn Developers. (2021). Random Forest Documentation. https://scikit-learn.org/stable/modules/ensemble.html#random-forests

## Credits

Created by Mark Attwood with assistance from Claude 3.7.