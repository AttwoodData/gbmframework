"""
Utility functions for the GBM Framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import logging

logger = logging.getLogger(__name__)

def prepare_data(data, feature_columns=None, target_column=None, test_size=0.2, 
                random_state=42, scale=True, stratify=True):
    """
    Prepare data for machine learning.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing features and target
    feature_columns : list, optional
        List of feature column names. If None, all columns except target are used.
    target_column : str, optional
        Name of target column. Required if feature_columns is None.
    test_size : float, optional (default=0.2)
        Proportion of data to use for testing
    random_state : int, optional (default=42)
        Random seed for reproducibility
    scale : bool, optional (default=True)
        Whether to scale features using StandardScaler
    stratify : bool, optional (default=True)
        Whether to stratify the split based on the target variable
        
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test)
    """
    # Check inputs
    if feature_columns is None and target_column is None:
        raise ValueError("Either feature_columns or target_column must be provided")
    
    # Determine feature and target columns
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
    elif target_column is None:
        remaining_cols = [col for col in data.columns if col not in feature_columns]
        if len(remaining_cols) == 1:
            target_column = remaining_cols[0]
            logger.info(f"Using {target_column} as target column")
        else:
            raise ValueError("Multiple potential target columns found. Please specify target_column.")
    
    # Split features and target
    X = data[feature_columns]
    y = data[target_column]
    
    # Split data
    if stratify and len(np.unique(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Scale features if requested
    if scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    return X_train, X_test, y_train, y_test

def compare_models(models_dict, X_test, y_test, metric='accuracy', 
                  figsize=(10, 6), plot=True):
    """
    Compare multiple trained models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model name to trained model instance
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    metric : str, optional (default='accuracy')
        Metric to use for comparison ('accuracy' or 'roc_auc')
    figsize : tuple, optional (default=(10, 6))
        Figure size for the plot
    plot : bool, optional (default=True)
        Whether to create a plot
        
    Returns:
    --------
    pandas.DataFrame: Comparison results
    """
    results = {}
    
    for name, model in models_dict.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif metric == 'roc_auc':
            y_prob = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_prob)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Store results
        results[name] = score
    
    # Create result DataFrame
    results_df = pd.DataFrame(results.items(), columns=['Model', metric.capitalize()])
    results_df = results_df.sort_values(by=metric.capitalize(), ascending=False)
    
    # Plot results if requested
    if plot:
        plt.figure(figsize=figsize)
        sns.barplot(x=metric.capitalize(), y='Model', data=results_df)
        plt.title(f'Model Comparison ({metric.capitalize()})')
        plt.xlim(max(0.5, min(results.values()) - 0.05), 
                min(1.0, max(results.values()) + 0.05))
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    return results_df

def check_feature_importance_agreement(models_dict, n_features=10):
    """
    Check agreement between feature importance rankings across different models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model name to trained model instance
    n_features : int, optional (default=10)
        Number of top features to consider
        
    Returns:
    --------
    pandas.DataFrame: Feature importance rankings across models
    """
    # Extract feature importance rankings
    importance_ranks = {}
    
    for name, model in models_dict.items():
        # Check if SHAP importance is available
        if hasattr(model, 'shap_importance_df') and model.shap_importance_df is not None:
            importance_df = model.shap_importance_df
        elif hasattr(model, 'importance_df') and model.importance_df is not None:
            importance_df = model.importance_df
        else:
            logger.warning(f"No feature importance found for {name}")
            continue
        
        # Get top features and their ranks
        top_features = importance_df.head(n_features)['Feature'].tolist()
        for i, feature in enumerate(top_features):
            if feature not in importance_ranks:
                importance_ranks[feature] = {}
            importance_ranks[feature][name] = i + 1
    
    # Create DataFrame from rankings
    rank_df = pd.DataFrame.from_dict(importance_ranks, orient='index')
    
    # Fill NaN with 0 (feature not in top N)
    rank_df = rank_df.fillna(0)
    
    # Calculate agreement score (lower is better)
    rank_df['Agreement_Score'] = rank_df.replace(0, np.nan).std(axis=1, skipna=True)
    rank_df['Mean_Rank'] = rank_df.replace(0, np.nan).mean(axis=1, skipna=True)
    
    # Sort by mean rank and agreement score
    rank_df = rank_df.sort_values(['Mean_Rank', 'Agreement_Score'])
    
    # Format the DataFrame
    for col in rank_df.columns:
        if col not in ['Agreement_Score', 'Mean_Rank']:
            rank_df[col] = rank_df[col].astype(int)
    
    return rank_df

def create_ensemble(models_dict, X, method='average'):
    """
    Create an ensemble prediction from multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model name to trained model instance
    X : pandas.DataFrame
        Features to predict on
    method : str, optional (default='average')
        Method for combining predictions ('average', 'vote')
        
    Returns:
    --------
    numpy.ndarray: Ensemble predictions
    """
    if method not in ['average', 'vote']:
        raise ValueError(f"Unsupported method: {method}. Choose from 'average' or 'vote'.")
    
    # Get predictions from all models
    predictions = {}
    for name, model in models_dict.items():
        if method == 'average':
            # Get probability predictions for positive class
            predictions[name] = model.predict_proba(X)[:, 1]
        else:  # 'vote'
            # Get class predictions (0 or 1)
            predictions[name] = model.predict(X)
    
    # Create DataFrame of predictions
    pred_df = pd.DataFrame(predictions)
    
    # Combine predictions
    if method == 'average':
        # Average probabilities
        ensemble_pred_proba = pred_df.mean(axis=1).values
        # Convert to class predictions
        ensemble_pred = (ensemble_pred_proba >= 0.5).astype(int)
        return ensemble_pred, ensemble_pred_proba
    else:  # 'vote'
        # Majority vote
        ensemble_pred = (pred_df.mean(axis=1) >= 0.5).astype(int)
        return ensemble_pred
