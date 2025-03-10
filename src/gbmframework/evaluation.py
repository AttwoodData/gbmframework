"""
Evaluation - Functions for evaluating machine learning models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import optional dependencies with error handling
try:
    import seaborn as sns
    has_seaborn = True
except ImportError:
    has_seaborn = False
    
try:
    from sklearn.metrics import (
        accuracy_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report, roc_curve
    )
    has_sklearn_metrics = True
except ImportError:
    has_sklearn_metrics = False


def evaluate_classification_model(model, X_test, y_test, threshold=0.5, figsize=(12, 10), plot=True):
    """
    Evaluate a classification model with multiple metrics and plots.
    
    Parameters:
    -----------
    model : trained model object
        The trained classification model to evaluate
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    y_test : pandas.Series or numpy.ndarray
        True test labels
    threshold : float, default=0.5
        Decision threshold for binary classification
    figsize : tuple, default=(12, 10)
        Figure size for plots
    plot : bool, default=True
        Whether to generate plots
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and figures
    """
    if not has_sklearn_metrics:
        raise ImportError("scikit-learn is required for model evaluation")
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Initialize results dictionary
    results = {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # If plotting is enabled
    if plot:
        # Check for matplotlib and seaborn
        if not has_seaborn:
            print("Warning: seaborn is not installed. Plots will use basic matplotlib.")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.tight_layout(pad=4.0)
        
        # Confusion Matrix
        if has_seaborn:
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[0, 0]
            )
        else:
            axes[0, 0].imshow(cm, cmap='Blues')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center')
            axes[0, 0].set_xticks([0, 1])
            axes[0, 0].set_xticklabels(['Negative', 'Positive'])
            axes[0, 0].set_yticks([0, 1])
            axes[0, 0].set_yticklabels(['Negative', 'Positive'])
            
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            # Get feature names if available
            if hasattr(X_test, 'columns'):
                feature_names = X_test.columns
            else:
                feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features (or fewer if less features exist)
            n_features = min(20, len(importance_df))
            if has_seaborn:
                sns.barplot(
                    x='importance',
                    y='feature',
                    data=importance_df.head(n_features),
                    ax=axes[0, 1]
                )
            else:
                axes[0, 1].barh(
                    importance_df.head(n_features)['feature'],
                    importance_df.head(n_features)['importance']
                )
                
            axes[0, 1].set_title('Feature Importance (Gini)')
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_ylabel('Feature')
        else:
            axes[0, 1].text(0.5, 0.5, 'Feature importance not available', 
                          horizontalalignment='center', verticalalignment='center')
        
        # Metrics as a table
        metrics_table = pd.DataFrame({
            'Metric': ['Accuracy', 'Recall', 'F1 Score', 'AUC'],
            'Value': [accuracy, recall, f1, auc]
        })
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(
            cellText=metrics_table.values,
            colLabels=metrics_table.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        axes[1, 0].set_title('Performance Metrics')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[1, 1].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        axes[1, 1].plot([0, 1], [0, 1], 'k--')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].legend(loc='lower right')
        
        # Add figure to results
        results['figure'] = fig
        
        # Display model architecture if available
        if hasattr(model, 'get_params'):
            print("Model Parameters:")
            for param, value in model.get_params().items():
                print(f"{param}: {value}")
    
    return results