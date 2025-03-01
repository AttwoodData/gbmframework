"""
Base class for Gradient Boosting Models with integrated SHAP analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class GBMBase:
    """Base class for Gradient Boosting Models with SHAP analysis."""
    
    def __init__(self, random_state=42):
        """Initialize the base class with common parameters."""
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.feature_names = None
        self.importance_df = None
        self.shap_values = None
        self.shap_explainer = None
        self.shap_importance_df = None
        self.importance_comparison = None
    
    def fit(self, X_train, X_test, y_train, y_test, params=None, 
            create_shap_plots=True, create_importance_plots=True, 
            sample_size=1000):
        """
        Train the model and analyze with feature importance and SHAP.
        
        This is an abstract method that should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return class probabilities."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.predict_proba(X)
    
    def plot_confusion_matrix(self, y_true, y_pred, title='Confusion Matrix'):
        """Plot confusion matrix with percentages."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_shap_dependence(self, feature, X, interaction_feature=None):
        """Create a SHAP dependence plot for a specific feature."""
        if self.shap_values is None:
            raise ValueError("SHAP values have not been calculated. Set create_shap_plots=True in fit().")
        
        # Import SHAP only when needed
        import shap
        
        plt.figure(figsize=(10, 6))
        if interaction_feature:
            shap.dependence_plot(
                feature, self.shap_values, X, 
                interaction_index=interaction_feature,
                show=False
            )
            plt.title(f'SHAP Dependence Plot: {feature} with {interaction_feature} Interaction')
        else:
            shap.dependence_plot(feature, self.shap_values, X, show=False)
            plt.title(f'SHAP Dependence Plot: {feature}')
        plt.tight_layout()
        plt.show()
    
    def explain_prediction(self, instance):
        """Explain an individual prediction using SHAP."""
        if self.model is None or self.shap_explainer is None:
            raise ValueError("Model or SHAP explainer not initialized.")
        
        # Import SHAP only when needed
        import shap
        
        # Process input
        if isinstance(instance, pd.Series):
            X = instance.to_frame().T
        else:
            X = instance
        
        # Get prediction
        pred_prob = self.predict_proba(X)[0, 1]
        pred_class = 'Positive Class' if pred_prob > 0.5 else 'Negative Class'
        
        # Calculate SHAP values for this instance
        shap_values = self.shap_explainer.shap_values(X)
        
        # Process SHAP values based on implementation
        shap_vals = self._process_shap_values_for_instance(shap_values)
        
        # Print prediction info
        print(f"Prediction: {pred_class}")
        print(f"Probability of positive class: {pred_prob:.2%}")
        
        # Plot SHAP values
        plt.figure(figsize=(10, 6))
        expected_value = self._get_expected_value()
        shap.force_plot(
            expected_value, 
            shap_vals, 
            X, 
            matplotlib=True,
            show=False
        )
        plt.title('Feature Contributions to Prediction')
        plt.tight_layout()
        plt.show()
        
        return {
            'shap_values': shap_vals,
            'prediction': pred_prob,
            'features': X.iloc[0].to_dict() if hasattr(X, 'iloc') else X
        }
    
    def _process_shap_values_for_instance(self, shap_values):
        """Process SHAP values for a single instance. Override in subclasses if needed."""
        return shap_values
    
    def _get_expected_value(self):
        """Get expected value from the SHAP explainer. Override in subclasses if needed."""
        return self.shap_explainer.expected_value
    
    def select_features_by_importance(self, threshold=0.9, use_shap=True):
        """
        Select features based on cumulative importance threshold.
        
        Parameters:
        threshold (float): Cumulative importance threshold (0-1)
        use_shap (bool): Whether to use SHAP-based importance (True) or model's native importance (False)
        
        Returns:
        list: Selected features
        """
        # Determine which importance dataframe to use
        if use_shap and self.shap_importance_df is not None:
            importance_df = self.shap_importance_df.copy()
        elif self.importance_df is not None:
            importance_df = self.importance_df.copy()
        else:
            raise ValueError("No feature importance calculated. Call fit() with create_importance_plots=True.")
        
        # Calculate normalized importance
        importance_df['Normalized'] = importance_df['Importance'] / importance_df['Importance'].sum()
        importance_df['Cumulative'] = importance_df['Normalized'].cumsum()
        
        # Select features based on threshold
        selected = importance_df[importance_df['Cumulative'] <= threshold]
        
        # Add one more feature if needed to meet threshold
        if len(selected) < len(importance_df):
            additional = importance_df.iloc[len(selected):len(selected)+1]
            selected = pd.concat([selected, additional])
        
        # Display selection info
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['Feature'], importance_df['Normalized'])
        plt.axhline(y=threshold/len(importance_df), color='r', linestyle='--')
        plt.title('Feature Importance Distribution')
        plt.ylabel('Normalized Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        print(f"Selected {len(selected)} out of {len(importance_df)} features")
        print(f"Cumulative importance: {selected['Cumulative'].max():.2%}")
        
        return selected['Feature'].tolist()
    
    def save_model(self, filename):
        """Save model to file. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement save_model()")
    
    def load_model(self, filename):
        """Load model from file. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def _print_data_info(self, X_train, X_test, y_train, y_test):
        """Print data split information."""
        print(f"\nData Split Information:")
        print("-" * 50)
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"\nTarget distribution in train:")
        print(pd.Series(y_train).value_counts(normalize=True))
        print(f"\nTarget distribution in test:")
        print(pd.Series(y_test).value_counts(normalize=True))
    
    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance on train and test sets."""
        # Model evaluation
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Training set performance
        print("\nTraining Set Performance:")
        print("-" * 50)
        print(classification_report(y_train, y_train_pred))
        self.plot_confusion_matrix(y_train, y_train_pred, "Training Set Confusion Matrix")
        
        # Test set performance
        print("\nTest Set Performance:")
        print("-" * 50)
        print(classification_report(y_test, y_test_pred))
        self.plot_confusion_matrix(y_test, y_test_pred, "Test Set Confusion Matrix")
        
        # Calculate metrics
        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_confusion_matrix': confusion_matrix(y_train, y_train_pred).tolist(),
            'test_confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist(),
        }
        
        # Check for overfitting
        print("\nOverfitting Check:")
        print("-" * 50)
        print(f"Training accuracy: {self.metrics['train_accuracy']:.4f}")
        print(f"Test accuracy: {self.metrics['test_accuracy']:.4f}")
        print(f"Accuracy difference: {(self.metrics['train_accuracy'] - self.metrics['test_accuracy']):.4f}")
