"""
XGBoost implementation with SHAP analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings

from gbmframework.base import GBMBase

class XGBoostModel(GBMBase):
    """XGBoost implementation with SHAP analysis."""
    
    def __init__(self, random_state=42):
        """Initialize XGBoost model."""
        super().__init__(random_state)
    
    def fit(self, X_train, X_test, y_train, y_test, params=None, 
            create_shap_plots=True, create_importance_plots=True, 
            sample_size=1000):
        """Train XGBoost model with SHAP analysis."""
        self.feature_names = X_train.columns.tolist()
        
        # Default XGBoost parameters
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'random_state': self.random_state
            # Removed deprecated 'use_label_encoder' parameter
        }
        
        # Update default parameters with provided ones
        xgb_params = default_params.copy()
        if params:
            xgb_params.update(params)
        
        # Handle class imbalance if specified
        if 'class_weight' in xgb_params:
            class_weights = xgb_params.pop('class_weight')
            # Convert sklearn-style class weights to XGBoost scale_pos_weight
            if isinstance(class_weights, dict) and 1 in class_weights and 0 in class_weights:
                xgb_params['scale_pos_weight'] = class_weights[1] / class_weights[0]
        
        # Print data info
        self._print_data_info(X_train, X_test, y_train, y_test)
        
        # Train XGBoost model
        print(f"\nTraining XGBoost model with parameters:")
        print("-" * 50)
        for key, value in xgb_params.items():
            print(f"{key}: {value}")
        
        # Suppress XGBoost warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = xgb.XGBClassifier(**xgb_params)
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        self._evaluate_model(X_train, X_test, y_train, y_test)
        
        # Importance analysis
        if create_importance_plots:
            self._analyze_feature_importance()
        
        # SHAP analysis
        if create_shap_plots:
            self._analyze_shap(X_test, sample_size)
        
        # Compare importances if both calculated
        if create_shap_plots and create_importance_plots and hasattr(self, 'shap_importance_df'):
            self._compare_importance_methods()
        
        return self
    
    def _process_shap_values_for_instance(self, shap_values):
        """Process SHAP values for a single instance for XGBoost."""
        if isinstance(shap_values, list):
            return shap_values[1][0]  # For binary classification, take positive class
        return shap_values[0]
    
    def _get_expected_value(self):
        """Get expected value from the XGBoost SHAP explainer."""
        if isinstance(self.shap_explainer.expected_value, list):
            return self.shap_explainer.expected_value[1]
        return self.shap_explainer.expected_value
    
    def save_model(self, filename):
        """Save XGBoost model to file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        self.model.save_model(filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load XGBoost model from file."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(filename)
        print(f"Model loaded from {filename}")
        return self
    
    def _analyze_feature_importance(self):
        """Analyze XGBoost's built-in feature importance."""
        print("\nXGBoost Feature Importance (Built-in):")
        print("-" * 50)
        
        # Get feature importance
        xgb_importance = self.model.feature_importances_
        self.importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': xgb_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.importance_df, x='Importance', y='Feature')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.show()
        
        print(self.importance_df.to_string(index=False))
    
    def _analyze_shap(self, X_test, sample_size):
        """Perform SHAP analysis for XGBoost model."""
        import shap
        
        print("\nCalculating SHAP values...")
        print("-" * 50)
        
        # Sample data if needed
        if sample_size and sample_size < len(X_test):
            X_shap = X_test.sample(n=sample_size, random_state=self.random_state)
        else:
            X_shap = X_test
        
        # Initialize TreeExplainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.shap_explainer.shap_values(X_shap)
        
        # Handle different SHAP value formats
        shap_vals = self.shap_values
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1]  # For binary classification
        
        # Summary plot
        print("\nSHAP Summary Plot:")
        print("-" * 50)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_shap, show=False)
        plt.tight_layout()
        plt.show()
        
        # Bar plot
        print("\nSHAP Feature Importance Bar Plot:")
        print("-" * 50)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_shap, plot_type="bar", show=False)
        plt.tight_layout()
        plt.show()
        
        # Calculate SHAP-based feature importance
        shap_importance = np.abs(shap_vals).mean(axis=0)
        self.shap_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': shap_importance
        }).sort_values('Importance', ascending=False)
        
        print(self.shap_importance_df.to_string(index=False))
        
        # Generate dependence plots for top features
        top_features = self.shap_importance_df['Feature'].head(3).tolist()
        for feature in top_features:
            self.plot_shap_dependence(feature, X_shap)
        
        # Generate interaction plots if we have enough features
        if len(top_features) >= 2:
            feature1 = top_features[0]
            feature2 = top_features[1]
            self.plot_shap_dependence(feature1, X_shap, interaction_feature=feature2)
            self.plot_shap_dependence(feature2, X_shap, interaction_feature=feature1)
    
    def _compare_importance_methods(self):
        """Compare SHAP vs XGBoost native importance."""
        comparison = pd.merge(
            self.shap_importance_df, 
            self.importance_df, 
            on='Feature', 
            suffixes=('_SHAP', '_XGB')
        )
        comparison['Rank_SHAP'] = comparison['Importance_SHAP'].rank(ascending=False)
        comparison['Rank_XGB'] = comparison['Importance_XGB'].rank(ascending=False)
        comparison['Rank_Difference'] = abs(comparison['Rank_SHAP'] - comparison['Rank_XGB'])
        comparison = comparison.sort_values('Importance_SHAP', ascending=False)
        
        print("\nComparison of SHAP vs XGBoost Native Importance:")
        print("-" * 50)
        print(comparison)
        
        # Visualize the comparison
        plt.figure(figsize=(12, 6))
        pd.concat([
            comparison.set_index('Feature')['Importance_SHAP'].rename('SHAP'),
            comparison.set_index('Feature')['Importance_XGB'].rename('XGBoost')
        ], axis=1).plot(kind='bar')
        plt.title('SHAP vs XGBoost Native Feature Importance')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        self.importance_comparison = comparison
