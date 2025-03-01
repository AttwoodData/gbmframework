"""
LightGBM implementation with SHAP analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

from gbmframework.base import GBMBase

class LightGBMModel(GBMBase):
    """LightGBM implementation with SHAP analysis."""
    
    def __init__(self, random_state=42):
        """Initialize LightGBM model."""
        super().__init__(random_state)
    
    def fit(self, X_train, X_test, y_train, y_test, params=None, 
            create_shap_plots=True, create_importance_plots=True, 
            sample_size=1000):
        """Train LightGBM model with SHAP analysis."""
        self.feature_names = X_train.columns.tolist()
        
        # Default LightGBM parameters
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': -1,  # -1 means no limit
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': self.random_state,
            'verbose': -1
        }
        
        # Update default parameters with provided ones
        lgb_params = default_params.copy()
        if params:
            lgb_params.update(params)
        
        # Handle class imbalance if specified
        if 'class_weight' in lgb_params:
            class_weights = lgb_params.pop('class_weight')
            # LightGBM uses 'is_unbalance' or 'scale_pos_weight'
            if isinstance(class_weights, dict) and 1 in class_weights and 0 in class_weights:
                lgb_params['scale_pos_weight'] = class_weights[1] / class_weights[0]
        
        # Print data info
        self._print_data_info(X_train, X_test, y_train, y_test)
        
        # Train LightGBM model
        print(f"\nTraining LightGBM model with parameters:")
        print("-" * 50)
        for key, value in lgb_params.items():
            print(f"{key}: {value}")
        
        # Extract n_estimators and remove it from params since it's passed directly to LGBMClassifier
        n_estimators = lgb_params.pop('n_estimators', 100)
        
        # Suppress LightGBM warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = lgb.LGBMClassifier(**lgb_params, n_estimators=n_estimators)
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
        """Process SHAP values for a single instance for LightGBM."""
        if isinstance(shap_values, list):
            # For binary classification, take the positive class (index 1)
            if len(shap_values) > 1:
                return shap_values[1][0]
            return shap_values[0][0]
        return shap_values[0]
    
    def _get_expected_value(self):
        """Get expected value from the LightGBM SHAP explainer."""
        if isinstance(self.shap_explainer.expected_value, list):
            # For binary classification, take the positive class
            if len(self.shap_explainer.expected_value) > 1:
                return self.shap_explainer.expected_value[1]
        return self.shap_explainer.expected_value
    
    def save_model(self, filename):
        """Save LightGBM model to file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        self.model.booster_.save_model(filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load LightGBM model from file."""
        self.model = lgb.Booster(model_file=filename)
        print(f"Model loaded from {filename}")
        return self
    
    def _analyze_feature_importance(self):
        """Analyze LightGBM's built-in feature importance."""
        print("\nLightGBM Feature Importance (Built-in):")
        print("-" * 50)
        
        # Get feature importance - LightGBM provides both 'split' and 'gain'
        importances = self.model.booster_.feature_importance(importance_type='gain')
        self.importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.importance_df, x='Importance', y='Feature')
        plt.title('LightGBM Feature Importance (gain)')
        plt.tight_layout()
        plt.show()
        
        print(self.importance_df.to_string(index=False))
    
    def _analyze_shap(self, X_test, sample_size):
        """Perform SHAP analysis for LightGBM model."""
        import shap
        
        print("\nCalculating SHAP values...")
        print("-" * 50)
        print("Note: You may see a warning about LightGBM SHAP values format changing.")
        print("This is expected and handled in our code.")
        
        # Sample data if needed
        if sample_size and sample_size < len(X_test):
            X_shap = X_test.sample(n=sample_size, random_state=self.random_state)
        else:
            X_shap = X_test
        
        # Initialize TreeExplainer with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer shap values output has changed")
            self.shap_explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.shap_explainer.shap_values(X_shap)
        
        # Handle difference in SHAP values format for LightGBM
        if isinstance(self.shap_values, list):
            # For binary classification, take the positive class (index 1)
            if len(self.shap_values) > 1:
                shap_vals = self.shap_values[1]
            else:
                shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
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
        """Compare SHAP vs LightGBM native importance."""
        comparison = pd.merge(
            self.shap_importance_df, 
            self.importance_df, 
            on='Feature', 
            suffixes=('_SHAP', '_LGB')
        )
        comparison['Rank_SHAP'] = comparison['Importance_SHAP'].rank(ascending=False)
        comparison['Rank_LGB'] = comparison['Importance_LGB'].rank(ascending=False)
        comparison['Rank_Difference'] = abs(comparison['Rank_SHAP'] - comparison['Rank_LGB'])
        comparison = comparison.sort_values('Importance_SHAP', ascending=False)
        
        print("\nComparison of SHAP vs LightGBM Native Importance:")
        print("-" * 50)
        print(comparison)
        
        # Visualize the comparison
        plt.figure(figsize=(12, 6))
        pd.concat([
            comparison.set_index('Feature')['Importance_SHAP'].rename('SHAP'),
            comparison.set_index('Feature')['Importance_LGB'].rename('LightGBM')
        ], axis=1).plot(kind='bar')
        plt.title('SHAP vs LightGBM Native Feature Importance')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        self.importance_comparison = comparison
