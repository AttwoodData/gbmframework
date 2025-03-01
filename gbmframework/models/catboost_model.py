"""
CatBoost implementation with SHAP analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import catboost as cb
import warnings

from gbmframework.base import GBMBase

class CatBoostModel(GBMBase):
    """CatBoost implementation with SHAP analysis."""
    
    def __init__(self, random_state=42):
        """Initialize CatBoost model."""
        super().__init__(random_state)
    
    def fit(self, X_train, X_test, y_train, y_test, params=None, 
            create_shap_plots=True, create_importance_plots=True, 
            sample_size=1000, cat_features=None):
        """Train CatBoost model with SHAP analysis."""
        self.feature_names = X_train.columns.tolist()
        
        # Default CatBoost parameters
        default_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'rsm': 0.8,            # Random subspace method (similar to colsample_bytree)
            'subsample': 0.8,      # Equivalent to bagging_fraction
            'min_data_in_leaf': 1,
            'random_seed': self.random_state,
            'verbose': 100,        # Print every 100 iterations
            'task_type': 'CPU'     # Use 'GPU' if available
        }
        
        # Update default parameters with provided ones
        cb_params = default_params.copy()
        if params:
            cb_params.update(params)
        
        # Handle class imbalance if specified
        if 'class_weight' in cb_params:
            class_weights = cb_params.pop('class_weight')
            # CatBoost uses 'auto_class_weights' or directly specified weights
            if isinstance(class_weights, dict):
                if 'auto' in class_weights and class_weights['auto']:
                    cb_params['auto_class_weights'] = 'Balanced'
                else:
                    # Convert dict weights to CatBoost format
                    class_weight_list = [class_weights.get(i, 1.0) for i in sorted(class_weights.keys())]
                    cb_params['class_weights'] = class_weight_list
        
        # Print data info
        self._print_data_info(X_train, X_test, y_train, y_test)
        
        # Automatic categorical feature detection if not specified
        if cat_features is None:
            # Identify categorical features (object, category, or bool types)
            cat_features = [col for col, dtype in X_train.dtypes.items() 
                          if dtype == 'object' or dtype == 'category' or dtype == 'bool']
            
            if cat_features:
                print(f"\nAutomatically detected categorical features: {cat_features}")
        
        # Convert iterations parameter
        iterations = cb_params.pop('iterations', 100)
        
        # Train CatBoost model
        print(f"\nTraining CatBoost model with parameters:")
        print("-" * 50)
        for key, value in cb_params.items():
            print(f"{key}: {value}")
        
        # Suppress verbose output from CatBoost
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = cb.CatBoostClassifier(iterations=iterations, **cb_params)
            self.model.fit(
                X_train, y_train,
                cat_features=cat_features,
                eval_set=(X_test, y_test),
                verbose=False  # Suppress output, as we'll handle it ourselves
            )
        
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
        """Process SHAP values for a single instance for CatBoost."""
        # CatBoost typically returns a single array for binary classification
        return shap_values[0]
    
    def _get_expected_value(self):
        """Get expected value from the CatBoost SHAP explainer."""
        return self.shap_explainer.expected_value
    
    def save_model(self, filename):
        """Save CatBoost model to file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        self.model.save_model(filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load CatBoost model from file."""
        self.model = cb.CatBoostClassifier()
        self.model.load_model(filename)
        print(f"Model loaded from {filename}")
        return self
    
    def _analyze_feature_importance(self):
        """Analyze CatBoost's built-in feature importance."""
        print("\nCatBoost Feature Importance (Built-in):")
        print("-" * 50)
        
        # Get feature importance - CatBoost provides various types
        importances = self.model.get_feature_importance()
        self.importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.importance_df, x='Importance', y='Feature')
        plt.title('CatBoost Feature Importance')
        plt.tight_layout()
        plt.show()
        
        print(self.importance_df.to_string(index=False))
    
    def _analyze_shap(self, X_test, sample_size):
        """Perform SHAP analysis for CatBoost model."""
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
        
        # Summary plot
        print("\nSHAP Summary Plot:")
        print("-" * 50)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X_shap, show=False)
        plt.tight_layout()
        plt.show()
        
        # Bar plot
        print("\nSHAP Feature Importance Bar Plot:")
        print("-" * 50)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X_shap, plot_type="bar", show=False)
        plt.tight_layout()
        plt.show()
        
        # Calculate SHAP-based feature importance
        shap_importance = np.abs(self.shap_values).mean(axis=0)
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
        """Compare SHAP vs CatBoost native importance."""
        comparison = pd.merge(
            self.shap_importance_df, 
            self.importance_df, 
            on='Feature', 
            suffixes=('_SHAP', '_CB')
        )
        comparison['Rank_SHAP'] = comparison['Importance_SHAP'].rank(ascending=False)
        comparison['Rank_CB'] = comparison['Importance_CB'].rank(ascending=False)
        comparison['Rank_Difference'] = abs(comparison['Rank_SHAP'] - comparison['Rank_CB'])
        comparison = comparison.sort_values('Importance_SHAP', ascending=False)
        
        print("\nComparison of SHAP vs CatBoost Native Importance:")
        print("-" * 50)
        print(comparison)
        
        # Visualize the comparison
        plt.figure(figsize=(12, 6))
        pd.concat([
            comparison.set_index('Feature')['Importance_SHAP'].rename('SHAP'),
            comparison.set_index('Feature')['Importance_CB'].rename('CatBoost')
        ], axis=1).plot(kind='bar')
        plt.title('SHAP vs CatBoost Native Feature Importance')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        self.importance_comparison = comparison
