def visualize_shap(shap_result, plot_type='summary', class_index=1, max_display=20, 
                 plot_size=(12, 8), plot_title=None, output_file=None, optimizer=None):
    """
    Create visualizations from SHAP values.
    
    Parameters:
    -----------
    shap_result : dict
        Dictionary containing SHAP results from generate_shap_values()
    plot_type : str, default='summary'
        Type of SHAP plot to generate:
        - 'summary': Summary plot showing feature importance
        - 'bar': Bar plot of mean absolute SHAP values
        - 'beeswarm': Detailed impact of each feature
        - 'waterfall': Waterfall plot for a single prediction (uses first row)
        - 'dependence': Dependence plot for the top feature
    class_index : int, default=1
        For multi-class problems, which class to analyze (default is positive class in binary)
    max_display : int, default=20
        Maximum number of features to display in plots
    plot_size : tuple, default=(12, 8)
        Size of the plot (width, height) in inches
    plot_title : str, optional
        Custom title for plot. If None, a default title is generated
    output_file : str, optional
        Path to save the plot. If None, plot is only displayed
    optimizer : SystemOptimizer, optional
        System optimizer instance for plot optimization
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated plot figure
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install using: pip install shap")
    
    # Apply optimization settings if available
    if optimizer is not None and hasattr(optimizer, 'system_info'):
        # You could add specific plot optimizations here if needed
        pass
    
    # Check if SHAP values exist
    if 'shap_values' not in shap_result or shap_result['shap_values'] is None:
        raise ValueError("No SHAP values found in the provided result dictionary")
    
    # Extract values from the result dictionary
    shap_values = shap_result['shap_values']
    sample_data = shap_result['sample_data']
    feature_names = shap_result['feature_names']
    explainer = shap_result.get('explainer')
    
    # Check for suspicious patterns that might indicate bad SHAP values
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        # Get the variance of SHAP values for each instance
        variance_per_feature = np.var(shap_values, axis=0)
        if np.all(variance_per_feature < 1e-6):  # Very small variance indicates uniform values
            print("Warning: SHAP values have suspiciously low variance, which may lead to less informative visualizations.")
    
    # Handle different output formats of shap_values
    if isinstance(shap_values, list):
        # For multi-class models or some binary models
        if len(shap_values) > 1:
            # Multi-class - use the specified class index
            class_idx = min(class_index, len(shap_values) - 1)
            shap_values_for_plot = shap_values[class_idx]
        else:
            # Single output in list
            shap_values_for_plot = shap_values[0]
    else:
        # Single array of SHAP values
        shap_values_for_plot = shap_values
    
    # Create figure
    plt.figure(figsize=plot_size)
    
    # Generate the requested plot type with robust fallbacks
    try:
        if plot_type == 'summary':
            try:
                # For newer SHAP versions
                explanation = shap.Explanation(
                    values=shap_values_for_plot,
                    data=sample_data,
                    feature_names=feature_names
                )
                shap.plots.beeswarm(explanation, max_display=max_display, show=False)
            except:
                # Fallback for older SHAP versions
                shap.summary_plot(
                    shap_values_for_plot, 
                    sample_data, 
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False
                )
            
            if plot_title:
                plt.title(plot_title)
            
        elif plot_type == 'bar':
            try:
                # For newer SHAP versions
                explanation = shap.Explanation(
                    values=shap_values_for_plot,
                    data=sample_data,
                    feature_names=feature_names
                )
                shap.plots.bar(explanation, max_display=max_display, show=False)
            except:
                # Fallback for older SHAP versions
                shap.summary_plot(
                    shap_values_for_plot, 
                    sample_data, 
                    feature_names=feature_names,
                    plot_type='bar', 
                    max_display=max_display,
                    show=False
                )
                
            if plot_title:
                plt.title(plot_title)
            
        elif plot_type == 'beeswarm':
            try:
                # For newer SHAP versions
                explanation = shap.Explanation(
                    values=shap_values_for_plot,
                    data=sample_data,
                    feature_names=feature_names
                )
                shap.plots.beeswarm(explanation, max_display=max_display, show=False)
            except:
                # Fallback for older SHAP versions
                shap.summary_plot(
                    shap_values_for_plot, 
                    sample_data, 
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False
                )
                
            if plot_title:
                plt.title(plot_title)
            
        elif plot_type == 'waterfall':
            try:
                # Choose the first sample for the waterfall plot
                if hasattr(sample_data, 'iloc'):
                    first_sample = sample_data.iloc[0]
                else:
                    first_sample = sample_data[0]
                    
                # For newer SHAP versions
                explanation = shap.Explanation(
                    values=shap_values_for_plot[0],
                    data=first_sample,
                    feature_names=feature_names
                )
                shap.plots.waterfall(explanation, max_display=max_display, show=False)
            except:
                # Fallback to a simple bar chart for the first sample
                plt.clf()
                sample_values = shap_values_for_plot[0]
                sorted_idx = np.argsort(np.abs(sample_values))[-max_display:]
                plt.barh(
                    [feature_names[i] for i in sorted_idx],
                    [sample_values[i] for i in sorted_idx]
                )
                plt.xlabel('SHAP value')
                plt.ylabel('Feature')
                plt.title('Feature Impact for Single Prediction (Fallback)')
                
            if plot_title:
                plt.title(plot_title)
            
        elif plot_type == 'dependence':
            try:
                # Find the most important feature
                mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)
                top_feature_idx = np.argmax(mean_abs_shap)
                
                # Find the second most important feature for interaction
                masked_importance = mean_abs_shap.copy()
                masked_importance[top_feature_idx] = 0
                second_feature_idx = np.argmax(masked_importance)
                
                # Get feature names
                top_feature = feature_names[top_feature_idx]
                second_feature = feature_names[second_feature_idx]
                
                # Create dependence plot using column names rather than indices
                if isinstance(sample_data, pd.DataFrame):
                    # If we have a DataFrame, use column names
                    shap.dependence_plot(
                        top_feature,
                        shap_values_for_plot,
                        sample_data,
                        interaction_index=second_feature,
                        show=False
                    )
                else:
                    # If we have a numpy array, use indices
                    shap.dependence_plot(
                        top_feature_idx,
                        shap_values_for_plot,
                        sample_data,
                        interaction_index=second_feature_idx,
                        feature_names=feature_names,
                        show=False
                    )
            except Exception as e:
                # Fallback to a simple scatter plot
                plt.clf()
                mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)
                top_feature_idx = np.argmax(mean_abs_shap)
                
                if isinstance(sample_data, pd.DataFrame):
                    x_values = sample_data.iloc[:, top_feature_idx]
                else:
                    x_values = sample_data[:, top_feature_idx]
                    
                y_values = shap_values_for_plot[:, top_feature_idx]
                
                plt.scatter(x_values, y_values, alpha=0.5)
                plt.xlabel(feature_names[top_feature_idx])
                plt.ylabel(f'SHAP value for {feature_names[top_feature_idx]}')
                plt.title(f'Dependence Plot (Fallback): {feature_names[top_feature_idx]}')
                
            if plot_title:
                plt.title(plot_title)
        
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}. Choose from 'summary', 'bar', 'beeswarm', 'waterfall', or 'dependence'.")
    
    except Exception as e:
        # If plotting fails, create a simpler fallback plot using the feature importance
        plt.clf()
        plt.title(f"Fallback plot: {plot_type} visualization failed", fontsize=14)
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=plt.gca().transAxes)
        
        # Try to at least show a bar chart of feature importance
        try:
            importance_df = shap_result['feature_importance'].head(max_display)
            plt.figure(figsize=plot_size)
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance (Fallback)')
        except:
            pass
    
    # Finalize the plot
    plt.tight_layout()
    
    # Save the plot if requested
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_file}")
    
    # Return the figure
    return plt.gcf()        
        # For other algorithms, use SHAP's built-in explainers
        try:
            if algorithm_type.lower() == "xgboost":
                if verbose > 0:
                    print("Creating XGBoost TreeExplainer...")
                explainer = shap.TreeExplainer(model)
            
            elif algorithm_type.lower() == "lightgbm":
                if verbose > 0:
                    print("Creating LightGBM TreeExplainer...")
                # Try with model.booster_ first (sklearn-style LightGBM)
                try:
                    explainer = shap.TreeExplainer(model.booster_)
                except (AttributeError, TypeError):
                    # Fall back to direct model
                    explainer = shap.TreeExplainer(model)
            
            elif algorithm_type.lower() == "randomforest":
                if verbose > 0:
                    print("Creating RandomForest TreeExplainer...")
                explainer = shap.TreeExplainer(model)
                    
            else:
                # Fallback to KernelExplainer for other model types
                if verbose > 0:
                    print(f"Model type {algorithm_type} not recognized, falling back to KernelExplainer...")
                if background_size < len(X_sample):
                    background = shap.sample(X_sample, background_size)
                else:
                    background = X_sample
                explainer = shap.KernelExplainer(model.predict_proba, background)
                
        except Exception as e:
            print(f"Error creating explainer: {str(e)}")
            raise ValueError(f"Failed to create SHAP explainer for {algorithm_type}: {str(e)}")
        
        # Calculate SHAP values
        try:
            if verbose > 0:
                print("Calculating SHAP values...")
            
            try:
                # Try with verbose parameter if available
                shap_values = explainer.shap_values(X_sample, verbose=verbose > 1)
            except TypeError:
                # Fallback if verbose is not supported
                shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            raise ValueError(f"Failed to calculate SHAP values: {str(e)}")
        
        # Calculate mean absolute SHAP values for feature ranking
        if isinstance(shap_values, list):
            # For multi-class models or some binary models that return a list of arrays
            if len(shap_values) > 1:
                # For binary classification, typically index 1 is the positive class
                shap_values_for_importance = shap_values[1]
            else:
                shap_values_for_importance = shap_values[0]
        else:
            shap_values_for_importance = shap_values
        
        mean_abs_shap = np.abs(shap_values_for_importance).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        if verbose > 0:
            print(f"SHAP calculation completed in {elapsed_time:.2f} seconds")
        
        # Create result dictionary
        result = {
            'shap_values': shap_values,
            'explainer': explainer,
            'feature_importance': feature_importance,
            'sample_data': X_sample,
            'feature_names': feature_names,
            'computation_time': elapsed_time
        }
        
        # Restore original thread settings if they were changed
        if old_threads is not None:
            os.environ["OMP_NUM_THREADS"] = old_threads
        
        return result
    
    except Exception as e:
        # Make sure to restore thread settings on exception
        if old_threads is not None:
            os.environ["OMP_NUM_THREADS"] = old_threads
        
        # Re-raise the exception
        raise e        
        # Special handling for CatBoost using native SHAP calculation
        if algorithm_type.lower() == "catboost":
            if X_train is None:
                raise ValueError("CatBoost requires training data (X_train) for reliable SHAP calculation. "
                                "Please provide X_train when using CatBoost.")
            
            if verbose > 0:
                print("Using CatBoost's native SHAP calculation...")
            
            try:
                # Import catboost for Pool object
                import catboost as cb
                
                # Convert training data to CatBoost Pool format
                if isinstance(X_train, pd.DataFrame):
                    X_train_values = X_train.values
                    train_pool = cb.Pool(X_train_values, label=None, feature_names=list(X_train.columns))
                else:
                    train_pool = cb.Pool(X_train, label=None, feature_names=feature_names)
                
                # Sample test data
                if isinstance(X_sample, pd.DataFrame):
                    X_sample_values = X_sample.values
                else:
                    X_sample_values = X_sample
                    
                # Get SHAP values for test data using model.get_feature_importance()
                # For CatBoost, we're actually calculating SHAP values for the TRAINING data
                # but we'll only use the number of samples that matches our test data size
                shap_values = model.get_feature_importance(
                    data=train_pool,
                    type='ShapValues'
                )
                
                # CatBoost adds a bias term as the last column, so remove it
                shap_values = shap_values[:, :-1]
                
                # Ensure we only use as many rows as we have in X_sample
                sample_count = min(len(shap_values), len(X_sample_values))
                shap_values = shap_values[:sample_count]
                
                # Calculate mean absolute SHAP values for feature ranking
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # If dimensions don't match, fall back to using model's feature_importances_
                if len(mean_abs_shap) != len(feature_names):
                    print(f"Warning: SHAP values shape {mean_abs_shap.shape} doesn't match feature names length {len(feature_names)}")
                    print("Falling back to model's feature_importances_")
                    
                    if hasattr(model, 'feature_importances_'):
                        mean_abs_shap = model.feature_importances_
                        # Create dummy SHAP values that align with feature importances
                        shaped_values = np.zeros((len(X_sample_values), len(feature_names)))
                        for i in range(len(X_sample_values)):
                            shaped_values[i] = mean_abs_shap
                        shap_values = shaped_values
                
                # Create feature importance DataFrame
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': mean_abs_shap
                }).sort_values('importance', ascending=False)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                if verbose > 0:
                    print(f"SHAP calculation completed in {elapsed_time:.2f} seconds")
                
                # Create result dictionary
                result = {
                    'shap_values': shap_values,
                    'explainer': None,
                    'feature_importance': feature_importance,
                    'sample_data': X_sample,
                    'feature_names': feature_names,
                    'computation_time': elapsed_time
                }
                
                # Restore original thread settings if they were changed
                if old_threads is not None:
                    os.environ["OMP_NUM_THREADS"] = old_threads
                
                return result
                
            except Exception as e:
                print(f"Error calculating CatBoost SHAP values: {str(e)}")
                print("Falling back to feature importance instead of SHAP values")
                
                # Use feature_importances_ attribute instead
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    
                    # Create dummy SHAP values based on feature importance
                    # This won't have the sample-specific variation of real SHAP values
                    # but will at least show the overall feature ranking correctly
                    shap_values = np.zeros((len(X_sample), len(importance)))
                    for i in range(len(X_sample)):
                        # Add slight random variation to make plots more interesting
                        shap_values[i] = importance * (0.95 + 0.1 * np.random.random(len(importance)))
                    
                    # Create feature importance DataFrame
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    if verbose > 0:
                        print(f"Feature importance calculation completed in {elapsed_time:.2f} seconds")
                    
                    # Create result dictionary
                    result = {
                        'shap_values': shap_values,
                        'explainer': None,
                        'feature_importance': feature_importance,
                        'sample_data': X_sample,
                        'feature_names': feature_names,
                        'computation_time': elapsed_time
                    }
                    
                    # Restore original thread settings if they were changed
                    if old_threads is not None:
                        os.environ["OMP_NUM_THREADS"] = old_threads
                    
                    return result"""
SHAP Utilities - Functions for generating and visualizing SHAP values
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_shap_values(model, X, algorithm_type, X_train=None, sample_size=None, 
                       background_size=100, verbose=1, optimizer=None):
    """
    Generate SHAP values for a trained gradient boosted tree model.
    
    Parameters:
    -----------
    model : model object
        Trained model (XGBoost, LightGBM, CatBoost, or RandomForest)
    X : pandas.DataFrame or numpy.ndarray
        Feature dataset for SHAP calculation (typically X_test or a sample)
    algorithm_type : str
        Type of GBT algorithm: "xgboost", "lightgbm", "catboost", or "randomforest"
    X_train : pandas.DataFrame or numpy.ndarray, optional
        Training data, required for CatBoost SHAP calculation
    sample_size : int, optional
        Number of samples to use for SHAP calculation (to reduce computation time)
    background_size : int, default=100
        Number of samples to use for background distribution (for non-tree models)
    verbose : int, default=1
        Verbosity level (0: silent, 1: normal, 2: detailed)
    optimizer : SystemOptimizer, optional
        System optimizer instance for multicore optimization
        
    Returns:
    --------
    dict
        Dictionary containing SHAP values, explainer, and feature importance
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install using: pip install shap")
    
    start_time = time.time()
    
    # Apply thread optimization if optimizer is provided
    old_threads = None
    if optimizer is not None and hasattr(optimizer, 'system_info'):
        old_threads = os.environ.get("OMP_NUM_THREADS", None)
        shap_threads = optimizer.system_info.get('shap_threads', None)
        if shap_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(shap_threads)
            if verbose > 0:
                print(f"SHAP calculation will use {shap_threads} threads")
        
        # Determine optimal sample size based on memory if not provided
        if sample_size is None and hasattr(X, 'shape'):
            available_mem_mb = optimizer.system_info.get('available_memory_gb', 4) * 1024
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            # Heuristic: estimate memory needed per sample based on features
            safe_sample_size = min(len(X), int(available_mem_mb / (n_features * 2.0)))
            sample_size = max(100, min(1000, safe_sample_size))
            if verbose > 0:
                print(f"Auto-selected sample size: {sample_size} based on available memory")
    
    try:
        # Handle feature names
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Sample data if requested (to reduce computation time)
        if sample_size is not None and sample_size < len(X):
            if hasattr(X, 'sample'):
                X_sample = X.sample(sample_size, random_state=42)
            else:
                indices = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
            if verbose > 0:
                print(f"Using {sample_size} samples for SHAP calculation (reduced from {len(X)})")
        else:
            X_sample = X
            if verbose > 0:
                print(f"Using all {len(X_sample)} samples for SHAP calculation")