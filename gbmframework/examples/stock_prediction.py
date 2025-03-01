"""
Stock market prediction example using the GBM Framework.

This example demonstrates how to use the GBM Framework to predict stock price movements
by creating technical indicators and training gradient boosting models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import the GBM Framework
try:
    from gbmframework import GBMFactory, utils
    from gbmframework.tuning import GBMTuner
except ImportError:
    import sys
    import os
    # Add the parent directory to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from gbmframework import GBMFactory, utils
    from gbmframework.tuning import GBMTuner

def download_stock_data(ticker='SPY', period='5y', interval='1d'):
    """
    Download stock data using yfinance.
    
    Parameters:
    -----------
    ticker : str, optional (default='SPY')
        Stock ticker symbol
    period : str, optional (default='5y')
        Time period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    interval : str, optional (default='1d')
        Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
    --------
    pandas.DataFrame: Stock price data
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.info("yfinance not found. Attempting to install...")
        import subprocess
        subprocess.check_call(["pip", "install", "yfinance"])
        import yfinance as yf
    
    logger.info(f"Downloading {ticker} data for the past {period}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    logger.info(f"Downloaded {len(df)} rows of data.")
    return df

def create_features(df):
    """
    Create technical indicators as features for stock prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with stock price data
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Calculate returns
    data['Return'] = data['Close'].pct_change()
    data['Return_Prev'] = data['Return'].shift(1)
    data['Return_Prev2'] = data['Return'].shift(2)
    data['Return_Prev3'] = data['Return'].shift(3)
    data['Return_Prev_Week'] = data['Close'].pct_change(5)
    data['Return_Prev_Month'] = data['Close'].pct_change(21)
    
    # Calculate moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Calculate ratios of price to moving averages
    data['Price_to_MA5'] = data['Close'] / data['MA5']
    data['Price_to_MA10'] = data['Close'] / data['MA10']
    data['Price_to_MA20'] = data['Close'] / data['MA20']
    data['Price_to_MA50'] = data['Close'] / data['MA50']
    
    # Calculate moving average crossovers
    data['MA5_cross_MA10'] = (data['MA5'] > data['MA10']).astype(int)
    data['MA10_cross_MA20'] = (data['MA10'] > data['MA20']).astype(int)
    data['MA20_cross_MA50'] = (data['MA20'] > data['MA50']).astype(int)
    
    # Calculate volatility
    data['Volatility_5d'] = data['Return'].rolling(window=5).std()
    data['Volatility_10d'] = data['Return'].rolling(window=10).std()
    data['Volatility_20d'] = data['Return'].rolling(window=20).std()
    
    # Volume indicators
    data['Volume_Change'] = data['Volume'].pct_change()
    data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA5']
    
    # Calculate RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Trading range
    data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Range_MA5'] = data['High_Low_Range'].rolling(window=5).mean()
    
    # Target variable: Will price go up in the next day?
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop NaN values
    data = data.dropna()
    
    logger.info(f"Created {len(data.columns) - len(df.columns)} technical indicators")
    
    return data

def prepare_data(data):
    """
    Prepare data for machine learning.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with stock data and technical indicators
        
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test, feature_columns)
    """
    # Features to use
    feature_columns = [
        'Return', 'Return_Prev', 'Return_Prev2', 'Return_Prev3', 
        'Return_Prev_Week', 'Return_Prev_Month',
        'Price_to_MA5', 'Price_to_MA10', 'Price_to_MA20', 'Price_to_MA50',
        'MA5_cross_MA10', 'MA10_cross_MA20', 'MA20_cross_MA50',
        'Volatility_5d', 'Volatility_10d', 'Volatility_20d',
        'Volume_Change', 'Volume_Ratio', 'RSI', 
        'High_Low_Range', 'Range_MA5'
    ]
    
    # Target column
    target_column = 'Target'
    
    # We'll use a time-based split since this is time series data
    # Use the last 20% of data for testing
    split_idx = int(len(data) * 0.8)
    
    # Split the data
    X_train = data.iloc[:split_idx][feature_columns]
    X_test = data.iloc[split_idx:][feature_columns]
    y_train = data.iloc[:split_idx][target_column]
    y_test = data.iloc[split_idx:][target_column]
    
    logger.info(f"Training data: {X_train.shape[0]} samples from {data.index[0]} to {data.index[split_idx-1]}")
    logger.info(f"Testing data: {X_test.shape[0]} samples from {data.index[split_idx]} to {data.index[-1]}")
    
    # Scale the features
    from sklearn.preprocessing import StandardScaler
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
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def train_models(X_train, X_test, y_train, y_test):
    """
    Train all available GBM models.
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        Training and testing features
    y_train, y_test : pandas.Series
        Training and testing targets
        
    Returns:
    --------
    dict: Dictionary of trained models
    """
    # Parameters for each model type
    params = {
        'xgboost': {
            'max_depth': 4,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': {0: 1, 1: 1}  # Balanced by default
        },
        'lightgbm': {
            'num_leaves': 31,
            'max_depth': 4,
            'learning_rate': 0.01,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'class_weight': {0: 1, 1: 1}  # Balanced by default
        },
        'catboost': {
            'depth': 4,
            'learning_rate': 0.01,
            'iterations': 200,
            'subsample': 0.8,
            'rsm': 0.8,
            'class_weight': {'auto': True}  # Automatic class balancing
        }
    }
    
    # Models to train
    model_types = ['xgboost', 'lightgbm', 'catboost']
    models = {}
    
    # Train each model type
    for model_type in model_types:
        try:
            logger.info(f"Training {model_type.upper()} model")
            
            # Create model
            model = GBMFactory.create_model(model_type)
            
            # Get appropriate parameters
            model_params = params[model_type]
            
            # Fit model
            model.fit(
                X_train=X_train, 
                X_test=X_test, 
                y_train=y_train, 
                y_test=y_test,
                params=model_params,
                create_shap_plots=True,
                create_importance_plots=True,
                sample_size=min(1000, len(X_test))
            )
            
            # Add to models dictionary
            models[model_type.upper()] = model
            
        except ImportError:
            logger.warning(f"{model_type.upper()} not available - skipping")
    
    return models

def run_trading_simulation(model, X_test, y_test, df):
    """
    Run a trading simulation using model predictions.
    
    Parameters:
    -----------
    model : GBMBase
        Trained model to use for predictions
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        True test labels
    df : pandas.DataFrame
        Original stock data
        
    Returns:
    --------
    pandas.DataFrame: Simulation results
    """
    # Get test data indices
    test_indices = X_test.index
    
    # Get predictions for the test period
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Set up simulation dataframe
    sim_data = pd.DataFrame({
        'Close': df.loc[test_indices, 'Close'],
        'True_Direction': y_test,
        'Pred_Direction': y_pred,
        'Pred_Probability': y_pred_proba,
        'Return': df.loc[test_indices, 'Return']
    })
    
    # Calculate strategy returns (invest when model predicts up)
    sim_data['Strategy_Return'] = sim_data['Pred_Direction'].shift(1) * sim_data['Return']
    
    # Fill first value with 0 (no investment on first day)
    sim_data['Strategy_Return'].iloc[0] = 0
    
    # Calculate cumulative returns
    sim_data['Cumulative_Market'] = (1 + sim_data['Return']).cumprod()
    sim_data['Cumulative_Strategy'] = (1 + sim_data['Strategy_Return']).cumprod()
    
    # Calculate metrics
    market_return = sim_data['Cumulative_Market'].iloc[-1] - 1
    strategy_return = sim_data['Cumulative_Strategy'].iloc[-1] - 1
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display results
    logger.info(f"Trading Simulation Results:")
    logger.info(f"Prediction accuracy: {accuracy:.2%}")
    logger.info(f"Market return: {market_return:.2%}")
    logger.info(f"Strategy return: {strategy_return:.2%}")
    logger.info(f"Outperformance: {strategy_return - market_return:.2%}")
    
    # Plot cumulative returns
    plt.figure(figsize=(14, 7))
    plt.plot(sim_data.index, sim_data['Cumulative_Market'], label='Buy & Hold')
    plt.plot(sim_data.index, sim_data['Cumulative_Strategy'], label='Model Strategy')
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return sim_data

def tune_model(X_train, X_test, y_train, y_test, model_type='xgboost', max_evals=30):
    """
    Tune a model using Hyperopt.
    
    Parameters:
    -----------
    X_train, X_test : pandas.DataFrame
        Training and testing features
    y_train, y_test : pandas.Series
        Training and testing targets
    model_type : str, optional (default='xgboost')
        Type of model to tune
    max_evals : int, optional (default=30)
        Maximum number of evaluations for Hyperopt
        
    Returns:
    --------
    tuple: (best_model, best_params)
    """
    try:
        import hyperopt
    except ImportError:
        logger.warning("Hyperopt not available. Cannot perform hyperparameter tuning.")
        logger.warning("Install with: pip install hyperopt")
        return None, None
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create model
    try:
        model = GBMFactory.create_model(model_type)
    except ImportError:
        logger.warning(f"{model_type.upper()} not available")
        return None, None
    
    # Create tuner
    tuner = GBMTuner(
        model_class=model,
        X_train=X_train_opt,
        X_val=X_val,
        y_train=y_train_opt,
        y_val=y_val,
        metric='roc_auc'
    )
    
    # Run optimization
    logger.info(f"Running hyperparameter optimization for {model_type.upper()}")
    best_params = tuner.optimize(max_evals=max_evals)
    
    # Plot optimization history
    tuner.plot_optimization_history()
    
    # Plot parameter importance
    tuner.plot_parameter_importance()
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    best_model = tuner.train_best_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    return best_model, best_params

def run_stock_prediction():
    """
    Run the complete stock prediction example.
    
    Returns:
    --------
    tuple: (models, best_model, simulation_data)
    """
    # Download stock data
    df = download_stock_data(ticker='SPY', period='5y')
    
    # Create features
    data = create_features(df)
    
    # Prepare data for machine learning
    X_train, X_test, y_train, y_test, features = prepare_data(data)
    
    # Train models
    models = train_models(X_train, X_test, y_train, y_test)
    
    # Compare models
    if len(models) > 0:
        logger.info("Comparing models")
        from sklearn.metrics import roc_auc_score
        
        results = {}
        for name, model in models.items():
            # Calculate ROC AUC
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            results[name] = auc
        
        # Find best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        
        logger.info(f"Model comparison:")
        for name, auc in results.items():
            logger.info(f"{name}: ROC AUC = {auc:.4f}")
        
        logger.info(f"\nBest model: {best_model_name} with ROC AUC = {results[best_model_name]:.4f}")
        
        # Run trading simulation with best model
        logger.info(f"\nRunning trading simulation with {best_model_name} model")
        simulation_data = run_trading_simulation(best_model, X_test, y_test, df)
        
        # Check feature importance agreement
        if len(models) > 1:
            logger.info("\nChecking feature importance agreement")
            agreement = utils.check_feature_importance_agreement(models, n_features=10)
            logger.info(f"Top 10 important features by agreement:")
            logger.info(agreement.head(10))
            
        # Optional: Tune the best model
        logger.info("\nTuning the best model (set max_evals=0 to skip)")
        max_evals = 10  # Set to 0 to skip tuning
        if max_evals > 0:
            tuned_model, tuned_params = tune_model(
                X_train, X_test, y_train, y_test, 
                model_type=best_model_name.lower(),
                max_evals=max_evals
            )
            
            if tuned_model is not None:
                logger.info("Running trading simulation with tuned model")
                tuned_simulation = run_trading_simulation(tuned_model, X_test, y_test, df)
                
                return models, tuned_model, tuned_simulation
        
        return models, best_model, simulation_data
    else:
        logger.error("No models were trained. Please install at least one GBM framework.")
        return {}, None, None

if __name__ == "__main__":
    logger.info("Running stock prediction example")
    models, best_model, simulation_data = run_stock_prediction()
