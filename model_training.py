import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from datetime import datetime
import os
import matplotlib.pyplot as plt
import ta
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator
from ta.others import DailyReturnIndicator
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def load_processed_data():
    """Load data from processed_data.json"""
    try:
        with open('data/processed_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: processed_data.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in processed_data.json")
        return None

def combine_timeframes(symbol_data):
    """Combine data from multiple timeframes for a symbol"""
    combined_data = {}
    base_interval = '1h'  # Use 1h as base timeframe
    
    if base_interval not in symbol_data:
        return None
        
    # Start with base timeframe data
    base_df = pd.DataFrame(symbol_data[base_interval]['data'])
    base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])
    base_df.set_index('timestamp', inplace=True)
    
    # Add features from other timeframes
    for interval in symbol_data.keys():
        if interval == base_interval:
            continue
            
        interval_df = pd.DataFrame(symbol_data[interval]['data'])
        interval_df['timestamp'] = pd.to_datetime(interval_df['timestamp'])
        interval_df.set_index('timestamp', inplace=True)
        
        # Resample to match base timeframe
        resampled = interval_df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Add suffix to columns
        resampled = resampled.add_suffix(f'_{interval}')
        
        # Join with base dataframe
        base_df = base_df.join(resampled, how='left')
    
    return base_df

def prepare_features_and_labels(df):
    """Prepare feature matrix X and target vector y with enhanced features"""
    
    # Calculate all technical indicators
    df = calculate_features(df)
    
    # Create target with dynamic threshold based on volatility and ATR
    df['atr_pct'] = df['atr'] / df['close']
    volatility_threshold = df['volatility'].rolling(window=20).mean()
    atr_threshold = df['atr_pct'].rolling(window=20).mean()
    combined_threshold = (volatility_threshold + atr_threshold) * 0.5
    
    # Create multiple target variables for different profit levels
    df['target_1'] = ((df['close'].shift(-1) - df['close']) / df['close'] > combined_threshold).astype(int)
    df['target_2'] = ((df['close'].shift(-2) - df['close']) / df['close'] > combined_threshold * 1.5).astype(int)
    df['target_3'] = ((df['close'].shift(-3) - df['close']) / df['close'] > combined_threshold * 2).astype(int)
    
    # Combine targets (any profitable opportunity)
    df['target'] = ((df['target_1'] == 1) | (df['target_2'] == 1) | (df['target_3'] == 1)).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features
    feature_columns = [col for col in df.columns if not col.startswith('target')]
    
    # Separate features and target
    X = df[feature_columns]
    y = df['target']
    
    return X, y

def train_model():
    """Train the model with enhanced parameters and feature selection"""
    processed_data = load_processed_data()
    if not processed_data:
        return
    
    print("Preparing data for training...")
    all_data = []
    
    # Combine data from all symbols and timeframes
    for symbol, timeframes_data in processed_data.items():
        symbol_df = combine_timeframes(timeframes_data)
        if symbol_df is not None:
            symbol_df['symbol'] = symbol
            all_data.append(symbol_df)
    
    if not all_data:
        print("No valid data found for training")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Store symbols for reference but remove from training data
    symbols = combined_df['symbol']
    combined_df = combined_df.drop('symbol', axis=1)
    
    X, y = prepare_features_and_labels(combined_df)
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    class_dist = pd.Series(y).value_counts(normalize=True)
    print(class_dist)
    
    # Create time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # No shuffle for time series
    )
    
    # Check if we need SMOTE
    minority_ratio = min(class_dist)
    use_smote = minority_ratio < 0.4  # Only use SMOTE if minority class is less than 40%
    
    # Create pipeline based on class distribution
    pipeline_steps = []
    pipeline_steps.append(('scaler', StandardScaler()))
    
    if use_smote:
        print("\nClass imbalance detected. Using SMOTE...")
        pipeline_steps.append(('smote', SMOTE(random_state=42, sampling_strategy=0.8)))
    
    pipeline_steps.extend([
        ('feature_selection', SelectFromModel(GradientBoostingClassifier(random_state=42))),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=0,
            criterion='entropy',
            bootstrap=True,
            max_features='sqrt'
        ))
    ])
    
    pipeline = Pipeline(pipeline_steps)
    
    # Define parameter grid
    param_grid = {
        'classifier__n_estimators': [300],
        'classifier__max_depth': [12, 15],
        'classifier__min_samples_split': [5, 8],
        'classifier__min_samples_leaf': [3, 4],
        'classifier__class_weight': ['balanced', {0: 1.5, 1: 1}]
    }
    
    # Perform grid search with time series cross-validation
    print("\nPerforming grid search with time series cross-validation...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        error_score=0  # Return score of 0 for failed fits
    )
    
    try:
        grid_search.fit(X_train, y_train)
        print("\nBest parameters:", grid_search.best_params_)
        
        # Get feature importance and selected features
        feature_selector = grid_search.best_estimator_.named_steps['feature_selection']
        selected_features = X.columns[feature_selector.get_support()].tolist()
        
        print("\nSelected features:", len(selected_features))
        for feature in selected_features[:10]:
            print(f"- {feature}")
        
        # Make predictions
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        print("\nFinding optimal probability threshold...")
        thresholds = np.linspace(0.3, 0.7, 40)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"Optimal probability threshold: {best_threshold:.3f}")
        
        # Final predictions with optimal threshold
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model and metadata
        print("\nSaving model and metadata...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/trading_model_{timestamp}.pkl'
        
        model_data = {
            'pipeline': grid_search.best_estimator_,
            'selected_features': selected_features,
            'optimal_threshold': best_threshold,
            'feature_importance': dict(zip(selected_features, 
                grid_search.best_estimator_.named_steps['classifier'].feature_importances_)),
            'training_timestamp': timestamp,
            'model_params': grid_search.best_params_,
            'best_f1_score': best_f1
        }
        
        joblib.dump(model_data, model_path)
        
        # Also save to main model file
        joblib.dump(model_data, 'models/trading_model.pkl')
        
        # Plot feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': grid_search.best_estimator_.named_steps['classifier'].feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.savefig(f'graphs/feature_importance_{timestamp}.png')
        plt.close()
        
    except Exception as e:
        print(f"\nError during model training: {e}")
        print("\nChecking data for issues...")
        print(f"Number of features: {X.shape[1]}")
        print("\nFeature names:")
        print(X.columns.tolist())
        print("\nChecking for NaN values:")
        print(X.isna().sum().sum())
        print("\nChecking for infinite values:")
        print(np.isinf(X.values).sum())
        raise

def calculate_features(df):
    """Calculate technical indicators and features for model training"""
    try:
        # Price action features
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Trend Indicators
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Multiple timeframe SMAs
        for period in [5, 8, 13, 21, 34, 55, 89]:
            df[f'sma_{period}'] = SMAIndicator(df['close'], window=period).sma_indicator()
        
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        # Momentum Indicators
        rsi = RSIIndicator(df['close'])
        df['rsi'] = rsi.rsi()
        
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volatility Indicators
        bb = BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        atr = AverageTrueRange(df['high'], df['low'], df['close'])
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volume Indicators
        obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
        df['obv'] = obv.on_balance_volume()
        
        adi = AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume'])
        df['adi'] = adi.acc_dist_index()
        
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_zscore'] = (df['volume'] - df['volume_sma']) / df['volume_std']
        
        # Returns and Momentum
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Momentum features with different periods
        for period in [5, 10, 15, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
            df[f'volume_momentum_{period}'] = df['volume'].pct_change(periods=period)
        
        return df
        
    except Exception as e:
        print(f"Error calculating features: {e}")
        raise

def main():
    print("Starting model training process...")
    train_model()

if __name__ == "__main__":
    main() 