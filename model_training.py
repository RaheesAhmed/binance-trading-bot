import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime
import os
import matplotlib.pyplot as plt

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
        print("Error: data/processed_data.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in data/processed_data.json")
        return None

def prepare_features_and_labels(df):
    """Prepare feature matrix X and target vector y"""
    
    # Features to use for prediction
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_high', 'bb_low', 'bb_mid',
        'sma_20', 'sma_50', 'sma_200',
        'atr', 'price_momentum', 'volume_sma', 'volume_momentum'
    ]
    
    # Create target variable (1 if price goes up in next candle, 0 if down)
    df['target'] = df['close'].shift(-1) > df['close']
    df['target'] = df['target'].astype(int)
    
    # Drop the last row since we won't have a target for it
    df = df.iloc[:-1]
    
    # Separate features and target
    X = df[feature_columns]
    y = df['target']
    
    return X, y

def combine_all_symbols_data(data):
    """Combine data from all symbols into a single DataFrame"""
    all_data = []
    
    for symbol, symbol_data in data.items():
        df = pd.DataFrame(symbol_data['data'])
        df['symbol'] = symbol  # Add symbol column for reference
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def train_model():
    """Train the Random Forest model"""
    # Load processed data
    processed_data = load_processed_data()
    if not processed_data:
        return
    
    print("Preparing data for training...")
    
    # Combine all symbols data
    combined_df = combine_all_symbols_data(processed_data)
    
    # Prepare features and labels
    X, y = prepare_features_and_labels(combined_df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'models/trading_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    # Save feature names for later use
    feature_names = {
        'features': list(X.columns),
        'last_updated': datetime.now().isoformat()
    }
    with open('data/feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print("\nModel, scaler, and feature names saved successfully!")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance plot
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('graphs/feature_importance.png')
    plt.close()

def main():
    print("Starting model training process...")
    train_model()

if __name__ == "__main__":
    main() 