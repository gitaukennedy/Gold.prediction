"""Train and evaluate ML models for gold price prediction"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging
from typing import Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldModelTrainer:
    """Train and manage ML models for price prediction"""
    
    def __init__(self):
        """Initialize model trainer"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def train(self, train_data: pd.DataFrame) -> dict:
        """
        Train Random Forest model on historical data
        
        Args:
            train_data: Training data with features and target
            
        Returns:
            Training metrics dictionary
        """
        # Select features (exclude OHLCV raw data and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Target']
        self.feature_columns = [col for col in train_data.columns if col not in exclude_cols]
        
        X_train = train_data[self.feature_columns]
        y_train = train_data['Target']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        logger.info("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Training metrics
        y_pred = self.model.predict(X_train_scaled)
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred, average='weighted'),
            'recall': recall_score(y_train, y_pred, average='weighted'),
            'f1': f1_score(y_train, y_pred, average='weighted')
        }
        
        logger.info(f"Training completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def evaluate(self, test_data: pd.DataFrame) -> dict:
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data with features and target
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_test = test_data[self.feature_columns]
        y_test = test_data['Target']
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predictions (0=down, 1=up)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(features[self.feature_columns])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(features[self.feature_columns])
        return self.model.predict_proba(X_scaled)
    
    def save(self, filepath: str):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        logger.info(f"Model loaded from {filepath}")


def main():
    """Test model trainer"""
    from data_fetcher import GoldDataFetcher
    from data_processor import GoldDataProcessor
    
    # Fetch and process data
    fetcher = GoldDataFetcher(period="3mo")
    raw_data = fetcher.fetch_historical_data()
    processed = GoldDataProcessor.prepare_features(raw_data)
    train_data, test_data = GoldDataProcessor.split_train_test(processed)
    
    # Train and evaluate
    trainer = GoldModelTrainer()
    train_metrics = trainer.train(train_data)
    test_metrics = trainer.evaluate(test_data)
    
    print("\nTrain Metrics:", train_metrics)
    print("Test Metrics:", test_metrics)


if __name__ == "__main__":
    main()
