"""Make predictions on gold prices"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldPredictor:
    """Make predictions and generate trading signals"""
    
    def __init__(self, model_trainer):
        """
        Initialize predictor
        
        Args:
            model_trainer: Trained GoldModelTrainer instance
        """
        self.model = model_trainer
        
    def predict_next_move(self, latest_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict next price movement
        
        Args:
            latest_features: Latest data with all features
            
        Returns:
            Prediction dictionary with direction and confidence
        """
        try:
            # Get prediction and probability
            prediction = self.model.predict(latest_features)[0]
            probabilities = self.model.predict_proba(latest_features)[0]
            
            direction = "UP" if prediction == 1 else "DOWN"
            confidence = max(probabilities) * 100
            
            result = {
                'direction': direction,
                'prediction': prediction,
                'confidence': confidence,
                'probability_up': probabilities[1] * 100,
                'probability_down': probabilities[0] * 100
            }
            
            logger.info(f"Prediction: {direction} (Confidence: {confidence:.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def generate_signal(self, latest_data: pd.DataFrame, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Generate trading signal based on prediction and indicators
        
        Args:
            latest_data: Latest features data
            threshold: Confidence threshold for signal generation
            
        Returns:
            Trading signal dictionary
        """
        prediction = self.predict_next_move(latest_data)
        
        # Extract indicators
        latest_row = latest_data.iloc[-1]
        rsi = latest_row.get('RSI', np.nan)
        macd = latest_row.get('MACD', np.nan)
        macd_signal = latest_row.get('MACD_Signal', np.nan)
        
        confidence = prediction['confidence'] / 100
        signal = 'HOLD'
        
        # Generate signal
        if confidence >= threshold:
            if prediction['direction'] == 'UP' and rsi < 70:
                signal = 'BUY'
            elif prediction['direction'] == 'DOWN' and rsi > 30:
                signal = 'SELL'
        
        result = {
            'signal': signal,
            'confidence': prediction['confidence'],
            'direction': prediction['direction'],
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'explanation': self._explain_signal(signal, prediction, rsi)
        }
        
        logger.info(f"Signal: {signal} | Confidence: {prediction['confidence']:.2f}% | RSI: {rsi:.2f}")
        return result
    
    @staticmethod
    def _explain_signal(signal: str, prediction: Dict, rsi: float) -> str:
        """Generate explanation for the signal"""
        if signal == 'BUY':
            return f"Prediction shows UP movement (RSI: {rsi:.2f} < 70 - not overbought)"
        elif signal == 'SELL':
            return f"Prediction shows DOWN movement (RSI: {rsi:.2f} > 30 - not oversold)"
        else:
            return "Confidence below threshold or contradictory signals"


def main():
    """Test predictor"""
    from data_fetcher import GoldDataFetcher
    from data_processor import GoldDataProcessor
    from model_trainer import GoldModelTrainer
    
    # Setup
    fetcher = GoldDataFetcher(period="3mo")
    raw_data = fetcher.fetch_historical_data()
    processed = GoldDataProcessor.prepare_features(raw_data)
    train_data, test_data = GoldDataProcessor.split_train_test(processed)
    
    # Train model
    trainer = GoldModelTrainer()
    trainer.train(train_data)
    
    # Make predictions
    predictor = GoldPredictor(trainer)
    latest = test_data.tail(1)
    
    signal = predictor.generate_signal(latest)
    print("\nTrading Signal:")
    for key, value in signal.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
