"""
Modular Machine Learning Training Pipeline.
Handles data preprocessing, model training, evaluation, and serialization for AWS deployment.
"""

import os
import joblib
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Enterprise-grade model trainer with built-in preprocessing and validation.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer with specific hyperparameters.
        """
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        self.pipeline: Optional[Pipeline] = None

    def create_pipeline(self) -> Pipeline:
        """
        Define a Scikit-learn pipeline for data scaling and model training.
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(**self.model_params))
        ])
        logger.info(f"Created training pipeline with params: {self.model_params}")
        return pipeline

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model and return evaluation metrics.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.pipeline = self.create_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2_score": float(r2)
        }
        
        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics

    def save_model(self, output_path: str):
        """
        Serialize the trained pipeline to disk.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.pipeline, output_path)
        logger.info(f"Model successfully saved to {output_path}")

    @staticmethod
    def load_model(model_path: str) -> Pipeline:
        """
        Load a serialized model pipeline.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        
        pipeline = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return pipeline

def main():
    """Example execution script for model training."""
    # Generate dummy data for demonstration
    logger.info("Generating sample data for process optimization...")
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y = X.iloc[:, 0] * 2 + X.iloc[:, 1] * 0.5 + np.random.normal(0, 0.1, 100)
    
    trainer = ModelTrainer()
    metrics = trainer.train(X, y)
    
    # Save the model
    model_file = "models/process_optimizer_v1.joblib"
    trainer.save_model(model_file)

if __name__ == "__main__":
    main()
