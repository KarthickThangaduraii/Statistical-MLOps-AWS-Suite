"""
AWS Lambda Inference Handler.
Facilitates real-time, serverless model inference for manufacturing process optimization.
"""

import json
import logging
import os
import boto3
import joblib
import numpy as np

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 client for downloading model artifacts
s3 = boto3.client('s3')

# Global variable to cache the model across Lambda invocations
MODEL = None

def load_model():
    """
    Downloads and loads the model from S3.
    Caches the model in the global MODEL variable for subsequent invocations.
    """
    global MODEL
    if MODEL is None:
        model_bucket = os.environ.get('MODEL_BUCKET')
        model_key = os.environ.get('MODEL_KEY')
        local_path = '/tmp/model.joblib'
        
        logger.info(f"Downloading model from s3://{model_bucket}/{model_key}")
        s3.download_file(model_bucket, model_key, local_path)
        
        MODEL = joblib.load(local_path)
        logger.info("Model loaded successfully.")
    
    return MODEL

def lambda_handler(event, context):
    """
    Main entry point for AWS Lambda.
    
    Args:
        event: Dict containing the input data (usually 'body' or direct JSON).
        context: Lambda context object.
        
    Returns:
        JSON response with prediction result or error.
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # 1. Parse Input
        body = event.get('body')
        if body and isinstance(body, str):
            data = json.loads(body)
        else:
            data = event
            
        features = data.get('features')
        if features is None:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing "features" field in input data.'})
            }
        
        # 2. Load Model
        model = load_model()
        
        # 3. Perform Inference
        # Convert features to numpy array and reshape for prediction
        input_array = np.array(features).reshape(1, -1)
        prediction = model.predict(input_array)
        
        # 4. Return Result
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': float(prediction[0]),
                'model_version': os.environ.get('MODEL_VERSION', '1.0.0')
            })
        }
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error during inference.'})
        }
