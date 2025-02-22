from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import logging
import requests
import joblib
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "C:/NPCI/models"  # Use forward slashes

# Load model, scaler, and label encoder (but don't use them for now)
try:
    print("🔄 Loading model, scaler, and label encoder...")
    model = joblib.load(f"{MODEL_PATH}/fraud_detection_model.pkl")
    scaler = joblib.load(f"{MODEL_PATH}/scaler.joblib")
    label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.joblib")
    print("✅ Model, Scaler, and Label Encoder loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    label_encoder = None

# External API configurations
GEMINI_API_URL = "https://api.gemini.com/v1"  # Example Gemini API endpoint
GEMINI_API_KEY = "AIzaSyDTHtOJl3UQGR8qf8fdlHL6psS19VmVrm0"
FINANCIAL_API_URL = "https://api.alphavantage.co/query"  # Example Alpha Vantage API
FINANCIAL_API_KEY = "OJ7M2WPX8O7ZS0I5"

# In-memory storage for recent predictions (for demonstration purposes)
recent_predictions = []

def call_gemini_api(transaction_id):
    """Call Gemini API to validate or fetch transaction details."""
    headers = {
        'Authorization': f'Bearer {GEMINI_API_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.get(f"{GEMINI_API_URL}/transaction/{transaction_id}", headers=headers, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API error: {str(e)}")
        return None

def call_financial_api(transaction_data):
    """Call a financial API (e.g., Alpha Vantage) for fraud detection."""
    params = {
        'function': 'TIME_SERIES_INTRADAY',  # Example function (replace with fraud detection endpoint)
        'symbol': transaction_data.get('symbol', 'BTC'),  # Example: Use transaction symbol
        'apikey': FINANCIAL_API_KEY
    }
    try:
        response = requests.get(FINANCIAL_API_URL, params=params, verify=False)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Financial API error: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the index.html page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for fraud prediction."""
    try:
        # Get data from the form
        data = request.form.to_dict()

        # Initialize result dictionary
        result = {
            'transaction_type': data['type'],
            'amount': float(data['amount']),
            'prediction': 'Unknown',
            'probability': '0.00%',
            'risk_level': 'Unknown',
            'recommendation': 'No recommendation available.'
        }

        # Call Gemini API for transaction validation
        gemini_response = call_gemini_api(data.get('transaction_id', ''))
        gemini_prediction = 0  # Default to non-fraudulent
        if gemini_response and gemini_response.get('status') == 'success':
            gemini_prediction = 0  # Assume success means no fraud
        else:
            gemini_prediction = 1  # Assume failure means potential fraud

        # Call financial API for additional fraud detection
        financial_response = call_financial_api(data)
        financial_prediction = 0  # Default to non-fraudulent
        if financial_response and financial_response.get('risk_score', 0) > 0.7:  # Example risk threshold
            financial_prediction = 1

        # Combine predictions (majority voting)
        predictions = [gemini_prediction, financial_prediction]
        final_prediction = max(set(predictions), key=predictions.count)

        # Update result with final prediction
        result['prediction'] = 'Fraudulent' if final_prediction == 1 else 'Not Fraudulent'
        result['probability'] = f"{final_prediction * 100:.2f}%"
        result['risk_level'] = 'High' if final_prediction == 1 else 'Low'

        # Provide a default recommendation if APIs fail
        if gemini_response is None and financial_response is None:
            result['recommendation'] = "Unable to verify transaction. Please contact support."

        # Store recent prediction
        recent_predictions.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'transaction_type': data['type'],
            'amount': float(data['amount']),
            'prediction': result['prediction'],
            'risk_level': result['risk_level']
        })

        # Render the result template
        return render_template('result.html', result=result, recent_predictions=recent_predictions[-5:])  # Show last 5 predictions

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)