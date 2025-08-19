#!/usr/bin/env python3
"""
XGBoost Profit Prediction Server
Simple Flask server to run XGBoost models and provide API endpoints
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our XGBoost predictor
try:
    from xgboost_model import XGBoostProfitPredictor
except ImportError:
    print("⚠️ XGBoost model not available, using fallback")
    XGBoostProfitPredictor = None

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global model instance
predictor = None
model_loaded = False

@app.route('/')
def home():
    """Simple home page with API documentation"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>XGBoost Profit Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; display: inline-block; }
            .url { font-family: monospace; background: #e9ecef; padding: 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🤖 XGBoost Profit Prediction API</h1>
        <p>This server provides machine learning-powered profit predictions using XGBoost.</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <span class="url">/api/train</span>
            <p>Train a new XGBoost model with CSV data</p>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span>
            <span class="url">/api/predict</span>
            <p>Make profit predictions using the trained model</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/api/model/status</span>
            <p>Get current model status and performance metrics</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span>
            <span class="url">/api/model/features</span>
            <p>Get feature importance from the trained model</p>
        </div>
        
        <h2>Model Status:</h2>
        <p id="status">Loading...</p>
        
        <script>
            fetch('/api/model/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').innerHTML = 
                        `<strong>Status:</strong> ${data.status}<br>
                         <strong>MAE:</strong> $${data.mae || 'N/A'}<br>
                         <strong>R²:</strong> ${data.r2 || 'N/A'}`;
                })
                .catch(error => {
                    document.getElementById('status').innerHTML = 'Error loading status';
                });
        </script>
    </body>
    </html>
    """
    return html

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a new XGBoost model with CSV data"""
    global predictor, model_loaded
    
    try:
        # Check if we have the required data
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read CSV data
        df = pd.read_csv(file)
        print(f"📁 Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Initialize predictor if not available
        if XGBoostProfitPredictor is None:
            return jsonify({'error': 'XGBoost model not available'}), 500
        
        if predictor is None:
            predictor = XGBoostProfitPredictor()
        
        # Prepare features and train model
        features, target = predictor.prepare_features(df)
        
        if len(features) == 0:
            return jsonify({'error': 'No valid features could be extracted from the data'}), 400
        
        # Train the model
        success = predictor.train_model(features, target)
        
        if success:
            model_loaded = True
            
            # Save the model
            model_path = f"models/profit_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs('models', exist_ok=True)
            predictor.save_model(model_path)
            
            return jsonify({
                'success': True,
                'message': 'Model trained successfully',
                'performance': predictor.model_performance,
                'features_count': len(features.columns),
                'samples_count': len(features),
                'model_path': model_path
            })
        else:
            return jsonify({'error': 'Model training failed'}), 500
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_profit():
    """Make profit predictions using the trained model"""
    global predictor, model_loaded
    
    try:
        if not model_loaded or predictor is None:
            return jsonify({'error': 'No trained model available. Please train a model first.'}), 400
        
        # Get prediction parameters
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        num_buyers = data.get('num_buyers', 1)
        expected_revenue = data.get('expected_revenue', 10000)
        selected_verticals = data.get('selected_verticals', [])
        additional_features = data.get('additional_features', {})
        
        # Validate inputs
        if num_buyers < 1:
            return jsonify({'error': 'Number of buyers must be at least 1'}), 400
        
        if expected_revenue <= 0:
            return jsonify({'error': 'Expected revenue must be positive'}), 400
        
        if not selected_verticals:
            return jsonify({'error': 'At least one vertical must be selected'}), 400
        
        # Make prediction
        predicted_profit = predictor.predict_profit(
            num_buyers, expected_revenue, selected_verticals, additional_features
        )
        
        if predicted_profit is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Calculate confidence interval (simplified)
        mae = predictor.model_performance.get('val_mae', 1000)
        confidence_interval = {
            'lower': max(0, predicted_profit - mae),
            'upper': predicted_profit + mae,
            'confidence_level': 0.95
        }
        
        return jsonify({
            'success': True,
            'prediction': {
                'predicted_profit': predicted_profit,
                'predicted_margin': (predicted_profit / expected_revenue) * 100,
                'confidence_interval': confidence_interval,
                'model_mae': mae
            },
            'inputs': {
                'num_buyers': num_buyers,
                'expected_revenue': expected_revenue,
                'selected_verticals': selected_verticals
            }
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """Get current model status and performance metrics"""
    global predictor, model_loaded
    
    if not model_loaded or predictor is None:
        return jsonify({
            'status': 'No model loaded',
            'mae': None,
            'r2': None,
            'features_count': 0,
            'training_date': None
        })
    
    performance = predictor.model_performance
    return jsonify({
        'status': 'Model loaded and ready',
        'mae': performance.get('val_mae'),
        'r2': performance.get('val_r2'),
        'features_count': len(predictor.feature_names),
        'training_date': performance.get('training_date'),
        'samples_count': performance.get('n_samples'),
        'performance': performance
    })

@app.route('/api/model/features', methods=['GET'])
def feature_importance():
    """Get feature importance from the trained model"""
    global predictor, model_loaded
    
    if not model_loaded or predictor is None:
        return jsonify({'error': 'No trained model available'}), 400
    
    try:
        importance = predictor.get_feature_importance()
        return jsonify({
            'success': True,
            'feature_importance': importance,
            'total_features': len(importance)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get feature importance: {str(e)}'}), 500

@app.route('/api/model/summary', methods=['GET'])
def model_summary():
    """Get detailed model summary"""
    global predictor, model_loaded
    
    if not model_loaded or predictor is None:
        return jsonify({'error': 'No trained model available'}), 400
    
    try:
        summary = predictor.get_model_summary()
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get model summary: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'xgboost_available': XGBoostProfitPredictor is not None
    })

if __name__ == '__main__':
    print("🚀 Starting XGBoost Profit Prediction Server...")
    print("📊 API available at: http://localhost:5000")
    print("🔗 Endpoints:")
    print("   • POST /api/train - Train new model")
    print("   • POST /api/predict - Make predictions")
    print("   • GET  /api/model/status - Model status")
    print("   • GET  /api/model/features - Feature importance")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Start the server
    app.run(host='0.0.0.0', port=5000, debug=True)
