#!/usr/bin/env python3
"""
Test script for XGBoost Profit Prediction Model
Demonstrates the model with sample data and shows performance improvements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost model
try:
    from xgboost_model import XGBoostProfitPredictor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost model not available. Install dependencies first:")
    print("   pip install -r requirements.txt")
    XGBOOST_AVAILABLE = False

def create_sample_data(n_samples=1000):
    """
    Create realistic sample data for testing the XGBoost model
    """
    print("📊 Creating sample data...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate dates (business days only)
    start_date = datetime(2023, 1, 1)
    dates = []
    current_date = start_date
    while len(dates) < n_samples:
        if current_date.weekday() < 5:  # Monday to Friday
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Generate realistic data
    data = []
    
    for i in range(n_samples):
        date = dates[i]
        day_name = date.strftime('%A')
        
        # Random vertical with realistic distribution
        verticals = ['MEDICARE ENGLISH', 'FINAL EXPENSE ENGLISH', 'ACA ENGLISH', 'MEDICARE SPANISH', 'AUTO INSURANCE ENGLISH']
        vertical_weights = [0.35, 0.25, 0.20, 0.15, 0.05]  # Realistic market distribution
        vertical = np.random.choice(verticals, p=vertical_weights)
        
        # Publisher count (1-8, with realistic distribution)
        publisher_counts = [1, 2, 3, 4, 5, 6, 7, 8]
        publisher_weights = [0.15, 0.20, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02]
        num_publishers = np.random.choice(publisher_counts, p=publisher_weights)
        
        # Revenue based on vertical and publisher count
        base_revenue = {
            'MEDICARE ENGLISH': 25000,
            'FINAL EXPENSE ENGLISH': 18000,
            'ACA ENGLISH': 22000,
            'MEDICARE SPANISH': 28000,
            'AUTO INSURANCE ENGLISH': 15000
        }
        
        revenue = base_revenue[vertical]
        # Add variation based on publisher count and seasonality
        revenue *= (0.8 + 0.4 * np.random.random())  # ±20% base variation
        revenue *= (1.0 + 0.1 * (num_publishers - 4))  # Publisher effect
        revenue *= (1.0 + 0.15 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365))  # Seasonal effect
        
        # Profit margin varies by vertical and has realistic patterns
        base_margins = {
            'MEDICARE ENGLISH': 0.21,
            'FINAL EXPENSE ENGLISH': 0.22,
            'ACA ENGLISH': 0.24,
            'MEDICARE SPANISH': 0.25,
            'AUTO INSURANCE ENGLISH': 0.18
        }
        
        base_margin = base_margins[vertical]
        
        # Add realistic variations
        margin_variation = 0.05  # ±5% margin variation
        margin = base_margin + (np.random.random() - 0.5) * margin_variation
        
        # Publisher count effect on margin
        if num_publishers == 1:
            margin *= 1.05  # Single publisher bonus
        elif num_publishers >= 6:
            margin *= 0.94  # Many publishers penalty
        
        # Seasonal margin effects
        if date.month in [1, 2]:  # January/February (Medicare season)
            if 'MEDICARE' in vertical:
                margin *= 1.03
        
        # Calculate profit
        profit = revenue * margin
        
        # Call volume data (if available)
        incoming_calls = int(revenue / 100 + np.random.poisson(50))
        connected_calls = int(incoming_calls * (0.6 + 0.3 * np.random.random()))
        converted_calls = int(connected_calls * (0.25 + 0.2 * np.random.random()))
        
        # Add some noise to make it realistic
        profit += np.random.normal(0, profit * 0.1)  # 10% noise
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Day': day_name,
            'Vertical': vertical,
            'Revenue': round(revenue, 2),
            'Net Profit': round(profit, 2),
            'Media Buyer': f'Publisher{np.random.randint(1, 21)}',
            'Incoming': incoming_calls,
            'Connected': connected_calls,
            'Converted': converted_calls
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Created {len(df)} sample records")
    print(f"📊 Revenue range: ${df['Revenue'].min():.0f} - ${df['Revenue'].max():.0f}")
    print(f"💰 Profit range: ${df['Net Profit'].min():.0f} - ${df['Net Profit'].max():.0f}")
    print(f"🏢 Verticals: {df['Vertical'].nunique()}")
    print(f"👥 Publishers: {df['Media Buyer'].nunique()}")
    
    return df

def test_rule_based_prediction(df):
    """
    Simulate the current rule-based prediction system
    """
    print("\n📋 Testing Rule-Based Prediction System...")
    
    predictions = []
    actuals = []
    
    for _, row in df.iterrows():
        # Simulate rule-based prediction logic
        vertical = row['Vertical']
        revenue = row['Revenue']
        
        # Simple rule-based margins (similar to current system)
        base_margins = {
            'MEDICARE ENGLISH': 0.213,
            'FINAL EXPENSE ENGLISH': 0.216,
            'ACA ENGLISH': 0.244,
            'MEDICARE SPANISH': 0.245,
            'AUTO INSURANCE ENGLISH': 0.18
        }
        
        base_margin = base_margins.get(vertical, 0.22)
        
        # Simple publisher effect
        publisher_effect = 1.0
        if 'Publisher' in str(row['Media Buyer']):
            publisher_num = int(str(row['Media Buyer']).replace('Publisher', ''))
            if publisher_num <= 3:
                publisher_effect = 1.02
            elif publisher_num >= 6:
                publisher_effect = 0.94
        
        predicted_margin = base_margin * publisher_effect
        predicted_profit = revenue * predicted_margin
        
        predictions.append(predicted_profit)
        actuals.append(row['Net Profit'])
    
    # Calculate metrics
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate R²
    ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
    ss_tot = np.sum((np.array(actuals) - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"📊 Rule-Based Performance:")
    print(f"   • MAE: ${mae:.2f}")
    print(f"   • RMSE: ${rmse:.2f}")
    print(f"   • R²: {r2:.3f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }

def test_xgboost_model(df):
    """
    Test the XGBoost model with the same data
    """
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost model not available for testing")
        return None
    
    print("\n🤖 Testing XGBoost ML Model...")
    
    try:
        # Initialize and train XGBoost model
        predictor = XGBoostProfitPredictor()
        
        # Prepare features
        features, target = predictor.prepare_features(df)
        
        if len(features) == 0:
            print("❌ No features could be extracted from the data")
            return None
        
        print(f"🔧 Extracted {len(features.columns)} features")
        print(f"📊 Feature names: {', '.join(features.columns[:5])}...")
        
        # Train the model
        success = predictor.train_model(features, target)
        
        if not success:
            print("❌ Model training failed")
            return None
        
        # Make predictions on the same data
        predictions = []
        for _, row in df.iterrows():
            # Get unique publishers for this date
            date_publishers = df[df['Date'] == row['Date']]['Media Buyer'].nunique()
            
            predicted_profit = predictor.predict_profit(
                num_buyers=date_publishers,
                expected_revenue=row['Revenue'],
                selected_verticals=[row['Vertical']]
            )
            
            if predicted_profit is not None:
                predictions.append(predicted_profit)
            else:
                predictions.append(0)
        
        # Calculate metrics
        actuals = df['Net Profit'].values
        mae = np.mean(np.abs(np.array(predictions) - actuals))
        mse = np.mean((np.array(predictions) - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R²
        ss_res = np.sum((actuals - np.array(predictions)) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print(f"📊 XGBoost Performance:")
        print(f"   • MAE: ${mae:.2f}")
        print(f"   • RMSE: ${rmse:.2f}")
        print(f"   • R²: {r2:.3f}")
        
        # Show feature importance
        feature_importance = predictor.get_feature_importance()
        print(f"\n🏆 Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            print(f"   {i+1}. {feature}: {importance:.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        print(f"❌ XGBoost testing error: {e}")
        return None

def compare_models(rule_based_results, xgboost_results):
    """
    Compare the performance of both models
    """
    if not rule_based_results or not xgboost_results:
        print("❌ Cannot compare models - missing results")
        return
    
    print("\n" + "="*60)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Performance metrics comparison
    print(f"{'Metric':<15} {'Rule-Based':<15} {'XGBoost':<15} {'Improvement':<15}")
    print("-" * 60)
    
    # MAE comparison
    mae_improvement = ((rule_based_results['mae'] - xgboost_results['mae']) / rule_based_results['mae']) * 100
    print(f"{'MAE':<15} ${rule_based_results['mae']:<14.0f} ${xgboost_results['mae']:<14.0f} {mae_improvement:>+13.1f}%")
    
    # RMSE comparison
    rmse_improvement = ((rule_based_results['rmse'] - xgboost_results['rmse']) / rule_based_results['rmse']) * 100
    print(f"{'RMSE':<15} ${rule_based_results['rmse']:<14.0f} ${xgboost_results['rmse']:<14.0f} {rmse_improvement:>+13.1f}%")
    
    # R² comparison
    r2_improvement = ((xgboost_results['r2'] - rule_based_results['r2']) / rule_based_results['r2']) * 100
    print(f"{'R²':<15} {rule_based_results['r2']:<14.3f} {xgboost_results['r2']:<14.3f} {r2_improvement:>+13.1f}%")
    
    print("-" * 60)
    
    # Summary
    print(f"\n🎯 Summary:")
    print(f"   • XGBoost reduces MAE by {mae_improvement:.1f}%")
    print(f"   • XGBoost improves R² by {r2_improvement:.1f}%")
    print(f"   • Better predictions lead to more accurate business decisions")
    
    # Business impact
    avg_revenue = np.mean(rule_based_results['actuals'])
    mae_reduction = rule_based_results['mae'] - xgboost_results['mae']
    print(f"\n💰 Business Impact:")
    print(f"   • Average daily profit: ${avg_revenue:.0f}")
    print(f"   • MAE reduction: ${mae_reduction:.0f}")
    print(f"   • More accurate profit forecasting for business planning")

def main():
    """
    Main test function
    """
    print("🧪 XGBoost Profit Prediction Model Test")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data(n_samples=500)  # Use 500 samples for faster testing
    
    # Test rule-based system
    rule_based_results = test_rule_based_prediction(df)
    
    # Test XGBoost model
    xgboost_results = test_xgboost_model(df)
    
    # Compare models
    if rule_based_results and xgboost_results:
        compare_models(rule_based_results, xgboost_results)
        
        # Save sample data for further testing
        df.to_csv('sample_profit_data.csv', index=False)
        print(f"\n💾 Sample data saved to 'sample_profit_data.csv'")
        print(f"   You can use this file to test the XGBoost server")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()
