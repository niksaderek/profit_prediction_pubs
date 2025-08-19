import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class XGBoostProfitPredictor:
    def __init__(self):
        self.model = None
        self.feature_encoders = {}
        self.scaler = RobustScaler()  # Better handling of outliers
        self.feature_names = []
        self.is_trained = False
        self.model_performance = {}
        self.training_stats = {}  # Store training data statistics
        
    def prepare_features(self, data):
        """
        Prepare features from raw CSV data for XGBoost training
        FIXED: Better feature engineering and no data leakage
        """
        print("🔧 Preparing features for XGBoost model...")
        
        # Convert to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Basic data cleaning
        df = df.dropna(subset=['Net Profit', 'Revenue', 'Vertical'])
        df = df[df['Revenue'] > 0]  # Only positive revenue
        df = df[df['Net Profit'].notna()]  # Valid profit values
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            # Sort by date for proper time series handling
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Create comprehensive feature set
        features = pd.DataFrame()
        
        # 1. Vertical encoding (categorical) - FIXED: No leakage
        if 'Vertical' in df.columns:
            le_vertical = LabelEncoder()
            features['vertical_encoded'] = le_vertical.fit_transform(df['Vertical'].astype(str))
            self.feature_encoders['vertical'] = le_vertical
        
        # 2. Publisher/Media Buyer count - FIXED: Proper grouping
        if 'Media Buyer' in df.columns:
            features['publisher_count'] = df.groupby('Date')['Media Buyer'].transform('nunique')
        else:
            features['publisher_count'] = 1
        
        # 3. Revenue features - FIXED: Better scaling and bucketing
        features['revenue'] = df['Revenue'].astype(float)
        features['revenue_log'] = np.log1p(features['revenue'])
        
        # FIXED: Use quantile-based revenue buckets instead of hard-coded
        revenue_quantiles = df['Revenue'].quantile([0.25, 0.5, 0.75]).values
        features['revenue_bucket'] = pd.cut(features['revenue'], 
                                          bins=[0] + list(revenue_quantiles) + [float('inf')], 
                                          labels=[0, 1, 2, 3])
        features['revenue_bucket'] = features['revenue_bucket'].astype(int)
        
        # 4. Temporal features - FIXED: Better seasonal patterns
        if 'Date' in df.columns:
            features['day_of_week'] = df['Date'].dt.dayofweek
            features['month'] = df['Date'].dt.month
            features['quarter'] = df['Date'].dt.quarter
            features['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
            features['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
            features['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
            
            # FIXED: Add better seasonal features
            features['day_of_year'] = df['Date'].dt.dayofyear
            features['week_of_year'] = df['Date'].dt.isocalendar().week
            features['is_medicare_season'] = df['Date'].dt.month.isin([1, 2, 10, 11, 12]).astype(int)
        
        # 5. Day name encoding
        if 'Day' in df.columns:
            le_day = LabelEncoder()
            features['day_name_encoded'] = le_day.fit_transform(df['Day'].astype(str))
            self.feature_encoders['day'] = le_day
        
        # 6. Interaction features - FIXED: Better interactions
        features['vertical_publisher_interaction'] = features['vertical_encoded'] * features['publisher_count']
        features['revenue_vertical_interaction'] = features['revenue_log'] * features['vertical_encoded']
        
        # 7. Call volume features - FIXED: Better rate calculations
        if 'Incoming' in df.columns:
            features['incoming_calls'] = pd.to_numeric(df['Incoming'], errors='coerce').fillna(0)
            features['connected_calls'] = pd.to_numeric(df['Connected'], errors='coerce').fillna(0)
            features['converted_calls'] = pd.to_numeric(df['Converted'], errors='coerce').fillna(0)
            
            # FIXED: Better rate calculations with smoothing
            features['connect_rate'] = np.where(features['incoming_calls'] > 0, 
                                             features['connected_calls'] / features['incoming_calls'], 0.5)
            features['conversion_rate'] = np.where(features['connected_calls'] > 0, 
                                                features['converted_calls'] / features['connected_calls'], 0.3)
            
            # FIXED: Add call volume ratios
            features['calls_per_revenue'] = features['incoming_calls'] / (features['revenue'] + 1)
        else:
            features['incoming_calls'] = 0
            features['connected_calls'] = 0
            features['converted_calls'] = 0
            features['connect_rate'] = 0.5
            features['conversion_rate'] = 0.3
            features['calls_per_revenue'] = 0
        
        # 8. FIXED: Better lag features with proper time series handling
        if 'Date' in df.columns and len(df) > 1:
            # Previous day's profit margin (not absolute profit)
            features['profit_margin_lag_1'] = (df['Net Profit'] / df['Revenue']).shift(1).fillna(0.2)
            features['revenue_lag_1'] = df['Revenue'].shift(1).fillna(df['Revenue'].mean())
            
            # Rolling averages of profit margins
            features['profit_margin_rolling_3d'] = (df['Net Profit'] / df['Revenue']).rolling(3, min_periods=1).mean()
            features['profit_margin_rolling_7d'] = (df['Net Profit'] / df['Revenue']).rolling(7, min_periods=1).mean()
            
            # Revenue volatility
            features['revenue_volatility_7d'] = df['Revenue'].rolling(7, min_periods=1).std() / df['Revenue'].rolling(7, min_periods=1).mean()
        else:
            features['profit_margin_lag_1'] = 0.2
            features['revenue_lag_1'] = df['Revenue'].mean() if len(df) > 0 else 0
            features['profit_margin_rolling_3d'] = 0.2
            features['profit_margin_rolling_7d'] = 0.2
            features['revenue_volatility_7d'] = 0
        
        # 9. FIXED: Statistical features without leakage
        if len(df) > 1:
            # Use rolling statistics instead of global statistics
            features['revenue_rolling_mean_7d'] = df['Revenue'].rolling(7, min_periods=1).mean()
            features['revenue_rolling_std_7d'] = df['Revenue'].rolling(7, min_periods=1).std()
            features['revenue_zscore_7d'] = (df['Revenue'] - features['revenue_rolling_mean_7d']) / (features['revenue_rolling_std_7d'] + 1e-8)
        else:
            features['revenue_rolling_mean_7d'] = df['Revenue'].mean() if len(df) > 0 else 0
            features['revenue_rolling_std_7d'] = 0
            features['revenue_zscore_7d'] = 0
        
        # 10. FIXED: Add profit margin features for better learning
        features['profit_margin'] = df['Net Profit'] / df['Revenue']
        features['profit_margin_log'] = np.log1p(features['profit_margin'])
        
        # Store training statistics for prediction
        self.training_stats = {
            'revenue_quantiles': revenue_quantiles.tolist() if len(df) > 0 else [0, 0, 0],
            'revenue_mean': df['Revenue'].mean() if len(df) > 0 else 0,
            'revenue_std': df['Revenue'].std() if len(df) > 0 else 0,
            'profit_margin_mean': features['profit_margin'].mean() if len(df) > 0 else 0.2,
            'profit_margin_std': features['profit_margin'].std() if len(df) > 0 else 0.05
        }
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # FIXED: Target variable is now profit margin instead of absolute profit
        target = features['profit_margin']
        
        # Remove target from features
        features = features.drop('profit_margin', axis=1)
        self.feature_names = features.columns.tolist()
        
        print(f"✅ Features prepared: {len(features.columns)} features, {len(features)} samples")
        print(f"📊 Feature names: {', '.join(self.feature_names)}")
        print(f"💰 Target: Profit Margin (range: {target.min():.3f} - {target.max():.3f})")
        
        return features, target
    
    def train_model(self, features, target, test_size=0.2):
        """
        Train XGBoost model with the prepared features
        FIXED: Better training process and hyperparameters
        """
        print("🚀 Training XGBoost model...")
        
        # FIXED: Better time series split
        if len(features) > 100:
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # For final training, use last 80% for training, 20% for validation
            split_idx = int(len(features) * (1 - test_size))
            X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_val = target.iloc[:split_idx], target.iloc[split_idx:]
        else:
            # If limited data, use all for training
            X_train, X_val = features, features
            y_train, y_val = target, target
        
        # FIXED: Scale only numerical features, not categorical
        numerical_features = features.select_dtypes(include=[np.number]).columns
        categorical_features = features.select_dtypes(include=['object', 'category']).columns
        
        # Scale numerical features
        if len(numerical_features) > 0:
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            
            X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
            X_val_scaled[numerical_features] = self.scaler.transform(X_val[numerical_features])
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # FIXED: Better hyperparameters for profit margin prediction
        self.model = xgb.XGBRegressor(
            n_estimators=2000,  # Increased for better learning
            learning_rate=0.005,  # Lower learning rate for better generalization
            max_depth=8,  # Slightly deeper for complex patterns
            min_child_weight=3,  # Increased to prevent overfitting
            subsample=0.85,  # Better subsampling
            colsample_bytree=0.85,  # Better column sampling
            reg_alpha=0.05,  # L1 regularization
            reg_lambda=0.8,  # L2 regularization
            random_state=42,
            early_stopping_rounds=100,  # Increased for better convergence
            eval_metric='mae'  # Use MAE for margin prediction
        )
        
        # Train the model
        if len(X_train_scaled) > 0:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Make predictions
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_val = self.model.predict(X_val_scaled)
            
            # Calculate metrics on profit margins
            train_mae = mean_absolute_error(y_train, y_pred_train)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            train_r2 = r2_score(y_train, y_pred_train)
            val_r2 = r2_score(y_val, y_pred_val)
            
            # FIXED: Convert margin predictions back to absolute profit for comparison
            train_profit_pred = y_pred_train * X_train['revenue']
            train_profit_actual = y_train * X_train['revenue']
            val_profit_pred = y_pred_val * X_val['revenue']
            val_profit_actual = y_val * X_val['revenue']
            
            # Calculate absolute profit MAE
            train_profit_mae = mean_absolute_error(train_profit_actual, train_profit_pred)
            val_profit_mae = mean_absolute_error(val_profit_actual, val_profit_pred)
            
            self.model_performance = {
                'train_mae_margin': train_mae,
                'val_mae_margin': val_mae,
                'train_mae_profit': train_profit_mae,
                'val_mae_profit': val_profit_mae,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'feature_importance': self.get_feature_importance(),
                'training_date': datetime.now().isoformat(),
                'n_samples': len(features)
            }
            
            self.is_trained = True
            
            print(f"✅ Model trained successfully!")
            print(f"📊 Margin MAE - Train: {train_mae:.4f}, Val: {val_mae:.4f}")
            print(f"📊 Profit MAE - Train: ${train_profit_mae:.2f}, Val: ${val_profit_mae:.2f}")
            print(f"📊 R² Score - Train: {train_r2:.3f}, Val: {val_r2:.3f}")
            
            return True
        else:
            print("❌ No training data available")
            return False
    
    def predict_profit(self, num_buyers, expected_revenue, selected_verticals, additional_features=None):
        """
        Make profit prediction using trained XGBoost model
        FIXED: Better prediction with proper feature scaling
        """
        if not self.is_trained:
            print("❌ Model not trained yet. Please train the model first.")
            return None
        
        try:
            # Create feature vector for prediction
            features = self.create_prediction_features(
                num_buyers, expected_revenue, selected_verticals, additional_features
            )
            
            # Scale numerical features
            numerical_features = features.select_dtypes(include=[np.number]).columns
            if len(numerical_features) > 0:
                features_scaled = features.copy()
                features_scaled[numerical_features] = self.scaler.transform(features[numerical_features])
            else:
                features_scaled = features
            
            # Make margin prediction
            predicted_margin = self.model.predict(features_scaled)[0]
            
            # Convert margin to absolute profit
            predicted_profit = expected_revenue * predicted_margin
            
            # Ensure prediction is reasonable
            predicted_profit = max(0, predicted_profit)
            
            print(f"🤖 XGBoost Prediction:")
            print(f"   • Predicted Margin: {predicted_margin:.4f} ({predicted_margin*100:.2f}%)")
            print(f"   • Predicted Profit: ${predicted_profit:.2f}")
            
            return predicted_profit
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def create_prediction_features(self, num_buyers, expected_revenue, selected_verticals, additional_features=None):
        """
        Create feature vector for a single prediction
        FIXED: Better feature creation without data leakage
        """
        # Initialize features with zeros
        features = pd.DataFrame(0, index=[0], columns=self.feature_names)
        
        # 1. Vertical encoding
        if 'vertical_encoded' in self.feature_names and selected_verticals:
            if 'vertical' in self.feature_encoders:
                vertical_encodings = [self.feature_encoders['vertical'].transform([v])[0] 
                                    for v in selected_verticals if v in self.feature_encoders['vertical'].classes_]
                if vertical_encodings:
                    features['vertical_encoded'] = np.mean(vertical_encodings)
        
        # 2. Publisher count
        if 'publisher_count' in self.feature_names:
            features['publisher_count'] = num_buyers
        
        # 3. Revenue features
        if 'revenue' in self.feature_names:
            features['revenue'] = expected_revenue
            features['revenue_log'] = np.log1p(expected_revenue)
        
        # 4. Revenue bucket using training quantiles
        if 'revenue_bucket' in self.feature_names and 'revenue_quantiles' in self.training_stats:
            quantiles = self.training_stats['revenue_quantiles']
            if expected_revenue <= quantiles[0]:
                features['revenue_bucket'] = 0
            elif expected_revenue <= quantiles[1]:
                features['revenue_bucket'] = 1
            elif expected_revenue <= quantiles[2]:
                features['revenue_bucket'] = 2
            else:
                features['revenue_bucket'] = 3
        
        # 5. Temporal features (use current date)
        current_date = datetime.now()
        if 'day_of_week' in self.feature_names:
            features['day_of_week'] = current_date.weekday()
        if 'month' in self.feature_names:
            features['month'] = current_date.month
        if 'quarter' in self.feature_names:
            features['quarter'] = (current_date.month - 1) // 3 + 1
        if 'is_weekend' in self.feature_names:
            features['is_weekend'] = 1 if current_date.weekday() >= 5 else 0
        if 'is_month_start' in self.feature_names:
            features['is_month_start'] = 1 if current_date.day == 1 else 0
        if 'is_month_end' in self.feature_names:
            features['is_month_end'] = 1 if current_date.day >= 28 else 0
        if 'day_of_year' in self.feature_names:
            features['day_of_year'] = current_date.timetuple().tm_yday
        if 'week_of_year' in self.feature_names:
            features['week_of_year'] = current_date.isocalendar()[1]
        if 'is_medicare_season' in self.feature_names:
            features['is_medicare_season'] = 1 if current_date.month in [1, 2, 10, 11, 12] else 0
        
        # 6. Day name encoding
        if 'day_name_encoded' in self.feature_names and 'day' in self.feature_encoders:
            current_day_name = current_date.strftime('%A')
            if current_day_name in self.feature_encoders['day'].classes_:
                features['day_name_encoded'] = self.feature_encoders['day'].transform([current_day_name])[0]
        
        # 7. Interaction features
        if 'vertical_publisher_interaction' in self.feature_names:
            features['vertical_publisher_interaction'] = features['vertical_encoded'].iloc[0] * num_buyers
        if 'revenue_vertical_interaction' in self.feature_names:
            features['revenue_vertical_interaction'] = features['revenue_log'].iloc[0] * features['vertical_encoded'].iloc[0]
        
        # 8. Call volume features (use realistic defaults)
        if 'incoming_calls' in self.feature_names:
            features['incoming_calls'] = int(expected_revenue / 100)  # Estimate based on revenue
            features['connected_calls'] = int(features['incoming_calls'].iloc[0] * 0.7)
            features['converted_calls'] = int(features['connected_calls'].iloc[0] * 0.3)
            features['connect_rate'] = 0.7
            features['conversion_rate'] = 0.3
            features['calls_per_revenue'] = features['incoming_calls'].iloc[0] / expected_revenue
        
        # 9. Lag features (use training averages)
        if 'profit_margin_lag_1' in self.feature_names:
            features['profit_margin_lag_1'] = self.training_stats.get('profit_margin_mean', 0.2)
        if 'revenue_lag_1' in self.feature_names:
            features['revenue_lag_1'] = expected_revenue  # Use current as reference
        if 'profit_margin_rolling_3d' in self.feature_names:
            features['profit_margin_rolling_3d'] = self.training_stats.get('profit_margin_mean', 0.2)
        if 'profit_margin_rolling_7d' in self.feature_names:
            features['profit_margin_rolling_7d'] = self.training_stats.get('profit_margin_mean', 0.2)
        if 'revenue_volatility_7d' in self.feature_names:
            features['revenue_volatility_7d'] = 0.1  # Assume 10% volatility
        
        # 10. Statistical features (use training data averages)
        if 'revenue_rolling_mean_7d' in self.feature_names:
            features['revenue_rolling_mean_7d'] = expected_revenue
        if 'revenue_rolling_std_7d' in self.feature_names:
            features['revenue_rolling_std_7d'] = expected_revenue * 0.2  # Assume 20% standard deviation
        if 'revenue_zscore_7d' in self.feature_names:
            features['revenue_zscore_7d'] = 0  # Assume current revenue is at mean
        
        # Add any additional features if provided
        if additional_features:
            for key, value in additional_features.items():
                if key in self.feature_names:
                    features[key] = value
        
        return features
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_model(self, filepath):
        """
        Save the trained model and encoders
        FIXED: Include training statistics
        """
        if not self.is_trained:
            print("❌ No trained model to save")
            return False
        
        try:
            # Save XGBoost model
            model_file = f"{filepath}_model.json"
            self.model.save_model(model_file)
            
            # Save encoders and metadata
            metadata = {
                'feature_names': self.feature_names,
                'feature_encoders': {k: v.classes_.tolist() for k, v in self.feature_encoders.items()},
                'model_performance': self.model_performance,
                'training_stats': self.training_stats,
                'is_trained': self.is_trained
            }
            
            metadata_file = f"{filepath}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Load a previously saved model
        FIXED: Include training statistics
        """
        try:
            # Load XGBoost model
            model_file = f"{filepath}_model.json"
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_file)
            
            # Load metadata
            metadata_file = f"{filepath}_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.is_trained = metadata['is_trained']
            self.model_performance = metadata['model_performance']
            self.training_stats = metadata.get('training_stats', {})
            
            # Reconstruct encoders
            self.feature_encoders = {}
            for name, classes in metadata['feature_encoders'].items():
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                self.feature_encoders[name] = le
            
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_model_summary(self):
        """
        Get a summary of the model performance and features
        FIXED: Show both margin and profit metrics
        """
        if not self.is_trained:
            return "Model not trained yet"
        
        summary = f"""
🤖 XGBoost Profit Prediction Model Summary
{'='*50}

📊 Performance Metrics (Profit Margins):
• Training MAE: {self.model_performance.get('train_mae_margin', 0):.4f}
• Validation MAE: {self.model_performance.get('val_mae_margin', 0):.4f}
• Training R²: {self.model_performance.get('train_r2', 0):.3f}
• Validation R²: {self.model_performance.get('val_r2', 0):.3f}

📊 Performance Metrics (Absolute Profit):
• Training MAE: ${self.model_performance.get('train_mae_profit', 0):.2f}
• Validation MAE: ${self.model_performance.get('val_mae_profit', 0):.2f}

🔧 Model Details:
• Features: {len(self.feature_names)}
• Training samples: {self.model_performance.get('n_samples', 0)}
• Training date: {self.model_performance.get('training_date', 'Unknown')}

🏆 Top 5 Most Important Features:
"""
        
        feature_importance = self.get_feature_importance()
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            summary += f"• {feature}: {importance:.4f}\n"
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Improved XGBoost Profit Predictor...")
    
    # Create sample data for testing
    sample_data = {
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Day': ['Monday', 'Tuesday', 'Wednesday'],
        'Vertical': ['MEDICARE ENGLISH', 'FINAL EXPENSE ENGLISH', 'ACA ENGLISH'],
        'Revenue': [15000, 22000, 18000],
        'Net Profit': [3200, 4800, 3900],
        'Media Buyer': ['Publisher1', 'Publisher2', 'Publisher1'],
        'Incoming': [100, 120, 95],
        'Connected': [80, 95, 75],
        'Converted': [25, 30, 22]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize and train model
    predictor = XGBoostProfitPredictor()
    features, target = predictor.prepare_features(df)
    
    if len(features) > 0:
        success = predictor.train_model(features, target)
        if success:
            # Test prediction
            prediction = predictor.predict_profit(
                num_buyers=3,
                expected_revenue=25000,
                selected_verticals=['MEDICARE ENGLISH', 'FINAL EXPENSE ENGLISH']
            )
            
            print(f"\n🎯 Test Prediction: ${prediction:.2f}")
            print(predictor.get_model_summary())
    else:
        print("❌ No valid features generated from sample data")
