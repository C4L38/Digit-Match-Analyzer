import numpy as np
import pandas as pd
from collections import deque, Counter
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

class DerivAnalyzer:
    def __init__(self, window_size=300, lookback=50):
        self.window_size = window_size
        self.lookback = lookback
        self.data = deque(maxlen=window_size)
        self.digits = deque(maxlen=window_size)
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_prediction = None
        self.prediction_history = []
        
    def add_tick(self, price, digit, timestamp):
        """Add new tick data"""
        tick_data = {
            'price': price,
            'digit': digit,
            'timestamp': timestamp,
            'time': datetime.fromtimestamp(timestamp)
        }
        self.data.append(tick_data)
        self.digits.append(digit)
        
    def prepare_features(self):
        """Prepare features for ML"""
        if len(self.digits) < self.lookback + 10:
            return None, None
        
        features = []
        targets = []
        
        for i in range(self.lookback, len(self.digits) - 1):
            window = list(self.digits)[i-self.lookback:i]
            current = self.digits[i]
            next_digit = self.digits[i+1]
            
            # Basic features
            digit_counts = [window.count(d) for d in range(10)]
            current_count = window.count(current)
            
            # Statistical features
            absences = []
            for d in range(10):
                last_idx = max([idx for idx, val in enumerate(window) if val == d], default=-1)
                absences.append(len(window) - last_idx if last_idx != -1 else len(window))
            
            # Recent patterns
            recent = window[-5:] if len(window) >= 5 else window
            same_as_last = 1 if current == recent[-1] else 0 if recent else 0
            
            # Combine features
            feature_vector = [
                current,
                current_count,
                same_as_last,
                *digit_counts,
                *absences[:5],
                len(set(window[-10:])) if len(window) >= 10 else len(set(window))
            ]
            
            features.append(feature_vector)
            targets.append(next_digit)
        
        return np.array(features), np.array(targets)
    
    def train(self):
        """Train ML models"""
        X, y = self.prepare_features()
        if X is None or len(X) < 100:
            return False
        
        # Split and balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Balance classes
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train_bal)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.models['random_forest'].fit(X_train_scaled, y_train_bal)
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.models['xgboost'].fit(X_train_scaled, y_train_bal)
        
        # Test accuracy
        accuracies = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            accuracies[name] = acc
        
        self.is_trained = True
        return accuracies
    
    def predict_next(self):
        """Predict next digit"""
        if not self.is_trained or len(self.digits) < self.lookback:
            return None
        
        # Prepare current features
        X, _ = self.prepare_features()
        if X is None or len(X) == 0:
            return None
        
        current_features = X[-1:].copy()
        scaled_features = self.scaler.transform(current_features)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(scaled_features)[0]
            proba = model.predict_proba(scaled_features)[0]
            
            predictions[name] = {
                'digit': int(pred),
                'confidence': float(np.max(proba)),
                'all_probs': [float(p) for p in proba]
            }
        
        # Ensemble prediction
        all_preds = [p['digit'] for p in predictions.values()]
        ensemble_digit = Counter(all_preds).most_common(1)[0][0]
        
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'models': predictions,
            'ensemble': {
                'digit': ensemble_digit,
                'confidence': sum(p['confidence'] for p in predictions.values()) / len(predictions),
                'vote_distribution': dict(Counter(all_preds))
            },
            'current_digit': self.digits[-1] if self.digits else None,
            'data_points': len(self.digits)
        }
        
        self.last_prediction = prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def analyze_contract(self, contract_type):
        """Analyze specific contract type"""
        if self.last_prediction is None:
            return None
        
        pred_digit = self.last_prediction['ensemble']['digit']
        
        analysis = {
            'contract_type': contract_type,
            'predicted_digit': pred_digit,
            'prediction_time': self.last_prediction['timestamp'],
            'models_confidence': self.last_prediction['ensemble']['confidence']
        }
        
        # Contract-specific analysis
        if contract_type == "digit_match":
            analysis['recommendation'] = f"Bet on digit {pred_digit}"
            analysis['expected_payout'] = "9:1"
            analysis['probability'] = "10%"
            
        elif contract_type == "digit_differs":
            analysis['recommendation'] = f"Bet that digit is NOT {pred_digit}"
            analysis['expected_payout'] = "0.35:1"
            analysis['probability'] = "90%"
            
        elif contract_type == "digit_over":
            analysis['recommendation'] = f"Bet on OVER 4" if pred_digit > 4 else f"Bet on UNDER 5"
            analysis['expected_payout'] = "0.9:1"
            analysis['probability'] = "50%"
            
        elif contract_type == "digit_under":
            analysis['recommendation'] = f"Bet on UNDER 5" if pred_digit < 5 else f"Bet on OVER 4"
            analysis['expected_payout'] = "0.9:1"
            analysis['probability'] = "50%"
            
        elif contract_type == "digit_odd":
            analysis['recommendation'] = f"Bet on ODD" if pred_digit % 2 == 1 else f"Bet on EVEN"
            analysis['expected_payout'] = "0.9:1"
            analysis['probability'] = "50%"
            
        elif contract_type == "digit_even":
            analysis['recommendation'] = f"Bet on EVEN" if pred_digit % 2 == 0 else f"Bet on ODD"
            analysis['expected_payout'] = "0.9:1"
            analysis['probability'] = "50%"
        
        return analysis
    
    def get_statistics(self):
        """Get current statistics"""
        if len(self.digits) == 0:
            return None
        
        counter = Counter(self.digits)
        total = len(self.digits)
        
        stats = {
            'total_ticks': total,
            'digit_distribution': {str(d): counter.get(d, 0)/total*100 for d in range(10)},
            'most_common': counter.most_common(3),
            'least_common': counter.most_common()[:-4:-1],
            'current_streak': self._get_current_streak(),
            'last_update': datetime.now().isoformat()
        }
        
        return stats
    
    def _get_current_streak(self):
        """Get current digit streak"""
        if len(self.digits) < 2:
            return None
        
        last_digit = self.digits[-1]
        streak = 1
        
        for i in range(len(self.digits)-2, -1, -1):
            if self.digits[i] == last_digit:
                streak += 1
            else:
                break
        
        return {'digit': last_digit, 'length': streak}