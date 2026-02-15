import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'deriv-analysis-secret-key-2024'
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # Deriv API Configuration
    DERIV_APP_ID = 1089
    DERIV_DEFAULT_SYMBOL = "R_100"  # Volatility 100 Index
    
    # Available contract types
    CONTRACT_TYPES = {
        "digit_match": {
            "name": "Digit Matches",
            "description": "Predict if last digit matches your choice",
            "payout": "9:1",
            "probability": "10%"
        },
        "digit_differs": {
            "name": "Digit Differs",
            "description": "Predict if last digit differs from your choice",
            "payout": "0.35:1",
            "probability": "90%"
        },
        "digit_over": {
            "name": "Digit Over",
            "description": "Predict if last digit is over 4",
            "payout": "0.9:1",
            "probability": "50%"
        },
        "digit_under": {
            "name": "Digit Under",
            "description": "Predict if last digit is under 5",
            "payout": "0.9:1",
            "probability": "50%"
        },
        "digit_odd": {
            "name": "Digit Odd",
            "description": "Predict if last digit is odd",
            "payout": "0.9:1",
            "probability": "50%"
        },
        "digit_even": {
            "name": "Digit Even",
            "description": "Predict if last digit is even",
            "payout": "0.9:1",
            "probability": "50%"
        }
    }
    
    # Available volatility indices
    VOLATILITY_INDICES = {
        "R_10": "Volatility 10 Index",
        "R_25": "Volatility 25 Index", 
        "R_50": "Volatility 50 Index",
        "R_75": "Volatility 75 Index",
        "R_100": "Volatility 100 Index"
    }
    
    # Analysis parameters
    WINDOW_SIZE = 300
    LOOKBACK_PERIOD = 50
    UPDATE_INTERVAL = 1  # seconds