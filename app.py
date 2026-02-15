# Now import other modules
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import asyncio
import random
from datetime import datetime
import time
from threading import Thread
from deriv_api import DerivAPI
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'deriv-auto-connect-2024'
socketio = SocketIO(app, async_mode="threading")

# YOUR API TOKEN - ALREADY EMBEDDED
YOUR_API_TOKEN = "MzJgr7FVhpRcB0D"
APP_ID = 1089

# Global variables - Store data for ALL markets separately
market_data = {
    # 1s versions
    "R_10": [],      # Volatility 10 (1s)
    "R_25": [],      # Volatility 25 (1s)
    "R_50": [],      # Volatility 50 (1s)
    "R_75": [],      # Volatility 75 (1s)
    "R_100": [],     # Volatility 100 (1s)
    
    # Standard versions (non-1s)
    "1HZ10V": [],    # Volatility 10
    "1HZ25V": [],    # Volatility 25
    "1HZ50V": [],    # Volatility 50
    "1HZ75V": [],    # Volatility 75
    "1HZ100V": []    # Volatility 100
}

connection_status = "disconnected"
api = None
current_symbol = "R_100"  # Currently selected market for display
active_subscriptions = {}  # Track active subscriptions per market

# Analysis window constant
ANALYSIS_WINDOW = 300  # Use 300 ticks for analysis

# Display names for all markets
DISPLAY_NAMES = {
    "R_10": "Volatility 10 (1s)",
    "R_25": "Volatility 25 (1s)",
    "R_50": "Volatility 50 (1s)",
    "R_75": "Volatility 75 (1s)",
    "R_100": "Volatility 100 (1s)",
    "1HZ10V": "Volatility 10",
    "1HZ25V": "Volatility 25",
    "1HZ50V": "Volatility 50",
    "1HZ75V": "Volatility 75",
    "1HZ100V": "Volatility 100"
}

print("="*60)
print("🚀 DERIV MULTI-MARKET ANALYZER STARTING...")
print(f"🔑 API Token: {YOUR_API_TOKEN[:8]}...")
print("="*60)
print("📈 Tracking 10 markets:")
for symbol, name in DISPLAY_NAMES.items():
    print(f"   • {name} ({symbol})")
print("="*60)


# ==============================
# ENHANCED ANALYSIS METHODS
# ==============================

def analyze_hot_method(digits):
    """HOT METHOD: Most frequent digit (momentum following)"""
    if not digits or len(digits) < 10:
        return {'digit': 0, 'confidence': 0}
    
    # Use last 300 ticks
    if len(digits) > ANALYSIS_WINDOW:
        analysis_digits = digits[-ANALYSIS_WINDOW:]
    else:
        analysis_digits = digits
    
    # Count frequencies
    counts = {}
    for d in range(10):
        counts[d] = analysis_digits.count(d)
    
    # Find hottest digit
    hottest = max(counts, key=counts.get)
    hottest_count = counts[hottest]
    
    # Expected count per digit
    expected = len(analysis_digits) / 10
    
    # Calculate confidence
    if expected > 0:
        std_dev = np.sqrt(expected * 0.9)
        deviation = (hottest_count - expected) / std_dev if std_dev > 0 else 0
        confidence = min(95, 50 + (deviation * 10))
    else:
        confidence = 50
    
    return {
        'digit': hottest,
        'confidence': round(confidence, 1)
    }


def analyze_cold_method(digits):
    """COLD METHOD: Least frequent digit (mean reversion)"""
    if not digits or len(digits) < 10:
        return {'digit': 0, 'confidence': 0}
    
    # Use last 300 ticks
    if len(digits) > ANALYSIS_WINDOW:
        analysis_digits = digits[-ANALYSIS_WINDOW:]
    else:
        analysis_digits = digits
    
    # Count frequencies
    counts = {}
    for d in range(10):
        counts[d] = analysis_digits.count(d)
    
    # Find coldest digit
    coldest = min(counts, key=counts.get)
    coldest_count = counts[coldest]
    
    # Expected count per digit
    expected = len(analysis_digits) / 10
    
    # Calculate confidence
    if expected > 0:
        std_dev = np.sqrt(expected * 0.9)
        deviation = (expected - coldest_count) / std_dev if std_dev > 0 else 0
        confidence = min(95, 50 + (deviation * 10))
    else:
        confidence = 50
    
    return {
        'digit': coldest,
        'confidence': round(confidence, 1)
    }


def analyze_weighted_method(digits):
    """WEIGHTED RECENT METHOD: Exponential weighting"""
    if not digits or len(digits) < 10:
        return {'digit': 0, 'confidence': 0}
    
    # Use last 300 ticks
    if len(digits) > ANALYSIS_WINDOW:
        analysis_digits = digits[-ANALYSIS_WINDOW:]
    else:
        analysis_digits = digits
    
    # Apply exponential weights
    weights = []
    for i in range(len(analysis_digits)):
        weight = np.exp(i / len(analysis_digits))
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Calculate weighted counts
    weighted_counts = {d: 0 for d in range(10)}
    for i, digit in enumerate(analysis_digits):
        weighted_counts[digit] += weights[i]
    
    # Find digit with highest weighted count
    predicted = max(weighted_counts, key=weighted_counts.get)
    max_weight = weighted_counts[predicted]
    
    # Expected weight per digit
    expected_weight = 1.0 / 10
    
    # Calculate confidence
    if expected_weight > 0:
        deviation = (max_weight - expected_weight) / expected_weight * 100
        confidence = min(95, 50 + deviation)
    else:
        confidence = 50
    
    return {
        'digit': predicted,
        'confidence': round(confidence, 1)
    }


def get_best_prediction(digits):
    """Run all 3 methods and return the best one"""
    if not digits or len(digits) < 10:
        return {
            'digit': 0,
            'confidence': 0,
            'method': 'none',
            'method_name': 'Insufficient Data',
            'all_methods': {}
        }
    
    # Run all three methods
    hot = analyze_hot_method(digits)
    cold = analyze_cold_method(digits)
    weighted = analyze_weighted_method(digits)
    
    # Store all results
    results = {
        'hot': {
            'digit': hot['digit'],
            'confidence': hot['confidence'],
            'method_name': 'Hot (Momentum)'
        },
        'cold': {
            'digit': cold['digit'],
            'confidence': cold['confidence'],
            'method_name': 'Cold (Mean Reversion)'
        },
        'weighted': {
            'digit': weighted['digit'],
            'confidence': weighted['confidence'],
            'method_name': 'Weighted Recent'
        }
    }
    
    # Find method with highest confidence
    best_method = max(results, key=lambda m: results[m]['confidence'])
    best = results[best_method]
    
    return {
        'digit': best['digit'],
        'confidence': best['confidence'],
        'method': best_method,
        'method_name': best['method_name'],
        'all_methods': results
    }


def get_digit_distribution(market_symbol):
    """Get distribution of digits in last 300 ticks for a specific market"""
    if market_symbol not in market_data or not market_data[market_symbol]:
        return {d: 0 for d in range(10)}
    
    digits = [t['digit'] for t in market_data[market_symbol]]
    
    if len(digits) > ANALYSIS_WINDOW:
        analysis_digits = digits[-ANALYSIS_WINDOW:]
    else:
        analysis_digits = digits
    
    return {d: analysis_digits.count(d) for d in range(10)}


# ==============================
# DERIV CONNECTION
# ==============================

async def connect_deriv():
    global api, connection_status

    try:
        print("🔗 Connecting to Deriv API...")
        api = DerivAPI(app_id=APP_ID)

        auth_response = await api.authorize(YOUR_API_TOKEN)

        if 'error' in auth_response:
            print(f"❌ Authorization error: {auth_response['error']}")
            connection_status = "auth_error"
            return False

        print("✅ Successfully connected to Deriv!")
        print(f"📊 Account: {auth_response.get('authorize', {}).get('loginid', 'Demo Account')}")

        connection_status = "connected"
        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        connection_status = f"error: {str(e)}"
        return False


# ==============================
# BACKGROUND TASKS - Subscribe to ALL markets
# ==============================

def start_background_tasks():
    # First connect
    Thread(target=run_deriv_stream, daemon=True).start()
    # Then subscribe to all markets
    Thread(target=subscribe_to_all_markets, daemon=True).start()


def run_deriv_stream():
    asyncio.run(main_deriv_loop())


async def main_deriv_loop():
    global api, connection_status

    try:
        print("🔗 Connecting to Deriv API...")
        api = DerivAPI(app_id=APP_ID)

        auth_response = await api.authorize(YOUR_API_TOKEN)

        if 'error' in auth_response:
            print(f"❌ Authorization error: {auth_response['error']}")
            connection_status = "auth_error"
            return

        print("✅ Successfully connected to Deriv!")
        print(f"📊 Account: {auth_response.get('authorize', {}).get('loginid', 'Demo Account')}")

        connection_status = "connected"

        # Keep alive forever
        while True:
            await asyncio.sleep(3600)

    except Exception as e:
        print(f"❌ Deriv loop error: {e}")
        connection_status = f"error: {str(e)}"


def subscribe_to_all_markets():
    """Subscribe to ALL volatility indices simultaneously"""
    all_symbols = [
        "R_10", "R_25", "R_50", "R_75", "R_100",           # 1s versions
        "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"  # Standard versions
    ]
    
    for symbol in all_symbols:
        print(f"📡 Starting subscription for {DISPLAY_NAMES.get(symbol, symbol)} ({symbol})...")
        Thread(target=subscribe_to_market, args=(symbol,), daemon=True).start()
        time.sleep(2)  # Small delay to avoid rate limiting


def subscribe_to_market(symbol):
    """Subscribe to a specific market"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(subscribe_to_symbol(symbol))
    except Exception as e:
        print(f"❌ Subscription error for {symbol}: {e}")
    finally:
        loop.close()


async def subscribe_to_symbol(symbol):
    """Subscribe to ticks for a specific symbol"""
    global api
    
    try:
        print(f"🎯 Setting up subscription for {DISPLAY_NAMES.get(symbol, symbol)}...")
        
        # Create new API instance for this subscription
        symbol_api = DerivAPI(app_id=APP_ID)
        await symbol_api.authorize(YOUR_API_TOKEN)
        
        active_subscriptions[symbol] = symbol_api
        
        subscription = await symbol_api.subscribe({
            "ticks": symbol,
            "subscribe": 1
        })

        subscription.subscribe(
            on_next=lambda response: handle_market_tick(response, symbol),
            on_error=lambda e: print(f"❌ Stream error for {symbol}: {e}")
        )

        # Keep alive forever
        while True:
            await asyncio.sleep(3600)

    except Exception as e:
        print(f"❌ Subscription error for {symbol}: {e}")


# ==============================
# PROCESS TICKS - Market-specific storage
# ==============================

def handle_market_tick(response, symbol):
    """Handle tick response for a specific market"""
    if 'error' in response:
        print(f"⚠️ API Error for {symbol}: {response['error']}")
        return

    if 'tick' in response:
        process_market_tick(response['tick'], symbol)


def process_market_tick(tick, symbol):
    global market_data

    try:
        price = float(tick['quote'])
        epoch = tick['epoch']
        
        # Extract last digit
        price_str = f"{price:.2f}"
        digit = int(price_str.replace('.', '')[-1])

        tick_info = {
            'price': price,
            'digit': digit,
            'epoch': epoch,
            'time': datetime.fromtimestamp(epoch).strftime('%H:%M:%S'),
            'symbol': symbol,
            'display_name': DISPLAY_NAMES.get(symbol, symbol),
            'source': 'REAL'
        }

        # Store in the correct market bucket
        if symbol not in market_data:
            market_data[symbol] = []
        
        market_data[symbol].append(tick_info)

        # Keep last 500 ticks per market
        if len(market_data[symbol]) > 500:
            market_data[symbol] = market_data[symbol][-500:]

        # Only emit for the currently selected market to reduce bandwidth
        if symbol == current_symbol:
            socketio.emit('new_tick', {
                'price': price,
                'digit': digit,
                'time': tick_info['time'],
                'total_ticks': len(market_data[symbol]),
                'source': 'REAL',
                'symbol': symbol,
                'display_name': tick_info['display_name']
            })

            if len(market_data[symbol]) % 10 == 0:
                socketio.emit('stats_update', get_statistics_for_market(symbol))

    except Exception as e:
        print(f"❌ Tick processing error for {symbol}: {e}")


# ==============================
# STATISTICS - Market-specific
# ==============================

def get_statistics_for_market(symbol):
    """Get statistics for a specific market"""
    if symbol not in market_data or not market_data[symbol]:
        return {
            'total_ticks': 0, 
            'connection_status': connection_status,
            'market': symbol,
            'display_name': DISPLAY_NAMES.get(symbol, symbol)
        }

    # Use last 300 ticks for distribution
    if len(market_data[symbol]) > ANALYSIS_WINDOW:
        last_n = [t['digit'] for t in market_data[symbol][-ANALYSIS_WINDOW:]]
    else:
        last_n = [t['digit'] for t in market_data[symbol]]
    
    digit_counts = {d: last_n.count(d) for d in range(10)}

    most_common = max(digit_counts, key=digit_counts.get)
    least_common = min(digit_counts, key=digit_counts.get)

    return {
        'total_ticks': len(market_data[symbol]),
        'digit_distribution': digit_counts,
        'most_common': {'digit': most_common, 'count': digit_counts[most_common]},
        'least_common': {'digit': least_common, 'count': digit_counts[least_common]},
        'connection_status': connection_status,
        'current_symbol': symbol,
        'display_name': DISPLAY_NAMES.get(symbol, symbol),
        'last_update': datetime.now().isoformat()
    }


def get_statistics():
    """Legacy function - returns stats for current market"""
    return get_statistics_for_market(current_symbol)


# ==============================
# SIMULATED DATA (FALLBACK)
# ==============================

def simulate_data():
    """Fallback simulation - generates data for all markets"""
    print("🔄 Starting simulated data for all markets...")
    
    symbols = list(market_data.keys())

    while True:
        for symbol in symbols:
            price = 1000 + random.random() * 100
            digit = random.randint(0, 9)

            tick_info = {
                'price': price,
                'digit': digit,
                'epoch': time.time(),
                'time': datetime.now().strftime('%H:%M:%S'),
                'symbol': symbol,
                'display_name': DISPLAY_NAMES.get(symbol, symbol),
                'source': 'SIMULATED'
            }

            if symbol not in market_data:
                market_data[symbol] = []
            
            market_data[symbol].append(tick_info)

            if len(market_data[symbol]) > 500:
                market_data[symbol] = market_data[symbol][-500:]

            # Only emit for current market
            if symbol == current_symbol:
                socketio.emit('new_tick', {
                    'price': price,
                    'digit': digit,
                    'time': tick_info['time'],
                    'total_ticks': len(market_data[symbol]),
                    'source': 'SIMULATED',
                    'symbol': symbol,
                    'display_name': tick_info['display_name']
                })

        time.sleep(1)  # One tick per second total, cycling through markets


# ==============================
# ROUTES
# ==============================

@app.route('/')
def index():
    return render_template('index.html', display_names=DISPLAY_NAMES)


@app.route('/api/statistics')
def api_statistics():
    return jsonify({'status': 'success', 'data': get_statistics()})


@app.route('/api/statistics/<symbol>')
def api_statistics_for_symbol(symbol):
    """Get statistics for a specific market"""
    if symbol in market_data:
        return jsonify({'status': 'success', 'data': get_statistics_for_market(symbol)})
    return jsonify({'status': 'error', 'message': 'Market not found'})


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    contract_type = data.get('contract_type', 'digit_match')
    symbol = data.get('symbol', current_symbol)  # Allow specifying symbol
    
    if symbol not in market_data or not market_data[symbol] or len(market_data[symbol]) < 10:
        return jsonify({
            'status': 'error',
            'message': f'Not enough data for {DISPLAY_NAMES.get(symbol, symbol)}. Have {len(market_data.get(symbol, []))} ticks, need at least 10.',
            'current_ticks': len(market_data.get(symbol, []))
        })
    
    # Get digits for this specific market
    digits = [t['digit'] for t in market_data[symbol]]
    
    # Get best prediction from all 3 methods
    prediction = get_best_prediction(digits)
    
    # Get distribution for display (last 300)
    distribution = get_digit_distribution(symbol)
    
    # Get last 10 ticks for display
    last_10 = []
    if len(market_data[symbol]) >= 10:
        for t in market_data[symbol][-10:]:
            last_10.append({
                'digit': t['digit'],
                'time': t['time'],
                'price': t['price']
            })
    
    # Contract-specific recommendations
    recommendations = {
        'digit_match': f"Bet on digit {prediction['digit']}",
        'digit_differs': f"Bet digit is NOT {prediction['digit']}",
        'digit_over': f"Bet OVER 4" if prediction['digit'] > 4 else f"Bet UNDER 5",
        'digit_under': f"Bet UNDER 5" if prediction['digit'] < 5 else f"Bet OVER 4",
        'digit_odd': f"Bet ODD" if prediction['digit'] % 2 == 1 else f"Bet EVEN",
        'digit_even': f"Bet EVEN" if prediction['digit'] % 2 == 0 else f"Bet ODD"
    }
    
    return jsonify({
        'status': 'success',
        'prediction': {
            'digit': prediction['digit'],
            'confidence': prediction['confidence'],
            'method': prediction['method'],
            'method_name': prediction['method_name'],
            'recommendation': recommendations.get(contract_type, 'Analyze complete'),
            'timestamp': datetime.now().isoformat(),
            'all_methods': prediction['all_methods'],
            'market': symbol,
            'market_display': DISPLAY_NAMES.get(symbol, symbol)
        },
        'statistics': {
            'total_ticks': len(digits),
            'distribution': distribution,
            'last_10': last_10
        }
    })


@app.route('/api/change-symbol', methods=['POST'])
def change_symbol():
    global current_symbol
    
    data = request.json
    symbol = data.get('symbol', 'R_100')
    
    # All available volatility indices
    valid_symbols = list(market_data.keys())
    
    if symbol in valid_symbols:
        current_symbol = symbol
        print(f"📊 Switched to {DISPLAY_NAMES.get(symbol, symbol)} ({symbol})")
        
        return jsonify({
            'status': 'success',
            'message': f'Changed to {DISPLAY_NAMES.get(symbol, symbol)}',
            'symbol': symbol,
            'display_name': DISPLAY_NAMES.get(symbol, symbol),
            'total_ticks': len(market_data.get(symbol, []))
        })
    
    return jsonify({
        'status': 'error',
        'message': 'Invalid symbol'
    })


@app.route('/api/markets-data')
def markets_data():
    """Get data status for all markets"""
    result = {}
    for symbol in market_data.keys():
        result[symbol] = {
            'total_ticks': len(market_data[symbol]),
            'display_name': DISPLAY_NAMES.get(symbol, symbol),
            'last_tick': market_data[symbol][-1]['time'] if market_data[symbol] else None
        }
    return jsonify({
        'status': 'success',
        'markets': result
    })


@app.route('/api/connection-status')
def connection_status_api():
    return jsonify({
        'status': 'success',
        'connection': {
            'status': connection_status,
            'total_ticks_all': sum(len(market_data[s]) for s in market_data),
            'markets': {s: len(market_data[s]) for s in market_data if market_data[s]},
            'current_market': current_symbol,
            'current_market_display': DISPLAY_NAMES.get(current_symbol, current_symbol),
            'timestamp': datetime.now().isoformat()
        }
    })


# ==============================
# WEBSOCKET EVENTS
# ==============================

@socketio.on('connect')
def handle_ws_connect():
    print('🌐 Client connected')
    socketio.emit('welcome', {
        'message': 'Welcome to Deriv Multi-Market Analyzer',
        'connection_status': connection_status,
        'total_ticks_all': sum(len(market_data[s]) for s in market_data),
        'current_market': current_symbol,
        'current_market_display': DISPLAY_NAMES.get(current_symbol, current_symbol),
        'markets': {s: len(market_data[s]) for s in market_data if market_data[s]}
    })


@socketio.on('request_market_data')
def handle_market_request(data):
    """Client requests data for a specific market"""
    symbol = data.get('symbol', current_symbol)
    if symbol in market_data:
        stats = get_statistics_for_market(symbol)
        socketio.emit('market_data', {
            'symbol': symbol,
            'display_name': DISPLAY_NAMES.get(symbol, symbol),
            'stats': stats,
            'last_10': market_data[symbol][-10:] if market_data[symbol] else []
        })


# ==============================
# MAIN
# ==============================

if __name__ == '__main__':
    start_background_tasks()
    # Also start simulation as fallback
    Thread(target=simulate_data, daemon=True).start()

    print("\n" + "="*60)
    print("📱 DERIV MULTI-MARKET ANALYZER READY")
    print("="*60)
    print("🌐 URL: http://localhost:5000")
    print(f"📊 Status: {connection_status}")
    print("📈 Tracking 10 markets:")
    for symbol, name in DISPLAY_NAMES.items():
        print(f"   • {name} ({symbol})")
    print("✨ 3 silent methods per market: Hot, Cold, Weighted")
    print("🎯 Auto-best-pick based on confidence")
    print("="*60)

    socketio.run(app, debug=True, port=5000, use_reloader=False)