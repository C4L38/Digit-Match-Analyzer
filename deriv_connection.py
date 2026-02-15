"""
Real Deriv API Connection
"""
import asyncio
import json
from datetime import datetime
from deriv_api import DerivAPI

class DerivRealTime:
    def __init__(self, app_id=1089, token=None):
        """
        Initialize Deriv API connection
        
        Args:
            app_id: Deriv Application ID (1089 for default)
            token: Your API token (None for public data)
        """
        self.app_id = app_id
        self.token = token
        self.api = None
        self.connected = False
        self.symbol = "R_100"  # Default to Volatility 100 Index
        
    async def connect(self):
        """Connect to Deriv API"""
        try:
            self.api = DerivAPI(app_id=self.app_id)
            
            # Authorize if token provided
            if self.token:
                await self.api.authorize(self.token)
                print("✓ Connected to Deriv with authentication")
            else:
                print("✓ Connected to Deriv (public data only)")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    async def subscribe_to_ticks(self, symbol, callback):
        """
        Subscribe to real-time ticks
        
        Args:
            symbol: Market symbol (e.g., "R_100")
            callback: Function to call on each tick
        """
        if not self.connected:
            await self.connect()
        
        self.symbol = symbol
        
        print(f"📡 Subscribing to {symbol}...")
        
        try:
            # Subscribe to ticks
            await self.api.subscribe({
                "ticks": symbol,
                "subscribe": 1
            })
            
            # Listen for ticks
            async for response in self.api.subscribe({"ticks": symbol}):
                if 'error' in response:
                    print(f"Error: {response['error']}")
                    continue
                    
                if 'tick' in response:
                    tick = response['tick']
                    
                    # Process tick data
                    tick_data = {
                        'symbol': symbol,
                        'price': float(tick['quote']),
                        'epoch': tick['epoch'],
                        'datetime': datetime.fromtimestamp(tick['epoch']),
                        'digit': self._extract_digit(float(tick['quote']))
                    }
                    
                    # Call callback with tick data
                    if callback:
                        callback(tick_data)
                        
        except Exception as e:
            print(f"✗ Subscription error: {e}")
    
    def _extract_digit(self, price):
        """Extract last digit from price"""
        price_str = f"{price:.2f}"
        return int(price_str.replace('.', '')[-1])
    
    async def get_historical_ticks(self, symbol, count=100):
        """Get historical ticks"""
        try:
            response = await self.api.subscribe({
                "ticks_history": symbol,
                "end": "latest",
                "count": count,
                "granularity": 0  # Tick-by-tick
            })
            
            ticks = []
            async for data in self.api.subscribe({"ticks_history": symbol}):
                if 'history' in data:
                    for tick in data['history']['prices']:
                        ticks.append({
                            'price': float(tick),
                            'digit': self._extract_digit(float(tick))
                        })
                    break
            
            return ticks
            
        except Exception as e:
            print(f"✗ Historical data error: {e}")
            return []
    
    async def get_available_symbols(self):
        """Get list of available volatility indices"""
        try:
            response = await self.api.ticks_stream("R_100")  # Just to get API working
            return {
                "R_10": "Volatility 10 Index",
                "R_25": "Volatility 25 Index",
                "R_50": "Volatility 50 Index", 
                "R_75": "Volatility 75 Index",
                "R_100": "Volatility 100 Index"
            }
        except:
            # Return default if API fails
            return {
                "R_10": "Volatility 10 Index",
                "R_25": "Volatility 25 Index",
                "R_50": "Volatility 50 Index", 
                "R_75": "Volatility 75 Index",
                "R_100": "Volatility 100 Index"
            }
    
    async def disconnect(self):
        """Disconnect from API"""
        if self.api:
            await self.api.clear()
            self.connected = False
            print("✓ Disconnected from Deriv")

# Singleton instance
deriv_client = None

def get_deriv_client(token=None):
    """Get or create Deriv client instance"""
    global deriv_client
    if deriv_client is None:
        deriv_client = DerivRealTime(token=token)
    return deriv_client