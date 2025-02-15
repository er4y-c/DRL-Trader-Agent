import pandas as pd
import numpy as np

class RSI:
    def __init__(self, data, window=14):
        self.data = data
        self.window = window
    
    def calculate(self):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        RS = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + RS))
        return self.data

class MACD:
    def __init__(self, data, short_window=12, long_window=26, signal_window=9):
        self.data = data
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
    
    def calculate(self):
        self.data['short_ema'] = self.data['close'].ewm(span=self.short_window, min_periods=1).mean()
        self.data['long_ema'] = self.data['close'].ewm(span=self.long_window, min_periods=1).mean()
        self.data['macd'] = self.data['short_ema'] - self.data['long_ema']
        self.data['signal'] = self.data['macd'].ewm(span=self.signal_window, min_periods=1).mean()
        return self.data

class BollingerBands:
    def __init__(self, data, window=20, num_std=2):
        self.data = data
        self.window = window
        self.num_std = num_std
    
    def calculate(self):
        self.data['ma'] = self.data['close'].rolling(window=self.window).mean()
        self.data['bb_upper'] = self.data['ma'] + 2 * self.data['close'].rolling(window=self.window).std()
        self.data['bb_lower'] = self.data['ma'] - 2 * self.data['close'].rolling(window=self.window).std()
        return self.data

class ATR:
    def __init__(self, data, window=14):
        self.data = data
        self.window = window
    
    def calculate(self):
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.data['atr'] = true_range.rolling(window=self.window).mean()
        return self.data

class LondonAsiaSession:
    def __init__(self, data, window=14):
        self.data = data
        self.window = window
        self.data['session'] = 0
    
    def is_weekend(self, date):
        return date.dayofweek >= 5

    def calculate(self):
        for index, row in self.data.iterrows():
            if not self.is_weekend(row['date']):
                if 2 <= row['date'].hour < 5:
                    self.data.at[index, 'session'] = 2 # London open session
                elif 20 <= row['date'].hour < 24 or 0 <= row['date'].hour < 2:
                    self.data.at[index, 'session'] = 1 # Asia session


        return self.data