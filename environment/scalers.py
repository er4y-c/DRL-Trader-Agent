import numpy as np
from .state import Observations

class MinMaxScaler:
    """
    This class normalizes the state data between 0 and 1
    """
    def __init__(self, min: float, max: float):
        self._min = min
        self._max = max
    
    def transform(self, observations: Observations) -> np.ndarray:

        assert isinstance(observations, Observations) == True, "observations must be an instance of Observations"

        transformed_data = []
        for state in observations:
            open = (state.open - self._min) / (self._max - self._min)
            high = (state.high - self._min) / (self._max - self._min)
            low = (state.low - self._min) / (self._max - self._min)
            close = (state.close - self._min) / (self._max - self._min)
            volume = (state.volume - self._min) / (self._max - self._min)
            rsi = (state.rsi - self._min) / (self._max - self._min)
            macd = (state.macd - self._min) / (self._max - self._min)
            signal = (state.signal - self._min) / (self._max - self._min)
            ma = (state.ma - self._min) / (self._max - self._min)
            bb_upper = (state.bb_upper - self._min) / (self._max - self._min)
            bb_lower = (state.bb_lower - self._min) / (self._max - self._min)
            atr = (state.atr - self._min) / (self._max - self._min)
            short_ema = (state.short_ema - self._min) / (self._max - self._min)
            long_ema = (state.long_ema - self._min) / (self._max - self._min)
            session = state.session
            transformed_data.append([open, high, low, close, volume, rsi, macd, signal, ma, bb_upper, bb_lower, atr, short_ema, long_ema, session, state.allocation_percentage])

        return np.array(transformed_data)
    
    def __call__(self, observations) -> np.ndarray:
        return self.transform(observations)