from typing import Generator
import pandas as pd
from environment.state import State

class PdDataFeeder:
    """
    PdDataFeeder class gets a Pandas Dataframe and calculates the states for the feeding environment.
    """
    def __init__(
            self, 
            df: pd.DataFrame,
            min: float = None,
            max: float = None,
            indicators: list = [],
            ) -> None:
        self._min = min
        self._max = max
        self._indicators = indicators
        self._df = self.add_indicator(df)

        assert isinstance(self._df, pd.DataFrame) == True, "df must be a pandas.DataFrame"
        assert 'date' in self._df.columns, "df must have 'date' column"
        assert 'open' in self._df.columns, "df must have 'open' column"
        assert 'high' in self._df.columns, "df must have 'high' column"
        assert 'low' in self._df.columns, "df must have 'low' column"
        assert 'close' in self._df.columns, "df must have 'close' column"

    @property
    def min(self) -> float:
        return self._min or self._df['low'].min()
    
    @property
    def max(self) -> float:
        return self._max or self._df['high'].max()

    def add_indicator(self, df, **kwargs) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'])
        for indicator_cls in self._indicators:
            indicator = indicator_cls(df, **kwargs)
            df = indicator.calculate()
        df.dropna(inplace=True)
        return df

    def __len__(self) -> int:
        return len(self._df)
    
    def __getitem__(self, idx: int, args=None) -> State:
        data = self._df.iloc[idx]

        state = State(
            date=data['date'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data.get('volume', 0.0),
            ma=data['ma'],
            bb_upper=data['bb_upper'],
            bb_lower=data['bb_lower'],
            atr=data['atr'],
            short_ema=data['short_ema'],
            long_ema=data['long_ema'],
            rsi=data['rsi'],
            macd=data['macd'],
            signal=data['signal'],
        )

        return state
    
    def __iter__(self) -> Generator[State, None, None]:
        """ Create a generator that iterate over the Sequence."""
        for index in range(len(self)):
            yield self[index]