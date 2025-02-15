from .state import State
import numpy as np

class Metric:
    """
    Base class for all metrics. Metrics are used to evaluate the performance of a trading strategy.
    """
    def __init__(self, name: str="metric") -> None:
        self.name = name
        self.reset()

    def update(self, state: State):
        assert isinstance(state, State), f'state must be State, received: {type(state)}'

        return state

    @property
    def result(self):
        raise NotImplementedError
    
    def reset(self, prev_state: State=None):
        assert prev_state is None or isinstance(prev_state, State), f'prev_state must be None or State, received: {type(prev_state)}'

        return prev_state
    

class DifferentActions(Metric):
    """The Different Actions metric is a measure of the number of times the agent took different actions like buy or sell."""
    def __init__(self, name: str="different_actions") -> None:
        super().__init__(name=name)

    def update(self, state: State):
        super().update(state)

        if not self.prev_state:
            self.prev_state = state
        else:
            if state.allocation_percentage != self.prev_state.allocation_percentage:
                self.different_actions += 1

            self.prev_state = state

    @property
    def result(self):
        return self.different_actions
    
    def reset(self, prev_state: State=None):
        super().reset(prev_state)

        self.prev_state = prev_state
        self.different_actions = 0


class AccountValue(Metric):
    """The Account Value is a measure of the total value of the trading account at the end of the trading period."""
    def __init__(self, name: str="account_value") -> None:
        super().__init__(name=name)

    def update(self, state: State):
        super().update(state)

        self.account_value = state.account_value

    @property
    def result(self):
        return self.account_value
    
    def reset(self, prev_state: State=None):
        super().reset(prev_state)
        
        self.account_value = prev_state.account_value if prev_state else 0.0

class AccountValueChange(Metric):
    """The Account Value Change is a measure of the percentage change in the account value relative to the initial account value."""
    def __init__(self, name: str="account_value_changement") -> None:
        super().__init__(name=name)

    def update(self, state: State):
        super().update(state)

        self.account_value = state.account_value

    @property
    def result(self):
        return ((self.account_value - 10000) / 10000) * 100
    
    def reset(self, prev_state: State=None):
        super().reset(prev_state)
        
        self.account_value = prev_state.account_value if prev_state else 0.0

class MaxDrawdown(Metric):
    """ The Maximum Drawdown (MDD) is a measure of the largest peak-to-trough decline in the 
    value of a portfolio or investment during a specific period

    The Maximum Drawdown Ratio represents the proportion of the peak value that was lost during 
    the largest decline. It is a measure of the risk associated with a particular investment or 
    portfolio. Investors and fund managers use the Maximum Drawdown and its ratio to assess the 
    historical downside risk and potential losses that could be incurred.
    """
    def __init__(self, name: str="max_drawdown") -> None:
        super().__init__(name=name)

    def update(self, state: State):
        super().update(state)

        # Use min to find the trough value
        self.max_account_value = max(self.max_account_value, state.account_value)

        # Calculate drawdown
        drawdown = (state.account_value - self.max_account_value) / self.max_account_value

        # Update max drawdown if the current drawdown is greater
        self.max_drawdown = min(self.max_drawdown, drawdown)

    @property
    def result(self):
        return self.max_drawdown
    
    def reset(self, prev_state: State=None):
        super().reset(prev_state)

        self.max_account_value = prev_state.account_value if prev_state else 0.0
        self.max_drawdown = 0.0


class SharpeRatio(Metric):
    """ The Sharpe Ratio, is a measure of the risk-adjusted performance of an investment or a portfolio. 
    It helps investors evaluate the return of an investment relative to its risk.

    A higher Sharpe Ratio indicates a better risk-adjusted performance. Investors and portfolio managers 
    often use the Sharpe Ratio to compare the risk-adjusted returns of different investments or portfolios. 
    It allows them to assess whether the additional return earned by taking on additional risk is justified.
    """
    def __init__(self, ratio_days=365.25, name: str='sharpe_ratio'):
        self.ratio_days = ratio_days
        super().__init__(name=name)

    def update(self, state: State):
        super().update(state)
        time_difference_days = (state.date - self.prev_state.date).days
        if time_difference_days >= 1:
            self.daily_returns.append((state.account_value - self.prev_state.account_value) / self.prev_state.account_value)
            self.prev_state = state
        
    @property
    def result(self):
        if len(self.daily_returns) == 0:
            return 0.0

        mean = np.mean(self.daily_returns)
        std = np.std(self.daily_returns)
        if std == 0:
            return 0.0
        
        sharpe_ratio = mean / std * np.sqrt(self.ratio_days)
        
        return sharpe_ratio
    
    def reset(self, prev_state: State=None):
        super().reset(prev_state)
        self.prev_state = prev_state
        self.daily_returns = []

class AverageWinLossRatio(Metric):
    """The Average Win/Loss Ratio is a measure of the average gain on profitable trades
    relative to the average loss on unprofitable trades. It is used to assess the risk-reward
    profile of a trading strategy. A higher Average Win/Loss Ratio indicates a better risk-reward
    profile, as the strategy generates more profit than loss on average.
    """
    def __init__(self, name: str="average_win_loss_ratio") -> None:
        super().__init__(name=name)
        self.total_wins = 0  # Toplam kazançlı işlem sayısı
        self.total_losses = 0  # Toplam zararlı işlem sayısı
        self.total_win_amount = 0  # Toplam kazanç miktarı
        self.total_loss_amount = 0  # Toplam zarar miktarı

    def update(self, state: State):
        super().update(state)
        if not self.prev_state:
            self.prev_state = state
        else:    
        # Son alım-satım işlemi sonucuna göre kazanç veya zararın belirlenmesi
            if state.account_value > self.prev_state.account_value:
                self.total_wins += 1
                self.total_win_amount += state.account_value - self.prev_state.account_value
            elif state.account_value < self.prev_state.account_value:
                self.total_losses += 1
                self.total_loss_amount += self.prev_state.account_value - state.account_value
            self.prev_state = state    

    @property
    def result(self):
        # Zararlı işlem sayısı sıfırdan büyükse ve ortalama zarar sıfırdan büyükse
        if self.total_losses > 0 and self.total_loss_amount > 0:
            average_win_loss_ratio = self.total_wins / self.total_losses
        else:
            average_win_loss_ratio = 0.0

        return average_win_loss_ratio

    def reset(self, prev_state: State=None):
        super().reset(prev_state)
        self.prev_state = prev_state
        self.total_wins = 0
        self.total_losses = 0
        self.total_win_amount = 0
        self.total_loss_amount = 0

class WinCount(Metric):
    """The Win Rate is a measure of the percentage of profitable trades relative to the total number of trades.
    It is used to assess the success rate of a trading strategy. A higher Win Rate indicates a higher percentage
    of profitable trades, while a lower Win Rate indicates a lower percentage of profitable trades.
    """
    def __init__(self, name: str="win_count") -> None:
        super().__init__(name=name)
        self.total_trade = 0 # Toplam işlem sayısı
        self.total_wins = 0  # Toplam kazançlı işlem sayısı
        self.total_win_amount = 0  # Toplam kazanç miktarı

    def update(self, state: State):
        super().update(state)
        if not self.prev_state:
            self.prev_state = state
        else:    
        # Son alım-satım işlemi sonucuna göre kazanç veya zararın belirlenmesi
            if state.account_value > self.prev_state.account_value:
                self.total_wins += 1
                self.total_win_amount += state.account_value - self.prev_state.account_value
            self.total_trade += 1
            self.prev_state = state    

    @property
    def result(self):
        """if self.total_trade > 0:
            win_rate = self.total_wins * 100 / self.total_trade
        else:
            win_rate = 0.0"""

        return self.total_wins

    def reset(self, prev_state: State=None):
        super().reset(prev_state)
        self.prev_state = prev_state
        self.total_trade = 0
        self.total_wins = 0
        self.total_win_amount = 0

class LossCount(Metric):
    """The Loss Rate is a measure of the percentage of unprofitable trades relative to the total number of trades.
    It is used to assess the failure rate of a trading strategy. A higher Loss Rate indicates a higher percentage
    of unprofitable trades, while a lower Loss Rate indicates a lower percentage of unprofitable trades.
    """
    def __init__(self, name: str="loss_count") -> None:
        super().__init__(name=name)
        self.total_trade = 0 # Toplam işlem sayısı
        self.total_losses = 0  # Toplam zarar eden işlem sayısı
        self.total_loss_amount = 0  # Toplam zarar miktarı

    def update(self, state: State):
        super().update(state)
        if not self.prev_state:
            self.prev_state = state
        else:    
        # Son alım-satım işlemi sonucuna göre kazanç veya zararın belirlenmesi
            if state.account_value < self.prev_state.account_value:
                self.total_losses += 1
                self.total_loss_amount += self.prev_state.account_value - state.account_value
            self.total_trade += 1
            self.prev_state = state

    @property
    def result(self):
        """if self.total_trade > 0:
            loss_rate = self.total_losses * 100 / self.total_trade
        else:
            loss_rate = 0.0
        """
        return self.total_losses

    def reset(self, prev_state: State=None):
        super().reset(prev_state)
        self.prev_state = prev_state
        self.total_trade = 0
        self.total_losses = 0
        self.total_loss_amount = 0