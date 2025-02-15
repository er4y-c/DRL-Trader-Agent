from .state import Observations
import numpy as np

class Reward:
    def __init__(self) -> None:
        pass

    @property
    def __name__(self) -> str:
        return self.__class__.__name__
    
    def __call__(self, observations: Observations) -> float:
        raise NotImplementedError
    
    def reset(self, observations: Observations):
        pass
    
class AccountValueChangeReward(Reward):
    def __init__(self) -> None:
        super().__init__()
        self.ratio_days=365.25

    def reset(self, observations: Observations):
        super().reset(observations)
        self.returns = []
    
    def __call__(self, observations: Observations) -> float:
        assert isinstance(observations, Observations) == True, "observations must be an instance of Observations"

        last_state, next_state = observations[-2:]
        reward = (next_state.account_value - last_state.account_value) / last_state.account_value

        return reward
    
class StandartDeviationReward(Reward):
    def __init__(self, sigma_tgt=0.2, bp=0.0001, mu=1) -> None:
        super().__init__()
        self.sigma_tgt = sigma_tgt
        self.bp = bp
        self.mu = mu
        self.sigma_estimate = []
      
    def reset(self, observations: Observations):
        super().reset(observations)
        self.sigma_estimate = []

    def _calculate_sigma(self, rt):
        # Calculate exponentially weighted moving standard deviation with a 60-day window
        if len(self.sigma_estimate) == 0:
            sigma_t_minus_1 = np.std(rt)  # Initial estimate
        else:
            sigma_t_minus_1 = np.sqrt(0.9 * self.sigma_estimate[-1]**2 + 0.1 * np.std(rt)**2)  # Exponentially weighted moving standard deviation
        self.sigma_estimate.append(sigma_t_minus_1)
        return sigma_t_minus_1

    def __call__(self, observations: Observations) -> float:
        assert isinstance(observations, Observations) == True, "observations must be an instance of Observations"

        # Calculate additive profit (rt)
        rt = np.diff([state.close for state in observations])

        # Calculate current, previous and two steps ago standard deviations
        sigma_t_minus_1 = self._calculate_sigma(rt)
        if len(self.sigma_estimate) >= 2:
            sigma_t_minus_2 = self.sigma_estimate[-2]
        else:
            sigma_t_minus_2 = sigma_t_minus_1

        # Calculate volatility scaling
        At_minus_1 = observations[-1].account_value
        At_minus_2 = observations[-2].account_value if len(observations) >= 2 else 0
        volatility_scaling = (self.sigma_tgt / sigma_t_minus_1) * (At_minus_1 / At_minus_2)

        # Calculate transaction cost
        pt_minus_1 = observations[-2].close
        transaction_cost = self.bp * pt_minus_1

        # Calculate reward
        reward = self.mu * volatility_scaling * (rt - transaction_cost)
        avg_reward = np.mean(reward)
        return float(avg_reward)