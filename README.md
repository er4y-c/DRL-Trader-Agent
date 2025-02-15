# DRL Trader Agent
This project was developed in 2023.

## Description
This project aims to develop a simple PPO agent on a trading environment developed with Gymnasium and Pygame. In addition, this trading environment provides a rule-based environment for users to backtest their own strategies.

## Kullanılan Teknolojiler

- Python 3.12+
- [stable-baselines3](https://stable-baselines3.readthedocs.io/)
- [gymnasium](https://gymnasium.farama.org/)
- [pandas](https://pandas.pydata.org/)
- [pytorch](https://pytorch.org/)
- Binance API

## Installation

1. Clone this repo.
  ```command
    git clone https://github.com/er4y-c/DRL-Trader-Agent.git
  ```

2. Create and activate a virtual environment.

3. Install the dependencies

  ```command
    poetry install
  ```

4. Create a Dynaconf settings file (settings.toml) and add your Binance API and Secret keys.

## Data Sources

In the project, you can use the price data of any trading asset that offers OHLC data. Just make sure that the data set you will use has `date`, `open`, `high`, `low`, `close`, `volume` columns.

You can also use get_crypto_data.py to retrieve historical data from Binance for any parameter (e.g. `BTC_USDT_4h`). You can use the `crypto_fixer` helper function in `data_fixer.py` to make the data compatible with the trading environment. This file also has a helper function called `bist_fixer` which is needed to format the `BIST100` data into the proper format.

## Environment

Trading Environment basically consists of 7 components.

- **DataFeeder**
  DataFeeder is used to calculate and integrate `indicator data` into the `OHLC data` that you give to the environment as a data set and to translate this data into `States` that our `PPO agent` can process. 

- **Indicators**
  It includes helper classes to calculate and add to states the indicators that people use when trading. It includes auxiliary indicators such as `RSI`, `MACD`, `Bollinger Bands`, `ATR`, `LondonAsiaSession`.

- **Metrics**
  It contains financial success metrics that can help us measure the success of our PPO agent in train and test. There is a basic Metric class and all metrics in the system are inherited from this class. This way you can create your own metrics. The various metrics available in the system are:

  - `DifferentActions`: The Different Actions metric is a measure of the number of times the agent took different actions like buy or sell.

  - `AccountValue`: The Account Value is a measure of the total value of the trading account at the end of the trading period.

  - `AccountValueChange`: The Account Value Change is a measure of the percentage change in the account value relative to the initial account value.

  - `MaxDrawdown`: The Maximum Drawdown (MDD) is a measure of the largest peak-to-trough decline in the 
    value of a portfolio or investment during a specific period

    The Maximum Drawdown Ratio represents the proportion of the peak value that was lost during 
    the largest decline. It is a measure of the risk associated with a particular investment or 
    portfolio. Investors and fund managers use the Maximum Drawdown and its ratio to assess the 
    historical downside risk and potential losses that could be incurred.

  - `SharpeRatio`: The Sharpe Ratio, is a measure of the risk-adjusted performance of an investment or a portfolio. 
    It helps investors evaluate the return of an investment relative to its risk.

    A higher Sharpe Ratio indicates a better risk-adjusted performance. Investors and portfolio managers 
    often use the Sharpe Ratio to compare the risk-adjusted returns of different investments or portfolios. 
    It allows them to assess whether the additional return earned by taking on additional risk is justified.
  
  - `AverageWinLossRatio`: The Average Win/Loss Ratio is a measure of the average gain on profitable trades
    relative to the average loss on unprofitable trades. It is used to assess the risk-reward
    profile of a trading strategy. A higher Average Win/Loss Ratio indicates a better risk-reward
    profile, as the strategy generates more profit than loss on average.

- **Render**
  Using `Pygame` library, it allows to plot OHLC data in candlestick format and show agent actions on this chart.

- **Reward**
  Reward mechanism is one of the most important factors affecting the success of an RL agent. There are two reward functions in the system: `AccountValueChangeReward` and `StandardDeviationReward`.

  - `AccountValueChangeReward` calculates rewards and penalties based on the remaining balance in the account as a result of all actions taken by the agent.

  - `StandardDeviationReward` is a volatility-based reward mechanism based on the standard deviation strategy in finance.

    1. Calculates an rt value using bar closing prices in States

    2. Calculates the current, previous and two steps previous `Exponential Weighted Moving Standard Deviation (EWMSD)`:

      ![EWMSD](https://latex.codecogs.com/png.latex?\sigma_t=\sqrt{0.9\cdot\sigma_{t-1}^2+0.1\cdot\text{std}(r_t)^2})
    
    3. Apply account_value based volatility scaling :

      ![Volatility Scaling](https://latex.codecogs.com/png.latex?\frac{\sigma_{\text{tgt}}}{\sigma_{t-1}}\times\frac{A_{t-1}}{A_{t-2}})

    4. Calculate transaction cost:

      ![Transaction Cost](https://latex.codecogs.com/png.latex?\text{transaction\_cost}=\text{bp}\times\text{latest\_close\_price})
    
    5. Calculate reward:

      ![Reward Formula](https://latex.codecogs.com/png.latex?\text{reward}=\mu\times\text{volatility\_scaling}\times(r_t-\text{transaction\_cost}))

- **Scaler**
  This component normalizes the state values between 0 and 1 by applying `min max scaling`.
  
- **State & Observation**
  `States` and `Observations `are classes that represent scaled versions of my OHLC, indicators and account data.

## Training

You can customize the environment according to your needs by using Trading Environment's instruments such as metrics, indicators, rewards, etc. Run train.py to train a `PPO` agent that implements `MlpPolicy` with the OHLC data you provide as input. The system will ask you for the name of the data set you want to use (it will look for it in the data folder in the main directory) and the number of epochs. The last 720 rows of data in the dataset will be reserved for testing. The model performs best on 4 hours of OHLC data. The trained model will be stored in the `runs folder` in the main directory.

## Testing

The agents you train are stored under the runs folder. To test a trained agent with any data set, run test.py. The system will ask you for the name of the OHLC data you want to train (it will look for it in the data folder) and the date range you want to train. While testing an agent, you can see the actions taken by the agent and the price ranges in which the agent performs these actions on the chart rendered with `Pygame`.

## Rule Based Backtest

You can also backtest your own trading strategies in the Trading Environment. You can see an example of this in the rule_based.py file. In this file you can see a sample implementation of the London Breakout Strategy, which is a strategy to trade using the differences between the London and Asian stock market sessions.