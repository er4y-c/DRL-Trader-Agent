import pandas as pd
from datetime import datetime

from environment.data_feeder import PdDataFeeder
from environment.trading_env import TradingEnv
from environment.render import PygameRender
from environment.scalers import MinMaxScaler
from environment.reward import AccountValueChangeReward
from environment.metrics import DifferentActions, AccountValue, SharpeRatio, MaxDrawdown, AverageWinLossRatio
from environment.indicators import RSI, MACD, BollingerBands, ATR, LondonAsiaSession
from environment.strategies import SupportResistanceDetector


df = pd.read_csv('data/fiat/EURUSD5.csv')

start_date = input("Enter the start date (YYYY-MM-DD): ")
start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")

end_date = input("Enter the end date (YYYY-MM-DD): ")
end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

ratio_days = (end_date_dt - start_date_dt).days

df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
print("Total days:", ratio_days)
print("Start date:", df['date'].iloc[0])
print("End date:", df['date'].iloc[-1])

pd_data_feeder = PdDataFeeder(df, indicators=[RSI, MACD, BollingerBands, ATR, LondonAsiaSession])

env = TradingEnv(
    data_feeder = pd_data_feeder,
    output_transformer = MinMaxScaler(min=pd_data_feeder.min, max=pd_data_feeder.max),
    initial_balance = 1000.0,
    max_episode_steps = len(df),
    window_size = 2,
    reward_function = AccountValueChangeReward(),
    metrics = [
        DifferentActions(),
        AccountValue(),
        SharpeRatio(ratio_days=ratio_days),
        MaxDrawdown(),
    ]
)

pygameRender = PygameRender(frame_rate=120)

state, info = env.reset()
pygameRender.render(info)
rewards = 0.0
detector = SupportResistanceDetector()

while True:
    action = detector.detect(state)
    if action == 2:
        detector.reset()
    
    state, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    pygameRender.render(info)

    if terminated or truncated:

        for metric, value in info['metrics'].items():
            print(metric, value)
        hours = 0
        state, info = env.reset()
        rewards = 0.0
        pygameRender.reset()
        pygameRender.render(info)
        break