import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.trading_env import TradingEnv
from environment.data_feeder import PdDataFeeder
from environment.render import PygameRender
from environment.indicators import RSI, MACD, BollingerBands, ATR
from environment.scalers import MinMaxScaler
from environment.reward import AccountValueChangeReward, StandartDeviationReward
from environment.metrics import DifferentActions, AccountValue, SharpeRatio, AccountValueChange, MaxDrawdown, AverageWinLossRatio, WinCount, LossCount
from agent.helper import changement_calculator

pd.options.mode.copy_on_write = True

data_source = input("Parity name : (ex: BTCUSDT_4h)")
df_test = pd.read_csv(f'data/crypto/{data_source}.csv')

agent_number = input('Enter the agent number: ')
start_date = input("Enter the start date (YYYY-MM-DD): ")
start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")

end_date = input("Enter the end date (YYYY-MM-DD): ")
end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

ratio_days = (end_date_dt - start_date_dt).days

df = df_test[(df_test['date'] >= start_date) & (df_test['date'] <= end_date)]

print("Total days:", ratio_days)

print("Start date:", df['date'].iloc[0])
print("Start date price (close)", df['close'].iloc[0])

print("End date:", df['date'].iloc[-1])
print("End date price (close)", df['close'].iloc[-1])

changement_per = changement_calculator(df['close'].iloc[0], df['close'].iloc[-1])
print("Percentage changement: ", changement_per )

pd_data_feeder_test = PdDataFeeder(df, indicators=[RSI, MACD, BollingerBands, ATR])

env = TradingEnv(
    data_feeder=pd_data_feeder_test,
    output_transformer=MinMaxScaler(min=pd_data_feeder_test.min, max=pd_data_feeder_test.max),
    initial_balance=10000.0,
    max_episode_steps=len(df),
    window_size=50,
    reward_function=StandartDeviationReward(),
    metrics=[
        DifferentActions(),
        AccountValue(),
        AccountValueChange(),
        MaxDrawdown(),
        SharpeRatio(ratio_days=ratio_days),
        AverageWinLossRatio(),
        WinCount(),
        LossCount()
    ]
)

vec_env = DummyVecEnv([lambda: env])
pygameRender = PygameRender(frame_rate=120)

model = PPO.load(f"runs/{agent_number}/best_model")
actor_params = sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters())
print(f"Number of parameters in the actor network: {actor_params}")

obs, info = env.reset()
done = False
totalReward = 0.0
pygameRender.render(info)
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    pygameRender.render(info)
    totalReward += reward
    
    if terminated or truncated:
        for metric, value in info['metrics'].items():
            print(metric, value)
        state, info = env.reset()
        pygameRender.render(info)
        pygameRender.reset()
        break
