import pandas as pd
from stable_baselines3 import PPO
import torch as th
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from agent.helper import get_agent_number
from environment.trading_env import TradingEnv
from environment.data_feeder import PdDataFeeder
from environment.indicators import RSI, MACD, BollingerBands, ATR
from environment.scalers import MinMaxScaler
from environment.reward import StandartDeviationReward 
from environment.metrics import DifferentActions, AccountValue, AccountValueChange, MaxDrawdown, SharpeRatio, AverageWinLossRatio, WinCount, LossCount

data_source = input("Parity name : (ex: BTCUSDT_4h)")
df = pd.read_csv(f'data/crypto/{data_source}.csv')
df = df[:-720] # leave data for testing
epoch = int(input("Enter the epoch: "))
pd_data_feeder = PdDataFeeder(df, indicators=[RSI, MACD, BollingerBands, ATR])
ratio_days = (df['date'].iloc[-1] - df['date'].iloc[0]).days

def make_env():
    return TradingEnv(
        data_feeder=pd_data_feeder,
        output_transformer=MinMaxScaler(min=pd_data_feeder.min, max=pd_data_feeder.max),
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

vec_env = make_vec_env(make_env, n_envs=4)
run_number = get_agent_number("runs/")
print(f"Run number: {run_number}")
print(f"Total days: {ratio_days}")
print(f"Start date: {df['date'].iloc[0]}")
print(f"End date: {df['date'].iloc[-1]}")

eval_callback = EvalCallback(vec_env, best_model_save_path=f"runs/{run_number}",
                            log_path=f"runs/{run_number}/", eval_freq=len(df), n_eval_episodes=1,
                            deterministic=True, render=False, verbose=1)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

model_ppo = PPO("MlpPolicy", vec_env, verbose=1, n_steps=len(df), n_epochs=epoch, learning_rate = 0.0001, batch_size=64, policy_kwargs=policy_kwargs, device='cuda')
model_ppo.learn(total_timesteps=epoch*len(df), callback=eval_callback)