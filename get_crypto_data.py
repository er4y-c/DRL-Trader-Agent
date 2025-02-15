from binance.client import Client
from dynaconf import settings
import csv
from datetime import datetime, timezone

client = Client(settings.BINANCE_API_KEY, settings.BINANCE_SECRET_KEY)
klines = [
    ["date", "open", "high", "low", "close", "volume"],
]
param = input("Parite giriniz (örn: BTCUSDT): ")
start_date = input("Başlangıç tarihini giriniz (örn: 1 Mar, 2024): ")
end_date = input("Bitiş tarihini giriniz (örn: 1 Apr, 2024): ")

hist_data = client.get_historical_klines(param, Client.KLINE_INTERVAL_4HOUR, start_date, end_date)

def format_timestamp(timestamp):
    timestamp_seconds = timestamp / 1000
    return datetime.fromtimestamp(timestamp_seconds, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

for i in hist_data:
    formatted_date = format_timestamp(i[0])
    klines.append([formatted_date, i[1], i[2], i[3], i[4], i[5]])

with open(f'data/crypto/{param}_4h.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(klines)

print(f"Dosya adı: {param}_4h.csv", f"Path: data/crypto/{param}_4h.csv")
