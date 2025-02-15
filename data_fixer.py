import pandas as pd
import os

def bist_fixer(folder_path):
    file_list = os.listdir(folder_path)
    
    for file_name in file_list:
        if file_name.endswith('.csv'):
            try:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                df.rename(columns={'Tarih': 'date',
                                   'Açılış': 'open',
                                   'Yüksek': 'high',
                                   'Düşük': 'low',
                                   'Kapanış': 'close',
                                   'Hacim': 'volume'}, inplace=True)
                
                df.drop(columns=['Ağırlıklı Ortalama', 'Miktar'], inplace=True, errors='ignore')
                df['date'] = pd.to_datetime(df['date'], dayfirst=True)
                df.sort_values(by='date', inplace=True)
                new_file_path = os.path.join(folder_path, file_name)
                df.to_csv(new_file_path, index=False)
                
                print(f"{file_name} dosyası başarıyla işlendi. Güncellenmiş dosya: {file_name}")
            
            except Exception as e:
                print(f"Hata: {e}. {file_name} dosyası işlenemedi. Bir sonraki dosyaya geçiliyor.")
                continue

def crypto_fixer(folder_path):
    file_list = os.listdir(folder_path)
    
    for file_name in file_list:
        if file_name.endswith('.csv'):
            try:
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                df.rename(columns={'Date': 'date',
                                   'Open': 'open',
                                   'High': 'high',
                                   'Low': 'low',
                                   'Close': 'close',
                                   'Volume USD': 'volume'}, inplace=True)
                df.drop(columns=['Volume XRP', 'unix', 'symbol'], inplace=True, errors='ignore')
                df['date'] = pd.to_datetime(df['date'])
                df.sort_values(by='date', inplace=True)
                new_file_path = os.path.join(folder_path, file_name)
                df.to_csv(new_file_path, index=False)
                
                print(f"{file_name} dosyası başarıyla işlendi. Güncellenmiş dosya: {file_name}")
            
            except Exception as e:
                print(f"Hata: {e}. {file_name} dosyası işlenemedi. Bir sonraki dosyaya geçiliyor.")
                continue

folder_path = 'data/crypto'

crypto_fixer(folder_path)
