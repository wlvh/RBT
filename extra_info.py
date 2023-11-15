#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2023-10-24 13:54:38
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2023-10-26 08:29:37
FilePath: /trading/extra_info.py
Description: 
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''
import requests
import json
from datetime import datetime, timedelta
import math
import statistics
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def fetch_historical_data_from_cryptocompare(api_key, fsym, limit=30, toTs=None, extraParams=None, sign=False):
    api_url = "https://min-api.cryptocompare.com/data/blockchain/histo/day"
    headers = {'authorization': f'Apikey {api_key}'}
    params = {
        'fsym': fsym,
        'limit': limit,
        'toTs': toTs,
    }
    # 删除字典中的 None 值，以保证 API 调用的成功
    params = {k: v for k, v in params.items() if v is not None}
    
    response = requests.get(api_url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = json.loads(response.text)
        return data
    else:
        return f"Failed to get data: {response.status_code}"
    
def get_block_ratiodate(api_key='564bb9f7896bec544f6d04889f1479ff3471e55abe78c4fb741033ac114e4c6e', fsym="BTC", limit=700, update=False):
    if update == False:
        print("Fetching new historical data...")
        historical_data = fetch_historical_data_from_cryptocompare(api_key, fsym, limit)
    else:
        print("Updating historical data...")
        df_chain_ratio_old = pd.read_csv('df_chain_ratio.csv', index_col='date')
        print(f"Length of old df_chain_ratio: {len(df_chain_ratio_old)}, Start date: {df_chain_ratio_old.index.min()}, End date: {df_chain_ratio_old.index.max()}")
        
        last_date_str = df_chain_ratio_old.index[-1]
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d %H:%M:%S')
        current_date = datetime.utcnow()
        x = (current_date - last_date).days
        print(f"Today is {current_date}, last date in df_chain_ratio is {last_date}, {x} days have passed since last update.")
        historical_data = fetch_historical_data_from_cryptocompare(api_key, fsym, limit=x+10)

    data_list = []
    for i in range(len(historical_data['Data']['Data'])):
        human_readable_date = datetime.utcfromtimestamp(historical_data['Data']['Data'][i]['time']).strftime('%Y-%m-%d %H:%M:%S')
        historical_data['Data']['Data'][i]['time_human_readable'] = human_readable_date
        data_list.append(historical_data['Data']['Data'][i])

    selected_columns = ['zero_balance_addresses_all_time', 'unique_addresses_all_time', 
                        'new_addresses', 'active_addresses', 'transaction_count', 'transaction_count_all_time', 
                        'large_transaction_count', 'average_transaction_value', 'block_height', 'hashrate', 
                        'difficulty', 'block_time', 'block_size', 'current_supply']

    data_list  = pd.DataFrame(data_list)
    df_chain_ratio_new = data_list[selected_columns].pct_change()       
    df_chain_ratio_new.fillna(0, inplace=True)
    df_chain_ratio_new.replace([np.inf, -np.inf], 0, inplace=True)
    df_chain_ratio_new['date'] = pd.to_datetime(data_list['time'], unit='s')
    df_chain_ratio_new.set_index('date', inplace=True)
    print(f"Length of new df_chain_ratio: {len(df_chain_ratio_new)}, Start date: {df_chain_ratio_new.index.min()}, End date: {df_chain_ratio_new.index.max()}")

    if update:
        # Merge new and old data, and remove duplicates
        df_chain_ratio = pd.concat([df_chain_ratio_old, df_chain_ratio_new])
        df_chain_ratio = df_chain_ratio.loc[~df_chain_ratio.index.duplicated(keep='last')]
    else:
        df_chain_ratio = df_chain_ratio_new
    # Convert Index to DatetimeIndex
    df_chain_ratio.index = pd.to_datetime(df_chain_ratio.index)
    # Format the DatetimeIndex
    df_chain_ratio.index = df_chain_ratio.index.strftime('%Y-%m-%d %H:%M:%S')

    print("Last 5 rows of df_chain_ratio after dropping NA values:")
    print(df_chain_ratio.dropna().tail(5)) 
    df_chain_ratio.to_csv('df_chain_ratio.csv', index=True)
    
def process_BTC_data(file_path='BTCUSDT_1m.csv'):
    # 读取CSV文件到DataFrame
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    
    # 将分钟级别的数据转换为每日数据
    df_daily = df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    
    # 计算过去30天的总价格涨跌幅
    df_daily['30_day_total_price_return'] = (df_daily['close'].pct_change(periods=30))   # in percentage
    
    # 计算过去30天的总交易额涨跌幅
    df_daily['30_day_total_volume_return'] = (df_daily['volume'].pct_change(periods=30))   # in percentage
    
    # 计算过去30天价格涨跌幅的波动率（标准差）
    df_daily['30_day_price_volatility'] = df_daily['close'].pct_change().rolling(window=30).std() * np.sqrt(30)
    
    # 计算过去30天交易额涨跌幅的波动率（标准差）
    df_daily['30_day_volume_volatility'] = df_daily['volume'].pct_change().rolling(window=30).std() * np.sqrt(30)
    # 输出 df_daily.dropna() 的最后五行信息
    print("Last 5 rows of df_daily after dropping NA values:")
    print(df_daily.dropna().tail(5))    
    return df_daily.dropna() 

def PCA_BlockAndBitcoin():
    df_daily = process_BTC_data(file_path='BTCUSDT_1m.csv')
    df_chain_ratio = pd.read_csv('df_chain_ratio.csv', index_col='date')
    
    df_daily.index = pd.to_datetime(df_daily.index).normalize()
    df_chain_ratio.index = pd.to_datetime(df_chain_ratio.index).normalize()
    
    # 输出 df_daily 和 df_chain_ratio 的长度以及最初和最后一天的日期
    print(f"Length of df_daily: {len(df_daily)}, Start date: {df_daily.index.min()}, End date: {df_daily.index.max()}")
    print(f"Length of df_chain_ratio: {len(df_chain_ratio)}, Start date: {df_chain_ratio.index.min()}, End date: {df_chain_ratio.index.max()}")
    if df_chain_ratio.index.max() != df_daily.index.max():
        print("The last date in df_chain_ratio is not the same as the last date in df_daily. We will drop the nan value after merge.")
    merged_df = pd.merge(df_chain_ratio, df_daily, left_index=True, right_index=True, how='outer')
    merged_df.dropna(inplace=True)
    
    merged_df = merged_df[['zero_balance_addresses_all_time', 'unique_addresses_all_time',
           'new_addresses', 'active_addresses', 'transaction_count',
           'transaction_count_all_time', 'large_transaction_count',
           'average_transaction_value', 'block_height', 'hashrate', 'difficulty',
           'block_time', 'block_size', 'current_supply', '30_day_total_price_return',
           '30_day_total_volume_return', '30_day_price_volatility',
           '30_day_volume_volatility']]
    # 输出 merged_df 的长度以及最初和最后一天的日期
    print(f"Length of merged_df: {len(merged_df)}, Start date: {merged_df.index.min()}, End date: {merged_df.index.max()}")

    scaler = StandardScaler()
    roll_window = 30
    pca = PCA(n_components=len(merged_df.columns))
    # Initialize an empty DataFrame to store PCA results
    df_pca_results = []  
    # Perform rolling window PCA
    for start in range(0, len(merged_df) - roll_window):
        end = start + roll_window
        df_window = merged_df.iloc[start:end, :]
        # Standardizing the window
        df_window = scaler.fit_transform(df_window)
        # Apply PCA on the window
        pca_result = pca.fit_transform(df_window)
        # Calculate the mean for each principal component
        mean_pca_values = np.mean(pca_result, axis=0)
        z_score_scaler = StandardScaler()
        z_score_normalized_mean_pca_values = z_score_scaler.fit_transform(mean_pca_values.reshape(-1, 1)).flatten()
        explained_var = pca.explained_variance_ratio_
        # Prepare a dictionary to hold this window's results
        pca_window_result = {'date': merged_df.index[end-1]}
        for i, (norm_value, var) in enumerate(zip(z_score_normalized_mean_pca_values, explained_var)):
            pca_window_result[f'z_score_normalized_mean_PC{i+1}'] = norm_value
            pca_window_result[f'explained_var_PC{i+1}'] = var
        # Append to new_rows list
        df_pca_results.append(pca_window_result)
    df_pca_results = pd.DataFrame(df_pca_results)
    print("Last 5 rows of df_pca_results:")
    print(df_pca_results.tail(5))
    df_pca_results.set_index('date', inplace=True)
    df_pca_results.index = df_pca_results.index.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Our PCA results is from {df_pca_results.index[1]} to {df_pca_results.index[-1]}")
    df_pca_results.to_csv('df_pca_results.csv', index=True)   
    
if __name__ == "__main__":
    #get_block_ratiodate(update=True)
    PCA_BlockAndBitcoin()