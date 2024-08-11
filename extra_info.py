#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2023-10-24 13:54:38
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-08-04 06:33:05
FilePath: /trading/extra_info.py
Description: 
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''
import requests
import json
from datetime import datetime, timedelta, timezone
import math
import os
import statistics
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf

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
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        #current_date = datetime.utcnow()
        current_date = datetime.now(timezone.utc)
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
    # 计算过去30天价格涨跌幅的峰度
    df_daily['30_day_price_kurtosis'] = df_daily['close'].pct_change().rolling(window=30).kurt()
    # 计算过去30天交易额涨跌幅的峰度
    df_daily['30_day_volume_kurtosis'] = df_daily['volume'].pct_change().rolling(window=30).kurt()
    # 计算过去30天价格涨跌幅的偏度
    df_daily['30_day_price_skewness'] = df_daily['close'].pct_change().rolling(window=30).skew()
    # 计算过去30天交易额涨跌幅的偏度
    df_daily['30_day_volume_skewness'] = df_daily['volume'].pct_change().rolling(window=30).skew()
    # 输出 df_daily.dropna() 的最后五行信息
    print("Last 5 rows of df_daily after dropping NA values:")
    print(df_daily.dropna().tail(5))    
    return df_daily.dropna() 

def PCA_BlockAndBitcoin(update_mode=False):
    df_daily = process_BTC_data(file_path='BTCUSDT_1m.csv')
    df_chain_ratio = pd.read_csv('df_chain_ratio.csv', index_col='date')
    # 删除 df_daily 中的重复行
    df_daily = df_daily[~df_daily.index.duplicated(keep='first')]

    # 删除 df_chain_ratio 中的重复行
    df_chain_ratio = df_chain_ratio[~df_chain_ratio.index.duplicated(keep='first')]

    df_daily.index = pd.to_datetime(df_daily.index).normalize()
    df_chain_ratio.index = pd.to_datetime(df_chain_ratio.index).normalize()
    
    # 输出 df_daily 和 df_chain_ratio 的长度以及最初和最后一天的日期
    print(f"Length of df_daily: {len(df_daily)}, Start date: {df_daily.index.min()}, End date: {df_daily.index.max()}")
    print(f"Length of df_chain_ratio: {len(df_chain_ratio)}, Start date: {df_chain_ratio.index.min()}, End date: {df_chain_ratio.index.max()}")
    if df_chain_ratio.index.max() != df_daily.index.max():
        print("The last date in df_chain_ratio is not the same as the last date in df_daily. We will drop the nan value after merge.")
    merged_df = pd.merge(df_chain_ratio, df_daily, left_index=True, right_index=True, how='outer')
    merged_df.dropna(inplace=True)
    # 如果索引不是唯一的，找出重复的索引
    if not merged_df.index.is_unique:
        print("Duplicate indices found:")
        print(merged_df.index[merged_df.index.duplicated(keep=False)])
        print(f"{merged_df.tail(5)}")
        return
        
    merged_df = merged_df[['zero_balance_addresses_all_time', 'unique_addresses_all_time',
           'new_addresses', 'active_addresses', 'transaction_count',
           'transaction_count_all_time', 'large_transaction_count',
           'average_transaction_value', 'block_height', 'hashrate', 'difficulty',
           'block_time', 'block_size', 'current_supply', '30_day_total_price_return',
           '30_day_total_volume_return', '30_day_price_volatility','30_day_volume_volatility',
           '30_day_price_kurtosis', '30_day_volume_kurtosis', '30_day_price_skewness','30_day_volume_skewness']]
    # 输出 merged_df 的长度以及最初和最后一天的日期
    print(f"Length of merged_df: {len(merged_df)}, Start date: {merged_df.index.min()}, End date: {merged_df.index.max()}")

    scaler = StandardScaler()
    # 以30天为时间窗口对BTC和区块链数据进行PCA，也许这个时间窗口可以调整
    roll_window = 30
    pca = PCA(n_components=len(merged_df.columns))
    print(f"the names of columns in merged_df are {merged_df.columns}")
        # If update mode is on, read existing PCA results and calculate only for new dates
    if update_mode:
        df_pca_results = pd.read_csv('df_pca_results.csv', index_col='date')
        df_pca_results.index = pd.to_datetime(df_pca_results.index)
        
        # Find the last date in the existing PCA results
        last_pca_date = df_pca_results.index.max()
        print(f"the last date in the existing PCA results is {last_pca_date}, and we are going to update the PCA results to {merged_df.index.max()}")
        # Determine the new data to be processed
        new_data_start = last_pca_date - pd.Timedelta(days=roll_window - 1)
        merged_df_update = merged_df.loc[new_data_start:]

        # Perform PCA only on the new data
        new_pca_results = []
        for start in range(0, len(merged_df_update) - roll_window +1):
            end = start + roll_window
            df_window = merged_df_update.iloc[start:end, :]
            df_window = scaler.fit_transform(df_window)
            pca_result = pca.fit_transform(df_window)
            mean_pca_values = np.mean(pca_result, axis=0)
            z_score_scaler = StandardScaler()
            z_score_normalized_mean_pca_values = z_score_scaler.fit_transform(mean_pca_values.reshape(-1, 1)).flatten()
            explained_var = pca.explained_variance_ratio_
            pca_window_result = {'date': merged_df_update.index[end-1]}
            for i, (norm_value, var) in enumerate(zip(z_score_normalized_mean_pca_values, explained_var)):
                pca_window_result[f'z_score_normalized_mean_PC{i+1}'] = norm_value
                pca_window_result[f'explained_var_PC{i+1}'] = var
            new_pca_results.append(pca_window_result)
        
        # Append new results to existing dataframe
        new_pca_df = pd.DataFrame(new_pca_results)
        new_pca_df.set_index('date', inplace=True)
        df_pca_results = pd.concat([df_pca_results, new_pca_df])
    else:
        # Initialize an empty DataFrame to store PCA results
        df_pca_results = []  
        # 设定时间窗口，从数据的前段开始，每次向后滚动一个时间窗口
        for start in range(0, len(merged_df) - roll_window +1):
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
        
    df_pca_results.set_index('date', inplace=True)    
    df_pca_results.index = df_pca_results.index.strftime('%Y-%m-%d %H:%M:%S')

    print("Last 5 rows of df_pca_results:")
    print(df_pca_results.tail(5))   
    # Ensure index is in the correct format
    df_pca_results.index = pd.to_datetime(df_pca_results.index).normalize()  # Normalize to midnight
    # Create the full date range
    full_date_range = pd.date_range(start=df_pca_results.index.min(), end=df_pca_results.index.max(), freq='D')
    # Find missing dates
    missing_dates = full_date_range.difference(df_pca_results.index)
    if not missing_dates.empty:
        print(f"There are missing dates: {missing_dates}")
        return
        # Check for missing values
    if df_pca_results.isnull().any().any():
        print("There are missing values in the DataFrame.")
        return
    # Check for full duplicates including the index
    if df_pca_results.duplicated(keep=False).any():
        print("There are fully duplicated rows in the DataFrame.")
        return
    # Check for duplicates ignoring the index
    df_temp = df_pca_results.reset_index()  # Reset index to treat 'date' as a regular column
    if df_temp.duplicated(subset=df_temp.columns.difference(['date']), keep=False).any():
        print("There are rows with duplicate data but different indices.")
        return

    #df_pca_results.set_index('date', inplace=True)
    df_pca_results.index = df_pca_results.index.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Our PCA results is from {df_pca_results.index[1]} to {df_pca_results.index[-1]}")
    df_pca_results.to_csv('df_pca_results.csv', index=True)   
    
def yf_fetch_data(start_date, end_date):
    symbols = ['ETH-USD', 'EURUSD=X', 'JPY=X', 'CNY=X', 'GBPUSD=X', '^GSPC', '^IXIC', 'GC=F']
    # Initialize an empty DataFrame
    data = pd.DataFrame()
    for symbol in symbols:
        try:
            # Fetch the data for each symbol
            temp_data = yf.download(symbol, start=start_date, end=end_date)
            if temp_data.empty:
                print(f"No data found for {symbol}")
                continue
            # Rename columns to include the symbol for uniqueness
            temp_data.columns = [f"{symbol}_{col}" for col in temp_data.columns]
            # Combine all data into a single DataFrame
            if data.empty:
                data = temp_data
            else:
                data = data.join(temp_data, how='outer')
        except Exception as e:
            print(f"Failed to download data for {symbol}: {e}")
    # Forward-fill missing values
    data.ffill(inplace=True)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
    return data

def fetch_or_update_yf_data():
    filename = 'yfinance_data.csv'
    three_years_ago = datetime.now() - timedelta(days=3*365)
    start_date = three_years_ago.strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    if not os.path.exists(filename):
        print("Fetching data for the past three years...")
        data = yf_fetch_data(start_date, end_date)
        data.to_csv(filename)
    else:
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        data.index = pd.to_datetime(data.index)  # 确保索引是 datetime 类型

        last_date = pd.to_datetime(data.index[-1]).date()
        update_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')

        if update_start_date < end_date:
            print(f"Updating data from {update_start_date} to {end_date}...")
            new_data = yf_fetch_data(update_start_date, end_date)
            new_data.index = pd.to_datetime(new_data.index)  # 确保索引是 datetime 类型

            # 合并旧数据和新数据
            updated_data = pd.concat([data, new_data])

            # 确保所有索引是 datetime 类型
            updated_data.index = pd.to_datetime(updated_data.index)

            # 识别和处理重复索引
            duplicate_indices = updated_data.index[updated_data.index.duplicated(keep=False)]

            if not duplicate_indices.empty:
                duplicates = updated_data.loc[duplicate_indices]
                print("处理前的重复条目：")
                print(duplicates)

                # 按索引分组并选择每组中缺失值最少的行
                processed_duplicates = (
                    duplicates.groupby(duplicates.index)
                    .apply(lambda x: x.loc[x.isna().sum(axis=1).idxmin()])
                )
                
                # 删除所有重复项并添加回处理过的重复项
                updated_data = updated_data.drop(index=duplicate_indices)
                updated_data = pd.concat([updated_data, processed_duplicates])
            
            # 确保删除所有重复的索引
            updated_data = updated_data[~updated_data.index.duplicated(keep='first')]

            # 排序索引并填充缺失的日期
            updated_data.index = pd.to_datetime(updated_data.index, errors='coerce')
            updated_data = updated_data.sort_index()

            # 确保日期索引的连续性，使用前向填充方法填补缺失值
            all_dates = pd.date_range(start=updated_data.index.min(), end=updated_data.index.max(), freq='D')
            updated_data = updated_data.reindex(all_dates)
            updated_data.ffill(inplace=True)

            updated_data.to_csv(filename)
        else:
            print("数据已经是最新的。")


if __name__ == "__main__":
    #get_block_ratiodate(update=True)
    get_block_ratiodate(update=True)
    PCA_BlockAndBitcoin(update_mode=False)
    fetch_or_update_yf_data()