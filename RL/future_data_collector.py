#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2024-08-14 07:26:31
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-08-21 07:20:05
FilePath: /trading/RL_selector/future_data_collector.py
Description: 
Copyright (c) 2024 by ${124321452@qq.com}, All Rights Reserved. 
'''
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from zipfile import ZipFile
from scipy.stats import skew, kurtosis
import re
import concurrent.futures
from tqdm import tqdm
from stable_baselines3 import PPO

# 自定义的 PPO 类，继承自 stable_baselines3 的 PPO 实现
class CustomPPO(PPO):
    def _setup_model(self):
        # 调用父类的 _setup_model 方法来初始化模型
        super(CustomPPO, self)._setup_model()
        # 保存原始的 forward 方法，用于后续调用
        original_forward = self.policy.forward
        # 添加额外的检查
        # print("Checking policy attributes:")
        # print("Has action_net:", hasattr(self.policy, 'action_net'))
        # print("Has mlp_extractor:", hasattr(self.policy, 'mlp_extractor'))
        # print("Has value_net:", hasattr(self.policy, 'value_net'))
        if hasattr(self.policy, 'action_net'):
            print("action_net structure:", self.policy.action_net)
        # 检查 policy 的其他重要属性
        important_attributes = ['observation_space', 'action_space', 'features_extractor']
        for attr in important_attributes:
            print(f"Has {attr}:", hasattr(self.policy, attr))
            
        def check_weights(module):
            for name, param in module.named_parameters():
                if torch.isnan(param).any():
                    print(f"WARNING: NaN weights detected in {name}")
                if torch.isinf(param).any():
                    print(f"WARNING: Inf weights detected in {name}")
        
        self.policy.apply(check_weights)

        # 自定义的 forward 方法，带有额外的调试信息
        def forward_with_checks(*args, **kwargs):
            # 打印前向传播的输入形状，以帮助检查输入数据是否正确
            # print("Forward pass input shapes:")
            # for arg in args:
            #     if isinstance(arg, dict):  # 如果输入是字典类型，逐项打印键和值的形状
            #         for k, v in arg.items():
            #             print(f"  {k}: {v.shape}")
            #     elif isinstance(arg, torch.Tensor):  # 如果输入是张量，打印张量的形状
            #         print(f"  Tensor shape: {arg.shape}")
            # 调用原始的 forward 方法
            result = original_forward(*args, **kwargs)
            # 检查 forward 方法的输出是否包含 NaN，并打印输出的形状和数值范围
            # if isinstance(result, tuple):  # 如果结果是一个元组（例如多个返回值）
            #     for i, tensor in enumerate(result):
            #         print(f"Output {i} shape: {tensor.shape}")
            #         # 确保输出中没有 NaN 值
            #         assert not torch.isnan(tensor).any(), f"NaN detected in policy output {i}"
            #         # 为什么在这里会有多个result的key，第0个是torch.Size([1, 216])，似乎是216个策略的分类，第二个值为torch.Size([1, 1])，第三个torch.Size([1])，
            #         print(f"Output {i} range: {tensor.min().item()} to {tensor.max().item()}")
            # else:  # 如果结果不是元组（单个返回值）
            #     print(f"Output shape: {result.shape}")
            #     assert not torch.isnan(result).any(), "NaN detected in policy output"
            #     print(f"Output range: {result.min().item()} to {result.max().item()}")
            # 返回原始的 forward 结果
            return result
        # 用自定义的 forward 方法替换原来的 forward 方法
        self.policy.forward = forward_with_checks
        # 保存原始的 _get_action_dist_from_latent 方法，用于后续调用
        original_get_action_dist = self.policy._get_action_dist_from_latent
        # 自定义的 _get_action_dist_from_latent 方法，带有额外的调试信息
        def get_action_dist_with_checks(latent_pi):
            # 打印 latent_pi 的形状和数值范围，以帮助检查潜在特征是否正确
            #print(f"Latent pi shape: {latent_pi.shape}")
            #print(f"Latent pi range: {latent_pi.min().item()} to {latent_pi.max().item()}")
            # 检查并打印 latent_pi 中的 NaN 值
            if torch.isnan(latent_pi).any():
                print("WARNING: NaN values detected in latent_pi")
                print("NaN positions:", torch.isnan(latent_pi).nonzero())
            mean_actions = self.policy.action_net(latent_pi)
            #print(f"Mean actions shape: {mean_actions.shape}")
            #print(f"Mean actions range: {mean_actions.min().item()} to {mean_actions.max().item()}")
            # 检查并打印 mean_actions 中的 NaN 值
            if torch.isnan(mean_actions).any():
                print("WARNING: NaN values detected in mean_actions")
                print("NaN positions:", torch.isnan(mean_actions).nonzero())
            # 调用原始的 _get_action_dist_from_latent 方法
            action_dist = original_get_action_dist(latent_pi)
            # 如果动作分布有多个参数（例如 Categorical 或 Normal 分布）
            if hasattr(action_dist, 'distribution'):
                pass
                # for i, dist in enumerate(action_dist.distribution):
                #     print(f"Action distribution {i}:")
                #     if isinstance(dist, torch.distributions.Categorical):
                #         print("  Type: Categorical")
                #         print(f"  Probs shape: {dist.probs.shape}")
                #         print(f"  Probs range: {dist.probs.min().item()} to {dist.probs.max().item()}")
                #     elif isinstance(dist, torch.distributions.Normal):
                #         print("  Type: Normal")
                #         print(f"  Loc shape: {dist.loc.shape}")
                #         print(f"  Loc range: {dist.loc.min().item()} to {dist.loc.max().item()}")
                #         print(f"  Scale shape: {dist.scale.shape}")
                #         print(f"  Scale range: {dist.scale.min().item()} to {dist.scale.max().item()}")
                #     else:
                #         print(f"  Type: {type(dist)}")
            else:
                print("Action distribution does not have 'distribution' attribute")
                print(f"Action distribution type: {type(action_dist)}")
            
            return action_dist
        # 用自定义的 _get_action_dist_from_latent 方法替换原来的方法
        self.policy._get_action_dist_from_latent = get_action_dist_with_checks

# 使用自定义的 CustomPPO 类来创建模型实例
# MultiInputPolicy：表示输入由多个不同的观察组成（例如市场数据和策略数据）
# env：是你之前定义的环境实例
# verbose=1：设置输出的详细程度
# policy_kwargs：传递给策略类的额外参数
# device='cuda'：表示在 GPU 上运行


def calculate_funding_rate_indicators(df, window):
    """
    cumulative_funding_rate:
    可能的问题：前 window-1 个值会是 NaN。
    解决方法：使用 fillna(0) 填充。这假设没有足够数据时累积资金费率为0。
    
    funding_rate_volatility:
    可能的问题：如果窗口内所有值相同，标准差为0，结果为 NaN。
    解决方法：使用 fillna(0) 填充。这表示在该窗口内没有波动。
    
    funding_rate_ma:
    可能的问题：前 window-1 个值会是 NaN。
    解决方法：使用 fillna(method='bfill') 用后面的有效值填充。这假设初始的移动平均与第一个有效值相同。
    
    funding_rate_divergence:
    可能的问题：如果 funding_rate_ma 有 NaN，结果会是 NaN。
    解决方法：使用 fillna(0) 填充。这假设没有足够数据时divergence为0。
    
    funding_rate_momentum:
    可能的问题：第一个 window 个值会是 NaN，如果初始价格为0也会产生 NaN。
    解决方法：使用 fillna(0) 填充。这假设没有足够数据时动量为0。
    
    funding_rate_regime:
    可能的问题：当 fundingRate 恰好为0时没有考虑。
    解决方法：添加一个 'Neutral' 类别。
    
    funding_rate_acceleration:
    可能的问题：第一个值会是 NaN。
    解决方法：使用 fillna(0) 填充。这假设初始加速度为0。
    
    funding_rate_percentile:
    可能的问题：如果窗口内所有值相同，结果可能是 NaN。
    解决方法：使用 fillna(0.5) 填充。这假设当所有值相同时，它们位于中位数位置。
    
    funding_rate_mean_reversion:
    可能的问题：当 funding_rate_volatility 为0时，会出现除以0的情况。
    解决方法：使用 np.where 条件语句，当 funding_rate_volatility 为0时返回0。
    """
    df = df.sort_index()
    
    result = pd.DataFrame(index=df.index)
    result['cumulative_funding_rate'] = df['fundingRate'].rolling(window=window).sum().fillna(0)
    result['funding_rate_volatility'] = df['fundingRate'].rolling(window=window).std().fillna(0)
    result['funding_rate_ma'] = df['fundingRate'].rolling(window=window).mean().fillna(method='bfill')
    result['funding_rate_divergence'] = df['fundingRate'] - result['funding_rate_ma'].fillna(0)
    result['funding_rate_momentum'] = df['fundingRate'].pct_change(periods=window).fillna(0)
    
    #result['extreme_funding_rate'] = ((df['fundingRate'] - result['funding_rate_ma']).abs() > 2 * result['funding_rate_volatility']).astype(int)
    
    # Funding rate regime
    result['funding_rate_regime'] = np.where(df['fundingRate'] > 0, 'Positive', 
                                             np.where(df['fundingRate'] < 0, 'Negative', 'Neutral'))
    
    # Funding rate acceleration
    result['funding_rate_acceleration'] = result['funding_rate_momentum'].diff().fillna(0)
    
    # Funding rate percentile
    result['funding_rate_percentile'] = df['fundingRate'].rolling(window=window).rank(pct=True).fillna(0.5)
    
    # Funding rate mean reversion (handle case where volatility is 0)
    result['funding_rate_mean_reversion'] = np.where(
        result['funding_rate_volatility'] != 0,
        (result['funding_rate_ma'] - df['fundingRate']) / result['funding_rate_volatility'],
        0
    ).astype(float)
    
    return result
    

def analyze_funding_rate_vs_price(df_funding, df_price, window):
    """
    Analyze the relationship between funding rate and price for a specific time window.
    
    :param df_funding: DataFrame with funding rate data
    :param df_price: DataFrame with price data
    :param window: int, number of days for the rolling window
    :return: DataFrame with correlation analysis
    """
    df_merged = pd.merge(df_funding, df_price, left_index=True, right_index=True, how='inner')
    df_merged['price_return'] = df_merged['close'].pct_change()

    def rolling_corr_with_lag(x, y, lag):
        return x.corr(y.shift(lag))
    
    
    def safe_corr(x, y):  
        """
        在数学上，当其中一个变量的标准差为零时，相关系数是未定义的。在这种情况下，大多数统计软件（包括 pandas）会返回 NaN。
        所以，在窗口中，如果资金费率保持不变，那么会导致无法计算有效的相关系数。
        """
        if np.std(x) == 0 or np.std(y) == 0:
            return 0
        return np.corrcoef(x, y)[0, 1]

    correlation = df_merged['fundingRate'].rolling(window=window).corr(df_merged['price_return']).fillna(0)
    lag_1_correlation = df_merged['fundingRate'].rolling(window=window).apply(
        lambda x: rolling_corr_with_lag(x, df_merged.loc[x.index, 'price_return'], 1)
    ).fillna(0)
    lag_7_correlation = df_merged['fundingRate'].rolling(window=window).apply(
        lambda x: rolling_corr_with_lag(x, df_merged.loc[x.index, 'price_return'], 7)
    ).fillna(0)

    return pd.DataFrame({
        'price_funding_correlation': correlation,
        'lag_1_correlation': lag_1_correlation,
        'lag_7_correlation': lag_7_correlation
    })


'''
期货的metrics数据

sum_open_interest: 持仓总量(基础币)，double 类型。
sum_open_interest_value: 持仓总价值，double 类型。
count_toptrader_long_short_ratio: 大户账户数多空比，double 类型。
sum_toptrader_long_short_ratio: 大户持仓量多空比，double 类型。
count_long_short_ratio: 所有交易者账户数多空比，double 类型。
sum_taker_long_short_vol_ratio: 吃单方主买量/吃单方主卖量，double 类型。

以天为单位重新整合数据。sum_open_interest_value和sum_open_interest取close，count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio取mean和波动率，偏度（Skewness）和峰度（Kurtosis）
'''

def process_zip_files(directory):
    all_data = []
    
    file_pattern = re.compile(r'^[A-Z]+-metrics-\d{4}-\d{2}-\d{2}\.zip$')
    
    for filename in os.listdir(directory):
        if file_pattern.match(filename):
            with ZipFile(os.path.join(directory, filename), 'r') as zip_file:
                for csv_file in zip_file.namelist():
                    with zip_file.open(csv_file) as file:
                        df = pd.read_csv(file)
                        df = adjust_datetime(df)  # Apply adjustment to each file
                        all_data.append(df)
    
    if not all_data:
        raise ValueError("No matching zip files found in the directory.")
    
    return pd.concat(all_data, ignore_index=True)

def adjust_datetime(df):
    df['create_time'] = pd.to_datetime(df['create_time'])
    
    # Check if the last row has '00:00:00' time
    if df['create_time'].iloc[-1].time() == pd.Timestamp('00:00:00').time():
        # Subtract 1 second from the last row
        df.loc[df.index[-1], 'create_time'] -= pd.Timedelta(seconds=1)
    
    return df

def fill_missing_values(df):
    # Convert create_time to datetime if it's not already
    df['create_time'] = pd.to_datetime(df['create_time'])
    
    # Sort the dataframe by symbol and create_time
    df = df.sort_values(['symbol', 'create_time'])
    
    # Group by symbol and date
    grouped = df.groupby([df['symbol'], df['create_time'].dt.date])
    
    filled_data = []
    
    for (symbol, date), group in grouped:
        # Check if there's at least one non-null value for each column
        if group.notna().any().all():
            # Forward fill within the group
            filled_group = group.fillna(method='ffill')
            filled_data.append(filled_group)
        else:
            # If any column is completely null for the day, keep the original data
            filled_data.append(group)
    
    return pd.concat(filled_data)

def calculate_statistics(group):
    close_cols = ['sum_open_interest', 'sum_open_interest_value']
    mean_cols = ['count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio', 
                 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
    
    result = {}
    
    # Define a threshold for the difference between last value and median
    # You may need to adjust this threshold based on your data
    threshold = 0.5  # 50% difference
    
    # Close values
    for col in close_cols:
        last_value = group[col].iloc[-1]
        median_value = group[col].median()
        
        # Calculate the relative difference
        relative_diff = abs(last_value - median_value) / median_value
        
        if relative_diff > threshold:
            result[f'{col}_close'] = median_value
            print(f"Warning: Using median for {col} due to large difference from last value.")
        else:
            result[f'{col}_close'] = last_value
    
    # Mean, volatility, skewness, and kurtosis
    for col in mean_cols:
        result[f'{col}_mean'] = group[col].mean()
        result[f'{col}_volatility'] = group[col].std()
        result[f'{col}_skewness'] = skew(group[col])
        result[f'{col}_kurtosis'] = kurtosis(group[col])
    
    # Add symbol and date
    result['symbol'] = group['symbol'].iloc[0]
    result['date'] = group['create_time'].dt.date.iloc[0]
    
    return pd.Series(result)

def get_future_metrics(directory):
    
    try:
        # Process all zip files
        df = process_zip_files(directory)
        
        # Convert create_time to datetime
        df['create_time'] = pd.to_datetime(df['create_time'])
        
        # Fill missing values
        df_filled = fill_missing_values(df)
        
        # Group by symbol and date, then calculate statistics
        daily_stats = df_filled.groupby([df_filled['symbol'], df_filled['create_time'].dt.date]).apply(calculate_statistics)
        
        # Reset index to make symbol and date columns
        daily_stats = daily_stats.reset_index(drop=True)
        
        # Sort by symbol and date
        daily_stats = daily_stats.sort_values(['symbol', 'date'])
        
        # Check for remaining missing values
        missing_values = daily_stats.isnull()
        if missing_values.any().any():
            print("Missing values detected after filling:")
            for column in daily_stats.columns:
                if missing_values[column].any():
                    missing_dates = daily_stats[missing_values[column]]['date']
                    print(f"\nColumn '{column}' has missing values on:")
                    for date in missing_dates:
                        print(date.strftime('%Y-%m-%d'))
        else:
            print("No missing values detected after filling.")
        
        # Save to CSV
        output_file = 'crypto_metrics_daily.csv'
        daily_stats.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
def fetch_funding_rate_data(start_timestamp, end_timestamp):
    
    base_url = "https://fapi.binance.com"
    funding_rate_endpoint = "/fapi/v1/fundingRate"
    
    print("Fetching BTCUSDT funding rate data...")
    all_data = []
    current_start = start_timestamp

    while current_start < end_timestamp:
        params = {
            'symbol': 'BTCUSDT',
            'startTime': current_start,
            'endTime': end_timestamp,
            'limit': 1000  # API 的最大限制
        }
        response = requests.get(base_url + funding_rate_endpoint, params=params)
        response.raise_for_status()
        fr_data = response.json()
        
        if not fr_data:  # 如果没有更多数据，跳出循环
            break
        
        all_data.extend(fr_data)
        
        # 更新 current_start 为最后一条数据的时间 + 1 毫秒
        current_start = fr_data[-1]['fundingTime'] + 1
        
    params = {
        'symbol': 'BTCUSDT',
        # If startTime and endTime are not sent, the most recent limit datas are returned.
        'limit': 1000  # API 的最大限制
    }
    response = requests.get(base_url + funding_rate_endpoint, params=params)
    response.raise_for_status()
    fr_data = response.json()
    all_data.extend(fr_data)

    fr_df = pd.DataFrame(all_data)
    fr_df['fundingTime'] = pd.to_datetime(fr_df['fundingTime'], unit='ms')
    fr_df = fr_df.set_index('fundingTime')
    fr_df['fundingRate'] = fr_df['fundingRate'].astype(float)
    fr_df = fr_df[~fr_df.index.duplicated(keep='first')]
    
    # 确保数据按时间排序
    fr_df = fr_df.sort_index()
    # 按日期进行重采样，选择每日的最后一个资金费率, 将来也可以取均值
    daily_fr = fr_df.resample('D').last()
    
    return daily_fr

'''
def fetch_kline_data(symbol, interval, start_timestamp, end_timestamp):
    base_url = "https://fapi.binance.com"
    klines_endpoint = "/fapi/v1/klines"
    all_klines = []
    current_start = start_timestamp

    while current_start < end_timestamp:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_timestamp,
            'limit': 1000  # Maximum allowed by the API
        }
        
        try:
            response = requests.get(base_url + klines_endpoint, params=params)
            response.raise_for_status()
            klines = response.json()
            
            if not klines:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': 1000  # Maximum allowed by the API
                }
                response = requests.get(base_url + klines_endpoint, params=params)
                response.raise_for_status()
                klines = response.json()
                all_klines.extend(klines)
                break
            
            all_klines.extend(klines)
            
            # Update the start time for the next iteration
            current_start = klines[-1][0] + 1  # Use the timestamp of the last kline + 1 millisecond
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            break

    return all_klines
'''

def fetch_kline_data(symbol, interval, start_timestamp, end_timestamp):
    print('going to fetch future data from api')
    base_url = "https://fapi.binance.com"
    klines_endpoint = "/fapi/v1/klines"
    all_klines = []

    params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1500  # Maximum allowed by the API
        }
        
    response = requests.get(base_url + klines_endpoint, params=params)
    response.raise_for_status()
    klines = response.json()
    
    all_klines.extend(klines)
        
    return all_klines


def process_kline_data(klines, date_range):
    try:
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sort by timestamp to ensure monotonic index
        df = df.sort_values('timestamp')
        
        # Remove duplicates, keeping the first occurrence
        df = df.drop_duplicates(subset='timestamp', keep='first')
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Convert to float after deduplication to avoid any potential conflicts
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Reindex to ensure we have entries for all days, forward fill missing data
        df = df.reindex(date_range, method='ffill')
        
        # Print some information about the data
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Number of data points: {len(df)}")
        print(f"Number of unique timestamps: {df.index.nunique()}")
        
        return df
    
    except Exception as e:
        print(f"Error in process_kline_data: {str(e)}")
        print("Diagnostic information:")
        print(f"Number of klines: {len(klines)}")
        if len(klines) > 0:
            print(f"First kline: {klines[0]}")
            print(f"Last kline: {klines[-1]}")
        print(f"Date range: {date_range[0]} to {date_range[-1]}")
        return None

def fetch_and_process_single_symbol(args):
    symbol, start_timestamp, end_timestamp, date_range = args
    try:
        klines = fetch_kline_data(symbol, '1d', start_timestamp, end_timestamp)
        if not klines:
            return symbol, None, f"No data retrieved for {symbol}"
        
        df = process_kline_data(klines, date_range)
        if df is None:
            return symbol, None, f"Failed to process data for {symbol}"
        
        return symbol, df, None
    except Exception as e:
        return symbol, None, f"Error processing {symbol}: {str(e)}"

def fetch_and_process_data(usdt_perpetuals, start_timestamp, end_timestamp, date_range, max_workers=8):
    all_contract_data = {}
    failed_symbols = []

    # 准备参数
    args_list = [(symbol, start_timestamp, end_timestamp, date_range) for symbol in usdt_perpetuals]

    # 使用线程池执行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(fetch_and_process_single_symbol, args_list), 
                            total=len(usdt_perpetuals), 
                            desc="Processing symbols"))

    # 处理结果
    for symbol, df, error_message in results:
        if df is not None:
            all_contract_data[symbol] = df
            print(f"Successfully processed {symbol}")
        else:
            failed_symbols.append(symbol)
            print(error_message)

    return all_contract_data, failed_symbols

def get_btcusdt_data(start_date, end_date, time_windows):
    '''
    
    交易量/未平仓合约量比率：衡量市场参与度
    波动率变化：当前波动率与前一时期波动率的比较。
    如果有未平仓合约数据，可以计算平均杠杆率。
    未平仓合约量与交易量比率（OI/Volume Ratio）
    未平仓合约量变化率
    未平仓合约量与资金费率
    未平仓合约量的标准差
    净流入/流出：基于买入和卖出量的差异。
    资金流向强度：净流入/流出相对于总交易量的比率。
    好的波动率和坏的波动率

    '''
    base_url = "https://fapi.binance.com"
    exchange_info_endpoint = "/fapi/v1/exchangeInfo"
    klines_endpoint = "/fapi/v1/klines"
    open_interest_endpoint = "/futures/data/openInterestHist"
    funding_rate_endpoint = "/fapi/v1/fundingRate"
    
    try:
        print("Fetching USDT-M perpetual contracts list...")
        response = requests.get(base_url + exchange_info_endpoint)
        response.raise_for_status()
        exchange_info = response.json()
        # 我担心有未来数据
        usdt_perpetuals = [symbol['symbol'] for symbol in exchange_info['symbols'] 
                           if symbol['symbol'].endswith('USDT') and symbol['contractType'] == 'PERPETUAL']
        
        print(f"Found {len(usdt_perpetuals)} USDT-M perpetual contracts")
        
        # Convert dates to timestamps
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        max_window = max(time_windows)
        start_timestamp = int((datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=max_window)).timestamp() * 1000)
        
        # Create a date range for our index
        date_range = pd.date_range(start=datetime.fromtimestamp(start_timestamp/1000),
                                   end=datetime.fromtimestamp(end_timestamp/1000),
                                   freq='D')
        
        print(f"Fetching data from {date_range[0].strftime('%Y-%m-%d')} to {date_range[-1].strftime('%Y-%m-%d')}")
        
        oi_df = pd.read_csv('crypto_metrics_daily.csv')
        oi_df['date'] = pd.to_datetime(oi_df['date'])
        oi_df = oi_df.set_index('date')
        btc_oi_data = oi_df.reindex(date_range, method='ffill')

        # Fetch BTCUSDT funding rate data
        print("Fetching BTCUSDT funding rate data...")
        btc_fr_data = fetch_funding_rate_data(start_timestamp, end_timestamp)
        
        all_contract_data, failed_symbols = fetch_and_process_data(usdt_perpetuals, start_timestamp, end_timestamp, date_range)
        
        if len(failed_symbols) > 1:
            print(f"Failed to process the following symbols: {failed_symbols}")

        btc_data = all_contract_data['BTCUSDT']
        
        print("Starting to calculate metrics for each day and time window")
        results = []
        
        analysis_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        metrics = ['quote_volume', 'taker_buy_quote_volume', 'taker_buy_ratio', 'market_share', 'volatility',
                   'oi_volume_ratio', 'volatility_change', 'avg_leverage', 'oi_funding_rate_correlation', 'oi_std',          'net_flow', 'flow_strength','good_volatility', 'bad_volatility']
        
        for current_date in analysis_range:
            print(f"Calculating metrics for {current_date.strftime('%Y-%m-%d')}")
            
            result = {
                'date': current_date.strftime('%Y-%m-%d'),
                'start_date': '',
                'end_date': ''
            }
            for window in time_windows:
                for metric in metrics:
                    result[f'{window}_{metric}'] = None

                try:
                    end_idx = btc_data.index.get_loc(current_date)
                    start_idx = max(0, end_idx - window + 1)
                    start_fr_idx = max(0, end_idx - window - 7)
                    
                        
                    window_data = {symbol: df.iloc[start_idx:end_idx+1] for symbol, df in all_contract_data.items()}
                    window_btc_data = window_data['BTCUSDT']
                    window_oi_data = btc_oi_data.loc[window_btc_data.index[0]:window_btc_data.index[-1]]
                    window_fr_data = btc_fr_data.loc[window_btc_data.index[0]:window_btc_data.index[-1]]
                    
                    start_fr_date = window_btc_data.index[0] - timedelta(days=8)
                    temp_fr_data = btc_fr_data.loc[start_fr_date:window_btc_data.index[-1]]
                    temp_data = {symbol: df.iloc[start_fr_idx:end_idx+1] for symbol, df in all_contract_data.items()}
                    temp_btc_data = temp_data['BTCUSDT']
                    
                    fr_indicators = calculate_funding_rate_indicators(temp_fr_data, window)
                    for col in fr_indicators.columns:
                        result[f'{window}_{col}'] = fr_indicators[col].iloc[-1]
                    
                    # Analyze price vs funding rate for the specific window
                    fr_price_analysis = analyze_funding_rate_vs_price(temp_fr_data, temp_btc_data, window)
                    for col in fr_price_analysis.columns:
                        result[f'{window}_{col}'] = fr_price_analysis[col].iloc[-1]
                    
                    # Add metrics from crypto_metrics_daily.csv
                    close_cols = ['sum_open_interest', 'sum_open_interest_value']
                    mean_cols = ['count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio', 
                                 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
                    
                    for col in close_cols:
                        result[f'{window}_{col}_close'] = window_oi_data[f'{col}_close'].iloc[-1]
                    
                    for col in mean_cols:
                        result[f'{window}_{col}_mean'] = window_oi_data[f'{col}_mean'].mean()
                        result[f'{window}_{col}_volatility'] = window_oi_data[f'{col}_volatility'].mean()
                        result[f'{window}_{col}_skewness'] = window_oi_data[f'{col}_skewness'].mean()
                        result[f'{window}_{col}_kurtosis'] = window_oi_data[f'{col}_kurtosis'].mean()
                    
                    total_volumes = {symbol: df['quote_volume'].sum() for symbol, df in window_data.items()}
                    sorted_symbols = sorted(total_volumes, key=total_volumes.get, reverse=True)
                    top_20_symbols = sorted_symbols[:20]
                    
                    btc_volume = total_volumes['BTCUSDT']
                    btc_taker_buy_volume = window_btc_data['taker_buy_quote_volume'].sum()
                    
                    top_20_volume = sum(total_volumes[symbol] for symbol in top_20_symbols)
                    market_share = btc_volume / top_20_volume if top_20_volume > 0 else 0
                    
                    taker_buy_ratio = btc_taker_buy_volume / btc_volume if btc_volume > 0 else 0
                    
                    # Calculate volatility (standard deviation of daily returns)
                    if window == 1:
                        volatility = 0
                    else:
                        daily_returns = window_btc_data['close'].pct_change().dropna()
                        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized volatility
                    
                    # New metric calculations
                    oi_volume_ratio = window_oi_data['sum_open_interest_close'].mean() / window_btc_data['quote_volume'].mean()
                    volatility_change = volatility / window_btc_data['close'].pct_change().std(ddof=1) if window > 1 else 0
                    avg_leverage = window_btc_data['quote_volume'].sum() / window_oi_data['sum_open_interest_close'].mean()
                    oi_funding_rate_correlation = window_oi_data['sum_open_interest_close'].corr(window_fr_data['fundingRate'])
                    oi_std = window_oi_data['sum_open_interest_close'].std()
                    net_flow = btc_taker_buy_volume - (btc_volume - btc_taker_buy_volume)
                    flow_strength = net_flow / btc_volume
                    
                    # Calculate good and bad volatility
                    returns = window_btc_data['close'].pct_change().dropna()
                    good_returns = returns[returns > 0]
                    bad_returns = returns[returns < 0]
                    # Calculate good volatility, use 0 if there are no positive returns
                    if len(good_returns) > 0:
                        good_volatility = (good_returns ** 2).mean() ** 0.5 * (252 ** 0.5)
                    else:
                        good_volatility = 0
                    
                    # Calculate bad volatility, use 0 if there are no negative returns
                    if len(bad_returns) > 0:
                        bad_volatility = (bad_returns ** 2).mean() ** 0.5 * (252 ** 0.5)
                    else:
                        bad_volatility = 0
                    
                    # Update the result dictionary
                    result['start_date'] = window_btc_data.index[0].strftime('%Y-%m-%d')
                    result['end_date'] = window_btc_data.index[-1].strftime('%Y-%m-%d')
                    result[f'{window}_quote_volume'] = btc_volume
                    result[f'{window}_taker_buy_quote_volume'] = btc_taker_buy_volume
                    result[f'{window}_taker_buy_ratio'] = taker_buy_ratio
                    result[f'{window}_market_share'] = market_share
                    result[f'{window}_volatility'] = volatility
                    result[f'{window}_oi_volume_ratio'] = oi_volume_ratio
                    result[f'{window}_volatility_change'] = volatility_change
                    result[f'{window}_avg_leverage'] = avg_leverage
                    result[f'{window}_oi_funding_rate_correlation'] = oi_funding_rate_correlation
                    result[f'{window}_oi_std'] = oi_std
                    result[f'{window}_net_flow'] = net_flow
                    result[f'{window}_flow_strength'] = flow_strength
                    result[f'{window}_good_volatility'] = good_volatility
                    result[f'{window}_bad_volatility'] = bad_volatility
                    
                    print(f"Completed calculations for {current_date.strftime('%Y-%m-%d')} with {window}-day window")
                except Exception as e:
                    print(f"Error calculating metrics for {current_date.strftime('%Y-%m-%d')} with {window}-day window: {str(e)}")
            
            results.append(result)
        
        print("Creating final DataFrame")
        result_df = pd.DataFrame(results)
        
        print("Data retrieval and processing complete!")
        return result_df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data: {str(e)}")
        return None


def download_file(url, destination, retries=3, delay=1):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            with open(destination, 'wb') as f:
                f.write(response.content)
            print(f"{destination} finished")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to download after {retries} attempts.")
    return False

def download_binance_futuremetircs(start_date, end_date, download_dir, symbol='BTCUSDT', max_workers=16):
    base_url = "https://data.binance.vision/data/futures/um/daily/metrics/"
    
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    current_date = start_date  
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    tasks = []
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        file_name = f"{symbol}-metrics-{date_str}.zip"
        destination = os.path.join(download_dir, file_name)
        
        if os.path.exists(destination):
            print(f"File {file_name} already exists. Skipping.")
        else:
            url = f"{base_url}{symbol}/{file_name}"
            tasks.append((url, destination))
        
        current_date += timedelta(days=1)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda args: download_file(*args), tasks))
        
    print(f"Downloaded {results.count(True)} files successfully, {results.count(False)} files failed.")



# Example usage
if __name__ == "__main__":
    start_date = "2023-01-01"
    # Get today's date in UTC and subtract one day
    end_date_utc = datetime.utcnow() - timedelta(days=1)
    # Format end_date to match the format of start_date
    end_date = end_date_utc.strftime("%Y-%m-%d")
    directory = "/binance_data"
    #先运行download_binance_data，获取future相关的数据
    download_binance_futuremetircs(start_date,end_date,download_dir=directory)
    #将这些数据整理成csv文件，处理缺失值，然后计算各类指标
    get_future_metrics(directory)       
    
    time_windows = [1, 7, 30]  # 1 day, 7 days, 30 days
    print(f"Starting data retrieval for all USDT-M perpetual contracts from {start_date} to {end_date}")
    result = get_btcusdt_data(start_date, end_date, time_windows)
    # 输出缺失值数量
    print(result.isnull().sum().to_string())
    if result is not None:
        print("\nBTCUSDT Contract Data for Different Time Windows:")
        #print(result.to_string(index=False))
        print("\nYou can uncomment the last line in the script to save this data to a CSV file.")
        # Optionally save to CSV
        result.to_csv('btcusdt_contract_data.csv', index=False)
    else:
        print("Failed to retrieve data.")
        
        
    # result_df_check = pd.read_csv('btcusdt_contract_data.csv', index_col='date')
    # print(result_df_check.isnull().sum().to_string())
    # result_df_check = result_df_check.dropna(axis=1)
    # missing_indices = result_df_check[result_df_check['7_funding_rate_mean_reversion'].isnull()].index
