#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2024-08-21 04:54:05
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-08-21 13:41:42
FilePath: /trading/RL_selector/RL_data_process.py
Description: 
Copyright (c) 2024 by ${124321452@qq.com}, All Rights Reserved. 
'''
import json
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, QuantileTransformer, MaxAbsScaler
from typing import Tuple, List

def rolling_window_standardize(df, bool_columns, abs_mean_larger10, window_size=30, 
                               method='log', process_type=1):
    """
    对数据框进行滚动窗口标准化处理，包含对数收益率计算和特定变量处理。
    
    参数:
    df (pd.DataFrame): 输入的数据框。
    bool_columns (list): 包含布尔变量名称的列表，这些变量不会被标准化处理。
    abs_mean_larger10 (list): 绝对均值大于等于10的变量列表。
    window_size (int): 滚动窗口大小，默认为30天。
    methods (list): 标准化方法列表，支持'robust', 'log', 'quantile', 'maxabs'。
    process_type (int): 处理类型：
                        1 - 不在abs_mean_larger10的变量保留原值和对数收益率，>=10的只保留对数收益率
                        2 - 不进行对数收益率处理，保留原值
                        3 - 只留对数收益率，去除原值
                        
    返回:
    pd.DataFrame: 标准化处理后的数据框。
    """
    # 识别列类型
    date_columns = df.select_dtypes(include=['datetime64', 'object']).columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    non_numeric_columns = df.columns.difference(numeric_columns)
    
    # 初始化结果字典
    result_dict = {col: df[col] for col in df.columns}
    
    # 使用前向和后向填充来填充NaN值,会有轻微的未来数据
    numeric_df = df[numeric_columns].ffill().bfill()
    
    def signed_log(x):
        return np.sign(x) * np.log(np.abs(x) + 1)
    # 计算对数收益率，处理负值
    log_return = signed_log(numeric_df) - signed_log(numeric_df.shift(1))
    
    # 根据处理类型进行处理
    for column in numeric_columns:
        if column in bool_columns:
            continue  # 跳过布尔变量
        
        if process_type == 1:
            if column not in abs_mean_larger10:
                result_dict[f'{column}_log_return'] = log_return[column]
            else:
                result_dict[f'{column}_log_return'] = log_return[column]
                del result_dict[column]  # 删除原始列
        elif process_type == 2:
            pass  # 保留原值，不做处理
        elif process_type == 3:
            result_dict[f'{column}_log_return'] = log_return[column]
            del result_dict[column]  # 删除原始列
        else:
            raise ValueError("you must choose 1, 2, 3 for process_type")
    
    # 滚动窗口标准化
    columns_to_standardize = [col for col in result_dict.keys() 
                              if col not in date_columns and col not in bool_columns]
    
    # 预先初始化所有scaler
    scalers = {
        'robust': RobustScaler(),
        'quantile': QuantileTransformer(output_distribution='normal'),
        'maxabs': MaxAbsScaler()
    }

    # 对于每种方法预先创建一个函数
    def log_transform(x):
        return np.log1p(np.abs(x)) * np.sign(x)
    
    def scale_with_scaler(x, scaler):
        return scaler.fit_transform(x.reshape(-1, 1)).flatten()[-1]

    for column in columns_to_standardize:
        if method == 'log':
            new_col = f'{column}_log'
            result_dict[new_col] = result_dict[column].apply(log_transform)
        else:
            new_col = f'{column}_{method}'
            scaler = scalers[method]
            result_dict[new_col] = result_dict[column].rolling(window=window_size, min_periods=1).apply(
                lambda x: scale_with_scaler(x.values, scaler)
            )
    # 创建最终的DataFrame
    df_standardized = pd.DataFrame(result_dict)
    # 去除因滚动窗口或对数收益率带来的NaN值
    # 使用前向和后向填充来填充NaN值,会有轻微的未来数据
    df_standardized = df_standardized.ffill().bfill()
    return df_standardized


def extract_strategies_data(json_file_path: str, config_path: str = 'config.json') -> Tuple[pd.DataFrame, List]:
    # 读取配置文件
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
    except Exception as e:
        print(f"Failed to load config from {config_path}: {str(e)}")
        config = None

    if not config:
        target_names = {
            'weighted_win_ratio', 'weighted_value', 'weighted_drawdown', 'weighted_SharpeDIY',
            'weighted_drawdown_value', 'weighted_win_ratio_value', 'weighted_drawdown_win_ratio_value',
            'weighted_sqn', 'weighted_sqn_drawdown', 'weighted_profitable_1days_ratio',
            'weighted_profitable_2days_ratio', 'weighted_profitable_3days_ratio'
        }
        opt_periods = {'35','56','70'}
        strategy_names = ['VWAPStrategyLong', 'VWAPStrategyShort', 'RSIV_StrategyOneday', 'RSIV_StrategyFourhour', 'MAVStrategy', 'WRSIStrategy']
        print("No target_names found in config file. Using default targets.")
    else:
        target_names = config['target_names']
        opt_periods = config['opt_periods']
        strategy_names = config['strategy_names']
        print("found config file. Using config params.")        
    # 读取 JSON 文件
    try:
        with open(json_file_path, 'r') as file:
            ori_data = json.load(file)
    except Exception as e:
        print(f"Failed to load JSON data from {json_file_path}: {str(e)}")
        raise

    sorted_keys = sorted(ori_data.keys(), key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data = OrderedDict((key, ori_data[key]) for key in sorted_keys if key >= '2022-12-01 00:00:00')

    strategies_list = []
    wrong_list = []

    for end_date, time_data in data.items():
        for time in opt_periods:
            target_data = time_data.get(time)
            if not target_data:
                print(f"Skipping time {time} at date {end_date} because it is not found in time_data.")
                continue  # 如果time在time_data中不存在，跳过本次循环
            for target in target_names:
                if target not in target_data:
                    print(f"Skipping target {target} at time {time} and date {end_date} because it is not found in target_data.")
                    continue  # 如果target在target_data中不存在，跳过本次循环
                for type_, strategy_data in target_data[target].items():
                    if type_ not in strategy_names:
                        print(f"Skipping type {type_} for target {target} at time {time} and date {end_date} because it is not in strategy_names.")
                        continue  # 如果type_不在strategy_names中，跳过本次循环
                    try:
                        apply_data = strategy_data['apply']['7'][target]['0']
                        total_return = apply_data['total_return']
                        # 提取策略参数
                        params = strategy_data['apply']['7'][target]['1']
                        # 确保参数顺序一致
                        sorted_params = OrderedDict(sorted(params.items()))

                        strategy_info = {
                            'date': pd.to_datetime(end_date),
                            'time': int(time),
                            'target': target,
                            'type': type_,
                            'return': total_return
                        }
                        # 添加排序后的参数
                        for i, (param_name, param_value) in enumerate(sorted_params.items()):
                            strategy_info[f'param_{i}'] = param_value

                        strategies_list.append(strategy_info)
                    except Exception as e:
                        wrong_list.append([end_date, time, target, type_, str(e)])
                        print(f"Error processing strategy: {end_date}, {time}, {target}, {type_}. Error: {str(e)}")
    # 创建 DataFrame
    strategies_df = pd.DataFrame(strategies_list)
    # 确保所有策略都有相同数量的参数列
    # 填充缺失值
    strategies_df.fillna(-999, inplace=True)
    # 转换数据类型
    strategies_df['return'] = strategies_df['return'].astype(float)
    # 设置多重索引
    #strategies_df.set_index(['date', strategies_df.groupby('date').cumcount()], inplace=True)

    print(f"Processed {len(strategies_df)} strategies. {len(wrong_list)} errors encountered.")

    return strategies_df, wrong_list

def detect_and_adjust_all_columns(df, high_threshold=10, low_threshold=10, window=30):
    """
    对DataFrame中的所有数值列检测高脉冲和低脉冲变化，并调整异常值。

    参数:
    df (pandas.DataFrame): 包含时间序列数据的DataFrame
    high_threshold (float): 定义高脉冲的倍数阈值，默认为5
    low_threshold (float): 定义低脉冲的倍数阈值，默认为5
    window (int): 滚动窗口的大小（天数），默认为7

    返回:
    tuple: (调整后的DataFrame, 高脉冲DataFrame, 低脉冲DataFrame)
    """
    # 确保索引是日期时间类型
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 对DataFrame进行排序，以确保正确的时间顺序
    df = df.sort_index()

    # 选择数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # 创建新的DataFrame来存储调整后的数据和脉冲检测结果
    adjusted_df = df.copy()
    high_pulse_df = pd.DataFrame(
        False, index=df.index, columns=numeric_columns)
    low_pulse_df = pd.DataFrame(False, index=df.index, columns=numeric_columns)

    for column in numeric_columns:
        # 计算滚动窗口的平均值（不包括当前值）
        rolling_mean = df[column].shift(1).rolling(
            window=window, min_periods=1).mean()

        # 计算当前值与前window天的平均值之比
        pulse_ratio = df[column] / rolling_mean

        # 检测高脉冲和低脉冲
        high_pulse = pulse_ratio > high_threshold
        low_pulse = pulse_ratio < (1 / low_threshold)

        # 调整高脉冲值
        adjusted_df.loc[high_pulse,
                        column] = rolling_mean[high_pulse] * high_threshold

        # 调整低脉冲值
        adjusted_df.loc[low_pulse,
                        column] = rolling_mean[low_pulse] * (1 / low_threshold)

        # 记录脉冲检测结果
        high_pulse_df[column] = high_pulse
        low_pulse_df[column] = low_pulse

    return adjusted_df, high_pulse_df, low_pulse_df


if __name__ == '__main__':
    # 读取 btcusdt_contract_data.csv
    df_contract = pd.read_csv('btcusdt_contract_data.csv', parse_dates=['date'])
    df_contract.set_index('date', inplace=True)
    
    # 读取 df_chain_ratio.csv
    df_chain = pd.read_csv(
        'df_chain_ratio.csv', parse_dates=['date'])
    df_chain.set_index('date', inplace=True)
    
    # 读取 BTCUSDT_future_1m.csv，重采样为1天
    df_future = pd.read_csv('BTCUSDT_future_1m.csv', parse_dates=['datetime'])
    df_future.set_index('datetime', inplace=True)
    df_future = df_future.resample('1D').last()
    # 计算价格波动性（使用20天滚动窗口）
    df_future['30_price_volatility'] = df_future['close'].pct_change().rolling(window=30).std()
    # 计算交易量波动性（使用20天滚动窗口）
    df_future['30_volume_volatility'] = df_future['volume'].pct_change().rolling(window=30).std()
    # 计算价格波动性（使用20天滚动窗口）
    df_future['7_price_volatility'] = df_future['close'].pct_change().rolling(window=7).std()
    # 计算交易量波动性（使用20天滚动窗口）
    df_future['7_volume_volatility'] = df_future['volume'].pct_change().rolling(window=7).std()
    
    # 读取 yfinance_data.csv
    df_yfinance = pd.read_csv(
        'yfinance_data.csv', header=0, index_col=0)
    df_yfinance.index = pd.to_datetime(df_yfinance.index)
    
    # 确保所有数据框的索引都是 datetime 格式
    for df in [df_contract, df_chain, df_future, df_yfinance]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    
    # 合并所有数据框
    merged_df = pd.concat([df_contract, df_chain, df_future,
                          df_yfinance], axis=1, join='inner')
    
    # 假设 df 是你的 DataFrame
    #missing_values = merged_df.isnull().sum()
    
    # 筛选出有缺失值的列
    #missing_columns = missing_values[missing_values > 0]
    
    #drop_lists = [i for i in missing_columns.index]
    
    # 全部为缺失值的变量
    drop_lists = [
        'start_date',
        'end_date',
        "1_oi_funding_rate_correlation",
        "1_oi_std",
        "1_lag_1_correlation",
        "1_lag_7_correlation",
        "7_lag_1_correlation",
        "7_lag_7_correlation"]
    
    df_new = merged_df.drop(drop_lists, axis=1)
    
    # 剔除只有一个值的变量
    #unique_counts = df_new.nunique()
    #columns_with_few_unique_values = unique_counts[unique_counts < 2]
    #columns_with_few_unique_values.index
    #drop_lists = [i for i in columns_with_few_unique_values.index]
    
    #df_new = df_new.drop(drop_lists, axis=1)
    # 只有一个值的变量
    drop_lists = ['1_volatility', '1_volatility_change', '1_good_volatility',
           '1_bad_volatility', '1_funding_rate_volatility',
           '1_funding_rate_divergence', '1_funding_rate_percentile',
           '1_funding_rate_mean_reversion', '1_price_funding_correlation',
           'EURUSD=X_Volume', 'JPY=X_Volume', 'CNY=X_Volume', 'GBPUSD=X_Volume']
    
    df_new = df_new.drop(drop_lists, axis=1)
    # 保存合并后的数据框到CSV文件
    output_file = 'uncheck_data.csv'
    df_new.to_csv(output_file, index=True, date_format='%Y-%m-%d %H:%M:%S')
    
        
    # 检测高脉冲和低脉冲，并调整数据
    adjusted_df, high_pulse_df, low_pulse_df = detect_and_adjust_all_columns(
        df_new, high_threshold=10, low_threshold=10, window=30)
    
    # 打印结果
    for column, result in high_pulse_df.items():
        pulse_count = result.sum()
        if pulse_count > 0:
            print(f"列 '{column}' 中检测到 {pulse_count} 个高脉冲数据点")
            # print(merged_df.loc[result, column])
            print("\n")
    
    for column, result in low_pulse_df.items():
        pulse_count = result.sum()
        if pulse_count > 0:
            print(f"列 '{column}' 中检测到 {pulse_count} 个低脉冲数据点")
            # print(merged_df.loc[result, column])
            print("\n")
    
    # 检测每个变量的类型
    data_types = adjusted_df.dtypes
    
    # 统计每种类型的变量数量
    type_counts = data_types.value_counts()
    
    print(f"共有 {len(type_counts)} 种不同的数据类型。")
    
    # 打印每种数据类型的变量
    for dtype, count in type_counts.items():
        print(f"\n数据类型: {dtype} ({count} 个变量)")
        print(data_types[data_types == dtype].index.tolist())
        
    # 使用 replace() 方法将 'Positive' 替换为 1，将 'Negative' 替换为 0
    adjusted_df[['1_funding_rate_regime', '7_funding_rate_regime', '30_funding_rate_regime']] = adjusted_df[['1_funding_rate_regime', '7_funding_rate_regime', '30_funding_rate_regime']].replace({'Positive': 1, 'Negative': 0})
    
    # 保存合并后的数据框到CSV文件
    output_file = 'adjusted_df.csv'
    adjusted_df.to_csv(output_file, index=True, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\n数据已保存到 {output_file}")
    
    # # 打印转换后的前几行数据，以确保替换成功
    # print(adjusted_df[['1_funding_rate_regime', '7_funding_rate_regime', '30_funding_rate_regime']].head())
    # # 计算每列的均值
    # mean_values = adjusted_df.mean()
    # # 筛选出均值的绝对值超过1的列
    # filtered_columns = mean_values[mean_values.abs() > 10].index
    # # 获取这些列的描述性统计信息（均值，最大值，最小值，标准差）
    # filtered_stats = adjusted_df[filtered_columns].agg(['mean', 'max', 'min', 'std'])
    # # 打印结果
    # print("均值的绝对值超过1的列及其统计信息：")
    # print(filtered_stats)
