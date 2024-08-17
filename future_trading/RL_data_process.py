# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
from collections import OrderedDict
from datetime import datetime
import pandas as pd
import numpy as np

# 读取 btcusdt_contract_data.csv
df_contract = pd.read_csv('btcusdt_contract_data.csv', parse_dates=['date'])
df_contract.set_index('date', inplace=True)

# 读取 df_chain_ratio.csv
df_chain = pd.read_csv('df_chain_ratio.csv', parse_dates=['date'])
df_chain.set_index('date', inplace=True)

# 读取 BTCUSDT_future_1m.csv，重采样为1天
df_future = pd.read_csv('BTCUSDT_future_1m.csv', parse_dates=['datetime'])
df_future.set_index('datetime', inplace=True)
df_future = df_future.resample('1D').last()

# 读取 yfinance_data.csv
df_yfinance = pd.read_csv('yfinance_data.csv', header=0, index_col=0)
df_yfinance.index = pd.to_datetime(df_yfinance.index)

# 确保所有数据框的索引都是 datetime 格式
for df in [df_contract, df_chain, df_future, df_yfinance]:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

# 合并所有数据框
merged_df = pd.concat([df_contract, df_chain, df_future, df_yfinance], axis=1,join='inner')

# 假设 df 是你的 DataFrame
missing_values = merged_df.isnull().sum()

# 筛选出有缺失值的列
missing_columns = missing_values[missing_values > 0]

drop_lists = [i for i in missing_columns.index]

#全部为缺失值的变量
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
unique_counts = df_new.nunique()
columns_with_few_unique_values = unique_counts[unique_counts < 2]
columns_with_few_unique_values.index
drop_lists = [i for i in columns_with_few_unique_values.index]

df_new = df_new.drop(drop_lists, axis=1)
#只有一个值的变量
drop_lists = [
    "7_volatility_change",
    "30_volatility_change"]

df_new = df_new.drop(drop_lists, axis=1)




def detect_and_adjust_all_columns(df, high_threshold=5, low_threshold=5, window=7):
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
    high_pulse_df = pd.DataFrame(False, index=df.index, columns=numeric_columns)
    low_pulse_df = pd.DataFrame(False, index=df.index, columns=numeric_columns)
    
    for column in numeric_columns:
        # 计算滚动窗口的平均值（不包括当前值）
        rolling_mean = df[column].shift(1).rolling(window=window, min_periods=1).mean()
        
        # 计算当前值与前window天的平均值之比
        pulse_ratio = df[column] / rolling_mean
        
        # 检测高脉冲和低脉冲
        high_pulse = pulse_ratio > high_threshold
        low_pulse = pulse_ratio < (1 / low_threshold)
        
        # 调整高脉冲值
        adjusted_df.loc[high_pulse, column] = rolling_mean[high_pulse] * high_threshold
        
        # 调整低脉冲值
        adjusted_df.loc[low_pulse, column] = rolling_mean[low_pulse] * (1 / low_threshold)
        
        # 记录脉冲检测结果
        high_pulse_df[column] = high_pulse
        low_pulse_df[column] = low_pulse
    
    return adjusted_df, high_pulse_df, low_pulse_df

# 检测所有数值列的高脉冲数据
numeric_columns = df_new .select_dtypes(include=[np.number]).columns
high_pulse_results = {}
low_pulse_results = {}

    # 检测高脉冲和低脉冲，并调整数据
adjusted_df, high_pulse_df, low_pulse_df = detect_and_adjust_all_columns(df_new , high_threshold=10, low_threshold=10, window=30)

for i in adjusted_df.columns:
    print(i)

# 打印结果
for column, result in high_pulse_results.items():
    pulse_count = result.sum()
    if pulse_count > 0:
        print(f"列 '{column}' 中检测到 {pulse_count} 个高脉冲数据点")
        #print(merged_df.loc[result, column])
        print("\n")

for column, result in low_pulse_results.items():
    pulse_count = result.sum()
    if pulse_count > 0:
        print(f"列 '{column}' 中检测到 {pulse_count} 个低脉冲数据点")
        #print(merged_df.loc[result, column])
        print("\n")
        
# 保存合并后的数据框到CSV文件
output_file = 'adjusted_df.csv'
adjusted_df.to_csv(output_file, index=True, date_format='%Y-%m-%d %H:%M:%S')
print(f"\n数据已保存到 {output_file}")



# 打开 JSON 文件
# with open(r"C:\Users\Administrator.DESKTOP-4H80TP4\Downloads\2021_10_08_2023_05_28_opt", 'r') as f:
#     # 使用 json.load() 解析 JSON 数据
#     data = json.load(f)

# # 将键转换为 datetime 对象，然后排序
# sorted_keys = sorted(data.keys(), key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

# # 创建一个新的有序字典，按照排序后的键
# sorted_data = OrderedDict((key, data[key]) for key in sorted_keys if key >='2024-07-22 00:00:00')

# # 使用 open() 函数以写入模式打开文件，并使用 json.dump() 函数将数据写入文件
# filename = 'data.json'
# with open(filename, 'w', encoding='utf-8') as file:
#     json.dump(sorted_data, file, ensure_ascii=False, indent=4)

# print(f"数据已成功保存到 {filename} 文件中。")

def extract_strategies_data(json_file_path: str) -> pd.DataFrame:
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        ori_data = json.load(file)

    # 将键转换为 datetime 对象，然后排序
    sorted_keys = sorted(ori_data.keys(), key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    
    # 创建一个新的有序字典，按照排序后的键
    data = OrderedDict((key, ori_data[key]) for key in sorted_keys if key >='2022-12-01 00:00:00')

    strategies_list = []
    wrong_list = []
    for end_date, time_data in data.items():
        for time, target_data in time_data.items():
            for target, type_data in target_data.items():
                for type_, strategy_data in type_data.items():
                    try:
                        apply_data = strategy_data['apply']['7'][target]['0']           
                        total_return = apply_data['total_return']
    
                        strategy_info = {
                            'date': pd.to_datetime(end_date),
                            'time': int(time),
                            'target': target,
                            'type': type_,
                            'return': total_return
                        }
    
                        strategies_list.append(strategy_info)
                    except:
                        print(end_date)
                        print(time)
                        print(target)
                        print(type_)
                        wrong_list.append([end_date,time,target,type_])
        #raise
    # 创建 DataFrame
    strategies_df = pd.DataFrame(strategies_list)

    # 设置多重索引
    strategies_df.set_index(['date', strategies_df.groupby('date').cumcount()], inplace=True)

    return strategies_df,wrong_list

# 使用函数加载数据
json_file_path= '2021_10_08_2023_05_28_opt'
strategies_data1,wrong_list = extract_strategies_data(json_file_path)
strategies_data = load_strategies_data_from_json(file_path)
print(strategies_data)