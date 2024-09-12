#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2024-07-13 05:28:17
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-08-08 05:57:18
FilePath: /trading/future_trading/future_data_pipline.py
Description: 
Copyright (c) 2024 by ${124321452@qq.com}, All Rights Reserved. 
'''
from datetime import datetime, timedelta
import datetime as dt
import pandas as pd
import json
import requests
import time
import os

import traceback
import sys

def get_binance_bars(symbol, interval, startTime, endTime, filepath):
    url = "https://fapi.binance.com/fapi/v1/klines"  # Binance Futures API endpoint
    df_list = []
    
    try:
        while startTime < endTime:
            # Ensure startTime is always less than endTime
            current_endTime = min(startTime + dt.timedelta(minutes=1000), endTime)
            
            startTime_str = str(int(startTime.timestamp() * 1000))
            endTime_str = str(int(current_endTime.timestamp() * 1000))
            
            print(f"Fetching data from {startTime} to {current_endTime}")
            
            limit = '1000'
            req_params = {"symbol": symbol, 'interval': interval, 'startTime': startTime_str,
                          'endTime': endTime_str, 'limit': limit}
            
            response = requests.get(url, params=req_params)
            response.raise_for_status()  # This will raise an HTTPError for bad responses
            data = response.json()
            
            print(f"Received {len(data)} data points")
            
            if not data:  # check if data is empty
                break
            
            df = pd.DataFrame(data)
            df = df.iloc[:, 0:6]
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype("float")
            
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df.set_index('datetime', inplace=True)  # set datetime as index
            
            df_list.append(df)
            
            # set new startTime as the last datetime of the current data
            if not df.empty:
                startTime = max(df.index) + dt.timedelta(minutes=1)
            else:
                startTime = current_endTime
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        if df_list:
            result_df = pd.concat(df_list)
            return result_df
        else:
            print("No data was collected")
            return None

def update_data(filepath, symbol, interval):
    filename = os.path.abspath(filepath)
    
    # Check if the file exists
    if os.path.exists(filename):
        # Load existing data
        df_old = pd.read_csv(filename, sep=',', index_col='datetime', parse_dates=True)
        if not df_old.empty:
            # Get the last datetime in the existing data
            last_datetime = df_old.index[-10]
        else:
            last_datetime = dt.datetime(2022, 9, 1)  # Or whatever start date you prefer
    else:
        print(f"File {filename} does not exist. Creating a new file.")
        df_old = pd.DataFrame()
        last_datetime = dt.datetime(2022, 9, 1)  # Or whatever start date you prefer

    # Get current datetime
    current_datetime = pd.Timestamp.now(tz='UTC').floor('T').tz_localize(None)

    # If the data is not up-to-date, get new data
    if df_old.empty or last_datetime < current_datetime:
        print(f"Fetching new data from {last_datetime} to {current_datetime}")
        df_new = get_binance_bars(symbol, interval, last_datetime, current_datetime, filepath)
        
        if df_new is not None and not df_new.empty:
            # Combine old and new data, remove duplicates
            df_combined = pd.concat([df_old, df_new]).drop_duplicates(keep='last')
            
            # Save to a temporary file first
            temp_filepath = filepath + '.temp'
            df_combined.to_csv(temp_filepath)
            
            # If saving to temp file was successful, rename it to replace the original file
            os.replace(temp_filepath, filepath)
            print(f"Updated data saved to {filepath}")
        else:
            print("No new data to add.")
    else:
        print("Data is already up-to-date.")

    print(f"Finished, data is saved in {filepath}")


# 需要处理找不到的情况。
def find_row_by_date(df, date_string):
    date = pd.to_datetime(date_string)
    try:
        return df.index.get_loc(date)
    except KeyError:
        closest_date = df.index.asof(date)
        if pd.isnull(closest_date):  # 如果最接近的日期仍然是空值，则寻找最近的晚于给定日期的索引
            closest_date = df.iloc[df.index > date].first_valid_index()
        if pd.isnull(closest_date):
            print(f"No date found that is close to {date_string}")
            return None
        return df.index.get_loc(closest_date)


def load_data(filepath):
    df = pd.read_csv(filepath, sep=',')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('datetime', inplace=True)
    return df


'''
find_discontinuities(df) 是一个生成器函数，它接受一个 DataFrame df,并通过遍历 DataFrame 的索引来查找不连续点。如果两个连续的时间戳之间的间隔超过1分钟,它就会生成(yield)一个不连续点。这个函数的主要假设是数据按时间顺序排列,且时间索引是连续的。

fill_discontinuity(df, symbol, interval, limit=100, max_retries=3) 函数接受一个 DataFrame df,一个表示交易品种的 symbol,一个表示时间间隔的 interval,一个限制填充不连续点次数的 limit,以及一个限制重试次数的 max_retries。它首先使用 find_discontinuities 来找到数据中的不连续点，然后尝试从 Binance API 获取缺失的数据来填充这些不连续点。如果获取数据失败，它会重试，直到达到最大重试次数。如果重试仍然失败，它会跳过这个不连续点。
'''


def find_discontinuities(df):
    prev_time = df.index[0]
    for i in range(1, len(df)):
        curr_time = df.index[i]
        if (curr_time - prev_time) > pd.Timedelta(minutes=1):
            yield prev_time
        prev_time = curr_time


def fill_discontinuity(df, symbol, interval, limit=100, max_retries=3):
    count = 0
    discontinuities = find_discontinuities(df)
    url = "https://fapi.binance.com/fapi/v1/klines"  # Binance Futures API endpoint
    
    for discontinuity in discontinuities:
        if count >= limit:
            print(f"Reached the limit of {limit} fills. Stopping.")
            break
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                startTime = (discontinuity + pd.Timedelta(minutes=1)).floor('T')
                endTime = df.index[df.index > discontinuity][0].floor('T')
                
                print(f"Attempting to fill gap from {startTime} to {endTime}")
                
                new_data_list = []
                current_start = startTime
                
                while current_start < endTime:
                    current_end = min(current_start + dt.timedelta(minutes=1000), endTime)
                    
                    startTime_str = str(int(current_start.timestamp() * 1000))
                    endTime_str = str(int(current_end.timestamp() * 1000))
                    
                    req_params = {"symbol": symbol, 'interval': interval, 'startTime': startTime_str,
                                  'endTime': endTime_str, 'limit': '1000'}
                    
                    response = requests.get(url, params=req_params)
                    response.raise_for_status()
                    data = response.json()
                    
                    print(f"Received {len(data)} data points")
                    
                    if not data:
                        break
                    
                    temp_df = pd.DataFrame(data)
                    temp_df = temp_df.iloc[:, 0:6]
                    temp_df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        temp_df[col] = temp_df[col].astype("float")
                    
                    temp_df['datetime'] = pd.to_datetime(temp_df['datetime'], unit='ms')
                    temp_df.set_index('datetime', inplace=True)
                    
                    new_data_list.append(temp_df)
                    
                    current_start = max(temp_df.index) + dt.timedelta(minutes=1)
                
                if new_data_list:
                    new_data = pd.concat(new_data_list)
                    df = pd.concat([df[df.index <= discontinuity], new_data, df[df.index > endTime]])
                    df = df.sort_index().drop_duplicates()
                    print(f"Successfully filled gap from {startTime} to {endTime}")
                    break
                else:
                    print(f"No data retrieved for gap from {startTime} to {endTime}")
                    retry_count += 1
            
            except Exception as e:
                retry_count += 1
                print(f"Error occurred while filling gap: {str(e)}")
                if retry_count < max_retries:
                    print(f"Retrying... Attempt {retry_count + 1} of {max_retries}")
                    time.sleep(1)
                traceback.print_exc()
        
        if retry_count == max_retries:
            print(f"Skip discontinuity point after {max_retries} retries.")
        
        count += 1
    
    print(f'Finished filling {count} discontinuities.')
    return df


def check_data_format(df):
    if len(df) == len(df[df.index.second == 0]):
        print("all data formats are corrected")
    else:
        print("some data formats are uncorrected")
        return df[df.index.second != 0]


def find_duplicates(df):
    # 选择除 'datetime' 之外的所有列
    columns = [col for col in df.columns if col != 'datetime']

    # 找到重复项
    duplicates = df.duplicated(subset=columns, keep=False)

    # 获取重复行
    duplicate_rows = df[duplicates]

    # 一行一行地打印每个重复行
    for i, row in duplicate_rows.iterrows():
        print(row)
        print("\n")  # 在行之间添加一个空行以便于阅读


# 递归的解决字典内存在datetime格式的key格式
def handle_datetime(obj):
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k.isoformat() if isinstance(k, dt.datetime) else str(k): handle_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [handle_datetime(elem) for elem in obj]
    else:
        return obj

# 读取/dict目录下符合条件的json字典


def merge_dicts(directory):
    merged_dict = {}
    for filename in os.listdir(directory):
        if filename.startswith("my_dict1_") and filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                target = filename.replace("my_dict1_", "").replace(
                    ".json", "").replace("_", " ")
                merged_dict[target] = data
    return merged_dict


def generate_dates(start_date_str: str, time_span: int) -> list:

    # Parse the start date string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Initialize an empty list to hold the dates
    dates_list = []

    # Calculate the end date
    end_date = start_date - timedelta(days=time_span)

    # Generate dates
    current_date = start_date
    while current_date >= end_date:
        # Append the date to the list in the required format
        dates_list.append(current_date.strftime('%Y-%m-%d %H:%M:%S'))

        # Move 3 days back
        current_date -= timedelta(days=3)

    return dates_list[::-1]


if __name__ == '__main__':
    # 示例调用
    symbol = "BTCUSDT"
    interval = "1m"  # 1分钟间隔
    # 设置开始时间和结束时间
    start_time = dt.datetime(2022, 9, 1)
    end_time = dt.datetime.now()

    filepath = 'BTCUSDT_future_1m.csv'  # Replace with the actual file path
    print("Now we are going to update bitcoin data")
    try:
        update_data(filepath=filepath, symbol=symbol, interval=interval)
    except Exception as e:
        print(f"An error occurred during the update process: {e}")
        traceback.print_exc()

    # Load the data
    df = load_data(filepath)

    print("Fill discontinuity in the BTC data")
    df = fill_discontinuity(df, "BTCUSDT", "1m", limit=50)

    # Check the data format
    check_data_format(df)

    # Find duplicates in the data
    find_duplicates(df)

    # Save the processed data to a CSV file
    df.to_csv(filepath)
