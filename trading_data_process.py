#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 13:07:58 2023

@author: lyuhongwang
"""
import datetime as dt
import pandas as pd
import json
import requests
import time
import os

def get_binance_bars(symbol, interval, startTime, endTime):
    url = "https://api.binance.com/api/v3/klines"
    df_list = []
    while True:
        startTime_str = str(int(startTime.timestamp() * 1000))
        print(startTime)
        endTime_str = str(int(endTime.timestamp() * 1000))
        limit = '1000'
        req_params = {"symbol" : symbol, 'interval' : interval, 'startTime' : startTime_str, \
                      'endTime' : endTime_str, 'limit' : limit}
        data = json.loads(requests.get(url, params = req_params).text)
        time.sleep(1)
        print(len(data))
        if len(data) == 0:  # check if data is empty
            break
        df = pd.DataFrame(data)
        df = df.iloc[:, 0:6]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df.open = df.open.astype("float")
        df.high = df.high.astype("float")
        df.low = df.low.astype("float")
        df.close = df.close.astype("float")
        df.volume = df.volume.astype("float")
        df.datetime = [dt.datetime.utcfromtimestamp(x / 1000.0) for x in df.datetime]  # update datetime format
        df.set_index('datetime', inplace=True)  # set datetime as index
        df_list.append(df)
        startTime = max(df.index) + dt.timedelta(0, 5)  # set new startTime as the last datetime of the current data
    return pd.concat(df_list) if df_list else None  # return None if df_list is empty

def update_data(filepath):
    # Parse the symbol and interval from the filepath
    filename = os.path.basename(filepath)
    symbol, interval = filename.split('_')
    interval = interval.split('.')[0]
    # Load existing data
    df_old = pd.read_csv(filepath, sep=',', index_col='datetime', parse_dates=True)
    
    # Get the last datetime in the existing data
    last_datetime = df_old.index[-1]
    
    # Get current datetime
    current_datetime = pd.Timestamp.now(tz='UTC')  # get current datetime as pandas Timestamp
    # Truncate current_datetime to minute precision and remove timezone information
    current_datetime = current_datetime.floor('T').tz_localize(None)
    
    # If the data is not up-to-date, get new data
    if last_datetime < current_datetime:
        df_new = get_binance_bars(symbol, interval, last_datetime.to_pydatetime(), current_datetime.to_pydatetime())
        # Check if df_new is not empty (this means new data was found)
        if not df_new.empty:
            # Append new data to the old data
            df = pd.concat([df_old, df_new])
        else:
            df = df_old
    else:
        df = df_old
    
    # Save the updated data
    df.to_csv(filepath)
    print(f"Finished, now the data is saved in {filepath}")

#update_data('BTCUSDT_1m.csv')

'''
check = get_binance_bars('BTCUSDT', '1m', dt.datetime(2020,1,1), dt.datetime(2023,5,1))
check.to_csv('BTCUSDT_1m.csv')
update_data('BTCUSDT_1m.csv')
'''
#需要处理找不到的情况。
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


'''
row = find_row_by_date(df, '2023-05-01 00:01:00')
print(row)
'''

def load_data(filepath):
    df = pd.read_csv(filepath, sep=',')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('datetime', inplace=True)
    return df

#df = load_data("BTCUSDT_1m.csv")
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
    for discontinuity in discontinuities:
        if count > limit:
            break

        retry_count = 0
        while retry_count < max_retries:
            try:
                startTime = (discontinuity + pd.Timedelta(minutes=1)).replace(second=0)
                endTime = df.index[df.index > discontinuity][0].replace(second=0) - pd.Timedelta(minutes=1)
                new_data = get_binance_bars(symbol, interval, startTime.to_pydatetime(), endTime.to_pydatetime())
                df = pd.concat([df[df.index <= discontinuity], new_data, df[df.index > endTime]])
                break
            except:
                retry_count += 1
                print(f"ConnectionError occurred. Retry {retry_count}. Waiting for 1 second before retrying...")
                time.sleep(1)

        if retry_count == max_retries:
            print(f"Skip discontinuity point after {max_retries} retries.")
        count += 1

    print(f'Finished {count} times.')
    return df



#df = fill_discontinuity(df, "BTCUSDT", "1m",limit=50)
#df.to_csv('BTCUSDT_1m.csv')

def check_data_format(df):
    if len(df) == len(df[df.index.second == 0]):
        print("all data formats are corrected")
    else:
        print("some data formats are uncorrected")
        return df[df.index.second != 0]

#check_data_format(df)

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

#find_duplicates(df)

#递归的解决字典内存在datetime格式的key格式
def handle_datetime(obj):
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k.isoformat() if isinstance(k, dt.datetime) else str(k): handle_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [handle_datetime(elem) for elem in obj]
    else:
        return obj
    
#读取/dict目录下符合条件的json字典
def merge_dicts(directory):
    merged_dict = {}
    for filename in os.listdir(directory):
        if filename.startswith("my_dict1_") and filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                target = filename.replace("my_dict1_", "").replace(".json", "").replace("_", " ")
                merged_dict[target] = data
    return merged_dict

from datetime import datetime, timedelta
'''
    Generate a list of dates, starting from the given date, and going backwards every 3 days, 
    until the given time span is reached.
    
    
    Parameters:
    - start_date_str (str): The start date in string format, like 2023-08-29.
    - time_span (int): The number of days to go back.
    
    Returns:
    - list: A list of dates in YYYY-MM-DD HH:MM:SS format.
'''
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