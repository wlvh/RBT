#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2023-08-08 05:48:15
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-03-19 00:19:53
FilePath: /trading/trading_data_pipeline.py
Description: 
    This script is used to process trading data.
    It uses functions imported from `trading_data_process.py` to perform the following tasks:
        - Update the data
        - Load the data
        - Fill the discontinuity in the data
        - Check the data format
        - Find duplicates in the data
    The processed data is then saved to a CSV file.
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''
# Import necessary functions
from trading_data_process import update_data, load_data, fill_discontinuity, check_data_format, find_duplicates
from extra_info import get_block_ratiodate,PCA_BlockAndBitcoin,fetch_or_update_yf_data
# Define the file path
filepath = '/home/WLH_trade/0615/trading/BTCUSDT_1m.csv'  # Replace with the actual file path

print("Now we are going to update bitcoin data")
update_data(filepath)

# Load the data
df = load_data(filepath)

print("Fill discontinuity in the BTC data")
df = fill_discontinuity(df, "BTCUSDT", "1m", limit=50)

# Check the data format
check_data_format(df)

# Find duplicates in the data
find_duplicates(df)

# Save the processed data to a CSV file
df.to_csv('BTCUSDT_1m.csv')

print("以下数据需要北京时间8点30以后才会更新前日数据")
# 获取区块链信息
get_block_ratiodate(update=True)
print("以30天为时间窗口对BTC和区块链数据进行PCA，也许这个时间窗口可以调整")
PCA_BlockAndBitcoin(update_mode=False)
print("Yfinance数据需要北京时间8点以后才会更新前日数据")
fetch_or_update_yf_data()