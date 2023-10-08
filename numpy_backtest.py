#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2023-08-30 09:13:07
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2023-09-13 03:34:45
FilePath: /trading/numpy_backtest.py
Description: 
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''
import json
from dateutil.parser import parse
import optuna
import optuna.storages as storages
from datetime import datetime, timedelta
import math
import sqlite3
import sys
import numpy as np
import pandas as pd
import time
from decimal import Decimal
import trading_data_process as tdp
from numba import jit
pd.options.mode.chained_assignment = None  # 禁用 SettingWithCopyWarning


def calculate_sma_numpy(arr, window):
    return np.convolve(arr, np.ones(window)/window, mode='valid')


def calculate_vwap(df, period=None): #checked，量价都是使用同样的period参数，合理。
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    weighted_typical_price = typical_price * df['volume']
    # 计算 EMA
    ema_typical_price = weighted_typical_price.ewm(span=period, adjust=False).mean()
    ema_volume = df['volume'].ewm(span=period, adjust=False).mean()
    # 计算 VWAP
    vwap = ema_typical_price / ema_volume
    return vwap

def calculate_ema_numpy(arr, span, alpha=None):
    if alpha == None:
        alpha = 2 / (span + 1)
    ema = np.zeros_like(arr, dtype=np.float64)
    ema[0] = arr[0] if not np.isnan(arr[0]) else 0  # Initialize with the first element if it's not NaN, otherwise use 0
    for t in range(1, len(arr)):
        if np.isnan(arr[t]):
            ema[t] = ema[t-1]  # If current value is NaN, use the last EMA value
        else:
            ema[t] = (1 - alpha) * ema[t-1] + alpha * arr[t]
    return ema

def calculate_vwap_numpy(high, low, close, volume, span):
    typical_price = (high + low + close) / 3
    weighted_typical_price = typical_price * volume
    ema_typical_price = calculate_ema_numpy(weighted_typical_price, span)
    ema_volume = calculate_ema_numpy(volume, span)
    #print("ema_typical_price",ema_typical_price)
    #print("ema_volume",ema_volume)
    vwap = ema_typical_price / ema_volume
    #print("vwap",vwap)
    return vwap

#load_data('BTCUSDT_1m.csv', '2023-08-23 00:00:00', 60)

def calculate_atr(df, period=14):#checked
    # 计算前一天的收盘价
    prev_close = df['close'].shift(1)
    # 计算真实最高价
    true_high = df['high'].where(df['high'] > prev_close, prev_close)
    true_low = df['low'].where(df['low'] < prev_close, prev_close)
    # 计算真实范围
    true_range = true_high - true_low
    # 使用Smoothed Moving Average来计算平均真实范围
    smoothing_factor = 1.0 / period  # 根据SmoothedMovingAverage的定义
    atr = true_range.ewm(alpha=smoothing_factor, adjust=False).mean()
    return atr

def calculate_atr_numpy(high, low, close, period=14):
    # Calculating previous day's close
    prev_close = np.roll(close, shift=1)
    prev_close[0] = np.nan
    # Calculating true high and true low
    true_high = np.maximum(high, prev_close)
    true_low = np.minimum(low, prev_close)
    # Calculating true range
    true_range = true_high - true_low
    # Using Smoothed Moving Average to calculate ATR
    smoothing_factor = 1.0 / period
    atr = calculate_ema_numpy(true_range, period,alpha=smoothing_factor)
    return atr


class TradeLogic:
    def __init__(self, df, index,buy_risk_per_trade,sell_risk_per_trade,buy_stop_loss,sell_stop_loss,buy_take_profit,sell_take_profit):
        self.df = df
        self.index = index
        self.buy_risk_per_trade = buy_risk_per_trade
        self.sell_risk_per_trade = sell_risk_per_trade 
        self.buy_stop_loss = buy_stop_loss
        self.sell_stop_loss = sell_stop_loss
        self.buy_take_profit = buy_take_profit
        self.sell_take_profit = sell_take_profit
        self.initialize_values()

    def initialize_values(self):
        for col_name in ["position", "cash", "open_direction", "open_cost_price", "max_profit", "BTC_debt", "USDT_debt", "free_BTC", "net_value"]:
            self.df[col_name][self.index] = self.df[col_name][self.index - 1]
        self.calculate_net_value()
    
    def calculate_net_value(self):
        index = self.index  # 当前索引
        try:           
            if self.df['position'][index] > 0:
                self.df['net_value'][index] = self.df['cash'][index] + self.df['close'][index] * self.df['position'][index] - self.df['BTC_debt'][index] * self.df['close'][index] - self.df['USDT_debt'][index]
            elif self.df['position'][index] <= 0:
                self.df['net_value'][index] = self.df['cash'][index] + (self.df['free_BTC'][index] - self.df['BTC_debt'][index]) * self.df['close'][index] - self.df['USDT_debt'][index]
            else:
                self.df['net_value'][index] = self.df['cash'][index] - self.df['BTC_debt'][index] * self.df['close'][index] - self.df['USDT_debt'][index]
        except Exception as e:
            print(f"Current index: {index}")
            print(f"the date is: {self.df['datetime'][index]}")
            print('---------------------------------')
            for column in self.df.columns:
                print('column is: ',column)
                print(self.df[column][index-100:index+100])
            raise e
            # break statement should be inside a loop
            
  
    def borrow_Margin(self, coin, how_much_borrow):
        index = self.index
        if coin == 'BTC':
            self.df['BTC_debt'][index] += how_much_borrow
            self.df['free_BTC'][index] += how_much_borrow
        elif coin == 'USDT':
            self.df['USDT_debt'][index] += how_much_borrow
            self.df['cash'][index] += how_much_borrow
        else:
            print("Please input BTC for sell or USDT for buy.")
        self.calculate_net_value()

    def repay_Margin(self):
        index = self.index
        BTC_debt = self.df['BTC_debt'][index]
        USDT_debt = self.df['USDT_debt'][index]
        if BTC_debt > 0:
            self.create_close_order()
            self.df['free_BTC'][index] -= BTC_debt
            self.df['BTC_debt'][index] -= BTC_debt
        elif USDT_debt > 0:
            self.create_close_order()
            self.df['cash'][index] -= USDT_debt
            self.df['USDT_debt'][index] -= USDT_debt
        else:
            #print("No debt to repay.")
            pass
        self.calculate_net_value()

    def calculate_size(self, avg_volume, risk_per_trade):
        self.calculate_net_value()
        index = self.index  # 当前处理的行索引
        account_net_value = self.df['net_value'][index]  # 账户净值
        volume = self.df['volume'][index]  # 当前成交量
        last_close = self.df['close'][index]  # 最后收盘价
        # 检查账户净值是否为负
        if account_net_value <= 0:
            return "Negative value"
        # 计算交易规模
        size = (account_net_value * risk_per_trade * (volume / avg_volume)) / last_close
        # 如果计算出的规模太大，进行调整
        if size * last_close > 2 * account_net_value:
            size = (2 * account_net_value) / last_close
        # 如果计算出的规模太小（小于 11 USDT），进行调整
        if size * last_close < 11:
            size = 11 / last_close
        return size

    def create_buy_order(self, size):
        index = self.index
        size = float(size)
        if size < 0:
            size = abs(size)
        order = self.df['next_open'][index] * size * 1.0005
        self.df['cash'][index] -= order
        #计算成本价，逻辑要写3个：做空时的减仓，做多时的加仓，空仓时做多。
        #做空时减仓有没有可能减到做多，这个需要测试一下。
        pre_position = self.df['position'][index]
        if pre_position < 0:
            pre_total_cost = -pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            extra_cost = size * float(self.df['next_open'][index]) * 1.0005
            self.df['open_cost_price'][index] = (pre_total_cost-extra_cost)/abs(self.df['position'][index])
        elif pre_position > 0:
            pre_total_cost = pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            extra_cost = size * float(self.df['next_open'][index]) * 1.0005 
            self.df['open_cost_price'][index] = (pre_total_cost+extra_cost)/abs(self.df['position'][index])
        else:
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            self.df['open_cost_price'][index] = self.df['next_open'][index] * 1.0005         
        
        self.calculate_net_value()

    def create_sell_order(self, size):
        index = self.index
        size = float(size)
        if size < 0:
            size = abs(size)
        order = self.df['next_open'][index] * size * 0.9995
        self.df['cash'][index] += order
        #计算成本价，逻辑要写3个：做空时的继续做空，做多时的减仓，空仓时做空。
        #做多时减仓有没有可能减到做空，这个需要测试一下。
        pre_position = self.df['position'][index]
        if pre_position < 0:
            pre_total_cost = -pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] -= size
            self.df['open_direction'][index] = "SELL"
            extra_cost = size * float(self.df['next_open'][index]) * 0.9995
            self.df['open_cost_price'][index] = (pre_total_cost+extra_cost)/abs(self.df['position'][index])
        elif pre_position > 0:
            pre_total_cost = pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] -= size
            self.df['open_direction'][index] = "SELL"
            extra_cost = size * float(self.df['next_open'][index]) * 0.9995 
            self.df['open_cost_price'][index] = (pre_total_cost-extra_cost)/abs(self.df['position'][index])
        else:
            self.df['position'][index] -= size
            self.df['open_direction'][index] = "SELL"
            self.df['open_cost_price'][index] = self.df['next_open'][index] * 0.9995 
        self.df['free_BTC'][index] -= size
        self.calculate_net_value()

    def create_close_order(self):
        index = self.index
        my_position = self.df['position'][index]
        
        if my_position > 0:
            self.df['cash'][index] += my_position * self.df['next_open'][index] * 0.9995
            self.df['position'][index] = 0
            self.df['open_direction'][index] = None  # The direction of the open position (Long/Short)
            self.df['open_cost_price'][index] = None  # Cost price of the open position
        elif my_position < 0:
            repay = self.df['next_open'][index] * abs(my_position) * 1.0005
            self.df['free_BTC'][index] += abs(my_position)
            self.df['cash'][index] -= repay
            self.df['position'][index] = 0
            self.df['open_direction'][index] = None  # The direction of the open position (Long/Short)
            self.df['open_cost_price'][index] = None
        self.df['max_profit'][index] = 0
        self.calculate_net_value()
        
    def handle_absolute_stop_orders(self, buy_signal, sell_signal):
        index = self.index
        
        max_profit = self.df['max_profit'][index]
        last_close = self.df['close'][index]
        my_position = self.df['position'][index]

        if my_position > 0:  # Long position
            buy_price = self.df['open_cost_price'][index]
            profit = (last_close - buy_price) / buy_price
            if profit < 0:
                profit = 0
            if profit > max_profit:
                self.df['max_profit'][index] = profit
            elif max_profit - profit >= self.buy_take_profit and not buy_signal:
                self.repay_Margin()
            elif last_close < buy_price * (1.0 - self.buy_stop_loss):
                self.repay_Margin()
                
        elif my_position < 0:  # Short position
            sell_price = self.df['open_cost_price'][index]
            profit = (sell_price - last_close) / sell_price
            if profit < 0:
                profit = 0
            if profit > max_profit:
                self.df['max_profit'][index] = profit
            elif max_profit - profit >= self.sell_take_profit and not sell_signal:
                self.repay_Margin()
            elif last_close > sell_price * (1.0 + self.sell_stop_loss):
                self.repay_Margin()
        
        self.calculate_net_value()

    def execute_trade(self):
        index = self.index
        my_position = self.df['position'][index]
        
        buy_risk_per_trade = self.buy_risk_per_trade
        sell_risk_per_trade = self.sell_risk_per_trade
        
        buy_vwap = self.df['buy_vwap'][index]
        sell_vwap = self.df['sell_vwap'][index]
        
        buy_atr = self.df['buy_atr'][index]
        sell_atr = self.df['sell_atr'][index]

        buy_atr_sma = self.df['buy_atr_sma'][index]
        sell_atr_sma = self.df['sell_atr_sma'][index]
        
        last_close = self.df['close'][index]
        last_volume = self.df['volume'][index]
        pre_volume = self.df['volume'][index-1]
        buy_avg_volume = self.df['buy_avg_volume'][index]
        sell_avg_volume = self.df['sell_avg_volume'][index]

        buy_signal = self.df['buy_signal'][index]
        sell_signal = self.df['sell_signal'][index]
        # 在市场中
        if my_position != 0:
            #更新buy price 和 sell price作为止盈止损的依据，这里的价格是最近一次成交的价格。
            # 买入逻辑
            if buy_signal:
                if my_position > 0:
                    size = self.calculate_size(avg_volume=buy_avg_volume, risk_per_trade=buy_risk_per_trade)
                    try:
                        if size - my_position > 0 and (size-my_position)*last_close > 11:
                            extra_buy = size - my_position
                            #print(f"Detected extra long buy signal, we are going to long extra {extra_buy} BTC.")
                            self.borrow_Margin(coin='USDT', how_much_borrow=extra_buy*last_close)
                            self.create_buy_order(size=extra_buy)
                        else:
                            #print(f"Detected long buy signal, but extra size does not meet minium amount 10 USDT.")
                            pass
                    except:
                        print("error")
                        print(f"size is {size}")
                        print(f"the net value is {self.df['net_value'][index]}")
                        print(f"my_position is {my_position}")
                if my_position < 0:  # 空头仓位
                    # 关闭空头，开始买入
                    self.repay_Margin()
                    self.df['max_profit'][index] = 0
                    size = self.calculate_size(avg_volume=buy_avg_volume, risk_per_trade=buy_risk_per_trade)
                    self.borrow_Margin(coin='USDT', how_much_borrow=size*last_close,)
                    self.create_buy_order(size=size)

            # 卖出逻辑
            elif sell_signal:
                if my_position < 0:
                    size = self.calculate_size(avg_volume=sell_avg_volume, risk_per_trade=sell_risk_per_trade)
                    if my_position + size > 0 and (my_position+size)*last_close > 11:
                        extra_sell = my_position + size
                        #print(f"Detected extra short sell signal, we are going to short extra {extra_sell} BTC.")
                        self.borrow_Margin(coin='BTC', how_much_borrow=extra_sell)
                        self.create_sell_order(size=extra_sell)
                    else:
                        #print(f"Detected short sell signal, but extra size does not meet minium amount 10 USDT.")  
                        pass        
                if my_position > 0:  # 多头仓位
                    # 关闭多头，开始卖出
                    self.repay_Margin()
                    self.df['max_profit'][index] = 0
                    size = self.calculate_size(avg_volume=sell_avg_volume, risk_per_trade=sell_risk_per_trade)
                    self.borrow_Margin(coin='BTC', how_much_borrow=size)
                    self.create_sell_order(size=size)
                
            self.handle_absolute_stop_orders(buy_signal, sell_signal)
        # 不在市场中
        else:
            self.df['max_profit'][index] = 0
            self.df['open_cost_price'][index] = 0
            # 买入逻辑
            if buy_signal:
                size = self.calculate_size(avg_volume=buy_avg_volume, risk_per_trade=buy_risk_per_trade)
                self.borrow_Margin(coin='USDT', how_much_borrow=size*last_close)
                self.create_buy_order(size=size)
            
            # 卖出逻辑
            elif sell_signal:
                size = self.calculate_size(avg_volume=sell_avg_volume, risk_per_trade=sell_risk_per_trade)
                self.borrow_Margin(coin='BTC', how_much_borrow=size)
                self.create_sell_order(size=size)
            # 获取当前的 UTC 时间（东0区时间）
        self.calculate_net_value()
        #return self.df.iloc[index]  # Returning the updated DataFrame

def load_data(filepath, end_date, data_len):
    # Step 1: Load the existing data
    df = pd.read_csv(filepath, sep=',')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('datetime', inplace=True)
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S')
    
    # Step 2: Filter data based on end_date and data_len
    start_date = end_date - timedelta(days=data_len + 3)  # Adding 5 days as per your requirement
    extended_end_date = end_date + timedelta(minutes=1)  # Extending end_date by 1 minute    
    filtered_df = df.loc[start_date:extended_end_date ]      
    if filtered_df.empty:
        print(f"the date of {end_date} is not in the data, we are goting to trunc our data.")
        filtered_df = df.loc[start_date:]
    filtered_df['next_open'] = filtered_df['open'].shift(-1)
    
    # Step 3: Add additional columns that are missing (Populate these columns based on your logic)
    filtered_df['position'] = None  # Position in the trade
    filtered_df['net_value'] = None  # Net value of the trade to be computed based on your logic
    filtered_df['cash'] = None  # Cash available
    filtered_df['open_direction'] = None  # The direction of the open position (Long/Short)
    filtered_df['open_cost_price'] = None  # Cost price of the open position
    filtered_df['max_profit'] = None  # Maximum profit
    filtered_df['BTC_debt'] = None  # BTC debt
    filtered_df['USDT_debt'] = None  # USDT debt
    filtered_df['free_BTC'] = None
    filtered_df['buy_vwap'] = 0
    filtered_df['buy_atr'] = 0
    filtered_df['buy_atr_sma'] = 0
    filtered_df['sell_vwap'] = 0
    filtered_df['sell_atr'] = 0
    filtered_df['sell_atr_sma'] = 0
    return filtered_df

#load_data('BTCUSDT_1m.csv', '2023-08-23 00:00:00', 60)

def calculate_sma_numpy_with_padding(arr, window):
    sma = np.convolve(arr, np.ones(window)/window, mode='valid')
    pad_size = window - 1
    pad = np.full(pad_size, np.nan)
    sma_padded = np.concatenate((pad, sma))
    return sma_padded

# Function to calculate indicators for a specific range in numpy arrays
def calculate_indicators_for_range_inplace(start, end, arrays, buy_vwap_period,buy_atr_period,buy_atr_sma_period,sell_vwap_period,sell_atr_period,sell_atr_sma_period,max_length):
    arrays['buy_vwap'] = arrays['buy_vwap'].astype(float)
    arrays['sell_vwap'] = arrays['buy_vwap'].astype(float)
    arrays['buy_atr'] = arrays['buy_vwap'].astype(float)
    arrays['sell_atr'] = arrays['buy_vwap'].astype(float)
    arrays['buy_atr_sma'] = arrays['buy_vwap'].astype(float)
    arrays['sell_atr_sma'] = arrays['buy_vwap'].astype(float)
    start = start - max_length 
    end = end + 1
        
    arrays['buy_vwap'][start:end] = calculate_vwap_numpy(arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], arrays['volume'][start:end], buy_vwap_period)
    arrays['buy_atr'][start:end] = calculate_atr_numpy(arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], buy_atr_period)
    arrays['buy_atr_sma'][start:end] = calculate_sma_numpy_with_padding(arrays['buy_atr'][start:end], buy_atr_sma_period)
    arrays['sell_vwap'][start:end] = calculate_vwap_numpy(arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], arrays['volume'][start:end], sell_vwap_period)
    arrays['sell_atr'][start:end] = calculate_atr_numpy(arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], sell_atr_period)
    arrays['sell_atr_sma'][start:end] = calculate_sma_numpy_with_padding(arrays['sell_atr'][start:end], sell_atr_sma_period)

if __name__ == "__main__":
    data_all = load_data("BTCUSDT_1m.csv", '2023-08-29 00:00:00', 6)
    
    start_time_total = time.time()
    symbol = 'BTC/USDT'
    last_action = 'free'
    buy_price = None
    sell_price = None
    max_profit = 0
    pre_my_position = 0

    with open('/home/WLH_trade/0615/trading/best_strategy', 'r') as file:
        best_strategy = json.load(file)

    sell_volume_window = best_strategy['1']['sell_volume_window']
    buy_volume_window = best_strategy['1']['buy_volume_window']
    buy_volume_multiplier = best_strategy['1']['buy_volume_multiplier']
    sell_volume_multiplier = best_strategy['1']['sell_volume_multiplier']
    buy_stop_loss = best_strategy['1']['buy_stop_loss']
    sell_stop_loss = best_strategy['1']['sell_stop_loss']
    buy_take_profit = best_strategy['1']['buy_take_profit']
    sell_take_profit = best_strategy['1']['sell_take_profit']
    sell_vwap_period = best_strategy['1']['sell_vwap_period']
    buy_vwap_period = best_strategy['1']['buy_vwap_period']
    buy_risk_per_trade = best_strategy['1']['buy_risk_per_trade']
    sell_risk_per_trade = best_strategy['1']['sell_risk_per_trade']
    buy_atr_period = best_strategy['1']['buy_atr_period']
    sell_atr_period = best_strategy['1']['sell_atr_period']
    #buy_atr_sma_period = best_strategy['1']['buy_atr_sma_period']  
    #sell_atr_sma_period = best_strategy['1']['sell_atr_sma_period']
    buy_atr_sma_period = 5 
    sell_atr_sma_period = 5
    max_length = max(sell_volume_window,buy_volume_window,sell_vwap_period,buy_vwap_period,sell_atr_period,buy_atr_period,sell_atr_sma_period,buy_atr_sma_period) + 1
    print(f"now we used the best strategy: {best_strategy['0']}, and its parameters are: {best_strategy['1']}")
    '''
    # 1. 计算VWAP
    data_all['buy_vwap'] = calculate_vwap(data_all, period=buy_vwap_period)
    data_all['sell_vwap'] = calculate_vwap(data_all, period=sell_vwap_period)

    # 2. 计算ATR
    data_all['buy_atr'] = calculate_atr(data_all, period=buy_atr_period)
    data_all['sell_atr'] = calculate_atr(data_all, period=sell_atr_period)

    # 3. 计算ATR的SMA（Simple Moving Average）
    data_all['buy_atr_sma'] = data_all['buy_atr'].rolling(window=buy_atr_sma_period ).mean()
    data_all['sell_atr_sma'] = data_all['sell_atr'].rolling(window=sell_atr_sma_period ).mean()

    # 4. 计算平均交易量
    data_all['buy_avg_volume'] = data_all['volume'].rolling(window=buy_volume_window).mean()
    data_all['sell_avg_volume'] = data_all['volume'].rolling(window=sell_volume_window).mean()
    # 预先计算买入和卖出条件
    data_all['buy_condition1'] = data_all['close'] > data_all['buy_vwap']
    data_all['buy_condition2'] = data_all['volume'].shift(1) > data_all['buy_avg_volume'] * buy_volume_multiplier * (data_all['buy_atr'] / data_all['buy_atr_sma'])
    data_all['buy_condition3'] = data_all['volume'] > data_all['buy_avg_volume'] * buy_volume_multiplier * (data_all['buy_atr'] / data_all['buy_atr_sma'])
    data_all['buy_signal'] = data_all['buy_condition1'] & (data_all['buy_condition2'] | data_all['buy_condition3'])

    data_all['sell_condition1'] = data_all['close'] < data_all['sell_vwap']
    data_all['sell_condition2'] = data_all['volume'].shift(1) > data_all['sell_avg_volume'] * sell_volume_multiplier * (data_all['sell_atr'] / data_all['sell_atr_sma'])
    data_all['sell_condition3'] = data_all['volume'] > data_all['sell_avg_volume'] * sell_volume_multiplier * (data_all['sell_atr'] / data_all['sell_atr_sma'])
    data_all['sell_signal'] = data_all['sell_condition1'] & (data_all['sell_condition2'] | data_all['sell_condition3'])
    '''
    # 创建一个字典来存储转换后的NumPy数组
    numpy_arrays = {}
    numpy_arrays['datetime'] = data_all.index.strftime('%Y-%m-%d %H:%M:%S').to_numpy()

    # 遍历DataFrame的每一列，并将其转换为NumPy数组
    for column in data_all.columns:
        numpy_arrays[column] = data_all[column].to_numpy()

    # Find the index of the row where datetime is '2023-08-23 00:00:00'
    # Note: We first convert the 'datetime' numpy array to a list and then find the index of the target date.
    target_date = '2023-08-23 00:00:00'
    row_at_date = np.where(numpy_arrays['datetime'] == target_date)[0][0]  # Getting the first occurrence

    target_date = pd.to_datetime(target_date, format='%Y-%m-%d %H:%M:%S')
    end_location = datetime.strftime(target_date + timedelta(days=6), '%Y-%m-%d %H:%M:%S')
    end_idx = np.where(numpy_arrays['datetime'] == str(end_location) )[0][0] 

    calculate_indicators_for_range_inplace(start=row_at_date, end=end_idx, arrays=numpy_arrays, 
                                        buy_vwap_period=buy_vwap_period,buy_atr_period=buy_atr_period,
                                        buy_atr_sma_period=buy_atr_sma_period,
                                        sell_vwap_period=sell_vwap_period,sell_atr_period=sell_atr_period,
                                        sell_atr_sma_period=sell_atr_sma_period,
                                        max_length=max_length)

    numpy_arrays['buy_avg_volume'] = calculate_sma_numpy_with_padding(numpy_arrays['volume'], buy_volume_window)
    numpy_arrays['sell_avg_volume'] = calculate_sma_numpy_with_padding(numpy_arrays['volume'], sell_volume_window)

    # Re-calculating the conditions and signals
    numpy_arrays['buy_condition1'] = numpy_arrays['close'] > numpy_arrays['buy_vwap']
    numpy_arrays['buy_condition2'] = np.roll(numpy_arrays['volume'], shift=1) > numpy_arrays['buy_avg_volume'] * buy_volume_multiplier * (numpy_arrays['buy_atr'] / numpy_arrays['buy_atr_sma'])
    numpy_arrays['buy_condition3'] = numpy_arrays['volume'] > numpy_arrays['buy_avg_volume'] * buy_volume_multiplier * (numpy_arrays['buy_atr'] / numpy_arrays['buy_atr_sma'])
    numpy_arrays['buy_signal'] = numpy_arrays['buy_condition1'] & (numpy_arrays['buy_condition2'] | numpy_arrays['buy_condition3'])

    numpy_arrays['sell_condition1'] = numpy_arrays['close'] < numpy_arrays['sell_vwap']
    numpy_arrays['sell_condition2'] = np.roll(numpy_arrays['volume'], shift=1) > numpy_arrays['sell_avg_volume'] * sell_volume_multiplier * (numpy_arrays['sell_atr'] / numpy_arrays['sell_atr_sma'])
    numpy_arrays['sell_condition3'] = numpy_arrays['volume'] > numpy_arrays['sell_avg_volume'] * sell_volume_multiplier * (numpy_arrays['sell_atr'] / numpy_arrays['sell_atr_sma'])
    numpy_arrays['sell_signal'] = numpy_arrays['sell_condition1'] & (numpy_arrays['sell_condition2'] | numpy_arrays['sell_condition3'])


    # Set the initial values for the relevant columns at the index (row_at_date - 1)
    initial_values = {
        'cash': 10000,
        'position': 0,
        'open_direction': 0,
        'open_cost_price': 0,
        'max_profit': 0,
        'BTC_debt': 0,
        'USDT_debt': 0,
        'net_value': 10000,
        'free_BTC': 0
    }

    # Iterating through the dictionary keys to set the initial values
    for column, initial_value in initial_values.items():
        numpy_arrays[column][row_at_date - 1] = initial_value


    start_time = time.time()

    # 用NumPy数组的长度替换Pandas DataFrame的长度
    for start_idx in range(row_at_date, end_idx+1):  # Replace 'some_column' with an actual column name
        #print(numpy_arrays['datetime'][start_idx])
        # 传递NumPy数组而不是DataFrame
        trade = TradeLogic(
            df=numpy_arrays, 
            index=start_idx, 
            buy_risk_per_trade=buy_risk_per_trade, 
            sell_risk_per_trade=sell_risk_per_trade,
            buy_stop_loss = buy_stop_loss,
            sell_stop_loss = sell_stop_loss,
            buy_take_profit = buy_take_profit,
            sell_take_profit = sell_take_profit)  
        trade.execute_trade()  # This modifies numpy_arrays in-place
        
    final_data = pd.DataFrame.from_dict({key: pd.Series(value) for key, value in numpy_arrays.items()})
    final_data.to_csv('numpy_for_backtest_0903_np.csv', index=False)



    print(numpy_arrays['net_value'][end_idx])
    print('Time taken for total: ', time.time() - start_time_total, ' seconds')
    print('Time taken for loop: ', time.time() - start_time, ' seconds')