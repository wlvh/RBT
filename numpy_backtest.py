#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2023-08-30 09:13:07
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-06-17 05:29:20
FilePath: /trading/numpy_backtest.py
Description: 
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''
import copy
from typing import Tuple, List
import trading_data_process as tdp
from decimal import Decimal
import time
import pandas as pd
import numpy as np
import sys
import sqlite3
import math
from datetime import datetime, timedelta
import optuna.storages as storages
import optuna
from dateutil.parser import parse
import json
import os
# 动态设置环境变量，限制numpy和pandas使用单线程
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
pd.options.mode.chained_assignment = None  # 禁用 SettingWithCopyWarning


def MaxDrawdown(numpy_arrays: dict, start_date: str, end_date: str, rolling_window: int):
    start_time = time.time()

    # 将日期和净值转换为 Pandas DataFrame
    df = pd.DataFrame({'datetime': pd.to_datetime(numpy_arrays['datetime']),
                       'net_value': numpy_arrays['net_value']})

    # 设置日期为索引
    df.set_index('datetime', inplace=True)

    # 将数据重采样为每小时数据（例如取每小时最后一个值）
    hourly_df = df.resample('H').last()
    hourly_df.fillna(method='ffill', inplace=True)
    # 过滤指定日期范围内的数据
    hourly_df = hourly_df.loc[start_date:end_date]

    # 计算每个滚动窗口的最大回撤
    max_drawdowns = []
    rolling_window_hours = rolling_window * 24  # 将天数转换为小时
    if len(hourly_df) < rolling_window_hours:
        window_values = hourly_df['net_value'].dropna().values
        if len(window_values) > 1:
            rolling_max = np.maximum.accumulate(window_values)
            drawdown = rolling_max - window_values
            max_drawdown = np.max(drawdown) / rolling_max[np.argmax(drawdown)]
            max_drawdowns.append(max_drawdown)
    else:
        for start_idx in range(len(hourly_df) - rolling_window_hours + 1):
            end_idx = start_idx + rolling_window_hours
            window_values = hourly_df['net_value'].iloc[start_idx:end_idx].values

            # 找到窗口内的峰值点
            peak_idx = np.argmax(window_values)

            # 从峰值点开始计算最大回撤
            if peak_idx < len(window_values) - 1:  # 确保峰值不是在窗口的最后一点
                post_peak_values = window_values[peak_idx:]
                rolling_max = np.maximum.accumulate(post_peak_values)
                drawdown = rolling_max - post_peak_values
                max_drawdown = np.max(drawdown) / \
                    rolling_max[np.argmax(drawdown)]
            else:
                max_drawdown = 0

            max_drawdowns.append(max_drawdown)

    cost_time = time.time() - start_time
    # print(f"max drawdowns cost time is {cost_time}")
    return max(max_drawdowns)


def calculate_trades_win_rate_and_sqn_from_dict(numpy_arrays: dict, start_date: str, end_date: str):
    start_time = time.time()
    start_date = pd.to_datetime(start_date, format="%Y-%m-%d %H:%M:%S")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(numpy_arrays)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    df.iloc[-1, df.columns.get_loc('trade_end_flag')] = 1

    # Identify trade start and end rows
    start_rows = df[df['trade_start_flag'] == 1]
    end_rows = df[df['trade_end_flag'] == 1].reset_index(drop=True)

    trade_results = []
    trade_details = []
    for start, end in zip(start_rows.iterrows(), end_rows.iterrows()):
        start_index, start_row = start
        end_index, end_row = end

        # Calculate trade profit/loss
        profit = end_row['net_value'] - start_row['net_value']
        # profit *= 1 if start_row['open_direction'] == 'BUY' else -1
        trade_results.append(profit)

        # Trade details
        trade_details.append({
            'start_time': start_row['datetime'],
            'end_time': end_row['datetime'],
            'profit': profit,
            'open_cost_price': start_row['open_cost_price']
        })

    # Trade statistics
    total_trades = len(trade_results)
    winning_trades = len([p for p in trade_results if p > 0])
    win_rate = winning_trades / total_trades if total_trades else 0
    mean_R = np.mean(trade_results) if total_trades else 0
    std_dev_R = np.std(trade_results) if total_trades else 0
    sqn = (mean_R / std_dev_R) * np.sqrt(total_trades) if std_dev_R != 0 else 0

    # 计算持仓变化
    df['position_change'] = df['position'].diff()

    # 交易换手率计算
    buy_volume = df['position_change'][df['position_change'] > 0]
    sell_volume = df['position_change'][df['position_change'] < 0]

    total_purchases_value = sum(buy_volume * df['close'][buy_volume.index])
    total_sales_value = sum(-sell_volume * df['close'][sell_volume.index])

    # 使用资产的最大值作为基准
    max_asset_value = df['net_value'].max()

    turnover_rate = (total_purchases_value + total_sales_value) / \
        max_asset_value if max_asset_value else 0
    cost_time = time.time()-start_time
    print(f"trades calculation cost time is {cost_time}")
    return total_trades, win_rate, sqn, turnover_rate, trade_results, trade_details

# Note: To use this function, you need to pass the appropriate dictionary structure as the numpy_arrays argument.
# The dictionary should contain 'datetime', 'position', and 'net_value' keys with associated numpy array values.

def calculate_sma_numpy(arr, window):
    return np.convolve(arr, np.ones(window)/window, mode='valid')


def calculate_rsi_numpy(prices, factor, window=14):
    if prices.size == 0:
        raise ValueError("The input array 'prices' is empty.")
    # 计算价格变化
    window = window * factor
    window = round(window)
    delta = prices[factor:] - prices[:-factor]
    if len(delta) == len(prices):
        print(f"delta.shape: {delta.shape}, factor: {factor}")
        print(f"the first factor line of delta is {delta[:factor]}")
        print("going to cut the first factor line of delta")
        delta = delta[factor:]
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(window=window).mean().to_numpy()
    # 计算RS，并抑制除零警告
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = np.full(len(prices), np.nan)
    # 打印调试信息
    rsi[factor:] = 100 - (100 / (1 + rs))
    return rsi


def calculate_weighted_rsi_numpy(prices, volumes, factor, window=14):
    start_time = time.time()
    if prices.size == 0 or volumes.size == 0:
        raise ValueError("The input array 'prices' or 'volumes' is empty.")
    if prices.size != volumes.size:
        raise ValueError(
            "The input arrays 'prices' and 'volumes' must have the same size.")
    # 计算价格变化和交易量变化
    print(f"Going to calculate wris, factor is {factor}, window is {window}")
    window = window * factor
    window = round(window)
    delta_time = time.time()
    delta = prices[factor:] - prices[:-factor]
    delta_volumes = volumes[factor:]
    if len(delta) == len(prices):
        delta = delta[factor:]
        delta_volumes = delta_volumes[factor:]
    # print(f"delta calculation took {time.time()-delta_time} seconds")
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # 计算加权平均收益和损失
    avg_time = time.time()
    # print("Going to calculate avg_loss and avg_gain")
    # avg_gain = np.convolve(gain * delta_volumes, np.ones(window),
    #                        'valid') / np.convolve(delta_volumes, np.ones(window), 'valid')
    # avg_loss = np.convolve(loss * delta_volumes, np.ones(window),
    #                        'valid') / np.convolve(delta_volumes, np.ones(window), 'valid')
    gain_series = pd.Series(gain * delta_volumes)
    loss_series = pd.Series(loss * delta_volumes)
    volume_series = pd.Series(delta_volumes)
    avg_gain = gain_series.rolling(window=window).mean().to_numpy(
    ) / volume_series.rolling(window=window).mean().to_numpy()
    avg_loss = loss_series.rolling(window=window).mean().to_numpy(
    ) / volume_series.rolling(window=window).mean().to_numpy()
    # print(f"avg_loss and avg gain calculation took {time.time()-avg_time} seconds")
    # 计算RS，并抑制除零警告
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = np.full(len(prices), np.nan)
    # rsi[factor + window - 1:] = 100 - (100 / (1 + rs))
    rsi[factor:] = 100 - (100 / (1 + rs))
    cost_time = time.time() - start_time
    print(f"WRSI calculation took {cost_time} seconds")
    return rsi


def calculate_ma_numpy(prices, ma_size, factor=None):
    start_time = time.time()
    if prices.size == 0:
        raise ValueError("The input array 'prices' is empty.")
    if factor != None:
        ma_size = ma_size * factor
    ma_size = round(ma_size)
    # # 创建一个与窗口大小相同的权重数组，所有元素都是1
    # weights = np.ones(ma_size) / ma_size

    # # 使用卷积计算移动平均
    # ma = np.convolve(prices, weights, 'valid')

    # # 为了保持数组长度一致，我们可以在结果前面填充NaN
    # ma_full = np.full_like(prices, np.nan, dtype=np.float64)
    # ma_full[ma_size-1:] = ma    
    prices_series = pd.Series(prices)
    ma = prices_series.rolling(window=ma_size).mean().to_numpy()
    # To keep the array length consistent, fill the front with NaN
    ma_full = np.full_like(prices, np.nan, dtype=np.float64)
    ma_full[ma_size-1:] = ma[ma_size-1:]
    #print(f"ma calculation took {time.time()-start_time} seconds")
    return ma_full


def calculate_vwap(df, period=None):  # checked，量价都是使用同样的period参数，合理。
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    weighted_typical_price = typical_price * df['volume']
    # 计算 EMA
    ema_typical_price = weighted_typical_price.ewm(
        span=period, adjust=False).mean()
    ema_volume = df['volume'].ewm(span=period, adjust=False).mean()
    # 计算 VWAP
    vwap = ema_typical_price / ema_volume
    return vwap


def calculate_ema_numpy(arr, span, alpha=None):
    if alpha == None:
        alpha = 2 / (span + 1)
    else:
        alpha = alpha / (span+1)
    if arr.size == 0:
        raise ValueError("The input array 'arr' is empty.")
    ema = np.zeros_like(arr, dtype=np.float64)
    # Initialize with the first element if it's not NaN, otherwise use 0
    ema[0] = arr[0] if not np.isnan(arr[0]) else 0
    for t in range(1, len(arr)):
        if np.isnan(arr[t]):
            ema[t] = ema[t-1]  # If current value is NaN, use the last EMA value
        else:
            ema[t] = (1 - alpha) * ema[t-1] + alpha * arr[t]
    return ema


def calculate_vwap_numpy(high, low, close, volume, span):
    # 验证输入数组是否为空
    if high.size == 0 or low.size == 0 or close.size == 0 or volume.size == 0:
        raise ValueError(
            f"One or more input arrays are empty. Check {high}, {low}, {close}, and {volume}")

    typical_price = (high + low + close) / 3
    weighted_typical_price = typical_price * volume

    # 在调用 calculate_ema_numpy 前，确保 weighted_typical_price 和 volume 非空
    if weighted_typical_price.size == 0:
        raise ValueError("The 'weighted_typical_price' array is empty.")

    if volume.size == 0:
        raise ValueError("The 'volume' array is empty.")

    ema_typical_price = calculate_ema_numpy(weighted_typical_price, span)
    ema_volume = calculate_ema_numpy(volume, span)

    vwap = ema_typical_price / ema_volume
    return vwap


def calculate_atr(df, period=14):  # checked
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
    atr = calculate_ema_numpy(true_range, period, alpha=smoothing_factor)
    return atr


class HedgeLogic:
    def __init__(self, df, index, buy_risk_per_trade, sell_risk_per_trade, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, com=0.0005):
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer.")

        self.df = df
        self.index = index
        self.buy_risk_per_trade = buy_risk_per_trade
        self.sell_risk_per_trade = sell_risk_per_trade
        self.buy_stop_loss = buy_stop_loss
        self.sell_stop_loss = sell_stop_loss
        self.buy_take_profit = buy_take_profit
        self.sell_take_profit = sell_take_profit
        self.com = com
        self.initialize_values()

    def initialize_values(self):
        ini_index = max(self.index, 1)

        for col_name in ["position", "cash", "open_direction", "open_cost_price", "max_profit", "BTC_debt", "USDT_debt", "free_BTC", "net_value"]:
            self.df[col_name][ini_index] = self.df[col_name][ini_index - 1]
        self.calculate_net_value()

    def calculate_net_value(self):
        index = self.index  # 当前索引
        try:
            if self.df['position'][index] > 0:
                self.df['net_value'][index] = self.df['cash'][index] + self.df['close'][index] * self.df['position'][index] - \
                    self.df['BTC_debt'][index] * \
                    self.df['close'][index] - self.df['USDT_debt'][index]
            elif self.df['position'][index] <= 0:
                self.df['net_value'][index] = self.df['cash'][index] + (
                    self.df['free_BTC'][index] - self.df['BTC_debt'][index]) * self.df['close'][index] - self.df['USDT_debt'][index]
            else:
                self.df['net_value'][index] = self.df['cash'][index] - \
                    self.df['BTC_debt'][index] * \
                    self.df['close'][index] - self.df['USDT_debt'][index]
        except Exception as e:
            print(f"Current index: {index}")
            print(f"the date is: {self.df['datetime'][index]}")
            print('---------------------------------')
            for column in self.df.keys():
                print('column is: ', column)
                print(self.df[column][index-100:index+100])
            raise e
            # break statement should be inside a loop

    def get_current_dataframe(self):
        """
        返回当前修改过的 DataFrame。
        """
        return self.df

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
            # print("No debt to repay.")
            pass
        self.calculate_net_value()

    def execute_hedge(self, signal, size):
        if signal not in ['short', 'long']:
            raise ValueError("signal must be 'short' or 'long'.")
        if not isinstance(size, (int, float)) or size <= 0:
            raise ValueError("size must be a positive number.")

        hedge_size = self.calculate_hedge_size(signal, size)
        if signal == 'short':
            self.create_sell_order(hedge_size)
        elif signal == 'long':
            self.create_buy_order(hedge_size)
        else:
            print(
                "Please input short for hedge short position or long for hedge long position.")
            raise ValueError

    def calculate_hedge_size(self, signal, size):
        index = self.index  # 当前处理的行索引
        account_net_value = self.df['net_value'][index]  # 账户净值
        last_close = self.df['close'][index]  # 最后收盘价
        if size * last_close > 2 * account_net_value:
            size = (2 * account_net_value) / last_close
        # 如果计算出的规模太小（小于 11 USDT），进行调整
        if size * last_close < 11:
            size = 11 / last_close
        # 这里的signal指的是需要对冲时开仓的方向
        # 我们可以设计多套对冲方案，等量对冲，1-risk_per_trade对冲等。
        if signal == 'short':
            hedge_size = max(1, (self.sell_risk_per_trade /
                             self.buy_risk_per_trade)) * size
        elif signal == 'long':
            hedge_size = max(1, (self.buy_risk_per_trade /
                             self.sell_risk_per_trade)) * size
        else:
            print(
                "Please input short for hedge short position or long for hedge long position.")
            raise ValueError
        return hedge_size

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
            raise ValueError
        self.calculate_net_value()

    def create_buy_order(self, size):

        index = self.index
        last_close = self.df['close'][index]
        size = float(size)
        self.borrow_Margin(coin='USDT', how_much_borrow=size*last_close)

        if size < 0:
            size = abs(size)
        order = self.df['next_open'][index] * size * 1.0005
        self.df['cash'][index] -= order
        # 计算成本价，逻辑要写3个：做空时的减仓，做多时的加仓，空仓时做多。
        # 做空时减仓有没有可能减到做多，这个需要测试一下。
        pre_position = self.df['position'][index]
        if pre_position < 0:
            pre_total_cost = -pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            extra_cost = size * float(self.df['next_open'][index]) * 1.0005
            self.df['open_cost_price'][index] = (
                pre_total_cost-extra_cost)/abs(self.df['position'][index])
        elif pre_position > 0:
            pre_total_cost = pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            extra_cost = size * float(self.df['next_open'][index]) * 1.0005
            self.df['open_cost_price'][index] = (
                pre_total_cost+extra_cost)/abs(self.df['position'][index])
        else:
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            self.df['open_cost_price'][index] = self.df['next_open'][index] * 1.0005

        self.calculate_net_value()

    def create_sell_order(self, size):
        index = self.index
        size = float(size)
        self.borrow_Margin(coin='BTC', how_much_borrow=size)

        if size < 0:
            size = abs(size)
        order = self.df['next_open'][index] * size * 0.9995
        self.df['cash'][index] += order
        # 计算成本价，逻辑要写3个：做空时的继续做空，做多时的减仓，空仓时做空。
        # 做多时减仓有没有可能减到做空，这个需要测试一下。
        pre_position = self.df['position'][index]
        try:
            if pre_position < 0:
                pre_total_cost = -pre_position * \
                    self.df['open_cost_price'][index]
                self.df['position'][index] -= size
                self.df['open_direction'][index] = "SELL"
                extra_cost = size * float(self.df['next_open'][index]) * 0.9995
                if self.df['position'][index] == 0:
                    self.df['open_cost_price'][index] = None
                else:
                    self.df['open_cost_price'][index] = (
                        pre_total_cost+extra_cost)/abs(self.df['position'][index])
            elif pre_position > 0:
                pre_total_cost = pre_position * \
                    self.df['open_cost_price'][index]
                self.df['position'][index] -= size
                self.df['open_direction'][index] = "SELL"
                extra_cost = size * float(self.df['next_open'][index]) * 0.9995
                if self.df['position'][index] == 0:
                    self.df['open_cost_price'][index] = None
                else:
                    self.df['open_cost_price'][index] = (
                        pre_total_cost-extra_cost)/abs(self.df['position'][index])
            else:
                self.df['position'][index] -= size
                self.df['open_direction'][index] = "SELL"
                self.df['open_cost_price'][index] = self.df['next_open'][index] * 0.9995
        except:
            print(
                f"index is {index}, pre_position is {pre_position}, size is {size},position is {self.df['position'][index]}")
            raise
        self.df['free_BTC'][index] -= size
        self.calculate_net_value()

    def create_close_order(self):
        index = self.index
        my_position = self.df['position'][index]

        if my_position > 0:
            self.df['cash'][index] += my_position * \
                self.df['next_open'][index] * 0.9995
            self.df['position'][index] = 0
            # The direction of the open position (Long/Short)
            self.df['open_direction'][index] = None
            # Cost price of the open position
            self.df['open_cost_price'][index] = None
        elif my_position < 0:
            repay = self.df['next_open'][index] * abs(my_position) * 1.0005
            self.df['free_BTC'][index] += abs(my_position)
            self.df['cash'][index] -= repay
            self.df['position'][index] = 0
            # The direction of the open position (Long/Short)
            self.df['open_direction'][index] = None
            self.df['open_cost_price'][index] = None
        self.df['max_profit'][index] = 0
        self.calculate_net_value()

    def update_values(self, index, max_profit=None, open_cost_price=None, trade_start_flag=None, trade_end_flag=None):
        if max_profit is not None:
            self.df['max_profit'][index] = max_profit
        if open_cost_price is not None:
            self.df['open_cost_price'][index] = open_cost_price
        if trade_start_flag is not None:
            self.df['trade_start_flag'][index] = trade_start_flag
        if trade_end_flag is not None:
            self.df['trade_end_flag'][index] = trade_end_flag

    def handle_hedge_stop_orders(self, buy_signal, sell_signal):
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
                self.df['trade_end_flag'][index] = 1
            elif last_close < buy_price * (1.0 - self.buy_stop_loss):
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1

        elif my_position < 0:  # Short position
            sell_price = self.df['open_cost_price'][index]
            if sell_price == None:
                print(
                    f"self.df['datetime'][index] is {self.df['datetime'][index]}")
            profit = (sell_price - last_close) / sell_price
            if profit < 0:
                profit = 0
            if profit > max_profit:
                self.df['max_profit'][index] = profit
            elif max_profit - profit >= self.sell_take_profit and not sell_signal:
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1
            elif last_close > sell_price * (1.0 + self.sell_stop_loss):
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1
        self.calculate_net_value()


class TradeLogic:
    def __init__(self, df, index, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, hedge_function=False, com=0.0005, hedge_df=None, strategy_name="", params={}):
        self.df = df
        self.index = index
        self.buy_stop_loss = buy_stop_loss
        self.sell_stop_loss = sell_stop_loss
        self.buy_take_profit = buy_take_profit
        self.sell_take_profit = sell_take_profit
        self.strategy_name = strategy_name
        if strategy_name == "VWAP_Strategy":
            self.buy_risk_per_trade = params['buy_risk_per_trade']
            self.sell_risk_per_trade = params['sell_risk_per_trade']
        elif strategy_name == "RSIV_Strategy" or strategy_name == "WRSIStrategy":
            self.buy_rsi_threshold = params['buy_rsi_threshold']
            self.sell_rsi_threshold = params['sell_rsi_threshold']
            self.buy_Apoint = params['buy_Apoint']
            self.buy_Bpoint = params['buy_Bpoint']
            self.sell_Apoint = params['sell_Apoint']
            self.sell_Bpoint = params['sell_Bpoint']
            self.open_percent = params['open_percent']
        elif strategy_name == "MACDStrategy":
            pass
        elif strategy_name == 'MAVStrategy':
            pass
        else:
            raise ValueError("Please input the correct strategy name.")

        # 需要加入策略名字
        if hedge_function and hedge_df:
            self.hedge_logic = HedgeLogic(df=hedge_df, index=index, buy_risk_per_trade=buy_risk_per_trade, sell_risk_per_trade=sell_risk_per_trade,
                                          buy_stop_loss=buy_stop_loss, sell_stop_loss=sell_stop_loss, buy_take_profit=buy_take_profit, sell_take_profit=sell_take_profit)
        else:
            self.hedge_logic = None
        self.com = com
        self.initialize_values()

    def initialize_values(self):
        ini_index = self.index
        if ini_index == 0:
            ini_index = 1
        for col_name in ["position", "cash", "open_direction", "open_cost_price", "max_profit", "BTC_debt", "USDT_debt", "free_BTC", "net_value"]:
            self.df[col_name][ini_index] = self.df[col_name][ini_index - 1]
        self.calculate_net_value()

    def calculate_net_value(self):
        index = self.index  # 当前索引
        try:
            if self.df['position'][index] > 0:
                self.df['net_value'][index] = self.df['cash'][index] + self.df['close'][index] * self.df['position'][index] - \
                    self.df['BTC_debt'][index] * \
                    self.df['close'][index] - self.df['USDT_debt'][index]
            elif self.df['position'][index] <= 0:
                self.df['net_value'][index] = self.df['cash'][index] + (
                    self.df['free_BTC'][index] - self.df['BTC_debt'][index]) * self.df['close'][index] - self.df['USDT_debt'][index]
            else:
                self.df['net_value'][index] = self.df['cash'][index] - \
                    self.df['BTC_debt'][index] * \
                    self.df['close'][index] - self.df['USDT_debt'][index]
        except Exception as e:
            print(f"Current index: {index}")
            print(f"the date is: {self.df['datetime'][index]}")
            print('---------------------------------')
            for column in self.df.keys():
                print('column is: ', column)
                print(self.df[column][index-10:index+10])
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
            raise ValueError
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
            # print("No debt to repay.")
            pass
        self.calculate_net_value()

    def calculate_size(self, signal=None):
        self.calculate_net_value()
        index = self.index  # 当前处理的行索引
        account_net_value = self.df['net_value'][index]  # 账户净值
        volume = self.df['volume'][index]  # 当前成交量
        last_close = self.df['close'][index]  # 最后收盘价
        # 检查账户净值是否为负
        if account_net_value <= 0:
            raise ValueError(
                "Account net value is negative. Please check your account balance.")
        # 使用 getattr 动态获取策略方法，并传递参数字典。这种方式可以根据不同策略传递不同的参数，而不需要在函数内部写多个 if 判断。
        strategy_method = getattr(
            self, f"{self.strategy_name.lower()}_size_calculation", None)
        # 每种策略都有自己的计算方法，命名方式为 strategy_name_size_calculation。这使得每种策略的逻辑都被封装在各自的方法中。
        size = strategy_method(account_net_value=account_net_value,
                               last_close=last_close, volume=volume, index=index, signal=signal)
        # 如果计算出的规模太大，进行调整
        if size * last_close > 2 * account_net_value:
            size = (2 * account_net_value) / last_close
        # 如果计算出的规模太小（小于 11 USDT），进行调整
        if size * last_close < 11:
            size = 11 / last_close
        return size

    def vwap_strategy_size_calculation(self, account_net_value, last_close, volume, index, signal):
        if signal == 'buy':
            avg_volume = self.df['buy_avg_volume'][index]
            risk_per_trade = self.buy_risk_per_trade
        elif signal == 'sell':
            avg_volume = self.df['sell_avg_volume'][index]
            risk_per_trade = self.sell_risk_per_trade
        return (account_net_value * risk_per_trade * (volume / avg_volume)) / last_close

    def rsiv_strategy_size_calculation(self, account_net_value, last_close, volume, index, signal):
        rsi = self.df['rsi'][index]
        if signal == 'buy':
            ori_size = account_net_value * self.open_percent / last_close
            if rsi < self.buy_rsi_threshold - self.buy_Apoint:
                ori_size = account_net_value * \
                    ((1-self.open_percent)/2+self.open_percent) / last_close
            if rsi < self.buy_rsi_threshold - self.buy_Apoint - self.buy_Bpoint:
                ori_size = account_net_value / last_close
        elif signal == 'sell':
            ori_size = account_net_value * self.open_percent / last_close
            if rsi > self.sell_rsi_threshold + self.sell_Apoint:
                ori_size = account_net_value * \
                    ((1-self.open_percent)/2+self.open_percent) / last_close
            if rsi > self.sell_rsi_threshold + self.sell_Apoint + self.sell_Bpoint:
                ori_size = account_net_value / last_close
        return 2 * ori_size

    def wrsistrategy_size_calculation(self, *args, **kwargs):
        return self.rsiv_strategy_size_calculation(*args, **kwargs)

    def macdstrategy_size_calculation(self, account_net_value, last_close, volume, index, signal):
        if signal == 'buy':
            ori_size = account_net_value / last_close
        elif signal == 'sell':
            ori_size = account_net_value / last_close
        return 2 * ori_size

    def mavstrategy_size_calculation(self, account_net_value, last_close, volume, index, signal):
        if signal == 'buy':
            ori_size = account_net_value / last_close
        elif signal == 'sell':
            ori_size = account_net_value / last_close
        return 2 * ori_size

    def create_buy_order(self, size):
        index = self.index
        size = float(size)
        if size < 0:
            size = abs(size)
        order = self.df['next_open'][index] * size * 1.0005
        self.df['cash'][index] -= order
        # 计算成本价，逻辑要写3个：做空时的减仓，做多时的加仓，空仓时做多。
        # 做空时减仓有没有可能减到做多，这个需要测试一下。
        pre_position = self.df['position'][index]
        if pre_position < 0:
            pre_total_cost = -pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            extra_cost = size * float(self.df['next_open'][index]) * 1.0005
            self.df['open_cost_price'][index] = (
                pre_total_cost-extra_cost)/abs(self.df['position'][index])
        elif pre_position > 0:
            pre_total_cost = pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] += size
            self.df['open_direction'][index] = "BUY"
            extra_cost = size * float(self.df['next_open'][index]) * 1.0005
            self.df['open_cost_price'][index] = (
                pre_total_cost+extra_cost)/abs(self.df['position'][index])
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
        # 计算成本价，逻辑要写3个：做空时的继续做空，做多时的减仓，空仓时做空。
        # 做多时减仓有没有可能减到做空，这个需要测试一下。
        pre_position = self.df['position'][index]
        if pre_position < 0:
            pre_total_cost = -pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] -= size
            self.df['open_direction'][index] = "SELL"
            extra_cost = size * float(self.df['next_open'][index]) * 0.9995
            self.df['open_cost_price'][index] = (
                pre_total_cost+extra_cost)/abs(self.df['position'][index])
        elif pre_position > 0:
            pre_total_cost = pre_position * self.df['open_cost_price'][index]
            self.df['position'][index] -= size
            self.df['open_direction'][index] = "SELL"
            extra_cost = size * float(self.df['next_open'][index]) * 0.9995
            self.df['open_cost_price'][index] = (
                pre_total_cost-extra_cost)/abs(self.df['position'][index])
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
            self.df['cash'][index] += my_position * \
                self.df['next_open'][index] * 0.9995
            self.df['position'][index] = 0
            # The direction of the open position (Long/Short)
            self.df['open_direction'][index] = None
            # Cost price of the open position
            self.df['open_cost_price'][index] = None
        elif my_position < 0:
            repay = self.df['next_open'][index] * abs(my_position) * 1.0005
            self.df['free_BTC'][index] += abs(my_position)
            self.df['cash'][index] -= repay
            self.df['position'][index] = 0
            # The direction of the open position (Long/Short)
            self.df['open_direction'][index] = None
            self.df['open_cost_price'][index] = None
        self.df['max_profit'][index] = 0
        self.calculate_net_value()

    def handle_absolute_stop_orders(self, buy_signal, sell_signal):
        index = self.index
        sell_all_flag = False
        max_profit = self.df['max_profit'][index]
        last_close = self.df['close'][index]
        my_position = self.df['position'][index]

        if self.strategy_name == 'MACDStrategy':
            if self.df['close_signal'][index] == True:
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1
                sell_all_flag = True
                self.calculate_net_value()
                return sell_all_flag

        if my_position > 0:  # Long position
            buy_price = self.df['open_cost_price'][index]
            if buy_price == None:
                print("Value of my_position: ", my_position)
                print("Type of my_position: ", type(my_position))
                print(f"index is {index}")
                for key, value in self.df.items():
                    print(key)
                    print(self.df[key][index])
                    print(self.df[key][index-10:index+10])
            profit = (last_close - buy_price) / buy_price
            if profit < 0:
                profit = 0
            if profit > max_profit:
                self.df['max_profit'][index] = profit
            elif max_profit - profit >= self.buy_take_profit and not buy_signal:
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1
                sell_all_flag = True
            elif last_close < buy_price * (1.0 - self.buy_stop_loss):
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1
                sell_all_flag = True

        elif my_position < 0:  # Short position
            sell_price = self.df['open_cost_price'][index]
            if sell_price == None:
                print("Value of my_position: ", my_position)
                print("Type of my_position: ", type(my_position))
                print(f"index is {index}")
                for key, value in self.df.items():
                    print(key)
                    print(self.df[key][index])
                    print(self.df[key][index-10:index+10])
            profit = (sell_price - last_close) / sell_price
            if profit < 0:
                profit = 0
            if profit > max_profit:
                self.df['max_profit'][index] = profit
            elif max_profit - profit >= self.sell_take_profit and not sell_signal:
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1
                sell_all_flag = True
            elif last_close > sell_price * (1.0 + self.sell_stop_loss):
                self.repay_Margin()
                self.df['trade_end_flag'][index] = 1
                sell_all_flag = True

        self.calculate_net_value()
        return sell_all_flag

    def execute_trade(self):
        index = self.index
        my_position = self.df['position'][index]
        last_close = self.df['close'][index]
        buy_signal = self.df['buy_signal'][index]
        sell_signal = self.df['sell_signal'][index]
        # 在市场中
        if my_position != 0:
            # 更新buy price 和 sell price作为止盈止损的依据，这里的价格是最近一次成交的价格。
            # 买入逻辑
            if buy_signal:
                if my_position > 0:
                    size = self.calculate_size(signal='buy')
                    try:
                        # 10美元是binance的最小交易单位
                        if size - my_position > 0 and (size-my_position)*last_close > 11:
                            extra_buy = size - my_position
                            # print(f"Detected extra long buy signal, we are going to long extra {extra_buy} BTC.")
                            # print(f"extry_buy is {extra_buy}, last_close is {last_close}")
                            self.borrow_Margin(
                                coin='USDT', how_much_borrow=extra_buy*last_close)
                            self.create_buy_order(size=extra_buy)
                            if self.hedge_logic:
                                self.hedge_logic.execute_hedge(
                                    signal='short', size=size)
                        else:
                            # print(f"Detected long buy signal, but extra size does not meet minium amount 10 USDT.")
                            pass
                    except:
                        print("error")
                        print(f"size is {size}")
                        print(
                            f"the net value is {self.df['net_value'][index]}")
                        print(f"my_position is {my_position}")
                if my_position < 0:  # 空头仓位
                    # 关闭空头，开始买入
                    self.repay_Margin()
                    self.df['trade_end_flag'][index] = 1
                    self.df['max_profit'][index] = 0
                    size = self.calculate_size(signal='buy')
                    self.borrow_Margin(
                        coin='USDT', how_much_borrow=size*last_close,)
                    self.create_buy_order(size=size)
                    self.df['trade_start_flag'][index] = 1
                    if self.hedge_logic:
                        self.hedge_logic.repay_Margin()
                        self.hedge_logic.update_values(
                            index, max_profit=0, trade_start_flag=1, trade_end_flag=1)
                        self.hedge_logic.execute_hedge(
                            signal='short', size=size)

            # 卖出逻辑
            elif sell_signal:
                if my_position < 0:
                    size = self.calculate_size(signal='sell')
                    if my_position + size > 0 and (my_position+size)*last_close > 11:
                        extra_sell = my_position + size
                        # print(f"Detected extra short sell signal, we are going to short extra {extra_sell} BTC.")
                        self.borrow_Margin(
                            coin='BTC', how_much_borrow=extra_sell)
                        self.create_sell_order(size=extra_sell)
                        if self.hedge_logic:
                            self.hedge_logic.execute_hedge(
                                signal='long', size=size)
                    else:
                        # print(f"Detected short sell signal, but extra size does not meet minium amount 10 USDT.")
                        pass
                if my_position > 0:  # 多头仓位
                    # 关闭多头，开始卖出
                    self.repay_Margin()
                    self.df['trade_end_flag'][index] = 1
                    self.df['max_profit'][index] = 0
                    size = self.calculate_size(signal='sell')
                    self.borrow_Margin(coin='BTC', how_much_borrow=size)
                    self.create_sell_order(size=size)
                    self.df['trade_start_flag'][index] = 1
                    if self.hedge_logic:
                        self.hedge_logic.repay_Margin()
                        self.hedge_logic.update_values(
                            index, max_profit=0, trade_start_flag=1, trade_end_flag=1)
                        self.hedge_logic.execute_hedge(
                            signal='long', size=size)

            sell_all_flag = self.handle_absolute_stop_orders(
                buy_signal, sell_signal)
            if self.hedge_logic:
                self.hedge_logic.handle_hedge_stop_orders(
                    buy_signal, sell_signal)
                if sell_all_flag == True:
                    self.hedge_logic.repay_Margin()
                    self.hedge_logic.update_values(
                        index, max_profit=0, trade_start_flag=0, trade_end_flag=1)

        # 不在市场中
        else:
            self.df['max_profit'][index] = 0
            self.df['open_cost_price'][index] = 0
            # 买入逻辑
            if buy_signal:
                size = self.calculate_size(signal='buy')
                print(f"buy size is {size}, last_close is {last_close}")
                self.borrow_Margin(
                    coin='USDT', how_much_borrow=size*last_close)
                self.create_buy_order(size=size)
                self.df['trade_start_flag'][index] = 1
                if self.hedge_logic:
                    self.hedge_logic.update_values(
                        index, max_profit=0, open_cost_price=0, trade_start_flag=1)
                    self.hedge_logic.execute_hedge(signal='short', size=size)
            # 卖出逻辑
            elif sell_signal:
                size = self.calculate_size(signal='sell')
                print(f"sell size is {size}, last_close is {last_close}")
                self.borrow_Margin(coin='BTC', how_much_borrow=size)
                self.create_sell_order(size=size)
                self.df['trade_start_flag'][index] = 1
                if self.hedge_logic:
                    self.hedge_logic.update_values(
                        index, max_profit=0, open_cost_price=0, trade_start_flag=1)
                    self.hedge_logic.execute_hedge(signal='long', size=size)
            # 获取当前的 UTC 时间（东0区时间）
        self.calculate_net_value()


def load_data(filepath, end_date, data_len):
    # Step 1: Load the existing data
    df = pd.read_csv(filepath, sep=',')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('datetime', inplace=True)
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S')

    # Step 2: Filter data based on end_date and data_len
    # Adding 5 days as per your requirement
    start_date = end_date - timedelta(days=data_len + 30)
    # Extending end_date by 1 minute
    extended_end_date = end_date + timedelta(minutes=1)
    filtered_df = df.loc[start_date:extended_end_date]
    if filtered_df.empty:
        print(
            f"the date of {end_date} is not in the data, we are goting to trunc our data.")
        filtered_df = df.loc[start_date:]
    filtered_df['next_open'] = filtered_df['open'].shift(-1)

    # Step 3: Add additional columns that are missing (Populate these columns based on your logic)
    filtered_df['position'] = None  # Position in the trade
    # Net value of the trade to be computed based on your logic
    filtered_df['net_value'] = None
    filtered_df['cash'] = None  # Cash available
    # The direction of the open position (Long/Short)
    filtered_df['open_direction'] = None
    filtered_df['open_cost_price'] = None  # Cost price of the open position
    filtered_df['max_profit'] = None  # Maximum profit
    filtered_df['BTC_debt'] = None  # BTC debt
    filtered_df['USDT_debt'] = None  # USDT debt
    filtered_df['free_BTC'] = None
    filtered_df['trade_start_flag'] = None
    filtered_df['trade_end_flag'] = None
    filtered_df['buy_vwap'] = 0
    filtered_df['buy_atr'] = 0
    filtered_df['buy_atr_sma'] = 0
    filtered_df['sell_vwap'] = 0
    filtered_df['sell_atr'] = 0
    filtered_df['sell_atr_sma'] = 0
    filtered_df['rsi'] = 0
    filtered_df['ma_short'] = 0
    filtered_df['ma_long'] = 0
    filtered_df['buy_avg_volume'] = 0
    filtered_df['sell_avg_volume'] = 0
    print(f"the length of the data is {len(filtered_df)}")
    # print filtered_df 的datetime的min和max
    print(f"the min of the filtered_df is {filtered_df.index.min()}")
    print(f"the max of the filtered_df is {filtered_df.index.max()}")
    return filtered_df


def load_data_bydate(df, end_date, start_date, init_columns=None):
    # Step 1: Load the existing data
    # df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    # df.set_index('datetime', inplace=True)
    if init_columns is None:
        init_columns = {}
    end_date = pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S')
    # Step 2: Filter data based on end_date and data_len
    # Adding 5 days as per your requirement
    start_date = pd.to_datetime(start_date, format='%Y-%m-%d %H:%M:%S')
    start_date = start_date - timedelta(days=35)
    # Extending end_date by 1 minute
    extended_end_date = end_date + timedelta(minutes=1)
    filtered_df = df.loc[start_date:extended_end_date]
    if filtered_df.empty:
        print(
            f"the date of {end_date} is not in the data, we are goting to trunc our data.")
        filtered_df = df.loc[start_date:]
    filtered_df['next_open'] = filtered_df['open'].shift(-1)

    # Step 3: Add additional columns that are missing (Populate these columns based on your logic)
    filtered_df['position'] = None  # Position in the trade
    # Net value of the trade to be computed based on your logic
    filtered_df['net_value'] = None
    filtered_df['cash'] = None  # Cash available
    # The direction of the open position (Long/Short)
    filtered_df['open_direction'] = None
    filtered_df['open_cost_price'] = None  # Cost price of the open position
    filtered_df['max_profit'] = None  # Maximum profit
    filtered_df['BTC_debt'] = None  # BTC debt
    filtered_df['USDT_debt'] = None  # USDT debt
    filtered_df['free_BTC'] = None
    filtered_df['trade_start_flag'] = None
    filtered_df['trade_end_flag'] = None
    # Initialize additional specified columns
    for column, value in init_columns.items():
        filtered_df[column] = value
        # if column not in df.columns:
        #     filtered_df[column] = value
        # else:
        #     raise KeyError(f"Column '{column}' already exists in the DataFrame.")
    return filtered_df


def calculate_sma_numpy_with_padding(arr, window):
    sma = np.convolve(arr, np.ones(window)/window, mode='valid')
    pad_size = window - 1
    pad = np.full(pad_size, np.nan)
    sma_padded = np.concatenate((pad, sma))
    return sma_padded

# Function to calculate indicators for a specific range in numpy arrays


def calculate_indicators_for_range_inplace(start, end, arrays, buy_vwap_period, buy_atr_period, buy_atr_sma_period, sell_vwap_period, sell_atr_period=None, sell_atr_sma_period=None, max_length=None):
    arrays['buy_vwap'] = arrays['buy_vwap'].astype(float)
    arrays['sell_vwap'] = arrays['buy_vwap'].astype(float)
    arrays['buy_atr'] = arrays['buy_vwap'].astype(float)
    arrays['sell_atr'] = arrays['buy_vwap'].astype(float)
    arrays['buy_atr_sma'] = arrays['buy_vwap'].astype(float)
    arrays['sell_atr_sma'] = arrays['buy_vwap'].astype(float)
    start = start - max_length
    end = end + 1
    # 为了避免因为索引越界而导致的错误，我们需要确保 start 和 end 有效
    if start >= end or start < 0 or end > len(arrays['high']):
        len_arrays = len(arrays['high'])
        raise ValueError(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")

    # 在计算指标之前检查数组是否有足够的数据
    for key in ['high', 'low', 'close', 'volume']:
        if end > len(arrays[key]):
            raise ValueError(
                f"The data array for '{key}' does not have enough data to calculate indicators from {start} to {end}.")

    arrays['buy_vwap'][start:end] = calculate_vwap_numpy(
        arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], arrays['volume'][start:end], buy_vwap_period)
    arrays['buy_atr'][start:end] = calculate_atr_numpy(
        arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], buy_atr_period)
    arrays['buy_atr_sma'][start:end] = calculate_sma_numpy_with_padding(
        arrays['buy_atr'][start:end], buy_atr_sma_period)
    arrays['sell_vwap'][start:end] = calculate_vwap_numpy(
        arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], arrays['volume'][start:end], sell_vwap_period)
    arrays['sell_atr'][start:end] = calculate_atr_numpy(
        arrays['high'][start:end], arrays['low'][start:end], arrays['close'][start:end], sell_atr_period)
    arrays['sell_atr_sma'][start:end] = calculate_sma_numpy_with_padding(
        arrays['sell_atr'][start:end], sell_atr_sma_period)


def initialize_column_from_None(arrays, column_name):
    """
    Helper function to initialize a column in the array if it does not exist.
    """
    if column_name not in arrays:
        arrays[column_name] = np.zeros(len(arrays['close']), dtype=float)


def calculate_indicators_by_date(start, end, arrays, buy_vwap_period, buy_atr_period, buy_atr_sma_period, sell_vwap_period, sell_atr_period, sell_atr_sma_period, buy_volume_window, sell_volume_window, max_length, is_live=False):
    start_time = time.time()
    #  arrays 是一个包含 NumPy 数组的字典
    initialize_column_from_None(arrays, 'buy_vwap')
    initialize_column_from_None(arrays, 'sell_vwap')
    initialize_column_from_None(arrays, 'buy_atr')
    initialize_column_from_None(arrays, 'sell_atr')
    initialize_column_from_None(arrays, 'buy_atr_sma')
    initialize_column_from_None(arrays, 'sell_atr_sma')
    initialize_column_from_None(arrays, 'buy_avg_volume')
    initialize_column_from_None(arrays, 'sell_avg_volume')
    arrays['buy_vwap'] = arrays['buy_vwap'].astype(float)
    arrays['sell_vwap'] = arrays['buy_vwap'].astype(float)
    arrays['buy_atr'] = arrays['buy_vwap'].astype(float)
    arrays['sell_atr'] = arrays['buy_vwap'].astype(float)
    arrays['buy_atr_sma'] = arrays['buy_vwap'].astype(float)
    arrays['sell_atr_sma'] = arrays['buy_vwap'].astype(float)
    arrays['buy_avg_volume'] = arrays['volume'].astype(float)
    arrays['sell_avg_volume'] = arrays['volume'].astype(float)
    ori_start = start
    start = start - max_length
    ori_end = end
    end = end + 1
    # 为了避免因为索引越界而导致的错误，我们需要确保 start 和 end 有效
    if start >= end or start < 0 or end > len(arrays['high']):
        len_arrays = len(arrays['high'])
        print(
            f"Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")
        raise ValueError(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")

    # 在计算指标之前检查数组是否有足够的数据
    for key in ['high', 'low', 'close', 'volume']:
        if end > len(arrays[key]):
            raise ValueError(
                f"The data array for '{key}' does not have enough data to calculate indicators from {start} to {end}.")

    if buy_atr_sma_period == None:
        print(f"deteced none value for atr_sam_period, set them as same as atr period")
        buy_atr_sma_period = buy_atr_period
    if sell_atr_sma_period == None:
        sell_atr_sma_period = sell_atr_period
    # 实盘的话全量计算，回测的话只计算需要的部分
    if is_live:
        relevant_high = arrays['high']
        relevant_low = arrays['low']
        relevant_close = arrays['close']
        relevant_volume = arrays['volume']
    else:
        relevant_high = arrays['high'][start:end]
        relevant_low = arrays['low'][start:end]
        relevant_close = arrays['close'][start:end]
        relevant_volume = arrays['volume'][start:end]

    buy_vwap_slice = calculate_vwap_numpy(
        high=relevant_high, low=relevant_low, close=relevant_close, volume=relevant_volume, span=buy_vwap_period)
    buy_atr_slice = calculate_atr_numpy(
        high=relevant_high, low=relevant_low, close=relevant_close, period=buy_atr_period)
    buy_atr_sma_slice = calculate_sma_numpy_with_padding(
        arr=buy_atr_slice, window=buy_atr_sma_period)
    sell_vwap_slice = calculate_vwap_numpy(
        high=relevant_high, low=relevant_low, close=relevant_close, volume=relevant_volume, span=sell_vwap_period)
    sell_atr_slice = calculate_atr_numpy(
        high=relevant_high, low=relevant_low, close=relevant_close, period=sell_atr_period)
    sell_atr_sma_slice = calculate_sma_numpy_with_padding(
        arr=sell_atr_slice, window=sell_atr_sma_period)
    buy_avg_volume_slice = calculate_sma_numpy_with_padding(
        arr=relevant_volume, window=buy_volume_window)
    sell_avg_volume_slice = calculate_sma_numpy_with_padding(
        arr=relevant_volume, window=sell_volume_window)

    # Check for missing values in the calculated slices within the required range
    slices = {
        'Buy VWAP': buy_vwap_slice,
        'Buy ATR': buy_atr_slice,
        'Buy ATR SMA': buy_atr_sma_slice,
        'Sell VWAP': sell_vwap_slice,
        'Sell ATR': sell_atr_slice,
        'Sell ATR SMA': sell_atr_sma_slice,
        'Buy Avg Volume': buy_avg_volume_slice,
        'Sell Avg Volume': sell_avg_volume_slice
    }
    start_idx = ori_start - start
    end_idx = ori_end - start + 1

    if not is_live:
        for name, slice_array in slices.items():
            if np.isnan(slice_array[start_idx:end_idx]).any():
                print(
                    f"Missing values found in {name} slice within the range {ori_start} to {ori_end}.")
                print(
                    f"Calculated {name} slice: {slice_array[start_idx:end_idx]}")
                print(f"The first 100 lines of data are {slice_array[:100]}")
                print(f"The last 100 lines of data are {slice_array[-100:]}")
                raise ValueError(
                    f"Missing values detected in {name} calculation.")
        # Assign the calculated values back to the original arrays within the original range
        arrays['buy_vwap'][ori_start:ori_end +
                           1] = buy_vwap_slice[start_idx:end_idx]
        arrays['buy_atr'][ori_start:ori_end +
                          1] = buy_atr_slice[start_idx:end_idx]
        arrays['buy_atr_sma'][ori_start:ori_end +
                              1] = buy_atr_sma_slice[start_idx:end_idx]
        arrays['sell_vwap'][ori_start:ori_end +
                            1] = sell_vwap_slice[start_idx:end_idx]
        arrays['sell_atr'][ori_start:ori_end +
                           1] = sell_atr_slice[start_idx:end_idx]
        arrays['sell_atr_sma'][ori_start:ori_end +
                               1] = sell_atr_sma_slice[start_idx:end_idx]
        arrays['buy_avg_volume'][ori_start:ori_end +
                                 1] = buy_avg_volume_slice[start_idx:end_idx]
        arrays['sell_avg_volume'][ori_start:ori_end +
                                  1] = sell_avg_volume_slice[start_idx:end_idx]
    else:
        arrays['buy_vwap'] = buy_vwap_slice
        arrays['buy_atr'] = buy_atr_slice
        arrays['buy_atr_sma'] = buy_atr_sma_slice
        arrays['sell_vwap'] = sell_vwap_slice
        arrays['sell_atr'] = sell_atr_slice
        arrays['sell_atr_sma'] = sell_atr_sma_slice
        arrays['buy_avg_volume'] = buy_avg_volume_slice
        arrays['sell_avg_volume'] = sell_avg_volume_slice

    cost_time = time.time() - start_time
    print(f"Tech indicator calculation took {cost_time} seconds")
    return arrays


def calculate_RSIindicators_by_date(start, end, arrays, rsi_size, ma_long, ma_short, buy_avg_volume, sell_avg_volume, strategy_name, max_length, factor=None, is_live=False):
    start_time = time.time()
    initialize_column_from_None(arrays, 'rsi')
    initialize_column_from_None(arrays, 'ma_long')
    initialize_column_from_None(arrays, 'buy_atr')
    initialize_column_from_None(arrays, 'ma_short')
    initialize_column_from_None(arrays, 'buy_avg_volume')
    initialize_column_from_None(arrays, 'sell_avg_volume')
    arrays['rsi'] = arrays['rsi'].astype(float)
    arrays['ma_long'] = arrays['ma_long'].astype(float)
    arrays['ma_short'] = arrays['ma_short'].astype(float)
    arrays['buy_avg_volume'] = arrays['buy_avg_volume'].astype(float)
    arrays['sell_avg_volume'] = arrays['sell_avg_volume'].astype(float)
    ori_start = start
    start = max(start - max_length - 20000, 0)
    ori_end = end
    end = end + 1
    # 为了避免因为索引越界而导致的错误，我们需要确保 start 和 end 有效
    if start >= end or start < 0 or end > len(arrays['high']):
        len_arrays = len(arrays['high'])
        print(
            f"start:{start},end:{end},ori_start:{ori_start},ori_end:{ori_end},len_arrays:{len_arrays}")
        print(
            f"the ori_start date is {arrays['datetime'][ori_start]}, the ori_end date is {arrays['datetime'][ori_end]}")
        print(
            f"the first date of arrays is {arrays['datetime'][0]}, the last date of arrays is {arrays['datetime'][-1]}")
        print(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")
        raise ValueError(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")

    # 在计算指标之前检查数组是否有足够的数据
    for key in ['high', 'low', 'close', 'volume']:
        if end > len(arrays[key]):
            raise ValueError(
                f"The data array for '{key}' does not have enough data to calculate indicators from {start} to {end}.")
    if "Fourhour" in strategy_name and factor == None:
        factor = 60 * 2
    elif "Oneday" in strategy_name and factor == None:
        factor = 60 * 12
    elif "WRSI" in strategy_name and factor:
        # factor取整数
        factor = round(factor)
    else:
        raise ValueError(
            f"the strategy name {strategy_name} is not supported or factor {factor}.")
    print(
        f"RSI calculation will based on RSI window {rsi_size}, and factor {factor}")
    # 实盘的话全量计算，回测的话只计算需要的部分
    if is_live:
        relevant_close = arrays['close']
        relevant_volume = arrays['volume']
    else:
        relevant_close = arrays['close'][start:end]
        relevant_volume = arrays['volume'][start:end]
    if "WRSI" in strategy_name:
        rsi_slice = calculate_weighted_rsi_numpy(
            prices=relevant_close, volumes=relevant_volume, window=rsi_size, factor=factor)
    elif "RSIV" in strategy_name:
        rsi_slice = calculate_rsi_numpy(
            prices=relevant_close, window=rsi_size, factor=factor)
    else:
        raise ValueError(
            f"the strategy name {strategy_name} is not supported.")
    ma_long_slice = calculate_ma_numpy(
        prices=relevant_close, ma_size=ma_long, factor=factor)
    ma_short_slice = calculate_ma_numpy(
        prices=relevant_close, ma_size=ma_short, factor=factor)
    buy_avg_volume_slice = calculate_ma_numpy(
        prices=relevant_volume, ma_size=buy_avg_volume)
    sell_avg_volume_slice = calculate_ma_numpy(
        prices=relevant_volume, ma_size=sell_avg_volume)

    # Check for missing values in the calculated slices within the required range
    slices = {
        'RSI': rsi_slice,
        'MA Long': ma_long_slice,
        'MA Short': ma_short_slice,
        'Buy Avg Volume': buy_avg_volume_slice,
        'Sell Avg Volume': sell_avg_volume_slice
    }
    start_idx = ori_start - start
    end_idx = ori_end - start + 1

    if not is_live:
        for name, slice_array in slices.items():
            if np.isnan(slice_array[start_idx:end_idx]).any():
                print(
                    f"Missing values found in {name} slice within the range {ori_start} to {ori_end}.")
                print(
                    f"Calculated {name} slice: {slice_array[start_idx:end_idx]}")
                print(f"the first 100 line data is {slice_array[:100]}")
                print(f"the last 100 line data is {slice_array[-100:]}")
                raise ValueError(
                    f"Missing values detected in {name} calculation.")
        # ori_start = 1000, ori_end = 2000, max_lenth = 200
        # start = 800,end=2001
        # relevant_close = arrays['close'][start:end] = 1200
        # start_idx = ori_start - start = 200
        # end_idx = ori_end - start + 1 = 2001 - 800 + 1 = 1200
        arrays['rsi'][ori_start:ori_end + 1] = rsi_slice[start_idx:end_idx]
        arrays['ma_long'][ori_start:ori_end +
                          1] = ma_long_slice[start_idx:end_idx]
        arrays['ma_short'][ori_start:ori_end +
                           1] = ma_short_slice[start_idx:end_idx]
        arrays['buy_avg_volume'][ori_start:ori_end +
                                 1] = buy_avg_volume_slice[start_idx:end_idx]
        arrays['sell_avg_volume'][ori_start:ori_end +
                                  1] = sell_avg_volume_slice[start_idx:end_idx]
    else:
        arrays['rsi'] = rsi_slice
        arrays['ma_long'] = ma_long_slice
        arrays['ma_short'] = ma_short_slice
        arrays['buy_avg_volume'] = buy_avg_volume_slice
        arrays['sell_avg_volume'] = sell_avg_volume_slice

    cost_time = time.time() - start_time
    print(f"Tech indicator calculation took {cost_time} seconds")
    return arrays


def calculate_macd_numpy(data, short_period=12, long_period=26, signal_period=9):
    ema_short = calculate_ema_numpy(data, short_period)
    ema_long = calculate_ema_numpy(data, long_period)
    macd_line = ema_short - ema_long
    signal_line = calculate_ema_numpy(macd_line, signal_period)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def calculate_MAVindicators_by_date(start, end, arrays, short_period, long_period, buy_short_volume, sell_short_volume, buy_long_volume, sell_long_volume, strategy_name, max_length, is_live=False):
    start_time = time.time()
    # factor是一个系数，用于构建天级别macd指标
    initialize_column_from_None(arrays, 'short_ma')
    initialize_column_from_None(arrays, 'long_ma')
    initialize_column_from_None(arrays, 'buy_short_volume')
    initialize_column_from_None(arrays, 'sell_short_volume')
    initialize_column_from_None(arrays, 'buy_long_volume')
    initialize_column_from_None(arrays, 'sell_long_volume')
    arrays['short_ma'] = arrays['short_ma'].astype(float)
    arrays['long_ma'] = arrays['long_ma'].astype(float)
    arrays['buy_short_volume'] = arrays['buy_short_volume'].astype(float)
    arrays['sell_short_volume'] = arrays['sell_short_volume'].astype(float)
    arrays['buy_long_volume'] = arrays['buy_long_volume'].astype(float)
    arrays['sell_long_volume'] = arrays['sell_long_volume'].astype(float)
    ori_start = start
    start = start - max_length - 10000
    ori_end = end
    end = end + 1
    # 为了避免因为索引越界而导致的错误，我们需要确保 start 和 end 有效
    if start >= end or start < 0 or end > len(arrays['high']):
        len_arrays = len(arrays['high'])
        print(
            f"start:{start},end:{end},ori_start:{ori_start},ori_end:{ori_end},len_arrays:{len_arrays}")
        print(
            f"the ori_start date is {arrays['datetime'][ori_start]}, the ori_end date is {arrays['datetime'][ori_end]}")
        print(
            f"the first date of arrays is {arrays['datetime'][0]}, the last date of arrays is {arrays['datetime'][-1]}")
        print(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")
        raise ValueError(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")

    # 在计算指标之前检查数组是否有足够的数据
    for key in ['high', 'low', 'close', 'volume']:
        if end > len(arrays[key]):
            raise ValueError(
                f"The data array for '{key}' does not have enough data to calculate indicators from {start} to {end}.")

    # 实盘的话全量计算，回测的话只计算需要的部分
    if is_live:
        relevant_close = arrays['close']
        relevant_volume = arrays['volume']
    else:
        relevant_close = arrays['close'][start:end]
        relevant_volume = arrays['volume'][start:end]

    short_ma = calculate_ma_numpy(
        prices=relevant_close, ma_size=short_period, factor=None)
    long_ma = calculate_ma_numpy(
        prices=relevant_close, ma_size=long_period, factor=None)
    buy_short_volume = calculate_ma_numpy(
        prices=relevant_volume, ma_size=buy_short_volume, factor=None)
    sell_short_volume = calculate_ma_numpy(
        prices=relevant_volume, ma_size=sell_short_volume, factor=None)
    buy_long_volume = calculate_ma_numpy(
        prices=relevant_volume, ma_size=buy_long_volume, factor=None)
    sell_long_volume = calculate_ma_numpy(
        prices=relevant_volume, ma_size=sell_long_volume, factor=None)

    # Check for missing values in the calculated slices within the required range
    slices = {
        'short_ma': short_ma,
        'long_ma': long_ma,
        'buy_short_volume': buy_short_volume,
        'sell_short_volume': sell_short_volume,
        'buy_long_volume': buy_long_volume,
        'sell_long_volume': sell_long_volume,
    }
    start_idx = ori_start - start
    end_idx = ori_end - start + 1

    if not is_live:
        for name, slice_array in slices.items():
            if np.isnan(slice_array[start_idx:end_idx]).any():
                print(
                    f"Missing values found in {name} slice within the range {ori_start} to {ori_end}.")
                print(
                    f"Calculated {name} slice: {slice_array[start_idx:end_idx]}")
                print(
                    f"the first 100 line data is {slice_array[start_idx:start_idx+100]}")
                print(
                    f"the last 100 line data is {slice_array[end_idx-100:end_idx]}")
                raise ValueError(
                    f"Missing values detected in {name} calculation.")
        # ori_start = 1000, ori_end = 2000, max_lenth = 200
        # start = 800,end=2001
        # relevant_close = arrays['close'][start:end] = 1200
        # start_idx = ori_start - start = 200
        # end_idx = ori_end - start + 1 = 2001 - 800 + 1 = 1200
        arrays['short_ma'][ori_start:ori_end + 1] = short_ma[start_idx:end_idx]
        arrays['long_ma'][ori_start:ori_end + 1] = long_ma[start_idx:end_idx]
        arrays['buy_short_volume'][ori_start:ori_end +
                                   1] = buy_short_volume[start_idx:end_idx]
        arrays['sell_short_volume'][ori_start:ori_end +
                                    1] = sell_short_volume[start_idx:end_idx]
        arrays['buy_long_volume'][ori_start:ori_end +
                                  1] = buy_long_volume[start_idx:end_idx]
        arrays['sell_long_volume'][ori_start:ori_end +
                                   1] = sell_long_volume[start_idx:end_idx]
    else:
        arrays['short_ma'] = short_ma
        arrays['long_ma'] = long_ma
        arrays['buy_short_volume'] = buy_short_volume
        arrays['sell_short_volume'] = sell_short_volume
        arrays['buy_long_volume'] = buy_long_volume
        arrays['sell_long_volume'] = sell_long_volume
    cost_time = time.time() - start_time
    print(f"Tech indicator calculation took {cost_time} seconds")
    return arrays


def calculate_MACDindicators_by_date(start, end, arrays, short_period, long_period, signal_period, L_short_period, L_long_period, L_signal_period, factor, alpha_factor, buy_avg_volume, sell_avg_volume, strategy_name, max_length, is_live=False):
    start_time = time.time()
    # factor是一个系数，用于构建天级别macd指标
    factor_short_period = factor * L_short_period
    factor_long_period = factor * L_long_period
    factor_signal_period = factor * L_signal_period
    # factor_signal_period = L_signal_period
    initialize_column_from_None(arrays, 'macd_line')
    initialize_column_from_None(arrays, 'signal_line')
    initialize_column_from_None(arrays, 'factor_macd_line')
    initialize_column_from_None(arrays, 'factor_signal_line')
    initialize_column_from_None(arrays, 'buy_avg_volume')
    initialize_column_from_None(arrays, 'sell_avg_volume')
    arrays['macd_line'] = arrays['macd_line'].astype(float)
    arrays['signal_line'] = arrays['signal_line'].astype(float)
    arrays['factor_macd_line'] = arrays['factor_macd_line'].astype(float)
    arrays['factor_signal_line'] = arrays['factor_signal_line'].astype(float)
    arrays['buy_avg_volume'] = arrays['buy_avg_volume'].astype(float)
    arrays['sell_avg_volume'] = arrays['sell_avg_volume'].astype(float)
    ori_start = start
    start = start - max_length - 5000
    ori_end = end
    end = end + 1
    # 为了避免因为索引越界而导致的错误，我们需要确保 start 和 end 有效
    if start >= end or start < 0 or end > len(arrays['high']):
        len_arrays = len(arrays['high'])
        print(
            f"start:{start},end:{end},ori_start:{ori_start},ori_end:{ori_end},len_arrays:{len_arrays}")
        print(
            f"the ori_start date is {arrays['datetime'][ori_start]}, the ori_end date is {arrays['datetime'][ori_end]}")
        print(
            f"the first date of arrays is {arrays['datetime'][0]}, the last date of arrays is {arrays['datetime'][-1]}")
        print(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")
        raise ValueError(
            f"Invalid range for calculating indicators. Check start:{start}, end:{end}, len(arrays['high']):{len_arrays}")

    # 在计算指标之前检查数组是否有足够的数据
    for key in ['high', 'low', 'close', 'volume']:
        if end > len(arrays[key]):
            raise ValueError(
                f"The data array for '{key}' does not have enough data to calculate indicators from {start} to {end}.")

    print(
        f"MACD calculation will based on short period {short_period}, long period {long_period}, signal period {signal_period}")
    print(
        f"MACD calculation will based on factor_short period {factor_short_period}, factor_long period {factor_long_period}, factor_signal period {factor_signal_period}")
    # 实盘的话全量计算，回测的话只计算需要的部分
    if is_live:
        relevant_close = arrays['close']
        relevant_volume = arrays['volume']
    else:
        relevant_close = arrays['close'][start:end]
        relevant_volume = arrays['volume'][start:end]

    short_ema = calculate_ema_numpy(
        arr=relevant_close, span=short_period, alpha=alpha_factor)
    long_ema = calculate_ema_numpy(
        arr=relevant_close, span=long_period, alpha=alpha_factor)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema_numpy(
        arr=macd_line, span=signal_period, alpha=alpha_factor)

    factor_short_ema = calculate_ema_numpy(
        arr=relevant_close, span=factor_short_period, alpha=alpha_factor)
    factor_long_ema = calculate_ema_numpy(
        arr=relevant_close, span=factor_long_period, alpha=alpha_factor)
    factor_macd_line = factor_short_ema - factor_long_ema
    factor_signal_line = calculate_ema_numpy(
        arr=factor_macd_line, span=factor_signal_period, alpha=alpha_factor)

    buy_avg_volume_slice = calculate_ma_numpy(
        prices=relevant_volume, ma_size=buy_avg_volume)
    sell_avg_volume_slice = calculate_ma_numpy(
        prices=relevant_volume, ma_size=sell_avg_volume)

    # Check for missing values in the calculated slices within the required range
    slices = {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'factor_macd_line': factor_macd_line,
        'factor_signal_line': factor_signal_line,
        'Buy Avg Volume': buy_avg_volume_slice,
        'Sell Avg Volume': sell_avg_volume_slice
    }
    start_idx = ori_start - start
    end_idx = ori_end - start + 1

    if not is_live:
        for name, slice_array in slices.items():
            if np.isnan(slice_array[start_idx:end_idx]).any():
                print(
                    f"Missing values found in {name} slice within the range {ori_start} to {ori_end}.")
                print(
                    f"Calculated {name} slice: {slice_array[start_idx:end_idx]}")
                print(
                    f"the first 100 line data is {slice_array[start_idx:start_idx+100]}")
                print(
                    f"the last 100 line data is {slice_array[end_idx-100:end_idx]}")
                raise ValueError(
                    f"Missing values detected in {name} calculation.")
        # ori_start = 1000, ori_end = 2000, max_lenth = 200
        # start = 800,end=2001
        # relevant_close = arrays['close'][start:end] = 1200
        # start_idx = ori_start - start = 200
        # end_idx = ori_end - start + 1 = 2001 - 800 + 1 = 1200
        arrays['macd_line'][ori_start:ori_end +
                            1] = macd_line[start_idx:end_idx]
        arrays['signal_line'][ori_start:ori_end +
                              1] = signal_line[start_idx:end_idx]
        arrays['factor_macd_line'][ori_start:ori_end +
                                   1] = factor_macd_line[start_idx:end_idx]
        arrays['factor_signal_line'][ori_start:ori_end +
                                     1] = factor_signal_line[start_idx:end_idx]
        arrays['buy_avg_volume'][ori_start:ori_end +
                                 1] = buy_avg_volume_slice[start_idx:end_idx]
        arrays['sell_avg_volume'][ori_start:ori_end +
                                  1] = sell_avg_volume_slice[start_idx:end_idx]
    else:
        arrays['macd_line'] = macd_line
        arrays['signal_line'] = signal_line
        arrays['factor_macd_line'] = factor_macd_line
        arrays['factor_signal_line'] = factor_signal_line
        arrays['buy_avg_volume'] = buy_avg_volume_slice
        arrays['sell_avg_volume'] = sell_avg_volume_slice

    cost_time = time.time() - start_time
    print(f"Tech indicator calculation took {cost_time} seconds")
    return arrays

