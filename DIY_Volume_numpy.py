#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Author: wlvh 124321452@qq.com
Date: 2023-06-16 13:58:55
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-06-20 08:11:08
FilePath: /trading/DIY_Volume_numpy.py
Description:
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved.
'''

import optuna.storages as storages
from functools import partial
import numpy_backtest as npb
from optuna.samplers import TPESampler
from optuna.exceptions import TrialPruned
import sys
import optuna
import logging
import json
import time
import math
import matplotlib.pyplot as plt
import statistics
import numpy as np
import backtrader as bt
import pandas as pd
from datetime import datetime
import os
# 动态设置环境变量，限制numpy和pandas使用单线程
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


def initialize_numpy_arrays(data_all, start_date, end_date, startcash=1000000):
    numpy_arrays = {}
    numpy_arrays['datetime'] = data_all.index.strftime(
        '%Y-%m-%d %H:%M:%S').to_numpy()
    # 遍历DataFrame的每一列，并将其转换为NumPy数组
    for column in data_all.columns:
        numpy_arrays[column] = data_all[column].to_numpy()
    initial_values = {
        'cash': startcash,
        'position': 0,
        'open_direction': 0,
        'open_cost_price': 0,
        'max_profit': 0,
        'BTC_debt': 0,
        'USDT_debt': 0,
        'net_value': startcash,
        'free_BTC': 0
    }

    matched_indices = np.where(numpy_arrays['datetime'] == str(start_date))[0]
    if matched_indices.size > 0:
        initial_date = matched_indices[0]
    else:
        print(
            f"the first date is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
        raise ValueError("未在数据中找到起始日期。")
    # 查找日期索引
    end_date_str = str(end_date)
    end_idx_array = np.where(numpy_arrays['datetime'] == end_date_str)[0]
    if end_idx_array.size == 0:
        # 如果没有找到对应的日期，抛出一个错误或者进行其他处理
        print(f"the last date is {numpy_arrays['datetime'][-1]}")
        print(f"End date {end_date_str} not found in the dataset.")
        print(
            f"we will use the last available date {numpy_arrays['datetime'][-1]} to replace the end date {end_date_str}.")
        end_date = numpy_arrays['datetime'][-1]
        end_idx = np.where(numpy_arrays['datetime'] == end_date)[0][0]
    # 获取索引
    else:
        end_idx = end_idx_array[0]

    for column, initial_value in initial_values.items():
        numpy_arrays[column][initial_date] = initial_value
        numpy_arrays[column][initial_date-1] = initial_value
    return numpy_arrays, initial_date, end_idx


def MAV_Strategy(df, end_date, start_date, short_period, strategy_name, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, buy_short_volume, sell_short_volume, long_factor, buy_long_factor, sell_long_factor, startcash=1000000, com=0.0005, numpy_arrays=None):
    strategy_start = time.time()
    long_period = short_period * long_factor
    buy_long_volume = buy_short_volume * buy_long_factor
    sell_long_volume = sell_short_volume * sell_long_factor
    # params里储存传递给npb.TradeLogic的参数，这些参数一般来说不适合由numpy_arrays数组进行储存。
    params = {}
    params['short_period'] = short_period
    params['long_period'] = long_period
    params['buy_short_volume'] = buy_short_volume
    params['sell_short_volume'] = sell_short_volume
    params['buy_long_volume'] = buy_long_volume
    params['sell_long_volume'] = sell_long_volume
    # 之所以有这个if判断，是因为在滚动回测时numpy_arrays是可以从上个周期继承的
    if numpy_arrays == None:
        init_columns = {'short_period': 0,
                        'long_period': 0,
                        'buy_short_volume': 0,
                        'sell_short_volume': 0,
                        'buy_long_volume': 0,
                        'sell_long_volume': 0}
        data_all = npb.load_data_bydate(
            df=df, end_date=end_date, start_date=start_date, init_columns=init_columns)
        data_start, data_end = data_all.index.min(), data_all.index.max()
        print(f"we prepare data from {data_start} to {data_end}")
        # print(f"load data costed {load_data} seconds")
        # 用指定值初始化这个数据集，并给出正确的开始日期结束日期的index
        numpy_arrays, initial_date, end_idx = initialize_numpy_arrays(
            data_all=data_all, start_date=start_date, end_date=end_date, startcash=startcash)
    else:
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(start_date))[0]
        if matched_indices.size > 0:
            initial_date = matched_indices[0]
        else:
            print(
                f"the start date in numpy arrays is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
            raise ValueError("未在数据中找到起始日期。")
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(end_date))[0]
        if matched_indices.size > 0:
            end_idx = matched_indices[0]
        else:
            print(
                f"the end date in numpy arrays is {numpy_arrays['datetime'][-1]} and end_date is {end_date}")
            end_idx = len(numpy_arrays['datetime']) - 1
            # raise ValueError("未在数据中找到结束日期。")

    # max_length是为了计算各类技术指标而所需的最大长度
    max_length = max(long_period, buy_long_volume, sell_long_volume, 10000) + 1
    max_length = int(round(max_length, 0))
    numpy_arrays = npb.calculate_MAVindicators_by_date(
        start=initial_date,
        end=end_idx, arrays=numpy_arrays,
        short_period=short_period,
        long_period=long_period,
        buy_short_volume=buy_short_volume,
        sell_short_volume=sell_short_volume,
        buy_long_volume=buy_long_volume,
        sell_long_volume=sell_long_volume,
        strategy_name=strategy_name,
        max_length=max_length)

    numpy_arrays = calculate_MAVsignals(arrays=numpy_arrays,
                                        start_date=start_date,
                                        end_date=end_date)
    print(f"Going to cut arrays to {start_date} to {end_date}")
    matched_indices = np.where(numpy_arrays['datetime'] == str(start_date))[0]
    if matched_indices.size > 0:
        start_idx = matched_indices[0]
    else:
        print(
            f"the first date is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
        raise ValueError("未在数据中找到起始日期。")

    print(
        f"We are going to backtest the first day {numpy_arrays['datetime'][start_idx]} to the last day {numpy_arrays['datetime'][end_idx]}")

    execute = time.time()
    for idx_i in range(start_idx, end_idx+1):
        trade = npb.TradeLogic(
            df=numpy_arrays,
            index=idx_i,
            buy_stop_loss=buy_stop_loss,
            sell_stop_loss=sell_stop_loss,
            buy_take_profit=buy_take_profit,
            sell_take_profit=sell_take_profit,
            strategy_name='MAVStrategy',
            params=params,
            com=com)
        trade.execute_trade()
    execute_cost = time.time() - execute
    # 打印net_value
    # 如果在end_idx的时刻正好激活交易信号，那么open_direction和open_cost_price会被重置为None，因而cash被重置为None，导致net_value为None，所以在这种情况下net_value设置为end_idx-1的值
    numpy_arrays['net_value'][end_idx] = numpy_arrays['net_value'][end_idx-1]
    print(f"the end date is {numpy_arrays['datetime'][end_idx]}")
    print(f"the net value is {numpy_arrays['net_value'][end_idx]}")
    print(f"the trade execute costed {execute_cost} seconds")

    if str(numpy_arrays['datetime'][start_idx]) != str(start_date):
        print(
            f"the start date is {start_date} but numpy_arrays['datetime'][0] is {numpy_arrays['datetime'][0]}")
        raise ValueError(
            "The start and end dates do not match the datetime array.")
    if str(numpy_arrays['datetime'][end_idx]) != str(end_date):
        print(
            f"the end date is {end_date} but numpy_arrays['datetime'][end_idx] is {numpy_arrays['datetime'][end_idx]}")
    cost_time = time.time() - strategy_start
    print(f"MAV strategy costed {cost_time} seconds")
    # final_data = pd.DataFrame.from_dict(
    # {key: pd.Series(value) for key, value in numpy_arrays.items()})
    # final_data.to_csv('numpy_for_backtest_0608_np.csv', index=False)
    return numpy_arrays


def MACD_Strategy(df, end_date, start_date, short_period, long_period, signal_period, L_short_period, L_long_period, L_signal_period, strategy_name, factor, alpha_factor, buy_stop_loss=0.015, sell_stop_loss=0.015, buy_take_profit=1, sell_take_profit=1, buy_avg_volume=1, sell_avg_volume=1, startcash=1000000, com=0.0005, numpy_arrays=None):
    strategy_start = time.time()
    # params里储存传递给npb.TradeLogic的参数，这些参数一般来说不适合由numpy_arrays数组进行储存。
    params = {}
    params['short_period'] = short_period
    params['long_period'] = long_period
    params['signal_period'] = signal_period
    params['L_short_period'] = L_short_period
    params['L_long_period'] = L_long_period
    params['L_signal_period'] = L_signal_period
    # 之所以有这个if判断，是因为在滚动回测时numpy_arrays是可以从上个周期继承的
    if numpy_arrays == None:
        init_columns = {'short_period': 0,
                        'long_period': 0,
                        'signal_period': 0}
        data_all = npb.load_data_bydate(
            df=df, end_date=end_date, start_date=start_date, init_columns=init_columns)
        data_start, data_end = data_all.index.min(), data_all.index.max()
        print(f"we prepare data from {data_start} to {data_end}")
        # print(f"load data costed {load_data} seconds")
        # 用指定值初始化这个数据集，并给出正确的开始日期结束日期的index
        numpy_arrays, initial_date, end_idx = initialize_numpy_arrays(
            data_all=data_all, start_date=start_date, end_date=end_date, startcash=startcash)
    else:
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(start_date))[0]
        if matched_indices.size > 0:
            initial_date = matched_indices[0]
        else:
            print(
                f"the start date in numpy arrays is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
            raise ValueError("未在数据中找到起始日期。")
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(end_date))[0]
        if matched_indices.size > 0:
            end_idx = matched_indices[0]
        else:
            print(
                f"the end date in numpy arrays is {numpy_arrays['datetime'][-1]} and end_date is {end_date}")
            end_idx = len(numpy_arrays['datetime']) - 1
            # raise ValueError("未在数据中找到结束日期。")

    # max_length是为了计算各类技术指标而所需的最大长度
    max_length = max(long_period, 10000) + 1
    max_length = int(round(max_length, 0))
    numpy_arrays = npb.calculate_MACDindicators_by_date(
        start=initial_date,
        end=end_idx, arrays=numpy_arrays,
        short_period=short_period,
        long_period=long_period,
        signal_period=signal_period,
        L_short_period=L_short_period,
        L_long_period=L_long_period,
        L_signal_period=L_signal_period,
        factor=factor,
        alpha_factor=alpha_factor,
        buy_avg_volume=buy_avg_volume,
        sell_avg_volume=sell_avg_volume,
        strategy_name=strategy_name,
        max_length=max_length)

    numpy_arrays = calculate_MACDsignals(arrays=numpy_arrays,
                                         start_date=start_date,
                                         end_date=end_date)
    print(f"Going to cut arrays to {start_date} to {end_date}")
    matched_indices = np.where(numpy_arrays['datetime'] == str(start_date))[0]
    if matched_indices.size > 0:
        start_idx = matched_indices[0]
    else:
        print(
            f"the first date is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
        raise ValueError("未在数据中找到起始日期。")

    print(
        f"We are going to backtest the first day {numpy_arrays['datetime'][start_idx]} to the last day {numpy_arrays['datetime'][end_idx]}")

    execute = time.time()
    for idx_i in range(start_idx, end_idx+1):
        trade = npb.TradeLogic(
            df=numpy_arrays,
            index=idx_i,
            buy_stop_loss=buy_stop_loss,
            sell_stop_loss=sell_stop_loss,
            buy_take_profit=buy_take_profit,
            sell_take_profit=sell_take_profit,
            strategy_name='MACDStrategy',
            params=params,
            com=com)
        trade.execute_trade()
    execute_cost = time.time() - execute
    # 打印net_value
    # 如果在end_idx的时刻正好激活交易信号，那么open_direction和open_cost_price会被重置为None，因而cash被重置为None，导致net_value为None，所以在这种情况下net_value设置为end_idx-1的值
    numpy_arrays['net_value'][end_idx] = numpy_arrays['net_value'][end_idx-1]
    print(f"the end date is {numpy_arrays['datetime'][end_idx]}")
    print(f"the net value is {numpy_arrays['net_value'][end_idx]}")
    print(f"the trade execute costed {execute_cost} seconds")

    if str(numpy_arrays['datetime'][start_idx]) != str(start_date):
        print(
            f"the start date is {start_date} but numpy_arrays['datetime'][0] is {numpy_arrays['datetime'][0]}")
        raise ValueError(
            "The start and end dates do not match the datetime array.")
    if str(numpy_arrays['datetime'][end_idx]) != str(end_date):
        print(
            f"the end date is {end_date} but numpy_arrays['datetime'][end_idx] is {numpy_arrays['datetime'][end_idx]}")
    cost_time = time.time() - strategy_start
    print(f"MACD strategy costed {cost_time} seconds")
    final_data = pd.DataFrame.from_dict(
        {key: pd.Series(value) for key, value in numpy_arrays.items()})
    final_data.to_csv('numpy_for_backtest_0608_np.csv', index=False)
    return numpy_arrays


def RSIV_Strategy(df, end_date, start_date, rsi_window, ma_short_period, ma_long_period, buy_stop_loss, sell_stop_loss, buy_take_profit, buy_rsi_threshold, sell_rsi_threshold, sell_take_profit, open_percent, buy_Apoint, buy_Bpoint, sell_Apoint, sell_Bpoint, strategy_name, buy_avg_volume=1, sell_avg_volume=1, startcash=1000000, com=0.0005, numpy_arrays=None):
    strategy_start = time.time()
    # params里储存传递给npb.TradeLogic的参数，这些参数一般来说不适合由numpy_arrays数组进行储存。
    # open_percent 代表开仓的初始仓位，一旦做多开仓，Apoint代表额外加仓的点位，例如我们在RSI低于30时开仓，仓位为初始仓位；之后如果RSI低于30-Apoint，则继续加仓，加仓仓位为open_percent + (1-open_percent)/2; 如果RSI低于30-Apoint-Bpoint，则继续加仓到满仓。
    params = {}
    params['buy_rsi_threshold'] = buy_rsi_threshold
    params['sell_rsi_threshold'] = sell_rsi_threshold
    params['buy_Apoint'] = buy_Apoint
    params['buy_Bpoint'] = buy_Bpoint
    params['sell_Apoint'] = sell_Apoint
    params['sell_Bpoint'] = sell_Bpoint
    params['open_percent'] = open_percent
    if numpy_arrays == None:
        init_columns = {'rsi': 0,
                        'ma_short': 0,
                        'ma_long': 0,
                        'buy_avg_volume': 0,
                        'sell_avg_volume': 0}
        data_all = npb.load_data_bydate(
            df=df, end_date=end_date, start_date=start_date, init_columns=init_columns)
        data_start, data_end = data_all.index.min(), data_all.index.max()
        print(f"we prepare data from {data_start} to {data_end}")
        # print(f"load data costed {load_data} seconds")
        # 用指定值初始化这个数据集，并给出正确的开始日期结束日期的index
        numpy_arrays, initial_date, end_idx = initialize_numpy_arrays(
            data_all=data_all, start_date=start_date, end_date=end_date, startcash=startcash)
    else:
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(start_date))[0]
        if matched_indices.size > 0:
            initial_date = matched_indices[0]
        else:
            print(
                f"the start date in numpy arrays is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
            raise ValueError("未在数据中找到起始日期。")
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(end_date))[0]
        if matched_indices.size > 0:
            end_idx = matched_indices[0]
        else:
            print(
                f"the end date in numpy arrays is {numpy_arrays['datetime'][-1]} and end_date is {end_date}")
            end_idx = len(numpy_arrays['datetime']) - 1
            # raise ValueError("未在数据中找到结束日期。")

    # max_length是为了计算各类技术指标而所需的最大长度
    max_length = max(rsi_window*720, ma_long_period *
                     720, ma_short_period*720, 10000) + 1
    max_length = int(round(max_length, 0))
    numpy_arrays = npb.calculate_RSIindicators_by_date(
        start=initial_date,
        end=end_idx, arrays=numpy_arrays,
        rsi_size=rsi_window,
        ma_short=ma_short_period,
        ma_long=ma_long_period,
        buy_avg_volume=buy_avg_volume,
        sell_avg_volume=sell_avg_volume,
        strategy_name=strategy_name,
        max_length=max_length)
    # numpy_arrays['buy_condition1'] = (np.roll(numpy_arrays['rsi'], shift=1) < buy_rsi_threshold) | (numpy_arrays['rsi'] < buy_rsi_threshold)
    # numpy_arrays['buy_condition2'] = numpy_arrays['close'] > numpy_arrays['ma_long']
    # numpy_arrays['buy_condition3'] = np.roll(numpy_arrays['close'], shift=1) > numpy_arrays['ma_long']
    # numpy_arrays['buy_signal'] = numpy_arrays['buy_condition1'] & (numpy_arrays['buy_condition2'] | numpy_arrays['buy_condition3'])

    # # 计算卖出条件
    # numpy_arrays['sell_condition1'] = (np.roll(numpy_arrays['rsi'], shift=1) > sell_rsi_threshold) | (numpy_arrays['rsi'] > sell_rsi_threshold)
    # numpy_arrays['sell_condition2'] = numpy_arrays['close'] < numpy_arrays['ma_short']
    # numpy_arrays['sell_condition3'] = np.roll(numpy_arrays['close'], shift=1) < numpy_arrays['ma_short']
    # numpy_arrays['sell_signal'] = numpy_arrays['sell_condition1'] & (numpy_arrays['sell_condition2'] | numpy_arrays['sell_condition3'])
    numpy_arrays = calculate_RSIsignals(df=numpy_arrays,
                                        buy_rsi_threshold=buy_rsi_threshold,
                                        start_date=start_date,
                                        end_date=end_date,
                                        sell_rsi_threshold=sell_rsi_threshold)
    print(f"Going to cut arrays to {start_date} to {end_date}")
    # start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
    # end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M:%S')
    matched_indices = np.where(numpy_arrays['datetime'] == str(start_date))[0]
    if matched_indices.size > 0:
        start_idx = matched_indices[0]
    else:
        print(
            f"the first date is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
        raise ValueError("未在数据中找到起始日期。")
    # end_idx = np.where(numpy_arrays['datetime'] == str(end_date))[0][0]
    print(
        f"We are going to backtest the first day {numpy_arrays['datetime'][start_idx]} to the last day {numpy_arrays['datetime'][end_idx]}")
    # for columns in numpy_arrays.keys():
    # print(f"check the value of {columns} is {numpy_arrays[columns][0]}")
    # print(f"check the value of {columns} is {numpy_arrays[columns][-1]}")
    execute = time.time()
    for idx_i in range(start_idx, end_idx+1):
        trade = npb.TradeLogic(
            df=numpy_arrays,
            index=idx_i,
            buy_stop_loss=buy_stop_loss,
            sell_stop_loss=sell_stop_loss,
            buy_take_profit=buy_take_profit,
            sell_take_profit=sell_take_profit,
            strategy_name='RSIV_Strategy',
            params=params,
            com=com)
        trade.execute_trade()
    execute_cost = time.time() - execute
    # 打印net_value
    # 如果在end_idx的时刻正好激活交易信号，那么open_direction和open_cost_price会被重置为None，因而cash被重置为None，导致net_value为None，所以在这种情况下net_value设置为end_idx-1的值
    numpy_arrays['net_value'][end_idx] = numpy_arrays['net_value'][end_idx-1]
    print(f"the end date is {numpy_arrays['datetime'][end_idx]}")
    print(f"the net value is {numpy_arrays['net_value'][end_idx]}")
    print(f"the trade execute costed {execute_cost} seconds")
    # if str(numpy_arrays['datetime'][0]) != str(start_date) or str(numpy_arrays['datetime'][-1]) != str(end_date):
    if str(numpy_arrays['datetime'][start_idx]) != str(start_date):
        print(
            f"the start date is {start_date} but numpy_arrays['datetime'][0] is {numpy_arrays['datetime'][0]}")
        raise ValueError(
            "The start and end dates do not match the datetime array.")
    if str(numpy_arrays['datetime'][end_idx]) != str(end_date):
        print(
            f"the end date is {end_date} but numpy_arrays['datetime'][end_idx] is {numpy_arrays['datetime'][end_idx]}")
    # 将 numpy_arrays 转换为 DataFrame 并保存为 CSV
    # df_result = pd.DataFrame(numpy_arrays)
    # csv_filename = f"RSIV_Strategy_Result.csv"
    # df_result.to_csv(csv_filename, index=False)
    # print(f"Results saved to {csv_filename}")
    return numpy_arrays


def WRSIStrategy(df, end_date, start_date, rsi_window, factor, ma_short_period, ma_long_period, buy_stop_loss, sell_stop_loss, buy_take_profit, buy_rsi_threshold, sell_rsi_threshold, sell_take_profit, open_percent, buy_Apoint, buy_Bpoint, sell_Apoint, sell_Bpoint, strategy_name, buy_avg_volume=1, sell_avg_volume=1, startcash=1000000, com=0.0005, numpy_arrays=None):
    strategy_start = time.time()
    # params里储存传递给npb.TradeLogic的参数，这些参数一般来说不适合由numpy_arrays数组进行储存。
    # open_percent 代表开仓的初始仓位，一旦做多开仓，Apoint代表额外加仓的点位，例如我们在RSI低于30时开仓，仓位为初始仓位；之后如果RSI低于30-Apoint，则继续加仓，加仓仓位为open_percent + (1-open_percent)/2; 如果RSI低于30-Apoint-Bpoint，则继续加仓到满仓。
    params = {}
    params['buy_rsi_threshold'] = buy_rsi_threshold
    params['sell_rsi_threshold'] = sell_rsi_threshold
    params['buy_Apoint'] = buy_Apoint
    params['buy_Bpoint'] = buy_Bpoint
    params['sell_Apoint'] = sell_Apoint
    params['sell_Bpoint'] = sell_Bpoint
    params['open_percent'] = open_percent
    params['factort'] = factor
    if numpy_arrays == None:
        init_columns = {'rsi': 0,
                        'ma_short': 0,
                        'ma_long': 0,
                        'buy_avg_volume': 0,
                        'sell_avg_volume': 0}
        data_all = npb.load_data_bydate(
            df=df, end_date=end_date, start_date=start_date, init_columns=init_columns)
        data_start, data_end = data_all.index.min(), data_all.index.max()
        print(f"we prepare data from {data_start} to {data_end}")
        # print(f"load data costed {load_data} seconds")
        # 用指定值初始化这个数据集，并给出正确的开始日期结束日期的index
        numpy_arrays, initial_date, end_idx = initialize_numpy_arrays(
            data_all=data_all, start_date=start_date, end_date=end_date, startcash=startcash)
    else:
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(start_date))[0]
        if matched_indices.size > 0:
            initial_date = matched_indices[0]
        else:
            print(
                f"the start date in numpy arrays is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
            raise ValueError("未在数据中找到起始日期。")
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(end_date))[0]
        if matched_indices.size > 0:
            end_idx = matched_indices[0]
        else:
            print(
                f"the end date in numpy arrays is {numpy_arrays['datetime'][-1]} and end_date is {end_date}")
            end_idx = len(numpy_arrays['datetime']) - 1
            # raise ValueError("未在数据中找到结束日期。")

    # max_length是为了计算各类技术指标而所需的最大长度
    max_length = max(rsi_window*720, ma_long_period *
                     720, ma_short_period*720, 10000) + 1
    max_length = int(round(max_length, 0))
    numpy_arrays = npb.calculate_RSIindicators_by_date(
        start=initial_date,
        end=end_idx, arrays=numpy_arrays,
        rsi_size=rsi_window,
        factor=factor,
        ma_short=ma_short_period,
        ma_long=ma_long_period,
        buy_avg_volume=buy_avg_volume,
        sell_avg_volume=sell_avg_volume,
        strategy_name="WRSIStrategy",
        max_length=max_length)
    # numpy_arrays['buy_condition1'] = (np.roll(numpy_arrays['rsi'], shift=1) < buy_rsi_threshold) | (numpy_arrays['rsi'] < buy_rsi_threshold)
    # numpy_arrays['buy_condition2'] = numpy_arrays['close'] > numpy_arrays['ma_long']
    # numpy_arrays['buy_condition3'] = np.roll(numpy_arrays['close'], shift=1) > numpy_arrays['ma_long']
    # numpy_arrays['buy_signal'] = numpy_arrays['buy_condition1'] & (numpy_arrays['buy_condition2'] | numpy_arrays['buy_condition3'])

    # # 计算卖出条件
    # numpy_arrays['sell_condition1'] = (np.roll(numpy_arrays['rsi'], shift=1) > sell_rsi_threshold) | (numpy_arrays['rsi'] > sell_rsi_threshold)
    # numpy_arrays['sell_condition2'] = numpy_arrays['close'] < numpy_arrays['ma_short']
    # numpy_arrays['sell_condition3'] = np.roll(numpy_arrays['close'], shift=1) < numpy_arrays['ma_short']
    # numpy_arrays['sell_signal'] = numpy_arrays['sell_condition1'] & (numpy_arrays['sell_condition2'] | numpy_arrays['sell_condition3'])
    numpy_arrays = calculate_RSIsignals(df=numpy_arrays,
                                        buy_rsi_threshold=buy_rsi_threshold,
                                        start_date=start_date,
                                        end_date=end_date,
                                        sell_rsi_threshold=sell_rsi_threshold)
    print(f"Going to cut arrays to {start_date} to {end_date}")
    # start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
    # end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M:%S')
    matched_indices = np.where(numpy_arrays['datetime'] == str(start_date))[0]
    if matched_indices.size > 0:
        start_idx = matched_indices[0]
    else:
        print(
            f"the first date is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
        raise ValueError("未在数据中找到起始日期。")
    # end_idx = np.where(numpy_arrays['datetime'] == str(end_date))[0][0]
    print(
        f"We are going to backtest the first day {numpy_arrays['datetime'][start_idx]} to the last day {numpy_arrays['datetime'][end_idx]}")
    # for columns in numpy_arrays.keys():
    # print(f"check the value of {columns} is {numpy_arrays[columns][0]}")
    # print(f"check the value of {columns} is {numpy_arrays[columns][-1]}")
    execute = time.time()
    for idx_i in range(start_idx, end_idx+1):
        trade = npb.TradeLogic(
            df=numpy_arrays,
            index=idx_i,
            buy_stop_loss=buy_stop_loss,
            sell_stop_loss=sell_stop_loss,
            buy_take_profit=buy_take_profit,
            sell_take_profit=sell_take_profit,
            strategy_name="WRSIStrategy",
            params=params,
            com=com)
        trade.execute_trade()
    execute_cost = time.time() - execute
    # 打印net_value
    # 如果在end_idx的时刻正好激活交易信号，那么open_direction和open_cost_price会被重置为None，因而cash被重置为None，导致net_value为None，所以在这种情况下net_value设置为end_idx-1的值
    numpy_arrays['net_value'][end_idx] = numpy_arrays['net_value'][end_idx-1]
    print(f"the end date is {numpy_arrays['datetime'][end_idx]}")
    print(f"the net value is {numpy_arrays['net_value'][end_idx]}")
    print(f"the trade execute costed {execute_cost} seconds")
    # if str(numpy_arrays['datetime'][0]) != str(start_date) or str(numpy_arrays['datetime'][-1]) != str(end_date):
    if str(numpy_arrays['datetime'][start_idx]) != str(start_date):
        print(
            f"the start date is {start_date} but numpy_arrays['datetime'][0] is {numpy_arrays['datetime'][0]}")
        raise ValueError(
            "The start and end dates do not match the datetime array.")
    if str(numpy_arrays['datetime'][end_idx]) != str(end_date):
        print(
            f"the end date is {end_date} but numpy_arrays['datetime'][end_idx] is {numpy_arrays['datetime'][end_idx]}")
    # 将 numpy_arrays 转换为 DataFrame 并保存为 CSV
    # df_result = pd.DataFrame(numpy_arrays)
    # csv_filename = f"RSIV_Strategy_Result.csv"
    # df_result.to_csv(csv_filename, index=False)
    # print(f"Results saved to {csv_filename}")
    return numpy_arrays

# def calculate_RSIsignals(df, buy_rsi_threshold, sell_rsi_threshold,start_date=None,end_date=None,is_live=False):
#     # df是dict，内为numpy格式的数据
#     df = df.copy()
#     if not is_live:
#         valid_indices = np.where((df['datetime'] >= str(start_date)) & (df['datetime'] <= str(end_date)))
#         df_valid = {key: np.array(val)[valid_indices] for key, val in df.items()}
#     else:
#         valid_indices = np.ones(len(df['datetime']), dtype=bool)
#         df_valid = df
#     # 计算买入条件和信号
#     df_valid['buy_condition1'] = ((np.roll(df_valid['rsi'], shift=1) < buy_rsi_threshold) | (df_valid['rsi'] < buy_rsi_threshold)) & valid_indices
#     df_valid['buy_condition2'] = (df_valid['close'] > df_valid['ma_long']) & valid_indices
#     df_valid['buy_condition3'] = (np.roll(df_valid['close'], shift=1) > df_valid['ma_long']) & valid_indices
#     df_valid['buy_signal'] = df_valid['buy_condition1'] & (df_valid['buy_condition2'] | df_valid['buy_condition3'])

#     # 计算卖出条件和信号
#     df_valid['sell_condition1'] = ((np.roll(df_valid['rsi'], shift=1) > sell_rsi_threshold) | (df_valid['rsi'] > sell_rsi_threshold)) & valid_indices
#     df_valid['sell_condition2'] = (df_valid['close'] < df_valid['ma_short']) & valid_indices
#     df_valid['sell_condition3'] = (np.roll(df_valid['close'], shift=1) < df_valid['ma_short']) & valid_indices
#     df_valid['sell_signal'] = df_valid['sell_condition1'] & (df_valid['sell_condition2'] | df_valid['sell_condition3'])

#     for key in ['buy_condition1', 'buy_condition2', 'buy_condition3', 'buy_signal', 'sell_condition1', 'sell_condition2', 'sell_condition3', 'sell_signal']:
#         df[key] = np.zeros(len(df['datetime']), dtype=bool)
#         df[key][valid_indices] = df_valid[key]
#         #print(df['datetime'][-100:])
#         #print(df[key][-100:])
#     first_index = np.where(valid_indices)[0][0]
#     last_index = np.where(valid_indices)[0][-1]
#     print(f"First valid datetime: {df['datetime'][first_index]}")
#     print(f"Last valid datetime: {df['datetime'][last_index]}")
#     # 打印信号条件（如果是实盘）
#     if is_live:
#         last_close = df['close'][-1]
#         last_rsi = df['rsi'][-1]
#         ma_long = df['ma_long'][-1]
#         ma_short = df['ma_short'][-1]
#         pre_rsi = df['rsi'][-2]
#         pre_close = df['close'][-2]

#         print(f"Buy Signal Conditions:")
#         print(f"  Condition 1: {df['buy_condition1'][-1]} (pre_rsi: {pre_rsi}, last_rsi: {last_rsi}, threshold: {buy_rsi_threshold})")
#         print(f"  Condition 2: {df['buy_condition2'][-1]} (last_close: {last_close}, ma_long: {ma_long})")
#         print(f"  Condition 3: {df['buy_condition3'][-1]} (pre_close: {pre_close}, ma_long: {ma_long})")

#         print(f"Sell Signal Conditions:")
#         print(f"  Condition 1: {df['sell_condition1'][-1]} (pre_rsi: {pre_rsi}, last_rsi: {last_rsi}, threshold: {sell_rsi_threshold})")
#         print(f"  Condition 2: {df['sell_condition2'][-1]} (last_close: {last_close}, ma_short: {ma_short})")
#         print(f"  Condition 3: {df['sell_condition3'][-1]} (pre_close: {pre_close}, ma_short: {ma_short})")
#     return df


def calculate_RSIsignals(df, buy_rsi_threshold, sell_rsi_threshold, start_date=None, end_date=None, is_live=False):
    # df是dict，内为numpy格式的数据
    df = df.copy()
    # if not is_live:
    #     valid_indices = (df['datetime'] >= str(start_date)) & (df['datetime'] <= str(end_date))
    #     df_valid = {key: np.array(val)[valid_indices] for key, val in df.items()}
    # else:
    #     end_date = df['datetime'][-1]
    #     valid_indices = (df['datetime'] >= str(start_date)) & (df['datetime'] <= str(end_date))
    #     df_valid = {key: np.array(val)[valid_indices] for key, val in df.items()}
    # 将日期转换为 numpy 的 datetime64 类型
    df['datetime'] = np.array(df['datetime'], dtype='str')

    if not is_live:
        start_date = str(start_date)
        end_date = str(end_date)
        valid_indices = (df['datetime'] >= start_date) & (
            df['datetime'] <= end_date)
        df_valid = {key: np.array(val)[valid_indices]
                    for key, val in df.items()}
    else:
        start_date = str(start_date)
        end_date = df['datetime'][-1]
        start_date = df['datetime'][-100]
        valid_indices = (df['datetime'] >= start_date) & (
            df['datetime'] <= end_date)
        df_valid = {key: np.array(val)[valid_indices]
                    for key, val in df.items()}

    # 计算买入条件和信号
    df_valid['buy_condition1'] = (np.roll(df_valid['rsi'], shift=1) < buy_rsi_threshold) | (
        df_valid['rsi'] < buy_rsi_threshold)
    df_valid['buy_condition2'] = df_valid['close'] > df_valid['ma_long']
    df_valid['buy_condition3'] = np.roll(
        df_valid['close'], shift=1) > df_valid['ma_long']
    df_valid['buy_signal'] = df_valid['buy_condition1'] & (
        df_valid['buy_condition2'] | df_valid['buy_condition3'])

    # 计算卖出条件和信号
    df_valid['sell_condition1'] = (np.roll(df_valid['rsi'], shift=1) > sell_rsi_threshold) | (
        df_valid['rsi'] > sell_rsi_threshold)
    df_valid['sell_condition2'] = df_valid['close'] < df_valid['ma_short']
    df_valid['sell_condition3'] = np.roll(
        df_valid['close'], shift=1) < df_valid['ma_short']
    df_valid['sell_signal'] = df_valid['sell_condition1'] & (
        df_valid['sell_condition2'] | df_valid['sell_condition3'])

    for key in ['buy_condition1', 'buy_condition2', 'buy_condition3', 'buy_signal', 'sell_condition1', 'sell_condition2', 'sell_condition3', 'sell_signal']:
        df[key] = np.zeros(len(df['datetime']), dtype=bool)
        np.place(df[key], valid_indices, df_valid[key])

    # 打印信号条件（如果是实盘）
    if is_live:
        last_close = df['close'][-1]
        last_rsi = df['rsi'][-1]
        ma_long = df['ma_long'][-1]
        ma_short = df['ma_short'][-1]
        pre_rsi = df['rsi'][-2]
        pre_close = df['close'][-2]

        print(f"Buy Signal Conditions:")
        print(
            f"  Condition 1: {df['buy_condition1'][-1]} (pre_rsi: {pre_rsi}, last_rsi: {last_rsi}, threshold: {buy_rsi_threshold})")
        print(
            f"  Condition 2: {df['buy_condition2'][-1]} (last_close: {last_close}, ma_long: {ma_long})")
        print(
            f"  Condition 3: {df['buy_condition3'][-1]} (pre_close: {pre_close}, ma_long: {ma_long})")

        print(f"Sell Signal Conditions:")
        print(
            f"  Condition 1: {df['sell_condition1'][-1]} (pre_rsi: {pre_rsi}, last_rsi: {last_rsi}, threshold: {sell_rsi_threshold})")
        print(
            f"  Condition 2: {df['sell_condition2'][-1]} (last_close: {last_close}, ma_short: {ma_short})")
        print(
            f"  Condition 3: {df['sell_condition3'][-1]} (pre_close: {pre_close}, ma_short: {ma_short})")
    # df_data = pd.DataFrame(df)
    # df_data.to_csv('RSI_signals.csv')
    return df


def calculate_MAVsignals(arrays, start_date=None, end_date=None, is_live=False):
    # df是dict，内为numpy格式的数据
    df = arrays.copy()
    # 做多信号，short_period上穿long_period，且buy_short_volume大于buy_long_volume
    # 做空信号，short_period下穿long_period，且sell_short_volume大于sell_long_volume

    # 将日期转换为 numpy 的 datetime64 类型
    df['datetime'] = np.array(df['datetime'], dtype='str')

    if not is_live:
        start_date = str(start_date)
        end_date = str(end_date)
        valid_indices = (df['datetime'] >= start_date) & (
            df['datetime'] <= end_date)
        df_valid = {key: np.array(val)[valid_indices]
                    for key, val in df.items()}
    else:
        start_date = str(start_date)
        end_date = df['datetime'][-1]
        start_date = df['datetime'][-100]
        valid_indices = (df['datetime'] >= start_date) & (
            df['datetime'] <= end_date)
        df_valid = {key: np.array(val)[valid_indices]
                    for key, val in df.items()}

    data = pd.DataFrame(df_valid)
    data = data[['short_ma', 'long_ma']]
    # 计算买入条件和信号
    df_valid['buy_condition1'] = (data['short_ma'] > data['long_ma']) & (
        data['short_ma'].shift(1) <= data['long_ma'].shift(1))
    df_valid['buy_condition2'] = df_valid['buy_short_volume'] > df_valid['buy_long_volume']
    df_valid['buy_condition3'] = np.roll(
        df_valid['buy_short_volume'], 1) > np.roll(df_valid['buy_long_volume'], 1)
    df_valid['buy_signal'] = df_valid['buy_condition1'] & (
        df_valid['buy_condition2'] | df_valid['buy_condition3'])

    # 计算卖出条件和信号
    df_valid['sell_condition1'] = (data['short_ma'] < data['long_ma']) & (
        data['short_ma'].shift(1) >= data['long_ma'].shift(1))
    df_valid['sell_condition2'] = df_valid['sell_short_volume'] > df_valid['sell_long_volume']
    df_valid['sell_condition3'] = np.roll(
        df_valid['sell_short_volume'], 1) > np.roll(df_valid['sell_long_volume'], 1)
    df_valid['sell_signal'] = df_valid['sell_condition1'] & (
        df_valid['sell_condition2'] | df_valid['sell_condition3'])

    for key in ['buy_condition1', 'buy_condition2', 'buy_condition3', 'buy_signal', 'sell_condition1', 'sell_condition2', 'sell_condition3', 'sell_signal']:
        df[key] = np.zeros(len(df['datetime']), dtype=bool)
        np.place(df[key], valid_indices, df_valid[key])

    # 打印信号条件（如果是实盘）
    if is_live:
        last_close = df['close'][-1]
        short_period = df['short_ma'][-1]
        long_period = df['long_ma'][-1]
        pre_short_period = df['short_ma'][-2]
        pre_long_period = df['long_ma'][-2]
        buy_short_volume = df['buy_short_volume'][-1]
        buy_long_volume = df['buy_long_volume'][-1]
        pre_buy_short_volume = df['buy_short_volume'][-2]
        pre_buy_long_volume = df['buy_long_volume'][-2]
        sell_short_volume = df['sell_short_volume'][-1]
        sell_long_volume = df['sell_long_volume'][-1]
        pre_sell_short_volume = df['sell_short_volume'][-2]
        pre_sell_long_volume = df['sell_long_volume'][-2]

        print(f"Buy Signal Conditions:")
        print(f"  Condition 1: {df['buy_condition1'][-1]} (short_period crossed above long_period, last short_period: {short_period}, last long_period: {long_period}, pre short_period: {pre_short_period}, pre long_period: {pre_long_period})")
        print(
            f"  Condition 2: {df['buy_condition2'][-1]} (buy_short_volume: {buy_short_volume} > buy_long_volume: {buy_long_volume})")
        print(
            f"  Condition 3: {df['buy_condition3'][-1]} (pre_buy_short_volume: {pre_buy_short_volume} > pre_buy_long_volume: {pre_buy_long_volume})")

        print(f"Sell Signal Conditions:")
        print(f"  Condition 1: {df['sell_condition1'][-1]} (short_period crossed below long_period, last short_period: {short_period}, last long_period: {long_period}, pre short_period: {pre_short_period}, pre long_period: {pre_long_period})")
        print(
            f"  Condition 2: {df['sell_condition2'][-1]} (sell_short_volume: {sell_short_volume} > sell_long_volume: {sell_long_volume})")
        print(
            f"  Condition 3: {df['sell_condition3'][-1]} (pre_sell_short_volume: {pre_sell_short_volume} > pre_sell_long_volume: {pre_sell_long_volume})")

    return df


def calculate_MACDsignals(arrays, start_date=None, end_date=None, is_live=False):
    # 1，小时级别MACD金叉（first cross 设置为True）。
    # 2，一旦小时级别macd趋势线下跌，则second cross设置为False，first position record=None。
    # 3，5分钟级别MACD金叉（second cross设置为True，second cross需要在first cross为True时才能设置为True，并且second cross为True后first cross设置为False）并记录点位（first position record）。
    # 4，5分钟级别MACD再次金叉且高于first position record时买入（second cross设置为False），止损点1%（止损点其他函数处理）。
    # 4.1，如果再次金叉时点位没有高于first position record，则将当前金叉视为first cross，当前价格设置为first position record。
    # 5，买入后开始记录5分钟级别MACD死叉点和对应的close_fisrt_position_record。
    # 6，5分钟级别MACD第一次死叉记录first down cross为True, 并记录点位(first down position record)。
    # 7，5分钟级别MACD第二次死叉（即first down cross为True后的第一次），先将first down cross设置为False，然后：
    # 7.1，如果点位低于first down position record，则卖出。并将所有的记录设置为False（first cross，second cross，first position record，first down cross，first down positin record）
    # 7.2，如果点位高于first down position record，则first down cross设置为True，并更新记录(first down position record)；

    arrays = arrays.copy()

    arrays['datetime'] = np.array(arrays['datetime'], dtype='str')
    if not is_live:
        start_date = str(start_date)
        end_date = str(end_date)
    else:
        end_date = arrays['datetime'][-1]
        start_date = arrays['datetime'][-10000]

    valid_indices = (arrays['datetime'] >= start_date) & (
        arrays['datetime'] <= end_date)

    df_valid = {key: np.array(val)[valid_indices]
                for key, val in arrays.items()}
    print(
        f"we use these daterange for MACD signals: {df_valid['datetime'][0]} to {df_valid['datetime'][-1]}")
    # 创建状态列
    df_valid['buy_signal'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    df_valid['sell_signal'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    df_valid['close_signal'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    # 处理小时级别的MACD金叉
    # 初始化金叉和死叉信号

    # 初始化变量
    buy_first_cross = False
    buy_position = False
    buy_first_down_cross, buy_second_down_cross = False, False
    close_fisrt_position_record = None

    sell_first_cross = False
    sell_position = False
    sell_first_down_cross, sell_second_down_cross = False, False
    close_first_position_record_sell = None

    buy_price, sell_price = None, None

    df_valid['factor_macd_line'] = df_valid['factor_macd_line'].astype(
        float).round(2)
    df_valid['factor_signal_line'] = df_valid['factor_signal_line'].astype(
        float).round(2)
    print(f"the last 100 line data is {df_valid['factor_macd_line'][-100:]}")
    print(f"the last 100 line data is {df_valid['factor_signal_line'][-100:]}")
    for i in range(1, len(df_valid)):
        # 1，小时级别MACD金叉（first cross 设置为True）。
        # and df_valid['factor_macd_line'][i-1] < df_valid['factor_signal_line'][i-1]:
        if df_valid['factor_macd_line'][i] > 0.99 * df_valid['factor_signal_line'][i]:
            sell_first_cross = False
            buy_first_cross = True
            print(f"first cross happend!!!")
        # 2，一旦小时级别macd趋势线下跌，则buy first and buy second cross设置为False，first position record=None。
        if df_valid['factor_macd_line'][i] < df_valid['factor_signal_line'][i] and df_valid['factor_macd_line'][i-1] > df_valid['factor_signal_line'][i-1]:
            buy_first_cross = False
            sell_first_cross = True
            print(f"first down cross happend!!!")

        # if buy_first_cross:
        #     # 3，5分钟级别MACD金叉（second cross设置为True，second cross需要在first cross为True时才能设置为True，并且second cross为True后first cross设置为False）并记录点位（first position record）。
        #     if df_valid['macd_line'][i] > df_valid['signal_line'][i] and df_valid['macd_line'][i-1] < df_valid['signal_line'][i-1]:
        #         print(f"second cross happend!!!")
        #         df_valid['buy_signal'][i] = True
        #         buy_position = True
        #         buy_first_cross = False
        #         buy_price = df_valid['close'][i]
        # if buy_position == True:
        #     buy_first_cross = False
        #     if buy_price * 0.99 > df_valid['close'][i]:  # 1%止损点
        #         df_valid['close_signal'][i] = True
        #         buy_position = False
        #         buy_first_down_cross = False
        #         close_fisrt_position_record = None
        #         buy_price = None
        #     # 5，买入后开始记录5分钟级别MACD死叉点和对应的close_fisrt_position_record。
        #     if df_valid['macd_line'][i] < df_valid['signal_line'][i] and df_valid['macd_line'][i-1] > df_valid['signal_line'][i-1] and buy_first_down_cross == False and buy_second_down_cross == False:
        #         buy_first_down_cross = True
        #         close_fisrt_position_record = df_valid['close'][i]
        #     elif buy_first_down_cross == True and buy_second_down_cross == False:
        #         if df_valid['macd_line'][i] < df_valid['signal_line'][i] and df_valid['macd_line'][i-1] > df_valid['signal_line'][i-1]:
        #             # 7，5分钟级别MACD第二次死叉（即first down cross为True后的第一次），先将first down cross设置为False，然后：
        #             buy_second_down_cross = True
        #             buy_first_down_cross = False
        #             # 7.1，如果点位低于first down position record，则卖出。并将所有的记录设置为False（first cross，second cross，first position record，first down cross，first down positin record）
        #             # 理论上需要确认df_valid['buy_first_cross'][i] == True，但似乎是在这个if判断时是100%True
        #             if df_valid['close'][i] < close_fisrt_position_record:
        #                 df_valid['close_signal'][i] = True
        #                 buy_position = False
        #                 buy_first_down_cross = False
        #                 close_fisrt_position_record = None
        #                 buy_price = None
        #             # 7.2，如果点位高于first down position record，则first down cross设置为True，并更新记录(first down position record)；
        #             else:
        #                 buy_second_down_cross = False
        #                 buy_first_down_cross = True
        #                 close_fisrt_position_record = df_valid['close'][i]
        #     elif buy_first_down_cross == True and buy_second_down_cross == True:
        #         raise ValueError("The down cross cannot be both True.")
        # elif buy_position == False:
        #     buy_first_down_cross, buy_second_down_cross = False, False
        #     close_fisrt_position_record = None
        #     buy_price = None
        # # 做空逻辑
        # if sell_first_cross:
        #     # 处理5分钟级别的MACD死叉
        #     if df_valid['macd_line'][i] < df_valid['signal_line'][i] and df_valid['macd_line'][i-1] > df_valid['signal_line'][i-1]:
        #         df_valid['sell_signal'][i] = True
        #         sell_position = True
        #         sell_first_cross = False
        #         sell_price = df_valid['close'][i]
        # if sell_position == True:
        #     sell_first_cross = False
        #     if sell_price < df_valid['close'][i] * 0.99:  # 1%止损点
        #         df_valid['close_signal'][i] = True
        #         sell_position = False
        #         sell_first_down_cross = False
        #         close_fisrt_position_record = None
        #         sell_price = None
        #     # 处理做空后的MACD金叉点
        #     if df_valid['macd_line'][i] < df_valid['signal_line'][i] and df_valid['macd_line'][i-1] > df_valid['signal_line'][i-1] and sell_first_down_cross == False and sell_second_down_cross == False:
        #         sell_first_down_cross = True
        #         close_first_position_record_sell = df_valid['close'][i]
        #     elif sell_first_down_cross == True and sell_second_down_cross == False:
        #         if df_valid['macd_line'][i] < df_valid['signal_line'][i] and df_valid['macd_line'][i-1] > df_valid['signal_line'][i-1]:
        #             # 处理再次金叉
        #             sell_second_down_cross = True
        #             sell_first_down_cross = False
        #             # 如果点位高于first down position record则买入平仓
        #             if df_valid['close'][i] > close_first_position_record_sell:
        #                 df_valid['close_signal'][i] = True
        #                 sell_position = False
        #                 sell_first_down_cross = False
        #                 close_first_position_record_sell = None
        #                 sell_price = None
        #             # 如果点位低于first down position record则更新记录
        #             else:
        #                 sell_second_down_cross = False
        #                 sell_first_down_cross = True
        #                 close_first_position_record_sell = df_valid['close'][i]
        #     elif sell_first_down_cross == True and sell_second_down_cross == True:
        #         raise ValueError("The down cross cannot be both True.")
        # elif sell_position == False:
        #     sell_first_down_cross, sell_second_down_cross = False, False
        #     close_first_position_record_sell = None
        #     sell_price = None

    if is_live:
        last_close = df_valid['close'][-1]
        last_macd = df_valid['factor_macd_line'][-1]
        last_signal = df_valid['factor_signal_line'][-1]

        print(f"Live Trading Status:")
        print(f"  Last Close: {last_close}")
        print(f"  Last factor_MACD: {last_macd}")
        print(f"  Last factor_Signal: {last_signal}")

        if buy_first_cross:
            print(f"Buy First Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
        elif buy_second_cross:
            print(f"Buy Second Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
            print(f"  First Position Record: {buy_fisrt_position_record}")

        if sell_first_cross:
            print(f"Sell First Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
        elif sell_second_cross:
            print(f"Sell Second Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
            print(f"  First Position Record: {sell_fisrt_position_record}")

        if buy_position:
            print(f"Buy Position Active:")
            print(
                f"  Close First Position Record: {close_fisrt_position_record}")
            print(f"  First Down Cross: {buy_first_down_cross}")
            print(f"  Second Down Cross: {buy_second_down_cross}")
        elif sell_position:
            print(f"Sell Position Active:")
            print(
                f"  Close First Position Record: {close_first_position_record_sell}")
            print(f"  First Down Cross: {sell_first_down_cross}")
            print(f"  Second Down Cross: {sell_second_down_cross}")
        else:
            print(f"No Active Positions")

    # 将计算结果填充回原始的 df 字典
    for key in ['close_signal', 'buy_signal', 'sell_signal']:
        arrays[key] = np.zeros(len(arrays['datetime']), dtype=bool)
        arrays[key][valid_indices] = df_valid[key]

    return arrays


def calculate_MACDsignalsZZZ(arrays, start_date=None, end_date=None, is_live=False):
    # 1，小时级别MACD金叉（first cross 设置为True）。
    # 2，一旦小时级别macd趋势线下跌，则second cross设置为False，first position record=None。
    # 3，5分钟级别MACD金叉（second cross设置为True，second cross需要在first cross为True时才能设置为True，并且second cross为True后first cross设置为False）并记录点位（first position record）。
    # 4，5分钟级别MACD再次金叉且高于first position record时买入（second cross设置为False），止损点1%（止损点其他函数处理）。
    # 4.1，如果再次金叉时点位没有高于first position record，则将当前金叉视为first cross，当前价格设置为first position record。
    # 5，买入后开始记录5分钟级别MACD死叉点和对应的close_fisrt_position_record。
    # 6，5分钟级别MACD第一次死叉记录first down cross为True, 并记录点位(first down position record)。
    # 7，5分钟级别MACD第二次死叉（即first down cross为True后的第一次），先将first down cross设置为False，然后：
    # 7.1，如果点位低于first down position record，则卖出。并将所有的记录设置为False（first cross，second cross，first position record，first down cross，first down positin record）
    # 7.2，如果点位高于first down position record，则first down cross设置为True，并更新记录(first down position record)；

    arrays = arrays.copy()

    arrays['datetime'] = np.array(arrays['datetime'], dtype='str')
    if not is_live:
        start_date = str(start_date)
        end_date = str(end_date)
    else:
        end_date = arrays['datetime'][-1]
        start_date = arrays['datetime'][-10000]

    valid_indices = (arrays['datetime'] >= start_date) & (
        arrays['datetime'] <= end_date)

    df_valid = {key: np.array(val)[valid_indices]
                for key, val in arrays.items()}
    print(
        f"we use these daterange for MACD signals: {df_valid['datetime'][0]} to {df_valid['datetime'][-1]}")
    # 创建状态列
    df_valid['buy_signal'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    df_valid['sell_signal'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    df_valid['close_signal'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    # 处理小时级别的MACD金叉
    # 初始化金叉和死叉信号
    buy_first_cross = (df_valid['factor_macd_line'][1:] > df_valid['factor_signal_line'][1:]) & (
        df_valid['factor_macd_line'][:-1] <= df_valid['factor_signal_line'][:-1])
    sell_first_cross = (df_valid['factor_macd_line'][1:] < df_valid['factor_signal_line'][1:]) & (
        df_valid['factor_macd_line'][:-1] >= df_valid['factor_signal_line'][:-1])
    # 在df_valid中添加新的金叉和死叉信号列
    df_valid['buy_first_cross'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    df_valid['sell_first_cross'] = np.full(
        df_valid['datetime'].shape[0], False, dtype=bool)
    # 将金叉和死叉信号填入新列中，注意从第二个元素开始
    df_valid['buy_first_cross'][1:] = buy_first_cross
    df_valid['sell_first_cross'][1:] = sell_first_cross

    # 初始化变量
    buy_first_cross, buy_second_cross = False, False
    buy_fisrt_position_record = None
    buy_position = False
    buy_first_down_cross, buy_second_down_cross = False, False
    close_fisrt_position_record = None

    sell_first_cross, sell_second_cross = False, False
    sell_fisrt_position_record = None
    sell_position = False
    sell_first_down_cross, sell_second_down_cross = False, False
    close_first_position_record_sell = None
    buy_price, sell_price = None, None
    for i in range(1, len(df_valid)):
        # 1，小时级别MACD金叉（first cross 设置为True）。
        if df_valid['buy_first_cross'][i]:
            sell_first_cross = False
            sell_second_cross = False
            sell_fisrt_position_record = None
            if buy_second_cross == False:
                buy_first_cross = True
                print(f"first cross happend!!!")
        # 2，一旦小时级别macd趋势线下跌，则buy first and buy second cross设置为False，first position record=None。
        elif df_valid['sell_first_cross'][i]:
            buy_first_cross = False
            buy_second_cross = False
            buy_fisrt_position_record = None
            if sell_second_cross == False:
                sell_first_cross = True
                print(f"first down cross happend!!!")

        if buy_first_cross:
            # 3，5分钟级别MACD金叉（second cross设置为True，second cross需要在first cross为True时才能设置为True，并且second cross为True后first cross设置为False）并记录点位（first position record）。
            if df_valid['macd_line'][i] > df_valid['signal_line'][i]:
                buy_second_cross = True
                print(f"second cross happend!!!")
                buy_fisrt_position_record = df_valid['close'][i]
                buy_first_cross = False
        elif buy_second_cross == True:
            # 4，5分钟级别MACD再次金叉且高于first position record时买入（second cross设置为False），止损点1%（止损点其他函数处理）。
            # 4.1 如何定义再次金叉
            if (df_valid['macd_line'][i] > df_valid['signal_line'][i]) and (df_valid['macd_line'][i-1] <= df_valid['signal_line'][i-1]):
                buy_second_cross = False
                print(f"third cross happend!!!")
                if df_valid['close'][i] >= buy_fisrt_position_record:
                    df_valid['buy_signal'][i] = True
                    buy_position = True
                    buy_price = df_valid['close'][i]
                # 4.1，如果再次金叉时点位没有高于first position record，则将当前金叉视为second_cross，当前价格设置为first position record。
                # 理论上需要确认df_valid['buy_first_cross'][i] == True，但似乎是在这个if判断时是100%True
                elif df_valid['close'][i] < buy_fisrt_position_record:
                    buy_second_cross = True
                    buy_fisrt_position_record = df_valid['close'][i]
                else:
                    buy_second_cross = False
                    buy_first_cross = False
        if buy_position == True:
            buy_first_cross, buy_second_cross = False, False
            buy_fisrt_position_record = None
            if buy_price * 0.99 > df_valid['close'][i]:  # 1%止损点
                df_valid['close_signal'][i] = True
                buy_position = False
                buy_first_down_cross, buy_second_down_cross = False, False
                close_fisrt_position_record = None
                buy_price = None
            # 5，买入后开始记录5分钟级别MACD死叉点和对应的close_fisrt_position_record。
            if (df_valid['macd_line'][i] < df_valid['signal_line'][i]) and (df_valid['macd_line'][i-1] >= df_valid['signal_line'][i-1]):
                buy_first_down_cross = True
                close_fisrt_position_record = df_valid['close'][i]
            elif buy_first_down_cross == True and buy_second_down_cross == False:
                if (df_valid['macd_line'][i] < df_valid['signal_line'][i]) and (df_valid['macd_line'][i-1] >= df_valid['signal_line'][i-1]):
                    # 7，5分钟级别MACD第二次死叉（即first down cross为True后的第一次），先将first down cross设置为False，然后：
                    buy_second_down_cross = True
                    buy_first_down_cross = False
                    # 7.1，如果点位低于first down position record，则卖出。并将所有的记录设置为False（first cross，second cross，first position record，first down cross，first down positin record）
                    # 理论上需要确认df_valid['buy_first_cross'][i] == True，但似乎是在这个if判断时是100%True
                    if df_valid['close'][i] < close_fisrt_position_record:
                        df_valid['close_signal'][i] = True
                        buy_position = False
                        buy_first_down_cross, buy_second_down_cross = False, False
                        close_fisrt_position_record = None
                        buy_price = None
                    # 7.2，如果点位高于first down position record，则first down cross设置为True，并更新记录(first down position record)；
                    else:
                        buy_second_down_cross = False
                        buy_first_down_cross = True
                        close_fisrt_position_record = df_valid['close'][i]
            elif buy_first_down_cross == True and buy_second_down_cross == True:
                raise ValueError("The down cross cannot be both True.")
        elif buy_position == False:
            buy_first_down_cross, buy_second_down_cross = False, False
            close_fisrt_position_record = None
            buy_price = None
        # 做空逻辑
        if sell_first_cross:
            # 处理5分钟级别的MACD死叉
            if df_valid['macd_line'][i] < df_valid['signal_line'][i]:
                sell_second_cross = True
                sell_fisrt_position_record = df_valid['close'][i]
                sell_first_cross = False
        elif sell_second_cross == True:
            # 处理5分钟级别的再次死叉
            if (df_valid['macd_line'][i] < df_valid['signal_line'][i]) and (df_valid['macd_line'][i-1] >= df_valid['signal_line'][i-1]):
                sell_second_cross = False
                if df_valid['close'][i] < sell_fisrt_position_record:
                    df_valid['sell_signal'][i] = True
                    sell_position = True
                    sell_price = df_valid['close'][i]
                elif df_valid['close'][i] > sell_fisrt_position_record:
                    sell_second_cross = True
                    sell_fisrt_position_record = df_valid['close'][i]
                else:
                    sell_second_cross = False
                    sell_first_cross = False

        if sell_position == True:
            sell_first_cross, sell_second_cross = False, False
            sell_fisrt_position_record = None
            if sell_price < df_valid['close'][i] * 0.99:  # 1%止损点
                df_valid['close_signal'][i] = True
                sell_position = False
                sell_first_down_cross, sell_second_down_cross = False, False
                close_fisrt_position_record = None
                sell_price = None
            # 处理做空后的MACD金叉点
            if (df_valid['macd_line'][i] > df_valid['signal_line'][i]) and (df_valid['macd_line'][i-1] <= df_valid['signal_line'][i-1]):
                sell_first_down_cross = True
                close_first_position_record_sell = df_valid['close'][i]
            elif sell_first_down_cross == True and sell_second_down_cross == False:
                if (df_valid['macd_line'][i] > df_valid['signal_line'][i]) and (df_valid['macd_line'][i-1] <= df_valid['signal_line'][i-1]):
                    # 处理再次金叉
                    sell_second_down_cross = True
                    sell_first_down_cross = False
                    # 如果点位高于first down position record则买入平仓
                    if df_valid['close'][i] > close_first_position_record_sell:
                        df_valid['close_signal'][i] = True
                        sell_position = False
                        sell_first_down_cross, sell_second_down_cross = False, False
                        close_first_position_record_sell = None
                        sell_price = None
                    # 如果点位低于first down position record则更新记录
                    else:
                        sell_second_down_cross = False
                        sell_first_down_cross = True
                        close_first_position_record_sell = df_valid['close'][i]
            elif sell_first_down_cross == True and sell_second_down_cross == True:
                raise ValueError("The down cross cannot be both True.")
        elif sell_position == False:
            sell_first_down_cross, sell_second_down_cross = False, False
            close_first_position_record_sell = None
            sell_price = None

    if is_live:
        last_close = df_valid['close'][-1]
        last_macd = df_valid['factor_macd_line'][-1]
        last_signal = df_valid['factor_signal_line'][-1]

        print(f"Live Trading Status:")
        print(f"  Last Close: {last_close}")
        print(f"  Last factor_MACD: {last_macd}")
        print(f"  Last factor_Signal: {last_signal}")

        if buy_first_cross:
            print(f"Buy First Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
        elif buy_second_cross:
            print(f"Buy Second Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
            print(f"  First Position Record: {buy_fisrt_position_record}")

        if sell_first_cross:
            print(f"Sell First Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
        elif sell_second_cross:
            print(f"Sell Second Cross Detected:")
            print(f"  Current Close: {last_close}")
            print(f"  MACD Line: {last_macd}")
            print(f"  Signal Line: {last_signal}")
            print(f"  First Position Record: {sell_fisrt_position_record}")

        if buy_position:
            print(f"Buy Position Active:")
            print(
                f"  Close First Position Record: {close_fisrt_position_record}")
            print(f"  First Down Cross: {buy_first_down_cross}")
            print(f"  Second Down Cross: {buy_second_down_cross}")
        elif sell_position:
            print(f"Sell Position Active:")
            print(
                f"  Close First Position Record: {close_first_position_record_sell}")
            print(f"  First Down Cross: {sell_first_down_cross}")
            print(f"  Second Down Cross: {sell_second_down_cross}")
        else:
            print(f"No Active Positions")

    # 将计算结果填充回原始的 df 字典
    for key in ['close_signal', 'buy_signal', 'sell_signal']:
        arrays[key] = np.zeros(len(arrays['datetime']), dtype=bool)
        arrays[key][valid_indices] = df_valid[key]
    import matplotlib.pyplot as plt
    dates = np.array([np.datetime64(date) for date in df_valid['datetime']])

    # 创建图表
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 绘制 close 价格在顶部图
    ax1.plot(dates, df_valid['close'], label='Close', color='blue')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')
    ax1.set_title('Close Price')

    # 绘制 factor_macd_line 和 factor_signal_line 在中间图
    ax2.plot(dates, df_valid['factor_macd_line'],
             label='Factor MACD Line', color='green')
    ax2.plot(dates, df_valid['factor_signal_line'],
             label='Factor Signal Line', color='red')
    ax2.set_ylabel('Factor MACD and Signal Lines')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper left')
    ax2.set_title('Factor MACD and Signal Lines')

    # 绘制 macd_line 和 signal_line 在底部图
    ax3.plot(dates, df_valid['macd_line'],
             label='MACD Line', color='purple', linestyle='--')
    ax3.plot(dates, df_valid['signal_line'],
             label='Signal Line', color='orange', linestyle='--')
    ax3.set_ylabel('MACD and Signal Lines')
    ax3.tick_params(axis='y')
    ax3.legend(loc='upper left')
    ax3.set_title('MACD and Signal Lines')

    # 设置 x 轴标签
    ax3.set_xlabel('Date')

    # 调整布局
    plt.tight_layout()

    # 保存图表到文件
    plt.savefig('macd_chart_financial.png')
    return arrays


def calculate_VWAPsignals(df, buy_volume_multiplier, sell_volume_multiplier, start_date=None, end_date=None, is_live=False):
    # df是dict，内为numpy格式的数据
    df = df.copy()
    df['datetime'] = np.array(df['datetime'], dtype='str')

    if not is_live:
        start_date = str(start_date)
        end_date = str(end_date)
        valid_indices = (df['datetime'] >= start_date) & (
            df['datetime'] <= end_date)
        df_valid = {key: np.array(val)[valid_indices]
                    for key, val in df.items()}
    else:
        end_date = df['datetime'][-1]
        start_date = df['datetime'][-100]
        valid_indices = (df['datetime'] >= start_date) & (
            df['datetime'] <= end_date)
        df_valid = {key: np.array(val)[valid_indices]
                    for key, val in df.items()}
    # 计算买卖条件和信号
    # 在 df_valid 上计算买卖条件和信号
    df_valid['buy_condition1'] = df_valid['close'] > df_valid['buy_vwap']
    df_valid['buy_condition2'] = np.roll(df_valid['volume'], shift=1) > df_valid['buy_avg_volume'] * \
        buy_volume_multiplier * (df_valid['buy_atr'] / df_valid['buy_atr_sma'])
    df_valid['buy_condition3'] = df_valid['volume'] > df_valid['buy_avg_volume'] * \
        buy_volume_multiplier * (df_valid['buy_atr'] / df_valid['buy_atr_sma'])
    df_valid['buy_signal'] = df_valid['buy_condition1'] & (
        df_valid['buy_condition2'] | df_valid['buy_condition3'])

    df_valid['sell_condition1'] = df_valid['close'] < df_valid['sell_vwap']
    df_valid['sell_condition2'] = np.roll(df_valid['volume'], shift=1) > df_valid['sell_avg_volume'] * \
        sell_volume_multiplier * \
        (df_valid['sell_atr'] / df_valid['sell_atr_sma'])
    df_valid['sell_condition3'] = df_valid['volume'] > df_valid['sell_avg_volume'] * \
        sell_volume_multiplier * \
        (df_valid['sell_atr'] / df_valid['sell_atr_sma'])
    df_valid['sell_signal'] = df_valid['sell_condition1'] & (
        df_valid['sell_condition2'] | df_valid['sell_condition3'])

    # 将计算结果填充回原始的 df 字典
    for key in ['buy_condition1', 'buy_condition2', 'buy_condition3', 'buy_signal', 'sell_condition1', 'sell_condition2', 'sell_condition3', 'sell_signal']:
        df[key] = np.zeros(len(df['datetime']), dtype=bool)
        df[key][valid_indices] = df_valid[key]

    # 打印信号条件（如果是实盘）
    if is_live:
        last_close = df['close'][-1]
        buy_vwap = df['buy_vwap'][-1]
        sell_vwap = df['sell_vwap'][-1]
        buy_atr = df['buy_atr'][-1]
        sell_atr = df['sell_atr'][-1]
        buy_atr_sma = df['buy_atr_sma'][-1]
        sell_atr_sma = df['sell_atr_sma'][-1]
        last_volume = df['volume'][-1]
        pre_volume = df['volume'][-2]
        buy_avg_volume = df['buy_avg_volume'][-1]
        sell_avg_volume = df['sell_avg_volume'][-1]

        print(f"Buy Signal Conditions:")
        print(
            f"  Condition 1: {df['buy_condition1'][-1]} (last_close: {last_close}, buy_vwap: {buy_vwap})")
        print(
            f"  Condition 2: {df['buy_condition2'][-1]} (pre_volume: {pre_volume}, expected: {buy_avg_volume * buy_volume_multiplier * (buy_atr / buy_atr_sma)})")
        print(
            f"  Condition 3: {df['buy_condition3'][-1]} (last_volume: {last_volume}, expected: {buy_avg_volume * buy_volume_multiplier * (buy_atr / buy_atr_sma)})")

        print(f"Sell Signal Conditions:")
        print(
            f"  Condition 1: {df['sell_condition1'][-1]} (last_close: {last_close}, sell_vwap: {sell_vwap})")
        print(
            f"  Condition 2: {df['sell_condition2'][-1]} (pre_volume: {pre_volume}, expected: {sell_avg_volume * sell_volume_multiplier * (sell_atr / sell_atr_sma)})")
        print(
            f"  Condition 3: {df['sell_condition3'][-1]} (last_volume: {last_volume}, expected: {sell_avg_volume * sell_volume_multiplier * (sell_atr / sell_atr_sma)})")

    return df


def VWAPStrategy(df, end_date, start_date, sell_vwap_period, buy_vwap_period, sell_volume_window, buy_volume_window, sell_atr_period, buy_atr_period, buy_volume_multiplier, sell_volume_multiplier, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, buy_risk_per_trade, sell_risk_per_trade, buy_atr_sma_period, sell_atr_sma_period, startcash=100000, com=0.0005, numpy_arrays=None):
    strategy_start = time.time()
    # 建立一个数据集（不包含初始化），包含计算投资组合指标和技术指标所需的列。根据开始日期和结束日期切割交易数据
    params = {}
    params['buy_risk_per_trade'] = buy_risk_per_trade
    params['sell_risk_per_trade'] = sell_risk_per_trade
    if numpy_arrays == None:
        init_columns = {'buy_vwap': 0,
                        'buy_atr': 0,
                        'buy_atr_sma': 0,
                        'sell_vwap': 0,
                        'sell_atr': 0,
                        'sell_atr_sma': 0}
        data_all = npb.load_data_bydate(
            df=df, end_date=end_date, start_date=start_date, init_columns=init_columns)
        # print(f"the times is from : {start_date} to {end_date}")
        data_start, data_end = data_all.index.min(), data_all.index.max()
        print(f"we prepare data from {data_start} to {data_end}")
        # print(f"load data costed {load_data} seconds")
        # 用指定值初始化这个数据集，并给出正确的开始日期结束日期的index
        numpy_arrays, initial_date, end_idx = initialize_numpy_arrays(
            data_all=data_all, start_date=start_date, end_date=end_date, startcash=startcash)
    else:
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(start_date))[0]
        if matched_indices.size > 0:
            initial_date = matched_indices[0]
        else:
            print(
                f"the start date in numpy arrays is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
            raise ValueError("未在数据中找到起始日期。")
        matched_indices = np.where(
            numpy_arrays['datetime'] == str(end_date))[0]
        if matched_indices.size > 0:
            end_idx = matched_indices[0]
        else:
            print(
                f"the end date in numpy arrays is {numpy_arrays['datetime'][-1]} and end_date is {end_date}")
            end_idx = len(numpy_arrays['datetime']) - 1
            # raise ValueError("未在数据中找到结束日期。")
    # backtest_time = time_module.time()
    # max_length是为了计算各类技术指标而所需的最大长度
    max_length = max(sell_volume_window, buy_volume_window, sell_vwap_period,
                     buy_vwap_period, sell_atr_period, buy_atr_period, 1500) + 1
    max_length = int(round(max_length, 0))
    numpy_arrays = npb.calculate_indicators_by_date(start=initial_date, end=end_idx, arrays=numpy_arrays,
                                                    buy_vwap_period=buy_vwap_period, buy_atr_period=buy_atr_period,
                                                    buy_atr_sma_period=buy_atr_sma_period,
                                                    sell_vwap_period=sell_vwap_period, sell_atr_period=sell_atr_period,
                                                    sell_atr_sma_period=sell_atr_sma_period, buy_volume_window=buy_volume_window,
                                                    sell_volume_window=sell_volume_window,
                                                    max_length=max_length)
    # Re-calculating the conditions and signals
    numpy_arrays = calculate_VWAPsignals(df=numpy_arrays, buy_volume_multiplier=buy_volume_multiplier,
                                         start_date=start_date, end_date=end_date, sell_volume_multiplier=sell_volume_multiplier)
    # print(f"the times is from : {numpy_arrays['datetime'][0]} to {numpy_arrays['datetime'][-1]}")
    print(f"Going to cut arrays to {start_date} to {end_date}")
    # start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S')
    # end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M:%S')
    matched_indices = np.where(numpy_arrays['datetime'] == str(start_date))[0]
    if matched_indices.size > 0:
        start_idx = matched_indices[0]
    else:
        print(
            f"the first date is {numpy_arrays['datetime'][0]} and start_date is {start_date}")
        raise ValueError("未在数据中找到起始日期。")
    # end_idx = np.where(numpy_arrays['datetime'] == str(end_date))[0][0]
    print(
        f"We are going to backtest the first day {numpy_arrays['datetime'][start_idx]} to the last day {numpy_arrays['datetime'][end_idx]}")
    # for columns in numpy_arrays.keys():
    # print(f"check the value of {columns} is {numpy_arrays[columns][0]}")
    # print(f"check the value of {columns} is {numpy_arrays[columns][-1]}")
    execute = time.time()
    for idx_i in range(start_idx, end_idx+1):
        trade = npb.TradeLogic(
            df=numpy_arrays,
            index=idx_i,
            buy_stop_loss=buy_stop_loss,
            sell_stop_loss=sell_stop_loss,
            buy_take_profit=buy_take_profit,
            sell_take_profit=sell_take_profit,
            strategy_name='VWAP_Strategy',
            params=params,
            com=com)
        trade.execute_trade()
    execute_cost = time.time() - execute
    # 如果在end_idx的时刻正好激活交易信号，那么open_direction和open_cost_price会被重置为None，因而cash被重置为None，导致net_value为None，所以在这种情况下net_value设置为end_idx-1的值
    numpy_arrays['net_value'][end_idx] = numpy_arrays['net_value'][end_idx-1]
    print(f"the end date is {numpy_arrays['datetime'][end_idx]}")
    print(f"the net value is {numpy_arrays['net_value'][end_idx]}")
    print(f"the trade execute costed {execute_cost} seconds")
    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"{current_time} - the day {numpy_arrays['datetime'][end_idx]} of value is {numpy_arrays['net_value'][end_idx]}")
    # print(f"the backtest {sorted_times[-1]}_{lambda_1}_{smooth_alpha} costed {time_module.time() - backtest_time} seconds")
    # 检查一下numpy_arrays['datetime'][0],numpy_arrays['datetime'][-1]是否等于start_date,end_date，注意格式
    # if str(numpy_arrays['datetime'][0]) != str(start_date) or str(numpy_arrays['datetime'][-1]) != str(end_date):
    if str(numpy_arrays['datetime'][start_idx]) != str(start_date):
        print(
            f"the start date is {start_date} but numpy_arrays['datetime'][0] is {numpy_arrays['datetime'][0]}")
        raise ValueError(
            "The start and end dates do not match the datetime array.")
    if str(numpy_arrays['datetime'][end_idx]) != str(end_date):
        print(
            f"the end date is {end_date} but numpy_arrays['datetime'][end_idx] is {numpy_arrays['datetime'][end_idx]}")
    return numpy_arrays


def calculate_performance(results, start_date, end_date, num_days, strategy_name, startcash):
    start_time = time.time()
    results_a = results.copy()
    print(f"performance start is {start_date}, end is {end_date}")
    valid_indices = np.where((results_a['datetime'] >= str(
        start_date)) & (results_a['datetime'] <= str(end_date)))
    results = {key: np.array(val)[valid_indices]
               for key, val in results_a.items()}
    print(
        f"after cutting, the results min datetime is {results['datetime'][0]}, the max is {results['datetime'][-1]}")
    # 确保 start_date 是一个 datetime 对象
# 计算以30天为滚动窗口的最大回撤，取最小值。例如如果测试周期为31天，那么就是一个2位的list，返回其中的最小值（负数）。
    drawdown = npb.MaxDrawdown(
        numpy_arrays=results, start_date=start_date, end_date=end_date, rolling_window=35)
    drawdown = round(drawdown, 4)
    total_trades, win_rate, sqn, turnover_rate, trade_results, trade_details = npb.calculate_trades_win_rate_and_sqn_from_dict(
        numpy_arrays=results, start_date=start_date, end_date=end_date)
    final_value = results['net_value'][-1]

    # 将numpy_arrays['datetime']转换为datetime类型
    datetime = pd.to_datetime(results['datetime'])
    # 创建一个新的DataFrame，包含datetime和net_value
    date_netvalue = pd.DataFrame(
        {'datetime': datetime, 'net_value': results['net_value']})
    date_netvalue.set_index('datetime', inplace=True)
    # 将分钟级别的数据转换为每日数据
    df_daily = date_netvalue.resample('D').last().dropna()
    # 如果只有一天的数据，年化波动率和夏普比率为0或NaN
    if len(df_daily) <= 1:
        daily_returns = 0
        # 计算年化收益率
        annual_returns = 0
        # 计算每日波动率
        daily_volatility = 0
        # 计算年化波动率
        annual_volatility = 0
        VWR = 0
        Sharpe_ratio = 0
    else:
        # 计算每日收益率
        daily_returns = df_daily['net_value'].pct_change().dropna()
        # 计算年化收益率,虽然比特币是365天交易，但为了和其他资产比较，我们使用252天
        annual_returns = (1 + daily_returns).prod() ** (252 /
                                                        len(daily_returns)) - 1
        # 计算每日波动率
        daily_volatility = daily_returns.std()
        # 计算年化波动率
        annual_volatility = daily_volatility * np.sqrt(252)
        # 计算夏普比率
        Sharpe_ratio = (annual_returns - 0) / (annual_volatility + 0.0001)

    # 设置最大可接受的sigma_P（投资者限制）
    sigma_max = 0.3  # 这个值需要你自己设定
    # 设置权重随波动性增加而降低的速率（投资者容忍度）
    tau = 2  # 这个值需要你自己设定
    # 计算VWR
    VWR = annual_returns * (1 - (annual_volatility / sigma_max) ** tau)

    positions = results['position']
    net_values = results['net_value']

    df_hourly = date_netvalue.resample('H').last().dropna()
    df_hourly_returns = df_hourly['net_value'].pct_change().dropna()

    close_prices = results['close']
    df_BTC = pd.DataFrame({'close': close_prices}, index=datetime)
    # 将分钟级别的数据转换为每日数据
    df_BTC_hourly = df_BTC.resample('H').last().dropna()
    # 计算每日收益率
    df_BTC_hourly_returns = df_BTC_hourly['close'].pct_change().dropna()

    try:
        covariance = np.cov(df_BTC_hourly_returns, df_hourly_returns)[0][1]
        variance = np.var(df_BTC_hourly_returns)
        beta = covariance / variance
    except ValueError:
        print("df_BTC_hourly_returns min datetime:", df_BTC_hourly.index.min())
        print("df_BTC_hourly_returns max datetime:", df_BTC_hourly.index.max())
        print("df_hourly_returns min datetime:", df_hourly.index.min())
        print("df_hourly_returns max datetime:", df_hourly.index.max())
        raise ValueError(
            "The dimensions of df_BTC_hourly_returns and df_hourly_returns do not match.")

    # 计算平均回报率
    avg_portfolio_return = df_hourly_returns.mean()
    avg_market_return = df_BTC_hourly_returns.mean()

    # 无风险利率
    risk_free_rate = 0.02  # 示例值，实际值应根据当前市场条件确定
    # 计算Alpha
    alpha = avg_portfolio_return - \
        (risk_free_rate + beta * (avg_market_return - risk_free_rate))

    num_days = len(df_daily)

    df_daily['previous_net_value'] = df_daily['net_value'].shift(1)
# 比较当前天和前一天的净值，计算增加的天数
    profitable_days = (df_daily['net_value'] >
                       df_daily['previous_net_value']).sum()

    profitable_2_days = 0
    if len(df_daily) >= 2:
        for i in range(0, len(df_daily) - 2, 2):
            if i + 2 < len(df_daily) and df_daily['net_value'].iloc[i + 2] > df_daily['net_value'].iloc[i]:
                profitable_2_days += 1
        # 检查最后一组是否包含在内
        if df_daily['net_value'].iloc[-1] > df_daily['net_value'].iloc[-2]:
            profitable_2_days += 1
    else:
        profitable_2_days = 0

    # 计算profitable_3_days
    profitable_3_days = 0
    if len(df_daily) >= 3:
        for i in range(0, len(df_daily) - 3, 3):
            if i + 3 < len(df_daily) and df_daily['net_value'].iloc[i + 3] > df_daily['net_value'].iloc[i]:
                profitable_3_days += 1
        # 检查最后一组是否包含在内
        if df_daily['net_value'].iloc[-1] > df_daily['net_value'].iloc[len(df_daily) % 3]:
            profitable_3_days += 1
    else:
        profitable_3_days = 0

    profitable_1days_ratio = profitable_days / num_days
    profitable_2days_ratio = profitable_2_days / \
        max(1, math.ceil((num_days - 1) / 2))
    profitable_3days_ratio = profitable_3_days / \
        max(1, math.ceil((num_days - 1) / 3))

    performances = {
        'strategy_name': strategy_name,
        'value': final_value,
        'total_return': (final_value / startcash - 1.0),
        'sqn': sqn,
        'alpha': alpha,
        'profitable_1days_ratio': profitable_1days_ratio,
        'profitable_2days_ratio': profitable_2days_ratio,
        'profitable_3days_ratio': profitable_3days_ratio,
        'drawdown_alpha': (1-(drawdown))*alpha,
        'win_ratio_alpha': alpha * win_rate,
        'drawdown_win_ratio_alpha': (1-(drawdown)) * alpha * win_rate,
        'drawdown': -drawdown,
        'drawdown_value': (-drawdown) + (final_value / startcash - 1.0),
        'total_trades': total_trades,
        'winning_trades':  len([trade for trade in trade_results if trade > 0]),
        'Win ratio': win_rate,
        'vwr': VWR,
        'beta': beta,
        'turnover_rate': turnover_rate,
        'daily_turnover_rate': turnover_rate/num_days,
        'daily_trade': total_trades/num_days,
        'win ratio value': win_rate * final_value,
        'drawdown value': (1-(drawdown)) * final_value,
        'annual_volatility': annual_volatility,
        'SharpeDIY': Sharpe_ratio,
        'sqn_drawdown': sqn * (1-(drawdown)),
        'drawdown_win_ratio': (1-(drawdown)) * win_rate,
        'drawdown_win_ratio_value': (1-(drawdown)) * win_rate * final_value,
        'trade_results': trade_results,
        'trade_details': trade_details
    }
    # 每4天至少交易一次
    min_trades = num_days/3
    punish_rate = performances['total_trades'] * (1/min_trades)
    if punish_rate > 1:
        punish_rate = 1
    print(
        f"the num_days is {num_days}, the total trades number is {performances['total_trades']}, the punish rate is {punish_rate}")
    performances['weighted_win_ratio'] = punish_rate * \
        performances['Win ratio']
    performances['weighted_sqn'] = punish_rate * performances['sqn']
    performances['weighted_profitable_1days_ratio'] = (
        1-(drawdown)) * performances['profitable_1days_ratio']
    performances['weighted_profitable_2days_ratio'] = (
        1-(drawdown)) * performances['profitable_2days_ratio']
    performances['weighted_profitable_3days_ratio'] = (
        1-(drawdown)) * performances['profitable_3days_ratio']
    performances['weighted_alpha'] = punish_rate * performances['alpha']
    performances['weighted_drawdonw_alpha'] = punish_rate * \
        performances['drawdown_alpha']
    performances['weighted_drawdown_win_ratio_alpha'] = punish_rate * \
        performances['drawdown_win_ratio_alpha']
    performances['weighted_win_ratio_alpha'] = punish_rate * \
        performances['win_ratio_alpha']
    performances['weighted_value'] = punish_rate * performances['value']
    performances['weighted_drawdown'] = punish_rate * (1-drawdown)
    performances['weighted_SharpeDIY'] = punish_rate * \
        performances['SharpeDIY']
    performances['weighted_drawdown_value'] = punish_rate * \
        performances['drawdown value']
    performances['weighted_win_ratio_value'] = punish_rate * \
        performances['win ratio value']
    performances['weighted_vwr'] = punish_rate * performances['vwr']
    performances['weighted_sqn_drawdown'] = punish_rate * \
        performances['sqn_drawdown']
    performances['weighted_drawdown_win_ratio'] = punish_rate * \
        performances['drawdown_win_ratio']
    performances['weighted_drawdown_win_ratio_value'] = punish_rate * \
        performances['drawdown_win_ratio_value']
    performances['weighted_drawdown_value_beta'] = punish_rate * \
        performances['drawdown value'] * (10+performances['beta'])

    cost_time = time.time() - start_time
    print(f"performance calculation costed {cost_time} secondes")
    return performances


def main(df, end_date, start_date, sell_vwap_period, buy_vwap_period, sell_volume_window, buy_volume_window, sell_atr_period, buy_atr_period, buy_volume_multiplier, sell_volume_multiplier, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, buy_risk_per_trade, sell_risk_per_trade, buy_atr_sma_period, sell_atr_sma_period, strategy_name, printlog=True, para_opt=True, startcash=1000000, com=0.0005, opt_target=None):
    start_time = time.time()
    sell_volume_window = round(sell_volume_window)
    buy_volume_window = round(buy_volume_window)
    buy_vwap_period = round(buy_vwap_period)
    sell_vwap_period = round(sell_vwap_period)
    sell_atr_period = round(sell_atr_period)
    buy_atr_period = round(buy_atr_period)
    if buy_atr_sma_period != None:
        buy_atr_sma_period = round(buy_atr_sma_period)
        sell_atr_sma_period = round(sell_atr_sma_period)
    else:
        buy_atr_sma_period = round(buy_atr_period)
        sell_atr_sma_period = round(sell_atr_period)

    # 将字符串转换为Timestamp对象
    start_strp = pd.to_datetime(start_date)
    end_strp = pd.to_datetime(end_date)

    # 计算两个日期之间的差异
    delta = end_strp - start_strp
    num_days = delta.days
    print(f'our trade is {num_days} days.')

    results = VWAPStrategy(df=df,
                           end_date=end_date,
                           start_date=start_date,
                           sell_atr_period=sell_atr_period,
                           buy_atr_period=buy_atr_period,
                           sell_volume_window=sell_volume_window,
                           buy_volume_window=buy_volume_window,
                           buy_volume_multiplier=buy_volume_multiplier,
                           sell_volume_multiplier=sell_volume_multiplier,
                           buy_stop_loss=buy_stop_loss,
                           sell_stop_loss=sell_stop_loss,
                           buy_vwap_period=buy_vwap_period,
                           sell_vwap_period=sell_vwap_period,
                           buy_take_profit=buy_take_profit,
                           sell_take_profit=sell_take_profit,
                           buy_risk_per_trade=buy_risk_per_trade,
                           sell_risk_per_trade=sell_risk_per_trade,
                           buy_atr_sma_period=buy_atr_sma_period,
                           sell_atr_sma_period=sell_atr_sma_period,
                           startcash=100000)

    # results.fillna(method='ffill', inplace=True)
    performances = calculate_performance(results=results, start_date=start_date, end_date=end_date,
                                         num_days=num_days, strategy_name=strategy_name, startcash=startcash)
    end = time.time()
    total_time = end - start_time
    print(f"Cost time: {total_time}")

    if not para_opt:
        print('------------------------------------')
        print(f"Total_trades is {performances['total_trades']}, Win trades are {performances['winning_trades']}, Win ratio is {performances['Win ratio']}, drawdown is {performances['drawdown']}, beta is {performances['beta']}, SharpeDIY is {performances['SharpeDIY']}, sqn is {performances['sqn']}, value is {performances['value']}")
        print(
            f"profitable_1days_ratio is {performances['profitable_1days_ratio']}, profitable_2days_ratio are {performances['profitable_2days_ratio']}, profitable_3days_ratio is {performances['profitable_3days_ratio']}")
        '''
        # Create directory if not exist
        directory = "trading_pictures"
        if not os.path.exists(directory):
            os.makedirs(directory)
        def create_filename(strategy_name, start_date, end_date, opt_target, plot_index):
            return f"{strategy_name}_{start_date}_to_{end_date}_{opt_target}_{plot_index}.png"

        # 策略名称、当前时间、数据开始和结束日期
        strategy_name = 'VWAPStrategyLong'
        directory = "trading_pictures"

        if not os.path.exists(directory):
            os.makedirs(directory)
        # 创建绘图
        fig, axs = plt.subplots(4, 1, figsize=(18, 40), dpi=100)

        # 净值曲线图
        axs[0].plot(results['net_value'], label='Net Value')
        axs[0].set_title('Net Value Curve')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Net Value')
        axs[0].legend()

        # 每日收益率分布图
        axs[1].hist(daily_returns, bins=50, alpha=0.6, color='blue')
        axs[1].set_title('Daily Returns Distribution')
        axs[1].set_xlabel('Daily Returns')
        axs[1].set_ylabel('Frequency')

        # 净值与比特币价格对比图
        ax2 = axs[2].twinx()
        axs[2].plot(results['net_value'], label='Net Value', color='g')
        ax2.plot(df_daily['close'], label='BTC Price', color='b')
        axs[2].set_title('Net Value vs Bitcoin Price')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Net Value')
        ax2.set_ylabel('BTC Price')
        axs[2].legend(loc='upper left')
        ax2.legend(loc='upper right')

        # 循环保存图表
        for i in range(3):
            filename = create_filename(strategy_name, start_date, end_date, opt_target, i)
            filepath = os.path.join(directory, filename)
            fig.savefig(filepath, dpi=100)
            plt.close(fig)    
        '''
        return performances

    if para_opt:
        if opt_target is None:
            raise ValueError(
                "Please specify an optimization target. You can use: 'sharpe', 'value', 'sqn', 'drawdown'")
        print(f'Now the optimize target is: {opt_target}')
        return performances


def rsi_main(df, end_date, start_date, rsi_window, ma_short_period, ma_long_period, buy_rsi_threshold, sell_rsi_threshold, open_percent, buy_Apoint, buy_Bpoint, sell_Apoint, sell_Bpoint, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, strategy_name, printlog=True, para_opt=True, startcash=1000000, com=0.0005, opt_target=None):
    start_time = time.time()

    # 将字符串转换为Timestamp对象
    start_strp = pd.to_datetime(start_date)
    end_strp = pd.to_datetime(end_date)

    # 计算两个日期之间的差异
    delta = end_strp - start_strp
    num_days = delta.days
    print(f'our trade is {num_days} days.')

    results = RSIV_Strategy(df=df,
                            end_date=end_date,
                            start_date=start_date,
                            rsi_window=rsi_window,
                            ma_short_period=ma_short_period,
                            ma_long_period=ma_long_period,
                            buy_rsi_threshold=buy_rsi_threshold,
                            sell_rsi_threshold=sell_rsi_threshold,
                            open_percent=open_percent,
                            buy_Apoint=buy_Apoint,
                            buy_Bpoint=buy_Bpoint,
                            sell_Apoint=sell_Apoint,
                            sell_Bpoint=sell_Bpoint,
                            buy_stop_loss=buy_stop_loss,
                            sell_stop_loss=sell_stop_loss,
                            buy_take_profit=buy_take_profit,
                            sell_take_profit=sell_take_profit,
                            strategy_name=strategy_name,
                            startcash=100000)
    print(f"the results end date is {results['datetime'][-2]}")
    print(f"the results net value is {results['net_value'][-2]}")
    # results.fillna(method='ffill', inplace=True)
    performances = calculate_performance(results=results, start_date=start_date, end_date=end_date,
                                         num_days=num_days, strategy_name=strategy_name, startcash=startcash)
    end = time.time()
    total_time = end - start_time
    print(f"Cost time: {total_time}")

    if not para_opt:
        print('------------------------------------')
        print(f"Total_trades is {performances['total_trades']}, Win trades are {performances['winning_trades']}, Win ratio is {performances['Win ratio']}, drawdown is {performances['drawdown']}, beta is {performances['beta']}, SharpeDIY is {performances['SharpeDIY']}, sqn is {performances['sqn']}, value is {performances['value']}")
        print(
            f"profitable_1days_ratio is {performances['profitable_1days_ratio']}, profitable_2days_ratio are {performances['profitable_2days_ratio']}, profitable_3days_ratio is {performances['profitable_3days_ratio']}")
        return performances

    if para_opt:
        if opt_target is None:
            raise ValueError(
                "Please specify an optimization target. You can use: 'sharpe', 'value', 'sqn', 'drawdown'")
        print(f'Now the optimize target is: {opt_target}')
        return performances


def wrsi_main(df, end_date, start_date, rsi_window, factor, ma_short_period, ma_long_period, buy_rsi_threshold, sell_rsi_threshold, open_percent, buy_Apoint, buy_Bpoint, sell_Apoint, sell_Bpoint, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, strategy_name, printlog=True, para_opt=True, startcash=1000000, com=0.0005, opt_target=None):
    start_time = time.time()

    # 将字符串转换为Timestamp对象
    start_strp = pd.to_datetime(start_date)
    end_strp = pd.to_datetime(end_date)

    # 计算两个日期之间的差异
    delta = end_strp - start_strp
    num_days = delta.days
    print(f'our trade is {num_days} days.')

    results = WRSIStrategy(df=df,
                           end_date=end_date,
                           start_date=start_date,
                           rsi_window=rsi_window,
                           factor=factor,
                           ma_short_period=ma_short_period,
                           ma_long_period=ma_long_period,
                           buy_rsi_threshold=buy_rsi_threshold,
                           sell_rsi_threshold=sell_rsi_threshold,
                           open_percent=open_percent,
                           buy_Apoint=buy_Apoint,
                           buy_Bpoint=buy_Bpoint,
                           sell_Apoint=sell_Apoint,
                           sell_Bpoint=sell_Bpoint,
                           buy_stop_loss=buy_stop_loss,
                           sell_stop_loss=sell_stop_loss,
                           buy_take_profit=buy_take_profit,
                           sell_take_profit=sell_take_profit,
                           strategy_name="WRSIStrategy",
                           startcash=100000)
    print(f"the results end date is {results['datetime'][-2]}")
    print(f"the results net value is {results['net_value'][-2]}")
    # results.fillna(method='ffill', inplace=True)
    performances = calculate_performance(results=results, start_date=start_date, end_date=end_date,
                                         num_days=num_days, strategy_name=strategy_name, startcash=startcash)
    end = time.time()
    total_time = end - start_time
    print(f"Cost time: {total_time}")

    if not para_opt:
        print('------------------------------------')
        print(f"Total_trades is {performances['total_trades']}, Win trades are {performances['winning_trades']}, Win ratio is {performances['Win ratio']}, drawdown is {performances['drawdown']}, beta is {performances['beta']}, SharpeDIY is {performances['SharpeDIY']}, sqn is {performances['sqn']}, value is {performances['value']}")
        print(
            f"profitable_1days_ratio is {performances['profitable_1days_ratio']}, profitable_2days_ratio are {performances['profitable_2days_ratio']}, profitable_3days_ratio is {performances['profitable_3days_ratio']}")
        return performances

    if para_opt:
        if opt_target is None:
            raise ValueError(
                "Please specify an optimization target. You can use: 'sharpe', 'value', 'sqn', 'drawdown'")
        print(f'Now the optimize target is: {opt_target}')
        return performances


def macd_main(df, end_date, start_date, short_period, long_period, signal_period, L_short_period, L_long_period, L_signal_period, M, factor, alpha_factor, strategy_name, printlog=True, para_opt=True, startcash=1000000, com=0.0005, opt_target=None):
    start_time = time.time()

    short_period = short_period * M
    long_period = long_period * M
    signal_period = signal_period * M

    # 将字符串转换为Timestamp对象
    start_strp = pd.to_datetime(start_date)
    end_strp = pd.to_datetime(end_date)

    # 计算两个日期之间的差异
    delta = end_strp - start_strp
    num_days = delta.days
    print(f'our trade is {num_days} days.')

    results = MACD_Strategy(df=df,
                            end_date=end_date,
                            start_date=start_date,
                            short_period=short_period,
                            long_period=long_period,
                            signal_period=signal_period,
                            L_short_period=L_short_period,
                            L_long_period=L_long_period,
                            L_signal_period=L_signal_period,
                            factor=factor,
                            alpha_factor=alpha_factor,
                            strategy_name=strategy_name,
                            startcash=100000)
    print(f"the results end date is {results['datetime'][-2]}")
    print(f"the results net value is {results['net_value'][-2]}")
    # results.fillna(method='ffill', inplace=True)
    performances = calculate_performance(results=results, start_date=start_date, end_date=end_date,
                                         num_days=num_days, strategy_name=strategy_name, startcash=startcash)
    end = time.time()
    total_time = end - start_time
    print(f"Cost time: {total_time}")

    if not para_opt:
        print('------------------------------------')
        print(f"Total_trades is {performances['total_trades']}, Win trades are {performances['winning_trades']}, Win ratio is {performances['Win ratio']}, drawdown is {performances['drawdown']}, beta is {performances['beta']}, SharpeDIY is {performances['SharpeDIY']}, sqn is {performances['sqn']}, value is {performances['value']}")
        print(
            f"profitable_1days_ratio is {performances['profitable_1days_ratio']}, profitable_2days_ratio are {performances['profitable_2days_ratio']}, profitable_3days_ratio is {performances['profitable_3days_ratio']}")
        return performances

    if para_opt:
        if opt_target is None:
            raise ValueError(
                "Please specify an optimization target. You can use: 'sharpe', 'value', 'sqn', 'drawdown'")
        print(f'Now the optimize target is: {opt_target}')
        return performances


def mav_main(df, end_date, start_date, short_period, long_factor, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, buy_short_volume, sell_short_volume, buy_long_factor, sell_long_factor, strategy_name, printlog=True, para_opt=True, startcash=1000000, com=0.0005, opt_target=None):
    start_time = time.time()
    # 将字符串转换为Timestamp对象
    start_strp = pd.to_datetime(start_date)
    end_strp = pd.to_datetime(end_date)

    # 计算两个日期之间的差异
    delta = end_strp - start_strp
    num_days = delta.days
    print(f'our trade is {num_days} days.')

    results = MAV_Strategy(df=df,
                           end_date=end_date,
                           start_date=start_date,
                           short_period=short_period,
                           buy_stop_loss=buy_stop_loss,
                           sell_stop_loss=sell_stop_loss,
                           buy_take_profit=buy_take_profit,
                           sell_take_profit=sell_take_profit,
                           buy_short_volume=buy_short_volume,
                           sell_short_volume=sell_short_volume,
                           long_factor=long_factor,
                           buy_long_factor=buy_long_factor,
                           sell_long_factor=sell_long_factor,
                           strategy_name=strategy_name,
                           startcash=100000)
    print(f"the results end date is {results['datetime'][-2]}")
    print(f"the results net value is {results['net_value'][-2]}")
    # results.fillna(method='ffill', inplace=True)
    performances = calculate_performance(results=results, start_date=start_date, end_date=end_date,
                                         num_days=num_days, strategy_name=strategy_name, startcash=startcash)
    end = time.time()
    total_time = end - start_time
    print(f"Cost time: {total_time}")

    if not para_opt:
        print('------------------------------------')
        print(f"Total_trades is {performances['total_trades']}, Win trades are {performances['winning_trades']}, Win ratio is {performances['Win ratio']}, drawdown is {performances['drawdown']}, beta is {performances['beta']}, SharpeDIY is {performances['SharpeDIY']}, sqn is {performances['sqn']}, value is {performances['value']}")
        print(
            f"profitable_1days_ratio is {performances['profitable_1days_ratio']}, profitable_2days_ratio are {performances['profitable_2days_ratio']}, profitable_3days_ratio is {performances['profitable_3days_ratio']}")
        return performances

    if para_opt:
        if opt_target is None:
            raise ValueError(
                "Please specify an optimization target. You can use: 'sharpe', 'value', 'sqn', 'drawdown'")
        print(f'Now the optimize target is: {opt_target}')
        return performances


def early_stopping_check(study, trial, early_stopping_rounds=10, start_early_stopping_round=None):

    if isinstance(start_early_stopping_round, int) and trial.number < start_early_stopping_round:
        return  # Too early for stopping

    current_trial_number = trial.number
    best_trial_number = study.best_trial.number
    should_stop = (current_trial_number -
                   best_trial_number) >= early_stopping_rounds
    if should_stop:
        print(
            f"Early stopping triggered after {early_stopping_rounds} rounds without improvement.")
        study.stop()

# 示例：针对不同策略定义不同的 main_opt 函数


def main_opt_vwap_long(trial, df, strategy_name, end_date, start_date, target):
    kwargs = {
        'sell_volume_window': trial.suggest_int('sell_volume_window', 60, 1500, step=10),
        'buy_volume_window': trial.suggest_int('buy_volume_window', 60, 1500, step=10),
        'buy_volume_multiplier': trial.suggest_float('buy_volume_multiplier', 0.1, 12, step=0.1),
        'sell_volume_multiplier': trial.suggest_float('sell_volume_multiplier', 0.1, 12, step=0.1),
        "buy_atr_period": trial.suggest_int('buy_atr_period', 60, 1500, step=10),
        "sell_atr_period": trial.suggest_int('sell_atr_period', 60, 1500, step=10),
        'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.04, step=0.005),
        'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.04, step=0.005),
        'buy_take_profit': trial.suggest_float('buy_take_profit', 0.005, 0.03, step=0.005),
        'sell_take_profit': trial.suggest_float('sell_take_profit', 0.005, 0.03, step=0.005),
        "buy_risk_per_trade": trial.suggest_float('buy_risk_per_trade', 0.3, 1, step=0.01),
        "sell_risk_per_trade": trial.suggest_float('sell_risk_per_trade', 0.3, 1, step=0.01),
        "sell_vwap_period": trial.suggest_int('sell_vwap_period', 60, 1500, step=10),
        "buy_vwap_period": trial.suggest_int('buy_vwap_period', 60, 1500, step=10),
        "buy_atr_sma_period": trial.suggest_int('buy_atr_sma_period', 60, 1500, step=10),
        "sell_atr_sma_period": trial.suggest_int('sell_atr_sma_period', 60, 1500, step=10),
        'opt_target': target,
        "printlog": False
    }
    performances = main(**kwargs, df=df, end_date=end_date,
                        start_date=start_date, strategy_name=strategy_name)
    if performances is None:
        print(f'No trade detected in {target}.')
        return -100  # or any special value you prefer
    # print(f'Now the optimize {target} is: {performances[target]}')
    if target == 'std_dev':
        # return negative performance if target is 'std_dev'
        return -performances[target]
    else:
        return performances[target]


def main_opt_vwap_short(trial, df, strategy_name, end_date, start_date, target):
    kwargs = {
        'sell_volume_window': trial.suggest_int('sell_volume_window', 13, 50),
        'buy_volume_window': trial.suggest_int('buy_volume_window', 13, 50),
        'buy_volume_multiplier': trial.suggest_float('buy_volume_multiplier', 1.5, 12, step=0.1),
        'sell_volume_multiplier': trial.suggest_float('sell_volume_multiplier', 1.5, 12, step=0.1),
        "buy_atr_period": trial.suggest_int('buy_atr_period', 13, 50),
        "sell_atr_period": trial.suggest_int('sell_atr_period', 13, 50),
        'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.04, step=0.005),
        'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.04, step=0.005),
        'buy_take_profit': trial.suggest_float('buy_take_profit', 0.005, 0.03, step=0.005),
        'sell_take_profit': trial.suggest_float('sell_take_profit', 0.005, 0.03, step=0.005),
        "buy_risk_per_trade": trial.suggest_float('buy_risk_per_trade', 0.3, 1, step=0.01),
        "sell_risk_per_trade": trial.suggest_float('sell_risk_per_trade', 0.3, 1, step=0.01),
        "sell_vwap_period": trial.suggest_int('sell_vwap_period', 10, 30),
        "buy_vwap_period": trial.suggest_int('buy_vwap_period', 10, 30),
        "buy_atr_sma_period": None,
        "sell_atr_sma_period": None,
        "opt_target": target,
        "printlog": False
    }
    performances = main(**kwargs, df=df, end_date=end_date,
                        start_date=start_date, strategy_name=strategy_name)
    if performances is None:
        print(f'No trade detected in {target}.')
        return -100  # or any special value you prefer
    # print(f'Now the optimize {target} is: {performances[target]}')
    if target == 'std_dev':
        # return negative performance if target is 'std_dev'
        return -performances[target]
    else:
        return performances[target]


def main_opt_rsiv_oneday(trial, df, strategy_name, end_date, start_date, target):
    factor = 1
    kwargs = {
        'rsi_window': trial.suggest_float('rsi_window', 3, 30, step=0.2),
        'ma_short_period': trial.suggest_float('ma_short_period', 1, 30, step=0.2),
        'ma_long_period': trial.suggest_float('ma_long_period', 1, 30, step=0.2),
        'buy_rsi_threshold': trial.suggest_int('buy_rsi_threshold', 20, 40, step=2),
        "sell_rsi_threshold": trial.suggest_int('sell_rsi_threshold', 60, 80, step=2),
        "open_percent": trial.suggest_float('open_percent', 0.3, 1, step=0.05),
        "buy_Apoint": trial.suggest_int('buy_Apoint', 0, 10, step=1),
        "buy_Bpoint": trial.suggest_int('buy_Bpoint', 0, 10, step=1),
        "sell_Apoint": trial.suggest_int('sell_Apoint', 0, 10, step=1),
        "sell_Bpoint": trial.suggest_int('sell_Bpoint', 0, 10, step=1),
        'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.04, step=0.005),
        'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.04, step=0.005),
        'buy_take_profit': trial.suggest_float('buy_take_profit', 0.005, 0.03, step=0.005),
        'sell_take_profit': trial.suggest_float('sell_take_profit', 0.005, 0.03, step=0.005),
        "opt_target": target,
        "printlog": False
    }
    performances = rsi_main(**kwargs, df=df, end_date=end_date,
                            start_date=start_date, strategy_name=strategy_name)
    if performances is None:
        print(f'No trade detected in {target}.')
        return -100  # or any special value you prefer
    # print(f'Now the optimize {target} is: {performances[target]}')
    if target == 'std_dev':
        # return negative performance if target is 'std_dev'
        return -performances[target]
    else:
        return performances[target]


def main_opt_rsiv_fourhour(trial, df, strategy_name, end_date, start_date, target):
    kwargs = {
        'rsi_window': trial.suggest_float('rsi_window', 3, 30, step=0.2),
        'ma_short_period': trial.suggest_float('ma_short_period', 1, 30, step=0.2),
        'ma_long_period': trial.suggest_float('ma_long_period', 1, 30, step=0.2),
        'buy_rsi_threshold': trial.suggest_int('buy_rsi_threshold', 20, 40, step=2),
        "sell_rsi_threshold": trial.suggest_int('sell_rsi_threshold', 60, 80, step=2),
        "open_percent": trial.suggest_float('open_percent', 0.3, 1, step=0.05),
        "buy_Apoint": trial.suggest_int('buy_Apoint', 0, 10, step=1),
        "buy_Bpoint": trial.suggest_int('buy_Bpoint', 0, 10, step=1),
        "sell_Apoint": trial.suggest_int('sell_Apoint', 0, 10, step=1),
        "sell_Bpoint": trial.suggest_int('sell_Bpoint', 0, 10, step=1),
        'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.04, step=0.005),
        'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.04, step=0.005),
        'buy_take_profit': trial.suggest_float('buy_take_profit', 0.005, 0.03, step=0.005),
        'sell_take_profit': trial.suggest_float('sell_take_profit', 0.005, 0.03, step=0.005),
        "opt_target": target,
        "printlog": False
    }
    performances = rsi_main(**kwargs, df=df, end_date=end_date,
                            start_date=start_date, strategy_name=strategy_name)
    if performances is None:
        print(f'No trade detected in {target}.')
        return -100  # or any special value you prefer
    # print(f'Now the optimize {target} is: {performances[target]}')
    if target == 'std_dev':
        # return negative performance if target is 'std_dev'
        return -performances[target]
    else:
        return performances[target]


def main_opt_wrsi(trial, df, strategy_name, end_date, start_date, target):
    kwargs = {
        'rsi_window': trial.suggest_float('rsi_window', 5, 30, step=0.2),
        'factor': trial.suggest_int('factor', 60, 1020, step=30),
        'ma_short_period': trial.suggest_float('ma_short_period', 1, 30, step=0.2),
        'ma_long_period': trial.suggest_float('ma_long_period', 1, 30, step=0.2),
        'buy_rsi_threshold': trial.suggest_int('buy_rsi_threshold', 20, 40, step=2),
        "sell_rsi_threshold": trial.suggest_int('sell_rsi_threshold', 60, 80, step=2),
        "open_percent": trial.suggest_float('open_percent', 0.3, 1, step=0.05),
        "buy_Apoint": trial.suggest_int('buy_Apoint', 0, 10, step=1),
        "buy_Bpoint": trial.suggest_int('buy_Bpoint', 0, 10, step=1),
        "sell_Apoint": trial.suggest_int('sell_Apoint', 0, 10, step=1),
        "sell_Bpoint": trial.suggest_int('sell_Bpoint', 0, 10, step=1),
        'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.04, step=0.005),
        'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.04, step=0.005),
        'buy_take_profit': trial.suggest_float('buy_take_profit', 0.005, 0.03, step=0.005),
        'sell_take_profit': trial.suggest_float('sell_take_profit', 0.005, 0.03, step=0.005),
        "opt_target": target,
        "printlog": False
    }
    performances = wrsi_main(**kwargs, df=df, end_date=end_date,
                             start_date=start_date, strategy_name=strategy_name)
    if performances is None:
        print(f'No trade detected in {target}.')
        return -100  # or any special value you prefer
    # print(f'Now the optimize {target} is: {performances[target]}')
    if target == 'std_dev':
        # return negative performance if target is 'std_dev'
        return -performances[target]
    else:
        return performances[target]


def main_opt_mav(trial, df, strategy_name, end_date, start_date, target):
    kwargs = {
        'short_period': trial.suggest_int('short_period', 20, 1000, step=10),
        'buy_short_volume': trial.suggest_int('buy_short_volume', 20, 1000, step=10),
        'sell_short_volume': trial.suggest_int('sell_short_volume', 20, 1000, step=10),
        'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.04, step=0.005),
        'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.04, step=0.005),
        'buy_take_profit': trial.suggest_float('buy_take_profit', 0.005, 0.03, step=0.005),
        'sell_take_profit': trial.suggest_float('sell_take_profit', 0.005, 0.03, step=0.005),
        'long_factor': trial.suggest_float('long_factor', 1.4, 20, step=0.2),
        'buy_long_factor': trial.suggest_float('buy_long_factor', 1.4, 20, step=0.2),
        'sell_long_factor': trial.suggest_float('sell_long_factor', 1.4, 20, step=0.2),
        "opt_target": target,
        "printlog": False
    }

    performances = mav_main(**kwargs, df=df, end_date=end_date,
                            start_date=start_date, strategy_name=strategy_name)
    if performances is None:
        print(f'No trade detected in {target}.')
        return -100  # or any special value you prefer
    # print(f'Now the optimize {target} is: {performances[target]}')
    if target == 'std_dev':
        # return negative performance if target is 'std_dev'
        return -performances[target]
    else:
        return performances[target]


def main_opt_macd(trial, df, strategy_name, end_date, start_date, target):
    kwargs = {
        'short_period': trial.suggest_int('short_period', 5, 18, step=1),
        'long_period': trial.suggest_int('long_period', 19, 50, step=1),
        'signal_period': trial.suggest_int('signal_period', 5, 20, step=1),
        'L_short_period': trial.suggest_int('short_period', 5, 18, step=1),
        'L_long_period': trial.suggest_int('long_period', 19, 50, step=1),
        'L_signal_period': trial.suggest_int('signal_period', 5, 20, step=1),
        'factor': trial.suggest_int('factor', 1, 20, step=1),
        'alpha_factor': trial.suggest_int('alpha_factor', 1, 10, step=1),
        'M': trial.suggest_int('M', 2, 10, step=1),
        "opt_target": target,
        "printlog": False
    }
    # M = trial.suggest_int('M', 1, 10, step=1)
    performances = macd_main(**kwargs, df=df, end_date=end_date,
                             start_date=start_date, strategy_name=strategy_name)
    if performances is None:
        print(f'No trade detected in {target}.')
        return -100  # or any special value you prefer
    # print(f'Now the optimize {target} is: {performances[target]}')
    if target == 'std_dev':
        # return negative performance if target is 'std_dev'
        return -performances[target]
    else:
        return performances[target]


def make_main_opt(df, strategy_name, end_date, start_date, target):
    print("opt_0")
    if strategy_name == 'VWAPStrategyLong':
        return lambda trial: main_opt_vwap_long(trial, df=df, strategy_name=strategy_name, end_date=end_date, start_date=start_date, target=target)
    elif strategy_name == 'VWAPStrategyShort':
        return lambda trial: main_opt_vwap_short(trial, df=df, strategy_name=strategy_name, end_date=end_date, start_date=start_date, target=target)
    elif strategy_name == 'RSIV_StrategyOneday':
        return lambda trial: main_opt_rsiv_oneday(trial, df=df, strategy_name=strategy_name, end_date=end_date, start_date=start_date, target=target)
    elif strategy_name == 'RSIV_StrategyFourhour':
        return lambda trial: main_opt_rsiv_fourhour(trial, df=df, strategy_name=strategy_name, end_date=end_date, start_date=start_date, target=target)
    elif strategy_name == 'MACDStrategy':
        return lambda trial: main_opt_macd(trial, df=df, strategy_name=strategy_name, end_date=end_date, start_date=start_date, target=target)
    elif strategy_name == 'MAVStrategy':
        return lambda trial: main_opt_mav(trial, df=df, strategy_name=strategy_name, end_date=end_date, start_date=start_date, target=target)
    elif strategy_name == 'WRSIStrategy':
        return lambda trial: main_opt_wrsi(trial, df=df, strategy_name=strategy_name, end_date=end_date, start_date=start_date, target=target)
    else:
        raise ValueError("Invalid strategy name.")

def create_storage(db_name):
    # 创建 SQLite 连接字符串
    url = f"sqlite:///{db_name}"
    from sqlalchemy import create_engine, text
    # 创建 SQLAlchemy 引擎，并启用 WAL 模式
    engine = create_engine(url, connect_args={"timeout": 5})
    
    with engine.connect() as connection:
        connection.execute(text("PRAGMA journal_mode=WAL;"))
    
    return storages.RDBStorage(url, engine_kwargs={"connect_args": {"timeout": 5}})



def optimize_and_run(df, targets, end_date, start_date, num_evals=100, printlog=True, strategy_name=None):
    opt_results = {}
    # 将字符串转换为datetime对象
    start_date = str(start_date)
    end_date = str(end_date)
    start_strp = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_strp = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    # 计算两个日期之间的差异
    delta = end_strp - start_strp
    # 提取天数
    date_period = delta.days
    end_date = pd.to_datetime(end_date)
    for target in targets:
        print(f'Now the optimize target is: {target}')
        main_opt = make_main_opt(
            df=df, end_date=end_date, start_date=start_date, target=target, strategy_name=strategy_name)
        start = time.time()
        old_date = end_date - pd.Timedelta(days=7)
        cur_study_name = f'BTC_spot_1109_study_{end_date}_{date_period}_{strategy_name}_{target}'
        old_study_name = f'BTC_spot_1109_study_{old_date}_{date_period}_{strategy_name}_{target}'
        db_name = f"BTC_spot_1109_study_{target}.db"
        #storage = storages.RDBStorage(
        #    f"sqlite:///{db_name}", engine_kwargs={"connect_args": {"timeout": 5}})
        # 创建存储实例，并启用 WAL 模式
        storage = create_storage(db_name)
        try:
            old_study = optuna.load_study(
                study_name=old_study_name, storage=storage)
            print(
                f"the target is {target} and date is {end_date}, we find the best para in {old_date}, use it for initial")
        except Exception as e:
            print(f"Caught exception: {e}")
            old_study = None
        if old_study is not None and len(old_study.trials) > 1:
            try:
                if old_study.best_trial.state == optuna.trial.TrialState.COMPLETE:
                    best_params = old_study.best_params
                    last_trial_params = old_study.trials[-1].params
                    # 将最优参数和最后一次trial的参数作为新的trial添加到新的study
                    new_study = optuna.create_study(
                        study_name=cur_study_name, storage=storage, direction='maximize', load_if_exists=True)
                    print(f"the old study best_params: {best_params}")
                    new_study.enqueue_trial(best_params)
                    new_study.enqueue_trial(last_trial_params)
                else:
                    raise ValueError("No complete trials found.")
            except ValueError:
                print(
                    f'Encountered an issue while getting the best parameters from the old study for {cur_study_name}. Initializing a new study.')
                new_study = optuna.create_study(
                    study_name=cur_study_name, storage=storage, direction='maximize', load_if_exists=True)
        else:
            print(
                f'We did not find the best parameters for the old study {cur_study_name} and will initialize a new one.')
            new_study = optuna.create_study(
                study_name=cur_study_name, storage=storage, direction='maximize', load_if_exists=True)
        # Add stream handler of stdout to show the messages to see Optuna works expectedly.
        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout))
        print(f'Now optimizing target: {target} over {date_period} days.')
        try:
            last_trial = new_study.trials[-1]
            completed_iterations = last_trial.number
            print(f"We have already run {completed_iterations} iterations.")
            # train at least 3 times
            remaining_iterations = max(3, num_evals - completed_iterations)
            print(f"We are going to run {remaining_iterations} iterations.")
        except:
            remaining_iterations = num_evals
        if num_evals > 200:
            print(
                f"num_evals {num_evals} larger than 200, going to use early stop function")
            early_stopping_rounds = 100
            new_study.optimize(
                main_opt,
                n_trials=remaining_iterations,
                gc_after_trial=True, n_jobs=1,
                callbacks=[
                    partial(early_stopping_check, early_stopping_rounds=early_stopping_rounds)]
            )
        else:
            new_study.optimize(
                main_opt, n_trials=remaining_iterations, gc_after_trial=True, n_jobs=1)
        print("opt_1")
        optimal_pars = new_study.best_params
        details = new_study.best_value
        print(
            f"the study name is {strategy_name} and target is {target} and end_date is {end_date}")
        print(f"the best params for our study are {optimal_pars}")
        print(f"the best value is {details}")
        print("opt_2")
        opt_results[target] = (optimal_pars, details)
        print("opt_3")
        end = time.time()
        total_time = end - start
        print(f"Cost time: {total_time}")

    return opt_results


def evaluate_opt_results(df, opt_results, end_date, start_date, strategy_name):
    performance_results = {}

    for target, (params, details) in opt_results.items():
        print(f'Going to evaluate {target} from {start_date} to {end_date}')
        # 针对特定策略进行特殊处理
        if strategy_name == 'VWAPStrategyShort':
            params['buy_atr_sma_period'] = params['buy_atr_period']
            params['sell_atr_sma_period'] = params['sell_atr_period']

        param_keys = params.keys()
        print(
            f"we will use these params for our evaluation of {strategy_name} and target {target} and end_date {end_date}")
        print(params)
        if 'VWAP' in strategy_name:
            performances = main(
                df=df,
                end_date=end_date,
                start_date=start_date,
                **{key: params[key] for key in param_keys},
                printlog=False,
                para_opt=False,
                startcash=100000,
                strategy_name=strategy_name,
                com=0.0005,
                opt_target=target
            )
        elif 'RSIV' in strategy_name:
            performances = rsi_main(
                df=df,
                end_date=end_date,
                start_date=start_date,
                **{key: params[key] for key in param_keys},
                printlog=False,
                para_opt=False,
                startcash=100000,
                strategy_name=strategy_name,
                com=0.0005,
                opt_target=target
            )
        elif 'WRSI' in strategy_name:
            performances = wrsi_main(
                df=df,
                end_date=end_date,
                start_date=start_date,
                **{key: params[key] for key in param_keys},
                printlog=False,
                para_opt=False,
                startcash=100000,
                strategy_name=strategy_name,
                com=0.0005,
                opt_target=target
            )
        elif 'MACD' in strategy_name:
            performances = macd_main(
                df=df,
                end_date=end_date,
                start_date=start_date,
                **{key: params[key] for key in param_keys},
                printlog=False,
                para_opt=False,
                startcash=100000,
                strategy_name=strategy_name,
                com=0.0005,
                opt_target=target
            )
        elif 'MAV' in strategy_name:
            performances = mav_main(
                df=df,
                end_date=end_date,
                start_date=start_date,
                **{key: params[key] for key in param_keys},
                printlog=False,
                para_opt=False,
                startcash=100000,
                strategy_name=strategy_name,
                com=0.0005,
                opt_target=target
            )
        else:
            raise ValueError("Invalid strategy name.")
        # 保存参数到字典
        data = {key: params[key] for key in param_keys}

        performance_results[target] = {}
        performance_results[target][0] = performances
        performance_results[target][1] = data

    return performance_results


def find_row_by_date(df, date_string):
    date = pd.to_datetime(date_string)
    return df.index.get_loc(date)


'''
# 使用示例
row = find_row_by_date(df, '2023-05-01 00:01:00')
print(row)
performance_results12 = evaluate_opt_results(opt_results, df=df[find_row_by_date(df, '2022-12-01 00:01:00'):find_row_by_date(df, '2023-02-01 00:01:00')])

performance_7days12 = evaluate_opt_results(opt_results, df=df[find_row_by_date(df, '2023-02-01 00:01:00'):find_row_by_date(df, '2023-02-08 00:01:00')])
performance_1month12 = evaluate_opt_results(opt_results, df=df[find_row_by_date(df, '2023-02-01 00:01:00'):find_row_by_date(df, '2023-03-01 00:01:00')])

performance_all = evaluate_opt_results(opt_results, df=df)
'''


def flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}


'''
flattened_data = [flatten_dict(record) for record in performance_all]
df_P = pd.DataFrame(performance_all)
df_P.to_csv("Check.csv")
'''


def dict_to_csv(data, csv_filename):
    df = pd.json_normalize(data)
    df.to_csv(csv_filename, index=False)


# dict_to_csv(df_P, "Check1.csv")
'''
import json

# 你的原始字典
your_dict = opt_results

# 创建一个新的字典，只包含每个元组的第一个元素（也就是字典）
new_dict = {key: value[0] for key, value in your_dict.items()}

# 保存这个新字典到一个json文件
with open('my_dict.json', 'w') as f:
    json.dump(new_dict, f)
    
with open('/Users/lyuhongwang/Desktop/my_dict.json', 'r') as f:
    opt_results = json.load(f)

print(opt_results)
'''
