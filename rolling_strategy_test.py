#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:00:03 2023

@author: lyuhongwang
"""
import os
# 动态设置环境变量，限制numpy和pandas使用单线程
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import pandas as pd
from datetime import timedelta
from DIY_Volume_numpy import *
from trading_data_process import *
import gc
import json


"""
def rolling_strategy_test(df, rolling_window=3, data_period=60, num_evals=50, targets=[]):
    # 确定日期范围
    start_date = df.index.min()
    end_date = df.index.max()

    # 设置滚动窗口的初始日期
    rolling_start = start_date
    print(f'the date range of test is {start_date} to {end_date}')
    targets = [targets]

    # 滚动窗口主循环
    while rolling_start + timedelta(days=data_period) <= end_date - timedelta(days=rolling_window):
        # 设置滚动窗口的结束日期
        rolling_end = rolling_start + timedelta(days=data_period)
        print(f'the rolling date range is {rolling_start} to {rolling_end}')
        opt_date = str(rolling_start) + ' ' + str(rolling_end)
        # 选取滚动窗口期间的数据
        rolling_df = df.loc[rolling_start:rolling_end]

        # 运行参数优化函数
        opt_results = optimize_and_run(targets, rolling_df, num_evals, printlog=False) #参数优化函数

        performances_period = evaluate_opt_results(opt_results, df=rolling_df) #评估优化后的参数的表现
        # 计算基准收益
        benchmark_return = performances_period['market_performance']['return']
        print(f'during the rolling date {rolling_start} : {rolling_end}, the benchmark return is {benchmark_return}')

        # 为每个目标创建一个字典
        for target in targets:
                # 在文件名中包含目标名称
            if os.path.exists(f'dict_rolling/{opt_date}_{target}.json'):
                with open(f'dict_rolling/{opt_date}_{target}.json', 'r') as f:
                    performance_total = handle_datetime(performance_total)
                    performance_total = json.load(f)
            else:
                performance_total = {}             
                     
            if target not in performance_total:
                performance_total[target] = {}
            if data_period not in performance_total[target]:
                performance_total[target][data_period] = {}
            if 'opt' not in performance_total[target][data_period]:
                performance_total[target][data_period]['opt'] = {}
            if 'apply' not in performance_total[target][data_period]:
                performance_total[target][data_period]['apply'] = {}

            performance_total[target][data_period]['opt'] = performances_period

            real_end_3 = rolling_end + timedelta(days=3)
            real_end_6 = rolling_end + timedelta(days=6)
            if real_end_6 > end_date:
                real_end_6 = end_date
            real_period_3 = df.loc[rolling_end:real_end_3]
            real_period_6 = df.loc[rolling_end:real_end_6]
            print(f'using the best strategy for next stage {rolling_end} to {real_end_3} and {real_end_6}')
            performance_3 = evaluate_opt_results(opt_results, df=real_period_3)
            performance_6 = evaluate_opt_results(opt_results, df=real_period_6)

            performance_date_3 = str(rolling_end) + ' ' + str(real_end_3) #+ ' ' + str(strategy_name)
            performance_date_6 = str(rolling_end) + ' '+ str(real_end_6) #+ ' ' + str(strategy_name)

            performance_total[target][data_period]['apply'][performance_date_3] = performance_3
            performance_total[target][data_period]['apply'][performance_date_6] = performance_6

            # 将字典写入文件，文件名包含目标名称
            with open(f'dict_rolling/{opt_date}_{target}.json', 'w') as f:
                performance_total = handle_datetime(performance_total)
                json.dump(performance_total, f)
            # 显式地删除不再需要的对象
            del performance_total
            del performance_3
            del performance_6
        # 为下一个窗口设置新的起始日期
        rolling_start += timedelta(days=rolling_window)
        # 手动触发垃圾回收
        gc.collect()
"""

def rolling_strategy_test(df, strategy_name=None, data_period=60, num_evals=50, targets=[], end_date=None):
    # 通过end_date和data_period来确定start_date
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.Timedelta(days=data_period)
    
    targets = [targets]
    # 设置滚动窗口的初始日期
    rolling_start = start_date
    # 设置滚动窗口的结束日期
    rolling_end = rolling_start + pd.Timedelta(days=data_period)
    print(f'the rolling date range is {rolling_start} to {rolling_end}')
    opt_date = str(rolling_start) + ' ' + str(rolling_end)

    # 选取滚动窗口期间的数据
    print('check')
    # 运行参数优化函数
    #opt_results = optimize_and_run(targets, rolling_df, num_evals, printlog=False) #参数优化函数
    opt_results = optimize_and_run(df=df,targets=targets, end_date=end_date, start_date=start_date,num_evals=num_evals, printlog=False,strategy_name=strategy_name) #参数优化函数

    performances_period = evaluate_opt_results(df=df,opt_results=opt_results, end_date=end_date, start_date=start_date,strategy_name=strategy_name) #评估优化后的参数的表现
    # 计算基准收益
   
    print(f'during the rolling date {rolling_start} : {rolling_end}')

    # 为每个目标创建一个字典
    for target in targets:
        filename = f'dict_rolling/{opt_date}_|{strategy_name}|_{target}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                performance_total = json.load(f)
                performance_total = handle_datetime(performance_total)
        else:
            performance_total = {}             

        if target not in performance_total:
            performance_total[target] = {}
        if data_period not in performance_total[target]:
            performance_total[target][data_period] = {}
        if 'opt' not in performance_total[target][data_period]:
            performance_total[target][data_period]['opt'] = {}
        if 'apply' not in performance_total[target][data_period]:
            performance_total[target][data_period]['apply'] = {}

        performance_total[target][data_period]['opt'] = performances_period

        real_end_3 = rolling_end + pd.Timedelta(days=3)
        real_end_6 = rolling_end + pd.Timedelta(days=6)
        real_end_9 = rolling_end + pd.Timedelta(days=9)
        real_end_7 = rolling_end + pd.Timedelta(days=7)
        if real_end_9 > df.index.max():
            real_end_9 = df.index.max()
        if real_end_6 > df.index.max():
            real_end_6 = df.index.max()
        if real_end_7 > df.index.max():
            real_end_7 = df.index.max()
        print(f'using the best strategy for next stage {rolling_end} to {real_end_3} and {real_end_6} and {real_end_9}')
        performance_3 = evaluate_opt_results(df=df,opt_results=opt_results, end_date=real_end_3, start_date=rolling_end,strategy_name=strategy_name)
        performance_6 = evaluate_opt_results(df=df,opt_results=opt_results, end_date=real_end_6, start_date=rolling_end,strategy_name=strategy_name)
        performance_9 = evaluate_opt_results(df=df,opt_results=opt_results, end_date=real_end_9, start_date=rolling_end,strategy_name=strategy_name)
        performance_7 = evaluate_opt_results(df=df,opt_results=opt_results, end_date=real_end_7, start_date=rolling_end,strategy_name=strategy_name)
        performance_date_3 = str(rolling_end) + ' ' + str(real_end_3) #+ ' ' + str(strategy_name)
        performance_date_6 = str(rolling_end) + ' '+ str(real_end_6) #+ ' ' + str(strategy_name)
        performance_date_9 = str(rolling_end) + ' '+ str(real_end_9) #+ ' ' + str(strategy_name)
        performance_total[target][data_period]['apply'][performance_date_3] = performance_3
        performance_total[target][data_period]['apply'][performance_date_6] = performance_6
        performance_total[target][data_period]['apply'][performance_date_9] = performance_9
        performance_total[target][data_period]['apply']['7'] = performance_7
        directory = 'dict_rolling'
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 将字典写入文件，文件名包含目标名称
        with open(filename, 'w') as f:
            performance_total = handle_datetime(performance_total)
            json.dump(performance_total, f)
    # 手动触发垃圾回收
    gc.collect()





def select_strategy(performances_period, benchmark_return):
    """
    选择策略的函数

    参数:
        opt_results: 一个字典，每个键是一个优化目标，每个值是一个包含参数字典的元组。
        benchmark_return: 基准收益。

    返回:
        最佳策略的参数字典。
    """
    # 确保有至少一个策略
    assert len(performances_period) > 0, "No strategies provided."

    # 将每个策略的回报率计算出来
    performance_results = {key:v for key,v in performances_period.items() if key !='market_performance'}

    # 找出收益为正的策略
    positive_return_strategies = {k: v for k, v in performance_results.items() if v['total_return'] > 0}

    if len(positive_return_strategies) == 0:
        # 所有策略的收益都为负，停止交易3天
        return None, "Stop trading for 3 days."
    
    benchmark_return = max(0,benchmark_return)

    # 找出收益超过基准收益的策略
    outperforming_strategies = {k: v for k, v in positive_return_strategies.items() if v['total_return'] > benchmark_return}

    if len(outperforming_strategies) == 0:
        # 无策略超过基准收益，选择最小回撤的策略
        min_drawdown_strategy = min(positive_return_strategies, key=lambda x: positive_return_strategies[x]['drawdown'])
        return opt_results[min_drawdown_strategy][0], min_drawdown_strategy

    # 找出回撤低于15%且收益最高的策略
    low_drawdown_strategies = {k: v for k, v in outperforming_strategies.items() if v['drawdown'] < 0.15}
    if len(low_drawdown_strategies) > 0:
        max_return_strategy = max(low_drawdown_strategies, key=lambda x: low_drawdown_strategies[x]['total_return'])
        return opt_results[max_return_strategy][0], max_return_strategy

    # 找出收益最高的策略
    max_return_strategy = max(outperforming_strategies, key=lambda x: outperforming_strategies[x]['total_return'])
    return opt_results[max_return_strategy][0], max_return_strategy

#targets = ['Win ratio','sqn','value','drawdown','SharpeDIY']
#check = rolling_strategy_test(targets=targets,df=df[:60*24*30*2], rolling_window=6, data_period=30, num_evals=1)


def process_json_files(directory, output_file_path):
    # 读取指定目录下的所有json文件
    file_names = os.listdir(directory)

    new_dict = {}

    for file_name in file_names:
        # 检查每个json文件的名称是否符合规定的格式
        try:
            parts = file_name.split(".",1)[0]
            parts = parts.split("_",1)
            date_parts = parts[0].split(' ')
            start_date = date_parts[0] + ' ' + date_parts[1]
            end_date = date_parts[2] + ' ' + date_parts[3]
            target = parts[1]
        except ValueError:
            print(f'File {file_name} does not meet the required format.')
            continue

        # 读取每个json文件并转换成字典
        with open(os.path.join(directory, file_name)) as f:
            data = json.load(f)

        # 使用新的键访问字典的值，并将这些值存储在一个新的字典中
        for train_days, value in data[target].items():
            if end_date not in new_dict:
                new_dict[end_date] = {}
            if train_days not in new_dict[end_date]:
                new_dict[end_date][train_days] = {}
            if target not in new_dict[end_date][train_days]:
                new_dict[end_date][train_days][target] = {}

            if 'opt' in new_dict[end_date][train_days][target]:
                print(f"Error: Key 'opt' already exists in dictionary for file {file_name}.")
                return

            # 检查并计算weighted_sqn_drawdown, weighted_drawdown_win_ratio, weighted_drawdown_win_ratio_value
            opt = value['opt'][target]['0']
            if 'weighted_sqn_drawdown' not in opt:
                opt['weighted_sqn_drawdown'] = opt['weighted_drawdown'] * opt['sqn']
            if 'weighted_drawdown_win_ratio' not in opt:
                opt['weighted_drawdown_win_ratio'] = opt['weighted_drawdown'] * opt['Win ratio']
            if 'weighted_drawdown_win_ratio_value' not in opt:
                opt['weighted_drawdown_win_ratio_value'] = opt['weighted_drawdown'] * opt['win ratio value']

            new_dict[end_date][train_days][target]['opt'] = value['opt']

            for apply_key, apply_value in value['apply'].items():
                date_parts = apply_key.split(' ')
                apply_start_date = date_parts[0] + ' ' + date_parts[1]
                apply_end_date = date_parts[2] + ' ' + date_parts[3]
                days_diff = (pd.to_datetime(apply_end_date) - pd.to_datetime(apply_start_date)).days
                
                if 'apply' not in new_dict[end_date][train_days][target]:
                    new_dict[end_date][train_days][target]['apply'] = {}
                elif days_diff in new_dict[end_date][train_days][target]['apply']:
                    print(f"Error: Key {days_diff} already exists in 'apply' dictionary for file {file_name}.")
                    return

                # 检查并计算weighted_sqn_drawdown, weighted_drawdown_win_ratio, weighted_drawdown_win_ratio_value in apply_value
                apply_opt = apply_value[target]['0']
                if 'weighted_sqn_drawdown' not in apply_opt:
                    apply_opt['weighted_sqn_drawdown'] = apply_opt['weighted_drawdown'] * apply_opt['sqn']
                if 'weighted_drawdown_win_ratio' not in apply_opt:
                    apply_opt['weighted_drawdown_win_ratio'] = apply_opt['weighted_drawdown'] * apply_opt['Win ratio']
                if 'weighted_drawdown_win_ratio_value' not in apply_opt:
                    apply_opt['weighted_drawdown_win_ratio_value'] = apply_opt['weighted_drawdown'] * apply_opt['win ratio value']

                new_dict[end_date][train_days][target]['apply'][days_diff] = apply_value

    # 将新的字典保存为一个本地json文件
    with open(output_file_path, 'w') as f:
        json.dump(new_dict, f)
        
from datetime import datetime, timedelta

def get_dates(end, days,span):
    end_date = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    dates = []

    for i in range(days // span):  # 得到天数的商数，这是我们需要迭代的次数
        dates.append(end_date)
        end_date -= timedelta(days=span)  # 每次迭代，我们都从结束日期中减去3天

    # 日期格式化并返回为字符串列表
    dates = dates[::-1]
    dates_str = "' '".join([f"'{date.strftime('%Y-%m-%d %H:%M:%S')}'" for date in dates])
    return dates_str

# 使用函数
#end = "2023-05-31 00:00:00"
#days = 10
#print(get_dates(end, days))
# 