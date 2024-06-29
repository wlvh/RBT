#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2023-07-19 11:31:19
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-06-23 04:16:28
FilePath: /trading/process_json_files.py
Description: 

Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import numpy as np
from multiprocessing import Pool

#这个函数是用来将所有target参数优化文件合并
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
            target = parts[1].split('|_',1)[1]
            strategy_name = parts[1].split('|_',1)[0].replace("|",'')
        except ValueError:
            print(f'File {file_name} does not meet the required format.')
            continue

        # 读取每个json文件并转换成字典
        try:
            with open(os.path.join(directory, file_name)) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f'Error reading {file_name}: {e}')
            raise e
        #print(file_name)
        #print(parts)
        #print(end_date)
        #print(target)
        #print(strategy_name)
        # 使用新的键访问字典的值，并将这些值存储在一个新的字典中
        for train_days, value in data[target].items():
            if end_date not in new_dict:
                new_dict[end_date] = {}
            if train_days not in new_dict[end_date]:
                new_dict[end_date][train_days] = {}
            if target not in new_dict[end_date][train_days]:
                new_dict[end_date][train_days][target] = {}
            if strategy_name not in new_dict[end_date][train_days][target]:
                new_dict[end_date][train_days][target][strategy_name] = {}

            if 'opt' in new_dict[end_date][train_days][target]:
                print(f"Error: Key 'opt' already exists in dictionary for file {file_name}.")
                return

            # 检查并计算weighted_sqn_drawdown, weighted_drawdown_win_ratio, weighted_drawdown_win_ratio_value
            opt = value['opt'][target]['0']
            if 'weighted_sqn_drawdown' not in opt:
                opt['weighted_sqn_drawdown'] = float(opt['weighted_drawdown']) * float(opt['sqn'])
            if 'weighted_drawdown_win_ratio' not in opt:
                opt['weighted_drawdown_win_ratio'] = float(opt['weighted_drawdown']) * float(opt['Win ratio'])
            if 'weighted_drawdown_win_ratio_value' not in opt:
                opt['weighted_drawdown_win_ratio_value'] = float(opt['weighted_drawdown']) * float(opt['win ratio value'])
            if 'drawdown_win_ratio' not in opt:
                opt['drawdown_win_ratio'] = (1+float(opt['drawdown'])) * float(opt['Win ratio'])
            if  'sqn_drawdown' not in opt:
                opt['sqn_drawdown'] = float(opt['sqn']) * (1+float(opt['drawdown']))
            if 'drawdown_win_ratio_value' not in opt:
                opt['drawdown_win_ratio_value'] = (1+float(opt['drawdown'])) * float(opt['win ratio value'])
    
            new_dict[end_date][train_days][target][strategy_name]['opt'] = value['opt']

            for apply_key, apply_value in value['apply'].items():
                date_parts = apply_key.split(' ')
                if apply_key == '7':
                    days_diff = 7
                else:
                    apply_start_date = date_parts[0] + ' ' + date_parts[1]
                    apply_end_date = date_parts[2] + ' ' + date_parts[3]
                    days_diff = (pd.to_datetime(apply_end_date) - pd.to_datetime(apply_start_date)).days
                # Restricting days_diff to 3, 6, or 9
                if days_diff not in [3, 7, 9]:
                    print(f'found {days_diff} not in [3, 7, 9]')
                    continue
                if 'apply' not in new_dict[end_date][train_days][target][strategy_name]:
                    new_dict[end_date][train_days][target][strategy_name]['apply'] = {}
                elif days_diff in new_dict[end_date][train_days][target][strategy_name]['apply']:
                    print(f"Warning: Key {days_diff} already exists in 'apply' dictionary for file {file_name}. Skipping.")
                    continue
                # 检查并计算weighted_sqn_drawdown, weighted_drawdown_win_ratio, weighted_drawdown_win_ratio_value in apply_value
                apply_opt = apply_value[target]['0']
                #if 'weighted_sqn_drawdown' not in apply_opt:
                #    apply_opt['weighted_sqn_drawdown'] = apply_opt['weighted_drawdown'] * apply_opt['sqn']
                #if 'weighted_drawdown_win_ratio' not in apply_opt:
                #    apply_opt['weighted_drawdown_win_ratio'] = apply_opt['weighted_drawdown'] * apply_opt['Win ratio']
                #if 'weighted_drawdown_win_ratio_value' not in apply_opt:
                #    apply_opt['weighted_drawdown_win_ratio_value'] = apply_opt['weighted_drawdown'] * apply_opt['win ratio value']
                #if 'drawdown_win_ratio' not in apply_opt:
                #    apply_opt['drawdown_win_ratio'] = (1+apply_opt['drawdown']) * apply_opt['Win ratio']
                #if  'sqn_drawdown' not in apply_opt:
                #    apply_opt['sqn_drawdown'] = apply_opt['sqn'] * (1+apply_opt['drawdown'])
                #if 'drawdown_win_ratio_value' not in apply_opt:
                #    apply_opt['drawdown_win_ratio_value'] = (1+apply_opt['drawdown']) * apply_opt['win ratio value']

                new_dict[end_date][train_days][target][strategy_name]['apply'][days_diff] = apply_value

    # 将新的字典保存为一个本地json文件
    with open(output_file_path, 'w') as f:
        json.dump(new_dict, f)
 
#检查回测结果中是否有缺失的键       
def check_missing_keys(json_path,config_path):
    # Constants
        # Load the config data
    with open(config_path) as config_file:
        config_data = json.load(config_file)
    opt_periods = config_data['opt_periods']
    targets = config_data['target_names']
    functions = ['opt', 'apply']
    apply_periods = ['3', '6', '9']
    strategy_names = config_data['strategy_names']
    # Load the data
    with open(json_path) as file:
        data = json.load(file)

    # Initialize temp_zero
    temp_zero = None

    # Initialize a list to store error messages
    errors = []
    for date, date_data in data.items():
    # 如果date小于'2022-08-28 00:00:00'，就跳过当前循环
        if datetime.strptime(date, '%Y-%m-%d %H:%M:%S') < datetime.strptime('2022-09-01 00:00:00', '%Y-%m-%d %H:%M:%S'):
            continue
        for period in opt_periods:
            for target in targets:
                for strategy_name in strategy_names:
                    for function in functions:
                        if function == 'opt':
                            if period in date_data and target in date_data[period] and strategy_name in date_data[period][target] and function in date_data[period][target][strategy_name]:
                                if target in date_data[period][target][strategy_name][function]:
                                    if '0' in date_data[period][target][strategy_name][function][target]:
                                        current_keys = set(date_data[period][target][strategy_name][function][target]['0'].keys())
                                        if not temp_zero:
                                            temp_zero = current_keys
                                        else:
                                            if current_keys != temp_zero:
                                                missing_keys = temp_zero - current_keys
                                                errors.append(f'Error, missing keys {missing_keys} at: {date} -> {period} -> {target} -> {strategy_name} -> {function} -> {target} -> 0')
                                    else:
                                        errors.append(f'Error, missing "0" key at: {date} -> {period} -> {target} -> {strategy_name} -> {function} -> {target}')
                                else:
                                    errors.append(f'Error, missing "{target}" key at: {date} -> {period} -> {target} -> {strategy_name} -> {function}')
                            else:
                                errors.append(f'Error, missing "{function}" key at: {date} -> {period} -> {target} -> {strategy_name}')
                        elif function == 'apply':
                            for apply_period in apply_periods:
                                if period in date_data and target in date_data[period] and strategy_name in date_data[period][target] and function in date_data[period][target][strategy_name] and apply_period in date_data[period][target][strategy_name][function]:
                                    if target in date_data[period][target][strategy_name][function][apply_period]:
                                        if '0' in date_data[period][target][strategy_name][function][apply_period][target]:
                                            current_keys = set(date_data[period][target][strategy_name][function][apply_period][target]['0'].keys())
                                            if not temp_zero:
                                                temp_zero = current_keys
                                            else:
                                                if current_keys != temp_zero:
                                                    missing_keys = temp_zero - current_keys
                                                    #errors.append(f'Error, missing keys {missing_keys} at: {date} -> {period} -> {target} -> {function} -> {apply_period} -> {target} -> 0')
                                        else:
                                            continue
                                            #errors.append(f'Error, missing "0" key at: {date} -> {period} -> {target} -> {function} -> {apply_period} -> {target}')
                                    else:
                                        continue
                                        #errors.append(f'Error, missing "{target}" key at: {date} -> {period} -> {target} -> {function} -> {apply_period}')
                                else:
                                    continue
                                    #errors.append(f'Error, missing "{apply_period}" key at: {date} -> {period} -> {target} -> {function}')
                        else:
                            errors.append(f'Error, missing "{function}" key at: {date} -> {period} -> {target} -> {strategy_name}')
    
    return errors

#errors = check_missing_keys('/Users/lyuhongwang/Downloads/2021_10_08_2023_05_28_opt')

#处理错误信息，只提取date，period，target，返回一个list
def process_errors(errors):
    processed_errors = set()
    
    for error in errors:
        # Split the error message by ' -> '
        split_error = error.split(' -> ')
        
        # Only attempt to extract the date, period, and target if they exist in the error message
        if len(split_error) >= 2:
            # Extract the date and period
            date = split_error[0].split(': ')[1]
            period = split_error[1]
            
            # If the error message contains a target, extract it; otherwise, use a default value
            if len(split_error) >= 3:
                target = split_error[2]
            else:
                target = 'N/A'
            strategy_name = split_error[-1]
            # Create a string with the date, period, and target separated by commas
            error_string = f"{date},{period},{target},{strategy_name}"
            
            # Add the error string to the set (this will automatically remove duplicates)
            processed_errors.add(error_string)
    
    return list(set(processed_errors))

#processed_errors = process_errors(errors)

#检查回测结果中是否有缺失的日期
def check_dates(data_path):
    with open(data_path) as file:
        data = json.load(file)
    # Get the list of dates from the data
    dates = sorted([datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in data.keys()])

    # Initialize a list to store missing or extra dates
    missing_or_extra_dates = []

    # Iterate over the dates
    for i in range(1, len(dates)):
        # Calculate the difference between the current date and the previous date
        date_diff = dates[i] - dates[i-1]

        # If the difference is not equal to 3 days, add the dates to the list
        if date_diff != timedelta(days=7):
            missing_or_extra_dates.append((dates[i-1], dates[i]))

    return missing_or_extra_dates


def extract_apply_data(json_path, output_path,config_path):
    # Constants
    with open(config_path) as config_file:
        config_data = json.load(config_file)
    opt_periods = config_data['opt_periods']
    targets = config_data['target_names']
    functions = ['opt', 'apply']
    apply_periods = ['3', '6', '9']
    strategy_names = config_data['strategy_names']
    
    # Load the data
    with open(json_path) as file:
        data = json.load(file)

    # Initialize a dictionary to store the apply data
    apply_data = {}
    apply_data_para = {}

    for date, date_data in data.items():
        if datetime.strptime(date, '%Y-%m-%d %H:%M:%S') < datetime.strptime('2022-09-01 00:00:00', '%Y-%m-%d %H:%M:%S'):
            continue
        for period in opt_periods:
            for target in targets:
                for strategy_name in strategy_names:
                    function = 'apply'
                    for apply_period in apply_periods:
                        # If apply_period '6' data is missing, use '3' data
                        # 假设date_data, period, target已经定义并包含了需要的数据
                        print(date_data.keys())
                        if apply_period == '6' and apply_period not in date_data[period][target][strategy_name][function]:
                            date_data[period][target][strategy_name][function]['6'] = date_data[period][target][strategy_name][function]['3']
                        # If apply_period '9' data is missing, use '6' data
                        elif apply_period == '9' and apply_period not in date_data[period][target][strategy_name][function]:
                            date_data[period][target][strategy_name][function]['9'] = date_data[period][target][strategy_name][function]['6']

                        # Check if all the keys exist and continue if not
                        if not (period in date_data and target in date_data[period] and strategy_name in date_data[period][target] and function in date_data[period][target][strategy_name] and apply_period in date_data[period][target][strategy_name][function] and target in date_data[period][target][strategy_name][function][apply_period] and '0' in date_data[period][target][strategy_name][function][apply_period][target]):
                            print(f"Data not found for {date}, {period}, {target},{strategy_name}, {function}, {apply_period}.")
                            continue

                        # Build the key for the apply_data dictionary
                        if apply_period == '3':
                            key = f'{date}_{period}_{target}_{strategy_name}_apply_{apply_period}'
                            if str(date) not in apply_data:
                                apply_data[str(date)] = {}
                            if str(date) not in apply_data_para:
                                apply_data_para[str(date)] = {}
                            apply_data[str(date)][key] = date_data[period][target][strategy_name][function][apply_period][target]['0']
                            apply_data_para[str(date)][key] = date_data[period][target][strategy_name][function][apply_period][target]['1']
                        
    # Write the apply data to the output file
    with open(output_path, 'w') as file:
        json.dump(apply_data, file)
    with open(f'{output_path}_para', 'w') as file:
        json.dump(apply_data_para, file)


#extract_apply_data('2021_10_08_2023_05_28_opt', 'apply_data.json')

def extract_opt_data(json_path, output_path,config_path):
    # Constants
    with open(config_path) as config_file:
        config_data = json.load(config_file)
    opt_periods = config_data['opt_periods']
    targets = config_data['target_names']
    functions = ['opt', 'apply']
    apply_periods = ['3', '6', '9']
    strategy_names = config_data['strategy_names']
    # Load the data
    with open(json_path) as file:
        data = json.load(file)

    # Initialize a dictionary to store the opt data
    opt_data = {}

    for date, date_data in data.items():
        if datetime.strptime(date, '%Y-%m-%d %H:%M:%S') < datetime.strptime('2022-09-01 00:00:00', '%Y-%m-%d %H:%M:%S'):
            continue
        for period in opt_periods:
            for target in targets:
                for strategy_name in strategy_names:
                    function = 'opt'
                    # Check if all the keys exist and return if not
                    if not (period in date_data and target in date_data[period] and strategy_name in date_data[period][target] and function in date_data[period][target][strategy_name] and target in date_data[period][target][strategy_name][function] and '0' in date_data[period][target][strategy_name][function][target]):
                        print(f"Data not found for {date}, {period}, {target},{strategy_name}, {function}.")
                        return

                    # Build the key for the opt_data dictionary
                    key = f'{date}_{period}_{target}_{strategy_name}_{function}'
                    if str(date) not in opt_data:
                        opt_data[str(date)] = {}
                    opt_data[str(date)][key] = date_data[period][target][strategy_name][function][target]['0']

    # Write the opt data to the output file
    with open(output_path, 'w') as file:
        json.dump(opt_data, file)

# Call the function
#extract_opt_data('2021_10_08_2023_05_28_opt', 'opt_data.json')

def get_data(apply_json, end_time, target, period,strategy_name):
    # 排序JSON键
    sorted_times = sorted(apply_json.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

    wanted_data = {}
    s_target = target
    s_period = period
    data_len = 10 # 70 days
    # Step 1: Find the index of end_time in sorted_times
    try:
        end_time_index = sorted_times.index(end_time)
    except ValueError:
        return {}  # If end_time is not found, return an empty dictionary
    # Step 2: Create a sublist containing data_len number of time points before end_time, including end_time
    start_index = max(0, end_time_index - data_len + 1)
    subset_times = sorted_times[start_index:end_time_index + 1]
    with open("config.json") as config_file:
        config_data = json.load(config_file)
    right_strategy_name = config_data["strategy_names"]
    right_target = config_data["target_names"]
    
    # Step 3: Collect data
    for time in subset_times:
        for pices in apply_json.get(time, {}):  # Safely get the value, default to empty dictionary if key is not found
            digit = pices.split('_')[-1]
            if digit == '3':
                # local_target = "_".join(pices.split('_')[:-3][2:])
                # print("local_target")
                # print(local_target)
                local_period = pices.split('_')[1]
                for right_name in right_strategy_name:
                    if right_name in pices:
                        local_strategy_name = right_name
                target_pices = pices.replace(local_strategy_name,'')
                target_pices = target_pices.split('__')[0]
                #print(target_pices)
                local_target = '_'.join(target_pices.split('_')[2:])
                if local_period == s_period and local_target == s_target and local_strategy_name == strategy_name:
                    wanted_data[time] = apply_json[time][pices]
    return wanted_data


#end_time = "2023-10-02 00:00:00"
#check = get_data(apply_data_para_path='apply_data.json_para', end_time = end_time, data_len=41, 
#                 target='weighted_sqn_drawdown', period='45')

    
int_list = ['sell_volume_window','buy_volume_window','sell_vwap_period','buy_vwap_period',
                'buy_atr_period','sell_atr_period','buy_atr_sma_period','sell_atr_sma_period']
    
# Initialize dictionary to store the smoothed parameters for the next period
def get_smooth_para(alpha,target_data,int_list):
    
    target_data_items = list(target_data.items())
    
    smoothed_parameters_next_period = {}
    
    # Smoothing factor (alpha), usually between 0 and 1.
    
    # Step 2: Loop through each parameter to perform exponential smoothing
    # print("target_data_items")
    # print(target_data_items)
    # print("target_data_items[0]")
    # print(target_data_items[0])
    #print("target_data_items[0][1]")
    #print(target_data_items[0][1])
    for param in target_data_items[0][1].keys():  # Taking keys from the first data point as representative parameters
        
        # Initialize smoothed value with the first observed value
        smoothed_value = target_data_items[0][1][param]
        
        # Loop through each time point to update the smoothed value
        for time, parameters in target_data_items[1:]:  # Start from the second data point
            observed_value = parameters[param]
            
            # Perform exponential smoothing
            smoothed_value = alpha * observed_value + (1 - alpha) * smoothed_value
        
        # Step 3: Store the smoothed value as the prediction for the next period
        if param in int_list:
            smoothed_parameters_next_period[param] = int(round(smoothed_value,0))
        else:
            smoothed_parameters_next_period[param] = round(smoothed_value,6)
        
    return smoothed_parameters_next_period


def apply_smooth_for_alldata(start_all,apply_data_para_path, targets, periods,alphas,int_list,strategy_names):
    with open(apply_data_para_path, 'r') as file:
        apply_json = json.load(file)
    sorted_times = sorted(apply_json.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    print(f"sorted_times: {sorted_times}")
    start_point = np.where(np.array(sorted_times) == start_all)[0][0]
    new_sorted_times = sorted_times[start_point:]
    
    for date_time in new_sorted_times:
        print(date_time)
        for target in targets:
            for period in periods:
                for strategy_name in strategy_names:
                    para_data = get_data(apply_json=apply_json, end_time=date_time,target=target, period=period,strategy_name=strategy_name)
                    if len(para_data) == 0:
                        raise ValueError(f"No data found for {date_time}, {period}, {target}, {strategy_name}.")
                    for alpha in alphas:
                        #para_data = get_data(apply_data_para_path, end_time=date_time, data_len=data_len, target=target, period=period)
                        smoothed_para = get_smooth_para(alpha,target_data=para_data,int_list=int_list)
                        name = date_time+'_'+period+'_'+target+'_'+strategy_name+'_'+'apply'+'_'+'alpha_'+str(alpha)
                        apply_json[date_time][name] = smoothed_para
    print('finished')
    with open(apply_data_para_path, 'w') as file:
        json.dump(apply_json, file)
    

def process_date(date_time, apply_data_para_path, data_len, targets, periods, alphas, int_list):
    # Load the data for this date_time
    with open(apply_data_para_path, 'r') as file:
        apply_json = json.load(file)

    results = {}
    print(date_time)
    for target in targets:
        for period in periods:
            para_data = get_data(apply_data_para_path, end_time=date_time, data_len=data_len, target=target, period=period)
            for alpha in alphas:
                smoothed_para = get_smooth_para(alpha, target_data=para_data, int_list=int_list)
                name = f"{date_time}_{period}_{target}_apply_alpha_{alpha}"
                results[name] = smoothed_para

    return date_time, results

def apply_smooth_for_alldata_parallel(start_all, apply_data_para_path, data_len, targets, periods, alphas, int_list):
    with open(apply_data_para_path, 'r') as file:
        apply_json = json.load(file)

    sorted_times = sorted(apply_json.keys(), key=lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    start_point = np.where(np.array(sorted_times) == start_all)[0][0]
    new_sorted_times = sorted_times[start_point:]

    # Use multiprocessing pool
    with Pool() as pool:
        # Map the process_date function to the list of dates
        results = pool.starmap(process_date, [(date_time, apply_data_para_path, data_len, targets, periods, alphas, int_list) for date_time in new_sorted_times])

    # Update the apply_json with the results
    for date_time, date_results in results:
        apply_json[date_time].update(date_results)

    # Write the updated apply_json to file
    with open(apply_data_para_path, 'w') as file:
        json.dump(apply_json, file)

# Call the parallel function
#apply_smooth_for_alldata_parallel(start_all, apply_data_para_path, data_len, targets, periods, alphas, int_list)
