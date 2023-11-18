'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Author: wlvh 124321452@qq.com
Date: 2023-06-20 06:14:07
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2023-11-10 12:22:33
FilePath: /trading/rolling.py
Description: 
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:56:04 2023

@author: lyuhongwang
"""
import datetime as dt
from DIY_Volume_numpy import optimize_and_run,find_row_by_date,evaluate_opt_results
from rolling_strategy_test import *
import pandas as pd
import json
import argparse

def load_data(filepath):
    df = pd.read_csv(filepath, sep=',')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('datetime', inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="The target to optimize and run")
    parser.add_argument("data_file", help="The data file to load")
    parser.add_argument("end_date", help="The end date of the data")
    parser.add_argument("data_period", type=int, help="The data period for rolling strategy test")
    parser.add_argument("num_evals", type=int, help="How much trials would training for rolling strategy test")
    parser.add_argument("strategy_name", type=str, help="Which strategy should be apply")
    
    args = parser.parse_args()

    df = load_data(args.data_file)
    target = args.target    
    rolling_strategy_test(df, data_period=args.data_period, num_evals=args.num_evals, targets=target,end_date=args.end_date, strategy_name=args.strategy_name)

    
if __name__ == "__main__":
    main()
