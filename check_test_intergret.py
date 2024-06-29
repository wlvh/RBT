#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2023-08-01 14:08:17
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2023-08-20 13:02:24
FilePath: /trading/check_test_intergret.py

Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 

Description: 
This script processes JSON files from a specified directory, checks for missing keys, 
processes the resulting errors, and checks for missing dates. The output is a list of 
processed errors and a printout of the missing dates check result. The script primarily 
relies on functions imported from the `process_json_files.py` module.

Usage:
python3 check_test_intergret.py

Note: 
The directory for JSON files and output file path are hardcoded in this script. 
Please modify the `process_json_files` function call in the main section 
of the script to specify a different directory or output file.
'''
import json
from process_json_files import process_json_files, check_missing_keys, process_errors, check_dates

# Call the function process_json_files
process_json_files(directory='dict_rolling', output_file_path='2021_10_08_2023_05_28_opt')

# Call the function check_missing_keys
errors = check_missing_keys(json_path='2021_10_08_2023_05_28_opt',config_path="/home/WLH_trade/0615/trading/config.json")

# Process the errors
processed_errors = process_errors(errors)

# Save the processed errors to a file
with open('processed_errors.json', 'w') as f:
    json.dump(processed_errors, f)

# Call the function check_dates
date_check = check_dates(data_path='2021_10_08_2023_05_28_opt')

# Print the result of check_dates
print(date_check)
