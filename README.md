# RBT - Rolling Backtest Trade

# Backtesting Library based on NumPy and Numba
[中文](#chinese-section)
---

## English

### Introduction

This repository houses a backtesting library that is built on top of `NumPy` and `Numba`. This library is designed to provide a highly efficient and versatile environment for trading strategy simulation and evaluation.

### Features

- **Switchable Strategies**: One of the unique aspects of this library is the ability to switch strategies during the backtesting period. This capability allows for a more dynamic and adaptive approach to market conditions.
  
- **State Preservation**: Information such as current positions, cost basis (for stop-loss), and peak profits (for take-profit) can be seamlessly transferred to the subsequent strategy period. This ensures continuity and can be vital for strategies that involve hedging, scaling, or other complex operations.

### Operational Steps
Post-Strategy Launch: Actions to be Performed Each Trading Cycle:
#### 1. Run trading_data_pipeline.py to update Bitcoin data to the latest day (UTC+0), check and fill in missing data (do not fill data for periods when Binance exchange is down), verify data format, and check for duplicate data (minor amounts are acceptable).
#### 2. Directly execute the allocata.sh script, changing the optimization cycle to 5, to update the past 9 days of strategy apply cycles. (nohup ./allocate.sh > allocate_output.log 2>&1 &)
#### 3. Run calculate_divisible_dates.py to confirm update dates.
#### 4. Copy dates from opt_date_to_the_latest_date, remove commas, input them, and execute the allocata.sh script to optimize the trading strategy for specified dates, targets, and time periods. (nohup ./allocate.sh > allocate_output.log 2>&1 &)
#### 5. Execute check_test_intergret.py to integrate all JSON files and check for any missing trading strategies (due to the evicting nature of cloud computing platforms). Output: processed_errors.json.
#### 6. Run fill_process_opt_gap.sh, read processed_errors.json, and use rolling.py to fill in any gaps in the trading strategies. (nohup ./fill_process_opt_gap.sh > output_fill.txt 2>&1 &)
#### 7. Execute check_test_intergret.py again to check for any omissions after running fill_process_opt_gap.sh.
#### 8. Run process_opt_for_model.py to split the dictionary and add smoothing sets.
#### 9. Execute Using_rolling_strategy.sh to update parameters for the select_model. Remember to update the script's date. \
nohup ./Using_rolling_strategy.sh > Using_rolling_strategy.log 2>&1 &

#### 10 Find the optimal lambda_1 and scorethreshold.
nohup /usr/bin/python3 /home/WLH_trade/0615/trading/rolling_select_strategy_model.py 2>&1 | tee -a logfile.log &

#### 11. Execute apply_selected_para.py, outputting two JSON files: max_target_dict_list, apply_json_values_list.
12 hours before: Update trading data and strategy optimization parameters. (end_date is 6 days before the current date). Aggressive and moderate strategies need this update, conservative ones do not, due to their longer time span.

9 hours before: Update trading data and strategy optimization parameters, start updating the strategy selection model based on the results of the past 12 hours' strategy optimization parameters (end_date is 6 days before the current date).

3 hours before: Update trading data and strategy optimization parameters, start updating the strategy selection model based on the results of the past 9 hours' strategy optimization parameters (end_date is 6 days before the current date).

1 hour before: Update trading data and strategy optimization parameters, start updating the strategy selection model based on the results of the past 3 hours' strategy optimization parameters (end_date is 6 days before the current date).

At hour 0: Update trading data and strategy optimization parameters, start updating the strategy selection model based on the results of the past 1 hour's strategy optimization parameters (end_date is 6 days before the current date), and launch the trading strategy based on the strategy selection model of the past 1 hour.

#### 12. Live Trading Phase
nohup python3 -u test.py > trading_output.log 2>&1 &

#### 13. Database Information:
##### 1. Without external information, add norm and L2 regularization.
    cur_study_name = f'ModelSelect-study-2023-10-23_norm_{rolling_len}-{date}-lambda_1{lambda_1}'
    old_study_name = f'ModelSelect-study-2023-10-23_norm_{rolling_len}-{old_date}-lambda_1{lambda_1}'
    db_name = f"rolling_select_strategy_model-2023-10-23_norm_{rolling_len}-lambda_1{lambda_1}.db"
##### 2. Without external information, add L2 regularization.
    cur_study_name = f'ModelSelect-study-2023-10-19_norm_{rolling_len}-{date}-lambda_1{lambda_1}'
    old_study_name = f'ModelSelect-study-2023-10-19_norm_{rolling_len}-{old_date}-lambda_1{lambda_1}'
    db_name = f"rolling_select_strategy_model-2023-10-19_{rolling_len}-lambda_1{lambda_1}.db"
##### 3. With external information, add norm and L2 regularization. External information is integrated into L2 regularization optimization. The lambda for external information is currently set the same as for L2 regularization.
    cur_study_name = f'ModelSelect-study-2023-10-28_norm_{rolling_len}-{date}-lambda_1{lambda_1}_PCA'
    old_study_name = f'ModelSelect-study-2023-10-28_norm_{rolling_len}-{old_date}-lambda_1{lambda_1}_PCA'
    db_name = f"rolling_select_strategy_model-2023-10-28_norm_{rolling_len}-lambda_1{lambda_1}_PCA.db"

### Note

This library is currently made public with backtesting parts only, as strategies built upon it have already proven to be profitable.

---
<a name="chinese-section"></a>
## 中文
### 简介

该仓库包含一个基于 `NumPy` 和 `Numba` 的回测库。该库旨在提供一个高效且多功能的环境，用于交易策略的模拟和评估。

### 特性

- **可切换策略**: 该库的一个独特之处在于可以在回测期间切换策略。这一功能允许更动态和适应性地应对市场条件。
  
- **状态保存**: 当前仓位、成本基础（用于止损）以及最高利润（用于止盈）等信息可以无缝转移到下一个策略期间。这确保了连续性，并且对于涉及对冲、规模调整或其他复杂操作的策略至关重要。

### 操作步骤

#### 策略启动之后每交易周期执行：
##### 1，执行trading_data_pipeline.py，将比特币数据更新到最新一天（东0区），并检查和填充空缺数据（币安交易所停机时间没有比特币数据，这些数据不填充），检查数据格式，检查是否有重复数据（少量不影响）。
##### 2，直接执行allocata.sh脚本，将优化周期改为5，这样就可以更新过去9天的策略的apply周期。（nohup ./allocate.sh > allocate_output.log 2>&1 &）
##### 2，执行calculate_divisible_dates.py，确认更新日期
##### 3，从opt_date_to_the_latest_date复制日期，去除逗号，将其输入并执行allocata.sh脚本，将指定日期，target，time period的交易策略进行优化。（nohup ./allocate.sh > allocate_output.log 2>&1 &）
##### 3，执行check_test_intergret.py,将所有的json文件整合在一起，并检查计算的交易策略是否有遗漏（云计算平台有逐出特性）。输出为processed_errors.json。
##### 4，执行fill_process_opt_gap.sh,读取processed_errors.json,调用rolling.py将空缺的交易策略进行补全。(nohup ./fill_process_opt_gap.sh > output_fill.txt 2>&1 &)
##### 5，执行check_test_intergret.py,将所有的json文件整合在一起，并检查计算的交易策略是否有遗漏（云计算平台有逐出特性）。输出为processed_errors.json。
##### 6，执行process_opt_for_model.py 将字典进行切分。并且添加参数平滑集。
##### 7，执行Using_rolling_strategy.sh，更新select_model的参数。
记得更新脚本的日期 \
nohup ./Using_rolling_strategy.sh > Using_rolling_strategy.log 2>&1 &
##### 8.2 寻找最优lambda_1和socrethreshold
nohup /usr/bin/python3 /home/WLH_trade/0615/trading/rolling_select_strategy_model.py 2>&1 | tee -a logfile.log &

##### 7，执行apply_selected_para.py，输出2个json文件：max_target_dict_list, apply_json_values_list
###### 前12小时：交易数据更新，策略优化参数更新。（end_date日期为当前日期-6天）激进和中庸需要，保守不需要，因为保守时间跨度足够长。
策略优化参数更新大概需要45*200，9000秒-3小时。
###### 前9小时，交易数据更新，策略优化参数更新，选择策略模型根据前12小时的策略优化参数结果开始更新（end_date日期为当前日期-6天）
###### 前3小时，交易数据更新，策略优化参数更新，选择策略模型根据前9小时的策略优化参数结果开始更新（end_date日期为当前日期-6天）
###### 前1小时，交易数据更新，策略优化参数更新，选择策略模型根据前3小时的策略优化参数结果开始更新（end_date日期为当前日期-6天）
###### 前0小时，交易数据更新，策略优化参数更新，选择策略模型根据前1小时的策略优化参数结果开始更新（end_date日期为当前日期-6天），交易策略根据前1小时的选择策略模型上线。

#### 实盘阶段
nohup python3 -u test.py > trading_output.log 2>&1 &

#### 数据库资料：
##### 1，无外部信息，加入norm和L2正则化
    cur_study_name = f'ModelSelect-study-2023-10-23_norm_{rolling_len}-{date}-lambda_1{lambda_1}'
    old_study_name = f'ModelSelect-study-2023-10-23_norm_{rolling_len}-{old_date}-lambda_1{lambda_1}'
    db_name = f"rolling_select_strategy_model-2023-10-23_norm_{rolling_len}-lambda_1{lambda_1}.db"
##### 2，无外部信息，加入L2正则化
    cur_study_name = f'ModelSelect-study-2023-10-19_norm_{rolling_len}-{date}-lambda_1{lambda_1}'
    old_study_name = f'ModelSelect-study-2023-10-19_norm_{rolling_len}-{old_date}-lambda_1{lambda_1}'
    db_name = f"rolling_select_strategy_model-2023-10-19_{rolling_len}-lambda_1{lambda_1}.db"
##### 3, 有外部信息，加入norm和L2正则化。外部信息是带入L2正则化一起优化。外部信息的lambda目前设置为和L2正则化一样。
    cur_study_name = f'ModelSelect-study-2023-10-28_norm_{rolling_len}-{date}-lambda_1{lambda_1}_PCA'
    old_study_name = f'ModelSelect-study-2023-10-28_norm_{rolling_len}-{old_date}-lambda_1{lambda_1}_PCA'
    db_name = f"rolling_select_strategy_model-2023-10-28_norm_{rolling_len}-lambda_1{lambda_1}_PCA.db"

### 注意事项
由于基于该库构建的策略已经证明是盈利的，因此目前仅公开回测库。

