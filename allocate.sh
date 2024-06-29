#!/bin/bash
###
 # @Author: wlvh 124321452@qq.com
 # @Date: 2023-06-21 10:59:18
 # @LastEditors: wlvh 124321452@qq.com
 # @LastEditTime: 2024-06-23 03:44:33
 # @FilePath: /trading/allocate.sh
 # @Description:
 # 
 # Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
### 
# 设置环境变量，限制numpy和pandas使用单线程
exec 2>> shell_errors.txt
targets=('weighted_win_ratio' 'weighted_value' 'weighted_drawdown' 'weighted_SharpeDIY' 'weighted_drawdown_value' 'weighted_win_ratio_value' 'weighted_drawdown_win_ratio_value' 'weighted_sqn' 'weighted_sqn_drawdown' 'weighted_profitable_1days_ratio' 'weighted_profitable_2days_ratio' 'weighted_profitable_3days_ratio') 
data_file="BTCUSDT_1m.csv"
data_periods=(35 56 70)

# time_list=('2022-09-04 00:00:00' '2022-09-11 00:00:00' '2022-09-18 00:00:00' '2022-09-25 00:00:00' '2022-10-02 00:00:00' '2022-10-09 00:00:00' '2022-10-16 00:00:00' '2022-10-23 00:00:00' '2022-10-30 00:00:00' '2022-11-06 00:00:00' '2022-11-13 00:00:00' '2022-11-20 00:00:00' '2022-11-27 00:00:00' '2022-12-04 00:00:00' '2022-12-11 00:00:00' '2022-12-18 00:00:00' '2022-12-25 00:00:00' '2023-01-01 00:00:00' '2023-01-08 00:00:00' '2023-01-15 00:00:00' '2023-01-22 00:00:00' '2023-01-29 00:00:00' '2023-02-05 00:00:00' '2023-02-12 00:00:00' '2023-02-19 00:00:00' '2023-02-26 00:00:00' '2023-03-05 00:00:00' '2023-03-12 00:00:00' '2023-03-19 00:00:00' '2023-03-26 00:00:00' '2023-04-02 00:00:00' '2023-04-09 00:00:00' '2023-04-16 00:00:00' '2023-04-23 00:00:00' '2023-04-30 00:00:00' '2023-05-07 00:00:00' '2023-05-14 00:00:00' '2023-05-21 00:00:00' '2023-05-28 00:00:00' '2023-06-04 00:00:00' '2023-06-11 00:00:00' '2023-06-18 00:00:00' '2023-06-25 00:00:00' '2023-07-02 00:00:00' '2023-07-09 00:00:00' '2023-07-16 00:00:00' '2023-07-23 00:00:00' '2023-07-30 00:00:00' '2023-08-06 00:00:00' '2023-08-13 00:00:00' '2023-08-20 00:00:00' '2023-08-27 00:00:00' '2023-09-03 00:00:00' '2023-09-10 00:00:00' '2023-09-17 00:00:00' '2023-09-24 00:00:00' '2023-10-01 00:00:00' '2023-10-08 00:00:00' '2023-10-15 00:00:00' '2023-10-22 00:00:00' '2023-10-29 00:00:00' '2023-11-05 00:00:00' '2023-11-12 00:00:00' '2023-11-19 00:00:00' '2023-11-26 00:00:00' '2023-12-03 00:00:00' '2023-12-10 00:00:00' '2023-12-17 00:00:00' '2023-12-24 00:00:00' '2023-12-31 00:00:00' '2024-01-07 00:00:00' '2024-01-14 00:00:00' '2024-01-21 00:00:00' '2024-01-28 00:00:00' '2024-02-04 00:00:00' '2024-02-11 00:00:00' '2024-02-18 00:00:00' '2024-02-25 00:00:00' '2024-03-03 00:00:00' '2024-03-10 00:00:00' '2024-03-17 00:00:00' '2024-03-24 00:00:00' '2024-03-31 00:00:00' '2024-04-07 00:00:00' '2024-04-14 00:00:00' '2024-04-21 00:00:00' '2024-04-28 00:00:00' '2024-05-05 00:00:00' '2024-05-12 00:00:00' '2024-05-19 00:00:00' '2024-05-26 00:00:00' '2024-06-02 00:00:00' '2024-06-09 00:00:00'
# ) 

time_list=('2024-06-23 00:00:00')

num_evals=500
half_evals=500
extra_evals=500
cpu_cores=$(nproc)
max_jobs=$((cpu_cores - 0))
strategy_names=("VWAPStrategyLong" "VWAPStrategyShort" 'RSIV_StrategyOneday' 'RSIV_StrategyFourhour' "MAVStrategy" "WRSIStrategy")

# clear the log files, they are too big
rm -f errors.txt
rm -f output.txt

for end_date in "${time_list[@]}"; do
    for target in "${targets[@]}"; do
        for data_period in "${data_periods[@]}"; do
            for strategy_name in "${strategy_names[@]}"; do
                running_jobs=$(ps aux | grep "python3 rolling.py" | grep -v grep | wc -l)

                while (( running_jobs >= max_jobs )); do
                    sleep 10
                    running_jobs=$(ps aux | grep "python3 rolling.py" | grep -v grep | wc -l)
                done

                echo "Executing for end date $end_date, target $target, period $data_period, strategy $strategy_name" evals $half_evals

                python3 rolling.py "$target" "$data_file" "$end_date" "$data_period" "$half_evals" "$strategy_name" 2>> errors.txt >> output.txt &
                sleep 1
            done
        done
    done
done

for end_date in "${time_list[@]}"; do
    for target in "${targets[@]}"; do
        for data_period in "${data_periods[@]}"; do
            for strategy_name in "${strategy_names[@]}"; do
                running_jobs=$(ps aux | grep "python3 rolling.py" | grep -v grep | wc -l)

                while (( running_jobs >= max_jobs )); do
                    sleep 10
                    running_jobs=$(ps aux | grep "python3 rolling.py" | grep -v grep | wc -l)
                done

                echo "Executing for end date $end_date, target $target, period $data_period, strategy $strategy_name" evals $num_evals

                python3 rolling.py "$target" "$data_file" "$end_date" "$data_period" "$num_evals" "$strategy_name" 2>> errors.txt >> output.txt &
                sleep 1
            done
        done
    done
done

for end_date in "${time_list[@]}"; do
    for target in "${targets[@]}"; do
        for data_period in "${data_periods[@]}"; do
            for strategy_name in "${strategy_names[@]}"; do
                running_jobs=$(ps aux | grep "python3 rolling.py" | grep -v grep | wc -l)

                while (( running_jobs >= max_jobs )); do
                    sleep 10
                    running_jobs=$(ps aux | grep "python3 rolling.py" | grep -v grep | wc -l)
                done

                echo "Executing for end date $end_date, target $target, period $data_period, strategy $strategy_name" evals $extra_evals

                python3 rolling.py "$target" "$data_file" "$end_date" "$data_period" "$extra_evals" "$strategy_name" 2>> errors.txt >> output.txt &
                sleep 1
            done
        done
    done
done

#所有任务结束后写入log
# 所有任务结束后写入log
echo "All tasks completed"
# Wait for all end_date tasks to complete
wait

# pgrep -f 'python3 rolling.py' | wc -l
# nohup ./allocate.sh > allocate_output.log 2>&1 &
# pkill -f 'rolling.py'
# pkill -f 'allocate.sh'
# python3 -u rolling.py 'weighted_win_ratio' "BTCUSDT_1m.csv" '2024-05-26 00:00:00' "70" "100" "WRSIStrategy" 2> errors.txt > output.txt

#python3 -u rolling.py 'weighted_win_ratio' "BTCUSDT_1m.csv" '2024-05-26 00:00:00' "70" "100" "WRSIStrategy"
