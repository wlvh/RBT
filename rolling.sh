#!/bin/bash
start_time=$(date +%s)

targets=('weighted_win_ratio' 'weighted_sqn' 'weighted_value' 'weighted_drawdown' 'weighted_SharpeDIY' 'weighted_drawdown_value' 'weighted_win_ratio_value')
data_file="BTCUSDT_1m.csv"
start_date="2022-06-15 00:01:00"
end_date="2023-06-15 00:01:00"
data_periods=(30 45 60)
rolling_window=3

# Convert dates to Unix timestamps
start_date_sec=$(date -d"$start_date" +%s)
end_date_sec=$(date -d"$end_date" +%s)

for target in "${targets[@]}"; do
    for data_period in "${data_periods[@]}"; do
        # Calculate the difference in days and divide by 2
        half_days=$(( (end_date_sec - start_date_sec) / 2 / 86400 ))

        # Add data_period - rolling_window to half_days
        half_days=$(( half_days + data_period - rolling_window ))

        # Make sure the number of days is divisible by rolling_window
        half_days=$(( (half_days / rolling_window) * rolling_window ))

        # Calculate the new end date for the first half
        new_end_date=$(date -d"@$((start_date_sec + half_days * 86400))" +"%Y-%m-%d %H:%M:%S")

        echo "New end date for data period $data_period: $new_end_date"

        # Calculate the half_end_date, which is 6 days later than new_start_date
        half_end_date=$(date -d"@$((start_date_sec + half_days * 86400 ))" +"%Y-%m-%d %H:%M:%S")

        python3 rolling.py "$target" "$data_file" "$start_date" "$half_end_date" "$data_period" &

        # Calculate the new start date for the second half
        new_start_date=$(date -d"@$((start_date_sec + half_days * 86400 - (data_period + rolling_window) * 86400))" +"%Y-%m-%d %H:%M:%S")

        python3 rolling.py "$target" "$data_file" "$new_start_date" "$end_date" "$data_period" &
    done
done

wait
# 获取脚本结束时的时间
end_time=$(date +%s)

# 计算并打印脚本的总运行时间
total_time=$((end_time - start_time))
echo "Total running time: $total_time seconds"
