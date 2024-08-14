# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
from collections import OrderedDict
from datetime import datetime
import pandas as pd
# 打开 JSON 文件
with open(r"C:\Users\Administrator.DESKTOP-4H80TP4\Downloads\2021_10_08_2023_05_28_opt", 'r') as f:
    # 使用 json.load() 解析 JSON 数据
    data = json.load(f)

# 将键转换为 datetime 对象，然后排序
sorted_keys = sorted(data.keys(), key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

# 创建一个新的有序字典，按照排序后的键
sorted_data = OrderedDict((key, data[key]) for key in sorted_keys if key >='2024-07-22 00:00:00')

# 使用 open() 函数以写入模式打开文件，并使用 json.dump() 函数将数据写入文件
filename = 'data.json'
with open(filename, 'w', encoding='utf-8') as file:
    json.dump(sorted_data, file, ensure_ascii=False, indent=4)

print(f"数据已成功保存到 {filename} 文件中。")

def extract_strategies_data(json_file_path: str) -> pd.DataFrame:
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    strategies_list = []

    for end_date, time_data in data.items():
        for time, target_data in time_data.items():
            for target, type_data in target_data.items():
                for type_, strategy_data in type_data.items():
                    apply_data = strategy_data['apply']['7'][target]['0']
                    total_return = apply_data['total_return']

                    strategy_info = {
                        'date': pd.to_datetime(end_date),
                        'time': int(time),
                        'target': target,
                        'type': type_,
                        'return': total_return
                    }

                    strategies_list.append(strategy_info)

    # 创建 DataFrame
    strategies_df = pd.DataFrame(strategies_list)

    # 设置多重索引
    strategies_df.set_index(['date', strategies_df.groupby('date').cumcount()], inplace=True)

    return strategies_df

# 使用函数加载数据
file_path = 'data.json'
strategies_data1 = extract_strategies_data(file_path)
strategies_data = load_strategies_data_from_json(file_path)
print(strategies_data)

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple

class FinancialTradingEnv(gym.Env):
    def __init__(self, end_date: str, data_length: int, strategies_data: pd.DataFrame):
        super(FinancialTradingEnv, self).__init__()
        
        self.market_data = self.load_market_data(end_date, data_length)
        self.strategies_data = strategies_data
        self.dates = strategies_data.index.get_level_values('date').unique() #这里必须用策略的日期
        self.current_step = 0

        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.market_data.columns),), dtype=np.float32
        )

        # 定义动作空间
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(3, 3, 3), dtype=np.float32
        )

        # 预处理策略数据
        self.preprocess_strategies_data()

    def load_market_data(self, end_date: str, data_length: int) -> pd.DataFrame:
        # 这里应该实现实际的数据加载逻辑
        # 为了示例，我们生成一些模拟数据
        dates = pd.date_range(end=end_date, periods=data_length)
        data = {
            'price_change': np.random.normal(0, 0.02, data_length),
            'volume_change': np.random.normal(0, 0.1, data_length),
            'price_volatility': np.random.uniform(0.01, 0.05, data_length),
            'volume_volatility': np.random.uniform(0.05, 0.2, data_length)
        }
        return pd.DataFrame(data, index=dates)

    def preprocess_strategies_data(self):
        categories = {
            'time': sorted(self.strategies_data['time'].unique()),
            'target': sorted(self.strategies_data['target'].unique()),
            'type': sorted(self.strategies_data['type'].unique())
        }
        
        for category, values in categories.items():
            idx_name = f'{category}_idx'
            self.strategies_data[idx_name] = self.strategies_data[category].map({v: i for i, v in enumerate(values)})

        # 将索引转换为类别以提高性能
        self.strategies_data.set_index(['date', 'time_idx', 'target_idx', 'type_idx'], append=True, inplace=True)

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        return self.market_data.iloc[self.current_step].values.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_step += 1

        # 解析动作
        time_weights, target_weights, type_weights = action

        # 计算策略分数
        selected_strategies = self._calculate_strategy_scores(time_weights, target_weights, type_weights)

        # 计算投资组合收益
        reward = self._calculate_portfolio_return(selected_strategies)


        # 获取下一个观察
        obs = self._next_observation()

        # 检查是否结束
        done = self.current_step >= len(self.market_data) - 1
        # 添加额外信息
        info = {
            'selected_strategies_count': len(selected_strategies),
            'current_date': self.dates[self.current_step]
        }

        return obs, reward, done, info

    def _calculate_strategy_scores(self, time_weights: np.ndarray, target_weights: np.ndarray, type_weights: np.ndarray) -> np.ndarray:
        
        current_date = self.dates[self.current_step]
        # 使用多重索引进行高效查询
        idx = pd.IndexSlice
        current_strategies = self.strategies_data.loc[idx[current_date, :, :, :]]

        # 计算策略分数
        scores = (
            time_weights[current_strategies['time_idx']] + 
            target_weights[current_strategies['target_idx']] + 
            type_weights[current_strategies['type_idx']]
        )
        
        return current_strategies[scores > 0]

    def _calculate_portfolio_return(self, selected_strategies: np.ndarray) -> float:
        if len(selected_strategies) > 0:
            reward = selected_strategies['total_return'].mean()
        else:
            reward = 0
        return reward

# 使用示例
market_data = pd.DataFrame({
    'sp500': [0.01, -0.02, 0.005, ...],
    'vix': [15, 16, 14, ...],
    # 其他市场数据...
})

strategies_data = pd.DataFrame({
    'time': [35, 56, 77, ...],
    'target': ['value', 'sqn', 'win_ratio', ...],
    'type': ['RSI', 'VWAP', 'MA', ...],
    'return': [0.02, -0.01, 0.03, ...],
    # 其他策略数据...
})

env = FinancialTradingEnv(market_data, strategies_data)

# 与强化学习算法一起使用
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 评估模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()