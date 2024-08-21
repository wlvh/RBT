#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2024-08-14 08:37:29
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-08-15 15:14:08
FilePath: /trading/RL_selector/RL_select_model.py
Description: 
Copyright (c) 2024 by ${124321452@qq.com}, All Rights Reserved. 
'''
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta
class FinancialTradingEnv(gym.Env):
    def __init__(self, end_date: str, data_length: int, strategies_data: pd.DataFrame):
        super(FinancialTradingEnv, self).__init__()
        
        self.market_data = self.load_market_data(end_date, data_length)
        self.strategies_data = strategies_data
        self.dates = strategies_data.index.get_level_values('date').unique() #这里必须用策略的日期
        self.current_date = None
        self.date_index = 0

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
        self.date_index = 30  # Start from the 31st day to have 30 days of history
        self.current_date = self.dates[self.date_index]
        return self._next_observation()

    def _next_observation(self):
        # 获取当前日期和过去30天的数据
        end_date = self.current_date
        start_date = end_date - timedelta(days=30)
        
        # 获取这31天的市场数据
        observation_data = self.market_data.loc[start_date:end_date]
        
        # 将数据展平为一维数组
        return observation_data.values.flatten().astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # 解析动作
        time_weights, target_weights, type_weights = action

        # 计算策略分数
        selected_strategies = self._calculate_strategy_scores(time_weights, target_weights, type_weights)

        # 计算投资组合收益
        reward = self._calculate_portfolio_return(selected_strategies)
        # 更新日期
        self.date_index += 1
        if self.date_index < len(self.dates):
            self.current_date = self.dates[self.date_index]

        # 获取下一个观察
        obs = self._next_observation()

        # 检查是否结束
        done = self.date_index >= len(self.dates) - 1
        # 添加额外信息
        info = {
            'selected_strategies_count': len(selected_strategies),
            'current_date': self.dates[self.current_step]
        }

        return obs, reward, done, info

    def _calculate_strategy_scores(self, time_weights: np.ndarray, target_weights: np.ndarray, type_weights: np.ndarray) -> np.ndarray:
        # 使用多重索引进行高效查询
        idx = pd.IndexSlice
        current_strategies = self.strategies_data.loc[idx[self.current_date, :, :, :]]

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