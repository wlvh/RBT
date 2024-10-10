#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2024-08-14 08:37:29
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-08-28 14:50:20
FilePath: /trading/RL_selector/RL_select_model.py
Description: 
Copyright (c) 2024 by ${124321452@qq.com}, All Rights Reserved. 
'''
import gymnasium as gym  # 使用 gymnasium 替换 gym
from gymnasium import spaces  # 使用 gymnasium 的 spaces
import numpy as np
import copy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
from typing import Tuple,Callable,Any,Dict
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import os
import json
from features_extractor import CNNFeaturesExtractor, AutoEncoderFeaturesExtractor, VAEFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
#from stable_baselines3.common.callbacks import EvalCallback
from RL_data_process import extract_strategies_data, rolling_window_standardize
import logging
from logging.handlers import RotatingFileHandler
from stable_baselines3.common.vec_env import DummyVecEnv
import traceback
from functools import wraps
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
import torch
from stable_baselines3.common.logger import configure
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelBinarizer
import warnings
from stable_baselines3.common.vec_env import VecMonitor
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')
warnings.filterwarnings("ignore", category=UserWarning, message="Tried to write empty key-value dict")
# 配置日志记录
def setup_logging(log_file: str = 'rl_trading.log', level: int = logging.INFO):
    '''
    集中化的日志配置：
    setup_logging 函数配置了一个中央日志系统，包括文件和控制台输出。
    使用 RotatingFileHandler 来管理日志文件大小，防止单个日志文件过大。
    
    错误处理装饰器：
    error_handler 装饰器捕获并记录函数中的异常，提供了一致的错误处理方式。
    它记录了详细的错误信息和堆栈跟踪，有助于调试。
    '''
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 创建一个轮转文件处理器，每个日志文件最大 5MB，保留 5 个旧文件
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 创建一个控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建一个格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
# 错误处理装饰器
def error_handler(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

setup_logging()
logger = logging.getLogger(__name__)    
    
# 每次随机从所有策略里抽取100个（同日期），生成50个假的（这里先用伪代码替代，之后我会自己实现），action改为猜测这150个策略的分类（正收益，负收益，假收益）,奖励函数为三分类的f1值

# 对抗性训练:
# 增强模型鲁棒性：对抗性训练可以帮助模型应对输入数据的微小变化,这在金融市场这样的动态环境中可能很有价值。
# 减少过拟合：通过引入随机扰动,可以防止模型过度拟合训练数据。
# 模拟真实世界的噪声：金融数据经常包含噪声,对抗性样本可以模拟这种情况。

class AdversarialTrainingCallback(BaseCallback):
    def __init__(self, verbose=0, adversarial_prob=0.1, adversarial_std=0.01):
        super(AdversarialTrainingCallback, self).__init__(verbose)
        self.adversarial_prob = adversarial_prob  # 设定添加对抗扰动的概率
        self.adversarial_std = adversarial_std  # 设定扰动的标准差
        
    def _on_step(self) -> bool:
        if np.random.random() < self.adversarial_prob:  # 随机决定是否添加扰动
            obs = self.training_env.env_method('get_next_observation')[0]
            perturbed_obs = obs.copy()  # 创建观察的副本
            
            # 只对 'market' 部分添加扰动
            if 'market' in perturbed_obs and isinstance(perturbed_obs['market'], np.ndarray):
                perturbed_obs['market'] = perturbed_obs['market'] + np.random.normal(0, self.adversarial_std, size=perturbed_obs['market'].shape)
            # 'strategies' 部分保持不变
            self.training_env.env_method('set_state', perturbed_obs)
        return True
    
def calculate_day_end_reward(self):
    '''
日终奖励是在处理完一个交易日的所有策略后,给予模型的额外反馈。它的目的是:
鼓励模型考虑整个交易日的整体表现,而不仅仅是单个策略的表现。
反映更长期的目标,如每日收益、风险控制或资金利用率。
提供一个机会来评估模型在整个交易日的决策质量。
奖励尺度: 确保日终奖励的尺度与即时奖励相当,或使用适当的缩放因子。
    '''
    returns = [decision['return'] for decision in self.daily_decisions]
    if len(returns) > 1:
        sharpe = np.mean(returns) / np.std(returns)
        return sharpe * 0.1  # 将夏普比率缩小到合适的范围
    return 0
# 可以包含过去几天的top performing策略的信息

class FinancialTradingEnv(gym.Env):
    def __init__(self, end_date: str, weeks: int, strategies_data: pd.DataFrame, market_data: pd.DataFrame, config_path: str = '/home/WLH_trade/0615/trading/config.json', num_samples: int = 20, is_training = True):
        super(FinancialTradingEnv, self).__init__()
        
        self.end_date = pd.to_datetime(end_date)
        self.start_date = self.end_date - pd.Timedelta(weeks=weeks)
        
        self.num_samples = num_samples
        self.is_training = is_training
        market_data = market_data.copy()
        market_data.loc[:, 'month'] = market_data.index.month
        market_data.loc[:, 'is_month_start'] = market_data.index.is_month_start.astype(int)
        market_data.loc[:, 'is_month_end'] = market_data.index.is_month_end.astype(int)
        market_data.loc[:, 'is_quarter_start'] = market_data.index.is_quarter_start.astype(int)
        market_data.loc[:, 'is_quarter_end'] = market_data.index.is_quarter_end.astype(int)
        self.market_data = market_data
        # 确保 'date' 列是 datetime 类型
        strategies_data['date'] = pd.to_datetime(strategies_data['date'])
        # 筛选策略数据，限制在 start_date 和 end_date 之间
        self.strategies_data = strategies_data[
            (strategies_data['date'] >= self.start_date) & 
            (strategies_data['date'] <= self.end_date)
        ]
        self.max_strategies = self.strategies_data.groupby('date').size().max() 
        # self.dates为每个周日    
        self.dates = np.sort(self.strategies_data['date'].unique())
        # 添加额外的历史数据
        extra_history_start = self.start_date - pd.Timedelta(days=7)
        extra_history_data = strategies_data[strategies_data['date'] == extra_history_start]        
        self.strategies_data = pd.concat([extra_history_data, self.strategies_data]).sort_values('date')
        
        self.preprocess_strategies_data()
        # 虚假策略所需要用到的参数
        self.fake_strategies = None
        if self.is_training:
            self.generate_fake_strategies()  # 直接在初始化时生成假策略
        self.mixed_strategies = None # 在每个步骤中混合真实和虚假策略
        self.episode_count = 0 # 计数器，为episodes_per_epoch服务
        self.episodes_per_epoch = 100  # 每隔N个epoch重新生成虚假策略，可以根据需要调整
        self.np_random = None
        self.last_observation = self.current_observation = None
        self.last_info = {}
        
        self.current_date = None
        self.date_index = 0
        # 获取策略参数列
        self.market_columns = self.market_data.columns
        market_obs_dim = len(self.market_columns) * 30  # 30天的历史数据
        #strategy_obs_dim = 3 + len(self.param_columns) + 2  # 3 for time_idx, target_idx, type_idx, 2 for target strategy flag and history strategy flag.
        # 当前策略维度
        current_strategy_dim = (
            3 +  # time_idx, target_idx, type_idx
            len(self.param_columns) +
            2  # 1 for target strategy flag, 1 for current/historical flag
        )
        # 历史策略维度（包括return）
        historical_strategy_dim = current_strategy_dim + 1  # +1 for return
        # 总策略数（1个当前策略 + 10个历史策略）
        self.total_strategies = 11
        # 增加10个用于历史performers的位置
        #total_strategies = self.max_strategies + 10
        # 定义观察空间 
        self.observation_space = spaces.Dict({
            'market': spaces.Box(low=-np.inf, high=np.inf, shape=(market_obs_dim,), dtype=np.float32),
            'strategies': spaces.Box(low=-1000, high=np.inf, shape=(self.total_strategies, max(current_strategy_dim, historical_strategy_dim)), dtype=np.float32)
        })
        # 动作空间: 0 - 假策略, 1 - 正收益, 2 - 负收益, 3 - 零收益
        self.action_space = spaces.Discrete(4)
        # self.action_space = spaces.MultiDiscrete([4] * 1)
        print(f"Market observation dimension: {market_obs_dim}")
        print(f"Current strategy dimension: {current_strategy_dim}")
        print(f"Historical strategy dimension: {historical_strategy_dim}")
        print(f"Total strategies in observation: {self.total_strategies}")
        # 模型知道自己对第一个策略进行分类。但是这样做的话reward会变得很不稳定（因为我们的评估样本只有1个）。reward改为跑完52个step后（52周）整体计算的f1值，这样会让reward变得稳定。
        self.weeks = weeks
        self.episode_step = 0
        self.episode_true_labels = []
        self.episode_predicted_labels = []
        self.episode_portfolio_returns = []
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def get_next_observation(self):
        return self._next_observation() 
    
    def analyze_param_types(self, strategies_data):
        param_types = {}
        for type_idx in strategies_data['type_idx'].unique():
            type_data = strategies_data[strategies_data['type_idx'] == type_idx]
            type_param_types = {}
            for col in [col for col in type_data.columns if col.startswith('param_')]:
                # 首先检查列的数据类型
                if type_data[col].dtype == 'int64':
                    type_param_types[col] = int
                elif type_data[col].dtype == 'float64':
                    # 对于浮点数，我们检查是否所有非 NaN 值都是整数
                    if type_data[col].dropna().apply(lambda x: x.is_integer() if isinstance(x, float) else True).all():
                        type_param_types[col] = int
                    else:
                        type_param_types[col] = float
                else:
                    # 对于其他类型，我们假设它是浮点数
                    type_param_types[col] = float
            param_types[type_idx] = type_param_types
        return param_types
    
    def generate_fake_strategies(self,test_method=True):
        # 每一条strategy会有time，target，type三个变量，分别代表它们的训练时间长度，优化目标和策略类型。当产生fake策略时：      
        # 1，有50%的几率选择class change。修改time，target，type中的2个。因此获取unique的time，target，type，随机选择（但不能选中原来的选择）。例如，当前date为’20221204‘，随机选中class change后，随机挑中了time和target。其中，A策略time为35，target为sqn。那么给它随机分配的time不得再为35，target不得再为sqn。
        # 2，有50的几率选择param change。param change的逻辑流程在for col in self.param_columns当中。
        fake_strategies = []
        # 获取所有可能的 time, target, type 的 unique 值
        unique_time = self.strategies_data['time_idx'].unique()
        unique_target = self.strategies_data['target_idx'].unique()
        unique_type = self.strategies_data['type_idx'].unique()
        sam_num = self.max_strategies // 9 # 确保虚假策略占每次抽样的三分之一
        param_types = self.analyze_param_types(self.strategies_data)
        for date in self.dates:
            try:
                real_strategies = self.strategies_data[self.strategies_data['date'] == date]
                negative_strategies = real_strategies[real_strategies['return'] < 0]
                positive_strategies = real_strategies[real_strategies['return'] > 0]
                zero_strategies = real_strategies[real_strategies['return'] == 0]
                n_negative = min(sam_num, len(negative_strategies))
                n_positive = min(sam_num, len(positive_strategies))
                n_zero = min(sam_num, len(zero_strategies))
                fake = pd.DataFrame()
                if n_negative > 0:
                    fake = pd.concat([fake, negative_strategies.sample(n=n_negative, replace=True)])
                if n_positive > 0:
                    fake = pd.concat([fake, positive_strategies.sample(n=n_positive, replace=True)])
                if n_zero > 0:
                    fake = pd.concat([fake, zero_strategies.sample(n=n_zero, replace=True)])
                
                # 检查是否达到最小假策略数量，如果不足则随机抽取补充
                if len(fake) < (sam_num * 3):
                    # 'did not find enough data, going to resample')
                    additional_needed = (sam_num * 3) - len(fake)
                    additional_strategies = real_strategies.sample(n=additional_needed, replace=True)
                    fake = pd.concat([fake, additional_strategies])
                #fake.replace('NA', np.nan, inplace=True)
            except Exception as e:
                # 捕获所有异常并打印详细信息
                print(f"An error occurred: {str(e)}")
                print(f"Date: {date} (Type: {type(date)})")
                print(f"Real strategies DataFrame length: {len(real_strategies)}")
                print(f"Real strategies DataFrame: \n{real_strategies}")
                print(f"Index format in strategies_data: {self.strategies_data.index}")
                raise e  # 重新抛出异常以便进一步调试
            # 如果 test_method 为 True，进行深拷贝
            if test_method:
                fake_copy = copy.deepcopy(fake)
            # 遍历每个策略，并进行 Class Change 或 Param Change
            for idx, strategy in fake.iterrows():
                if np.random.rand() < 0.5:
                    # 50% 概率进行 Class Change
                    variables_to_change = np.random.choice(['time_idx', 'target_idx', 'type_idx'], size=2, replace=False)
                    
                    if 'time_idx' in variables_to_change:
                        possible_times = unique_time[unique_time != strategy['time_idx']]
                        if len(possible_times) > 0:
                            new_time = np.random.choice(possible_times)
                            fake.loc[idx, 'time_idx'] = new_time
    
                    if 'target_idx' in variables_to_change:
                        possible_targets = unique_target[unique_target != strategy['target_idx']]
                        if len(possible_targets) > 0:
                            new_target = np.random.choice(possible_targets)
                            fake.loc[idx, 'target_idx'] = new_target
    
                    if 'type_idx' in variables_to_change:
                        possible_types = unique_type[unique_type != strategy['type_idx']]
                        if len(possible_types) > 0:
                            new_type = np.random.choice(possible_types)
                            fake.loc[idx, 'type_idx'] = new_type
    
                else:
                    strategy_type = strategy['type_idx']
                    for col in self.param_columns:
                        # 检查值是否为-999（缺失值）
                        if strategy[col] != -999 and np.random.random() < 0.5:
                            param_type = param_types[strategy_type][col]
                            if param_type == int:
                                # 对于整数，我们使用整数倍数
                                multiplier = np.random.choice([0.7, 0.8, 1.2, 1.3])
                                fake.loc[idx, col] = int(strategy[col] * multiplier)
                            else:
                                # 对于浮点数，直接使用浮点数乘法
                                fake.loc[idx, col] *= np.random.choice([0.7, 0.8, 1.2, 1.3])

            fake['is_fake'] = True
            fake_strategies.append(fake)
            # 如果 test_method 为 True，进行对比
            if test_method:
                comparison_result = fake.compare(fake_copy)
                if comparison_result.empty:
                    print(f"No changes were made to the fake DataFrame for date {date}.")
                else:
                    #print("Changes detected in the fake DataFrame for date")
                    pass
        # 合并所有生成的 fake 数据
        self.fake_strategies = pd.concat(fake_strategies)
        self.fake_strategies = self.fake_strategies.fillna(-999)
        print('Fake strategies created successfully')
      
    def mix_strategies(self):
        date = self.dates[self.date_index]
        cur_fake = self.fake_strategies[self.fake_strategies['date'] == date]
        fake_len = len(cur_fake)
        real = self.strategies_data[self.strategies_data['date'] == date].sample(n=min(100, len(self.strategies_data[self.strategies_data['date'] == date])), replace=False)
        fake = cur_fake
        real = self.strategies_data[self.strategies_data['date'] == date].sample(n=(self.max_strategies-fake_len), replace=False)
        mixed = pd.concat([real, fake]).sample(frac=1).reset_index(drop=True)
        return mixed

    def preprocess_strategies_data(self):
        # 创建 DataFrame 的明确副本
        self.strategies_data = self.strategies_data.copy()
        # 确保所有的缺失值都被统一处理，同时不会影响后续的类别映射和其他处理步骤。
        self.strategies_data = self.strategies_data.fillna(-999)
        self.param_columns = [col for col in self.strategies_data.columns if col.startswith('param_')]    
        # 创建类别映射
        categories = {
            'time': sorted(self.strategies_data['time'].unique()),
            'target': sorted(self.strategies_data['target'].unique()),
            'type': sorted(self.strategies_data['type'].unique())
        }
        # 应用类别映射
        for category, values in categories.items():
            idx_name = f'{category}_idx'
            self.strategies_data.loc[:, idx_name] = self.strategies_data[category].map({v: i for i, v in enumerate(values)})
        # 如果 'is_fake' 列不存在，添加它
        if 'is_fake' not in self.strategies_data.columns:
            self.strategies_data.loc[:, 'is_fake'] = False    
        # 确保列的顺序一致
        columns_order = ['date', 'return', 'is_fake'] + sorted(self.param_columns) + ['time_idx', 'target_idx', 'type_idx']
        # 检查所有必需的列是否存在
        missing_columns = set(columns_order) - set(self.strategies_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in strategies_data: {missing_columns}")
        # 使用方括号选择列
        self.strategies_data = self.strategies_data[columns_order]
    
    def get_last_info(self):
        return self.last_info if hasattr(self, 'last_info') else {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
            
        self.date_index = 0
        self.current_date = self.dates[self.date_index]
        self.current_observation = None
        
        if self.is_training and (self.episode_count % self.episodes_per_epoch == 20):
            self.generate_fake_strategies() # 每隔self.episodes_per_epoch轮次重新生成虚假策略
        if self.is_training:
            self.mixed_strategies = self.mix_strategies()  # 包含真假策略
        else:
            self.mixed_strategies = self.get_real_strategies(prediction_index=0)  # 只包含真实策略
            
        observation = self._next_observation()
        self.last_observation = observation
        if not isinstance(observation, dict) or 'market' not in observation or 'strategies' not in observation:
            raise ValueError(f"Invalid observation returned by reset: {observation}")
            
        info = {}  
        self.episode_step = 0
        self.episode_true_labels = []
        self.episode_predicted_labels = []
        self.episode_portfolio_returns = []
        # 这样在model.predict(obs, deterministic=True)的时候先返回index为0的策略的分类
        # 然后进行step，step再返回下一个index的obs
        return observation, info
    
    def get_real_strategies(self, prediction_index):
        date = self.dates[self.date_index]
        all_strategies = self.strategies_data[self.strategies_data['date'] == date]
        # 获取指定的策略
        predicted_strategy = all_strategies.iloc[prediction_index]
        # 获取其他策略
        other_strategies = all_strategies[all_strategies.index != predicted_strategy.name]
        # 将指定的策略放在第一位，然后是其他策略
        real = pd.concat([predicted_strategy.to_frame().T, other_strategies])
        
        return real
    
    def set_state(self, obs):
        # This method allows the callback to set a perturbed state
        self.current_observation = obs
        
    def _get_market_obs(self):
        end_date = self.current_date
        start_date = end_date - pd.Timedelta(days=29)
        market_data = self.market_data.loc[start_date:end_date]
        #print(f"Market data shape before processing: {market_data.shape}")
        # 确保我们有30天的数据，如果不足则用0填充
        if len(market_data) < 30:
            padding = pd.DataFrame(0, index=pd.date_range(start=start_date, end=end_date, freq='D'), columns=market_data.columns)
            market_data = pd.concat([padding, market_data]).fillna(0)
            market_data = market_data.iloc[-30:]  # 确保只有最后30天
        #print(f"Market data shape after processing: {market_data.shape}")
        market_obs = market_data.values.flatten()
        # 确保与初始化的市场obs一致
        return np.array(market_obs, dtype=np.float32)
                
    def _next_observation(self):
        # if self.current_observation is not None:
        #     obs = self.current_observation
        #     self.current_observation = None
        #     return obs
        # 获取混合策略数据
        strategy_obs = []
        # 在这里我们需要做一个对比实验，一个提供target策略的信息和former策略的信息；另一个提供全量策略的信息和former策略的信息。理论上第一个在样本外会好一点，因为训练时的全量策略信息包含了虚假策略的信息而预测时不包含，可能会导致样本外的性能的降低。
        # for i, (_, strategy) in enumerate(self.mixed_strategies.iterrows()):
        #     strategy_features = [
        #         strategy['time_idx'],
        #         strategy['target_idx'],
        #         strategy['type_idx'],
        #         *[strategy[col] for col in self.param_columns],
        #         1  # 标记为目标策略
        #     ]
        #     strategy_obs.append(strategy_features)
        cur_strategy = self.mixed_strategies.iloc[0]
        self.true_label = self._get_true_label(cur_strategy)
        current_strategy_obs = [
                cur_strategy['time_idx'],
                cur_strategy['target_idx'],
                cur_strategy['type_idx'],
                *[cur_strategy[col] for col in self.param_columns],
                1,  # 标记为目标策略
                1   # 标记为当前策略（而不是历史策略）
                                ]
        strategy_obs = [current_strategy_obs]
        # 添加历史top performers
        former_date = self.current_date - pd.Timedelta(days=7)
        former_strategies = self.strategies_data[self.strategies_data['date'] == former_date]
        top_ranking = former_strategies.nlargest(5, 'return')
        bottom_ranking = former_strategies.nsmallest(5, 'return')
        former_ranking = pd.concat([top_ranking, bottom_ranking])
        # 按 return 列从高到低排序
        former_ranking_sorted = former_ranking.sort_values(by='return', ascending=False)
        if len(former_ranking_sorted) != 10:
            print(f"something wrong about former_ranking_sorted, the len of bottom is {len(bottom_ranking)}, the len of top is {len(top_ranking)}.")
            print(f"Current date: {self.current_date}")
            print(f"Former date: {former_date}")
            print(f"Unique dates in strategies_data: {self.strategies_data['date'].unique()}")
            print(f"Is former_date in strategies_data: {former_date in self.strategies_data['date'].values}")
            raise ValueError(f"something wrong about former_ranking_sorted, the len of bottom is {len(bottom_ranking)}, the len of top is {len(top_ranking)}.")
        for _, strategy in former_ranking_sorted.iterrows():
            strategy_features = [
                strategy['time_idx'],
                strategy['target_idx'],
                strategy['type_idx'],
                *[strategy[col] for col in self.param_columns],
                strategy['return'],
                0,  # 不是目标策略
                0   # 标记为历史策略
            ]
            strategy_obs.append(strategy_features)
        # 以下代码是为了保证strategy_obs里的每一个元素的长度一致
        max_dim = max(len(obs) for obs in strategy_obs)
        padded_strategy_obs = []
        for obs in strategy_obs:
            padded_obs = obs + [0] * (max_dim - len(obs))  # 用0填充
            padded_strategy_obs.append(padded_obs)
        strategy_obs = np.array(padded_strategy_obs, dtype=np.float32)
        # 如果策略数量少于11，用零向量填充
        while len(strategy_obs) < self.total_strategies:
            strategy_obs = np.vstack([strategy_obs, np.zeros_like(strategy_obs[0])])
        #print(f"the current_strategy in obs for {self.current_date}: {strategy_obs[0]} ")
        # 获取市场数据
        market_obs = self._get_market_obs()
        return {
            'market': np.array(market_obs, dtype=np.float32),
            'strategies': np.array(strategy_obs, dtype=np.float32)
        }
    
    def get_last_observation(self):
        return self.last_observation

    def step(self, action: int, prediction_index = None) -> Tuple[Dict, float, bool, bool, Dict]:
        if not self.is_training and prediction_index is None:
            raise ValueError("prediction_index must be provided in prediction mode")
            
        info = {
            'f1_scores': [],
            'portfolio_returns': [],
            'true_labels': [],
            'predicted_labels': []
        }
        # 初始化 done 为 False
        done = False
        # 在获取current_strategy之前不应该调用self.mix_strategies()或者self.get_real_strategies()函数，否则会与reset后第一个step相冲突
        current_strategy = self.mixed_strategies.iloc[0]
        #print(f"the current_strategy in step for {self.current_date}: {current_strategy} ")
        true_label = self.true_label
        portfolio_return = current_strategy['return'] if action == 1 else 0
        
        self.episode_true_labels.append(true_label)
        self.episode_predicted_labels.append(action)
        self.episode_portfolio_returns.append(portfolio_return)
        self.episode_step += 1
        
        if self.is_training:
            self.date_index += 1
            if self.date_index < len(self.dates):
                self.current_date = self.dates[self.date_index]
            else:
                done = True
        else:
            done = False
    
        if (self.date_index >= len(self.dates) - 1) or (self.episode_step >= self.weeks):
            done = True
    
        # reward = self._calculate_reward(done)
        reward = 1 if action == true_label else 0
        #print(reward)
        self.true_label = None
        # self.mix_strategies()到底在step阶段调用还是在step和reset两个阶段都调用，需要结合train和predict两个阶段来判断
        if self.is_training:
            self.mixed_strategies = self.mix_strategies()
        else:
            if prediction_index + 1 < len(self.mixed_strategies):
                self.mixed_strategies = self.get_real_strategies(prediction_index=prediction_index+1)
            else:
                done = True
            # 注意这里用的是prediction_index+1，因为要为next_obs进行调用
        next_obs = self._next_observation()
        info = self._get_info(true_label, action, portfolio_return, done)
        if not isinstance(next_obs, dict) or 'market' not in next_obs or 'strategies' not in next_obs:
            raise ValueError(f"Invalid observation returned by step: {next_obs}")
        return next_obs, reward, done, False, info
    
    def _get_true_label(self, strategy):
        return 0 if strategy['is_fake'] else (
            1 if strategy['return'] > 0 else (
            2 if strategy['return'] < 0 else 3))
    
    def _calculate_reward(self, done):
        # if done:
        f1_reward = self.calculate_f1_score(np.array(self.episode_predicted_labels), np.array(self.episode_true_labels))
        portfolio_reward = np.sum(self.episode_portfolio_returns)
        # print('the action for labels')
        # print(self.episode_predicted_labels)
        # print('the true labels')
        # print(self.episode_true_labels)
        # print(f"the f1 score is {f1_reward}")
        #print(f"f1 reward: {f1_reward}")
        return f1_reward + (0 * portfolio_reward)  # 调整权重
        # else:
        #     return 0   
    
    def _get_info(self, true_label, action, portfolio_return, done):
        info = {
            'true_label': true_label,
            'predicted_label': action,
            'portfolio_return': portfolio_return,
            'current_date': self.current_date,
            'episode_step': self.episode_step
        }
        if done:
            info['f1_score'] = f1_score(self.episode_true_labels, self.episode_predicted_labels, average='macro')
            info['total_portfolio_return'] = np.sum(self.episode_portfolio_returns)
        return info
    
    def _calculate_strategy_scores(self, time_weights: np.ndarray, target_weights: np.ndarray, type_weights: np.ndarray) -> pd.DataFrame:
        # Existing strategy score calculation logic...
        idx = pd.IndexSlice
        current_strategies = self.strategies_data.loc[idx[self.current_date, :, :, :]]

        scores = (
            time_weights[current_strategies.index.get_level_values('time_idx')] + 
            target_weights[current_strategies.index.get_level_values('target_idx')] + 
            type_weights[current_strategies.index.get_level_values('type_idx')]
        )
        
        return current_strategies[scores > 0]
    
    def _calculate_portfolio_return(self, selected_strategies: pd.DataFrame) -> float:
        # Existing portfolio return calculation logic...
        if len(selected_strategies) > 0:
            reward = selected_strategies['return'].mean()
        else:
            reward = 0
        return reward
    
    def calculate_f1_score(self, predictions, true_labels):
        '''
        处理稀有类别更健壮: 通过使用 LabelBinarizer，新的实现将每个类别转换为二进制格式，这样可以确保所有类别在计算 F1 分数时都被考虑，即使某些类别没有出现在 predictions 或 true_labels 中。zero_division=1 参数还可以避免除零错误，返回一个默认值（在这里是1）。
        支持多标签: 新的实现可以更好地支持多标签分类问题，因为它将每个类别都二值化处理，可以分别计算每个类别的 F1 分数，并通过 average='macro' 取平均。
        稳定性提升: 通过 binarization，即使某些类别在预测或真实标签中没有出现，f1_score 函数依然可以正常工作，避免潜在的异常。
        '''
        lb = LabelBinarizer()
        lb.fit(range(4))  # 假设有 4 个类别
        true_labels_bin = lb.transform(true_labels)
        predictions_bin = lb.transform(predictions)
        print(f"true_labels_bin: {true_labels_bin}")
        print(f"predictions_bin: {predictions_bin}")
        return f1_score(true_labels_bin, predictions_bin, average='macro', zero_division=1)

def load_market_data(start_date: str, end_date: str, file) -> pd.DataFrame:
    # 注意：这里的end_date应该是训练结束日期的前一天
    end_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    end_date, start_date = pd.to_datetime(end_date), pd.to_datetime(start_date)
    data = file
    data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    # 将日期列设置为索引
    data.set_index('date', inplace=True)
    # 确保数据是按日期排序的
    data.sort_index(inplace=True)
    # 选择指定日期范围内的数据
    mask = (data.index >= start_date) & (data.index < end_date)
    new_data = data.loc[mask]
    
    return new_data


# class CustomCallback(BaseCallback):
#     # 添加了一个 CustomCallback 类，用于在训练过程中监控 KL 散度（针对 FatTailedVAE）
#     # 现在，当使用 FatTailedVAE 时：
#     # 监控训练过程中的 KL 散度变化。
#     # 查看最终学习到的自由度参数，这反映了模型认为数据的"肥尾"程度。
#     # 生成新的样本，这可能对于数据增强或进一步的分析很有用。
#     def __init__(self, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         self.kl_divergences = []

#     def _on_step(self) -> bool:
#         # 这是 Stable-Baselines3 回调函数中的一个关键方法，会在每个训练步骤后被调用。
#         # 假设我们可以访问最后一批观察数据
#         last_obs = self.training_env.get_attr('last_observation')
#         # 通过模型的特征提取器对最后的观察数据进行编码，生成均值 mu 和对数方差 logvar。
#         # If we're using multiple environments, we need to process each one
#         for obs in last_obs:
#             if obs is not None:
#                 mu, logvar = self.model.policy.features_extractor.encode(obs)
#                 kl_div = self.model.policy.features_extractor.kl_divergence(mu, logvar).mean().item()
#                 self.kl_divergences.append(kl_div)
#         return True

# def custom_eval_function(model, env, n_episodes=5):
#     all_episode_rewards = []
#     all_true_labels = []
#     all_predicted_labels = []
#     all_portfolio_returns = []

#     for _ in range(n_episodes):
#         obs, info = env.reset()
#         print("Reset observation type:", type(obs))
#         print("Reset observation content:", obs)
#         done = False
#         episode_reward = 0
#         episode_true_labels = []
#         episode_predicted_labels = []
#         episode_portfolio_returns = []

#         while not done:
#             try:
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, reward, done, _, info = env.step(action)
#             except Exception as e:
#                 print("Error in predict or step:")
#                 print("Observation type:", type(obs))
#                 print("Observation content:", obs)
#                 print('market')
#                 print("Observation type:", type(obs['market']))
#                 print("Observation content:", obs['market'])
#                 print('strategies')
#                 print("Observation type:", type(obs['strategies']))
#                 print("Observation content:", obs['strategies'])
#                 raise e
#             episode_reward += reward
#             episode_true_labels.append(info['true_label'])
#             episode_predicted_labels.append(info['predicted_label'])
#             episode_portfolio_returns.append(info['portfolio_return'])

#         all_episode_rewards.append(episode_reward)
#         all_true_labels.extend(episode_true_labels)
#         all_predicted_labels.extend(episode_predicted_labels)
#         all_portfolio_returns.extend(episode_portfolio_returns)
#     # 计算平均奖励
#     mean_reward = np.mean(all_episode_rewards)
#     # 计算整体 F1 分数
#     overall_f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
#     # 计算每个类别的 F1 分数
#     class_f1_scores = f1_score(all_true_labels, all_predicted_labels, average=None)
#     # 计算总体投资组合回报
#     total_portfolio_return = np.sum(all_portfolio_returns)
#     # 计算夏普比率（假设无风险利率为0）
#     returns = np.array(all_portfolio_returns)
#     sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

#     results = {
#         'mean_reward': mean_reward,
#         'overall_f1': overall_f1,
#         'total_portfolio_return': total_portfolio_return,
#         'sharpe_ratio': sharpe_ratio
#     }
    
#     for i, score in enumerate(class_f1_scores):
#         results[f'f1_score_class_{i}'] = score

#     return results

def custom_eval_function(model, eval_env, n_eval_episodes=50):
    print(f"Eval env type: {type(eval_env)}")
    # 使用 SB3 的 evaluate_policy 函数
    try:
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
        print(f"Evaluate policy results: mean_reward={mean_reward}, std_reward={std_reward}")
    except Exception as e:
        print(f"Error in evaluate_policy: {e}")
        mean_reward, std_reward = 0, 0
    # 收集额外的指标
    all_true_labels = []
    all_predicted_labels = []
    all_portfolio_returns = []

    for episode in range(n_eval_episodes):
        reset_result = eval_env.reset()
        if isinstance(reset_result, tuple):
            print("the output of reset is a tuple")
            obs = reset_result[0]
            info = reset_result[1] if len(reset_result) > 1 else {}
        else:
            obs = reset_result
            info = {}
        # print(f"Initial observation type: {type(obs)}")
        # if isinstance(obs, dict):
        #     for key, value in obs.items():
        #         print(f"  {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
        # else:
        #     print(f"Observation content: {obs}")
        
        done = False
        episode_step = 0
        while not done:
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                if isinstance(eval_env, VecEnv):
                    done = done[0]
                    info = info[0]
                
                all_true_labels.append(info.get('true_label', None))
                all_predicted_labels.append(info.get('predicted_label', None))
                all_portfolio_returns.append(info.get('portfolio_return', 0))
                
                episode_step += 1
                # if episode_step % 10 == 0:  # 每10步打印一次
                #     print(f"Episode {episode + 1}, Step {episode_step}")
            except Exception as e:
                print(f"Error in episode {episode + 1}, step {episode_step}:")
                print(f"Observation type: {type(obs)}")
                print(f"Observation content: {obs}")
                print(f"Error: {e}")
                break

    # 计算其他指标
    overall_f1 = f1_score(all_true_labels, all_predicted_labels, average='macro')
    class_f1_scores = f1_score(all_true_labels, all_predicted_labels, average=None)
    total_portfolio_return = np.sum(all_portfolio_returns)
    returns = np.array(all_portfolio_returns)
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'overall_f1': overall_f1,
        'total_portfolio_return': total_portfolio_return,
        'sharpe_ratio': sharpe_ratio,
        **{f'f1_score_class_{i}': score for i, score in enumerate(class_f1_scores)}
    }

class CustomEvalCallback(EvalCallback):
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, **kwargs):
        super().__init__(eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, **kwargs)
        self.no_improvement_count = 0
        
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 使用自定义评估函数
            eval_env = self.model.get_env()
            eval_results = custom_eval_function(self.model, eval_env, n_eval_episodes=50)
            
            # 记录评估指标
            for metric, value in eval_results.items():
                self.logger.record(f"eval/{metric}", value)
            
            # 检查是否有改进
            if eval_results['mean_reward'] > self.best_mean_reward:
                self.best_mean_reward = eval_results['mean_reward']
                self.no_improvement_count = 0  # 有改进时重置计数器
                if self.verbose > 0:
                    print(f"New best mean reward: {self.best_mean_reward:.2f}")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            else:
                self.no_improvement_count += 1  # 没有改进时递增计数器
            
            # 学习率调整逻辑
            min_learning_rate = 1e-6
            if self.no_improvement_count >= 2:
                new_lr = max(self.model.learning_rate * 0.7, min_learning_rate)
                self.model.learning_rate = new_lr
                print(f"Reducing learning rate to {new_lr}")
                self.logger.record("train/learning_rate", new_lr)
            
            # 早期停止逻辑
            if self.no_improvement_count >= 20:
                print("Stopping early due to no improvement")
                return False

        return True


class OptimizedTrainingCallback(BaseCallback):
    def __init__(self, eval_freq=1000, verbose=0, tensorboard_callback=None):
        # eval_freq应该与模型n_steps一致，确保每次梯度更新后做一次评估
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_metrics = {
            'mean_reward': -np.inf,
            'overall_f1': -np.inf,
            'total_portfolio_return': -np.inf,
            'sharpe_ratio': -np.inf
        }
        self.no_improvement_count = 0
        self.tensorboard_callback = tensorboard_callback

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            eval_results = custom_eval_function(self.model, self.model.get_env())
            # 记录评估指标
            for metric, value in eval_results.items():
                self.logger.record(f"eval/{metric}", value)
            
            if self.tensorboard_callback:
                self.tensorboard_callback.update_eval_metrics(eval_results, self.n_calls)
            # 检查多个指标的改进情况
            improved = False
            #for metric in ['mean_reward', 'overall_f1', 'total_portfolio_return', 'sharpe_ratio']:
            for metric in ['mean_reward', 'overall_f1']:
                if eval_results[metric] > self.best_metrics[metric]:
                    self.best_metrics[metric] = eval_results[metric]
                    improved = True     
            # 更新 TensorBoard
            if self.tensorboard_callback:
                self.tensorboard_callback.update_eval_metrics(eval_results, self.n_calls)

            if improved:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            # 早期停止和学习率调整逻辑
            if self.no_improvement_count >= 10:
                print("Stopping early due to no improvement")
                return False
            # 引入最小学习率，防止学习率过小
            min_learning_rate = 1e-6
            # 只有在模型表现未改进时，才减少学习率
            if self.no_improvement_count >= 2:
                new_lr = max(self.model.learning_rate * 0.5, min_learning_rate)
                self.model.learning_rate = new_lr
                print(f"Reducing learning rate to {new_lr}")
                self.logger.record("train/learning_rate", new_lr)

        self.logger.dump(self.n_calls)
        return True

class CheckpointCallback(BaseCallback):
    # 定期保存模型检查点，便于恢复训练
    def __init__(self, save_freq=100000, save_path='./checkpoints', name_prefix='rl_model'):
        super(CheckpointCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f'{self.name_prefix}_{self.n_calls}_steps.zip')
            self.model.save(path)
        return True
    
class OptimizedTensorboardCallback(BaseCallback):
    # (base) PS C:\Users\Administrator.DESKTOP-4H80TP4\RBT> tensorboard --logdir=tensorboard_logs
    def __init__(self, log_dir: str, verbose: int = 0, log_freq: int = 1000):
        super(OptimizedTensorboardCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.kl_divergences = []
        self.portfolio_returns = []
        self.selected_strategies_counts = []
        self.f1_scores: Dict[str, list] = {f'class_{i}': [] for i in range(4)}
        self.f1_scores['overall'] = []
        self.log_freq = log_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_freq == 0:
            self._log_step_info()
        last_info = self._get_last_info()
        
        self.portfolio_returns.append(last_info.get('portfolio_return', 0))
        
        self.writer.add_scalar("environment/portfolio_return", self.portfolio_returns[-1], self.num_timesteps)

        return True
    
    def _get_last_info(self):
        if hasattr(self.training_env, 'envs'):
            # For vectorized environments
            return self.training_env.env_method('get_last_info')[0]
        elif hasattr(self.training_env, 'get_last_info'):
            # For non-vectorized environments with get_last_info method
            return self.training_env.get_last_info()
        elif hasattr(self.training_env, 'unwrapped'):
            # For wrapped environments
            return self.training_env.unwrapped.last_info
        else:
            # Fallback
            return {}

    def _log_step_info(self) -> None:
        try:
            self._log_learning_rate()
            self._log_episode_info()
            self._log_losses()
            self._log_gradient_norm()
            self._log_environment_info()
            self._log_kl_divergence()
        except Exception as e:
            print(f"Error in logging step info: {e}")

    def _log_learning_rate(self) -> None:
        if hasattr(self.model, 'learning_rate'):
            self.writer.add_scalar("training/learning_rate", self.model.learning_rate, self.num_timesteps)

    def _log_episode_info(self) -> None:
        if self.model.ep_info_buffer and len(self.model.ep_info_buffer) > 0:
            ep_reward = self.model.ep_info_buffer[-1]['r']
            self.writer.add_scalar("training/episode_reward", ep_reward, self.num_timesteps)
            self.episode_rewards.append(ep_reward)
            self.episode_count += 1

    def _log_losses(self) -> None:
        if hasattr(self.model, 'logger'):
            for loss_name in ['pg_loss', 'value_loss', 'entropy_loss', 'kl_div']:
                if loss_name in self.model.logger.name_to_value:
                    loss_value = self.model.logger.name_to_value[loss_name]
                    self.writer.add_scalar(f"losses/{loss_name}", loss_value, self.num_timesteps)

    def _log_gradient_norm(self) -> None:
        if hasattr(self.model.policy, 'optimizer'):
            total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.model.policy.parameters() if p.grad is not None) ** 0.5
            self.writer.add_scalar("training/gradient_norm", total_norm, self.num_timesteps)

    def _log_environment_info(self) -> None:
        last_info = self._get_last_info()
        self.portfolio_returns.append(last_info.get('portfolio_return', 0))
        self.writer.add_scalar("environment/portfolio_return", self.portfolio_returns[-1], self.num_timesteps)

    def _log_kl_divergence(self) -> None:
        if hasattr(self.model.policy.features_extractor, 'kl_divergence'):
            self._record_kl_divergence()

    def _record_kl_divergence(self) -> None:
        try:
            last_obs = self.training_env.get_attr('last_observation')
            for obs in last_obs:
                if obs is not None:
                    mu, logvar = self.model.policy.features_extractor.encode(obs)
                    kl_div = self.model.policy.features_extractor.kl_divergence(mu, logvar).mean().item()
                    self.kl_divergences.append(kl_div)
                    self.writer.add_scalar("metrics/kl_divergence", kl_div, self.num_timesteps)
        except Exception as e:
            print(f"Error in recording KL divergence: {e}")

    def update_eval_metrics(self, eval_results, n_calls):
        for metric, value in eval_results.items():
            self.writer.add_scalar(f"eval/{metric}", value, n_calls)

    def _update_f1_moving_averages(self, n_calls: int) -> None:
        window_size = 10  # 使用最近10个值计算移动平均
        for class_key, scores in self.f1_scores.items():
            if len(scores) >= window_size:
                moving_avg = sum(scores[-window_size:]) / window_size
                self.writer.add_scalar(f"eval/f1_score_moving_avg_{class_key}", moving_avg, n_calls)

    def on_training_end(self) -> None:
        self.writer.close()
        

class LossGradientCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=1):
        super(LossGradientCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.losses = {'policy_loss': [], 'value_loss': [], 'entropy_loss': []}
        self.gradient_norms = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            print(f"\nStep {self.n_calls}")
            
            # 从logger获取损失值
            if self.model.logger:
                for loss_name in ['pg_loss', 'value_loss', 'entropy_loss']:
                    if loss_name in self.model.logger.name_to_value:
                        loss_value = self.model.logger.name_to_value[loss_name]
                        self.losses[loss_name].append(loss_value)
                        print(f"Current {loss_name}: {loss_value:.4f}")
                    else:
                        print(f"No data for {loss_name}")
            else:
                print("Model logger not available")

            # 计算梯度范数
            total_norm = 0
            for param in self.model.policy.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = np.sqrt(total_norm)
            self.gradient_norms.append(total_norm)
            print(f"Current gradient norm: {total_norm:.4f}")

            # # 获取各个损失值
            # pg_loss = self.model.logger.name_to_value.get('train/policy_gradient_loss', None)
            # value_loss = self.model.logger.name_to_value.get('train/value_loss', None)
            # entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', None)
            # approx_kl = self.model.logger.name_to_value.get('train/approx_kl', None)
            # clip_fraction = self.model.logger.name_to_value.get('train/clip_fraction', None)
            # print("below data are get from self.model.logger.name_to_value")
            # print(f"Step: {self.n_calls}")
            # print(f"Policy Gradient Loss: {pg_loss}")
            # print(f"Value Loss: {value_loss}")
            # print(f"Entropy Loss: {entropy_loss}")
            # print(f"Approximate KL: {approx_kl}")
            # print(f"Clip Fraction: {clip_fraction}")
            # print("---")

        return True

    def on_training_end(self):
        print("\nTraining ended. Final statistics from self.losses:")
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                print(f"Average {loss_name}: {np.mean(loss_values):.4f}")
            else:
                print(f"No data collected for {loss_name}")
        if self.gradient_norms:
            print(f"Average gradient norm: {np.mean(self.gradient_norms):.4f}")
        else:
            print("No gradient norm data collected during training")

def train_model(strategies_data: pd.DataFrame, market_data: pd.DataFrame, end_date: str, weeks: int, 
                feature_extractor: str = 'cnn', adv_param: list = [0,0]):
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(weeks=weeks) - timedelta(days=30)).strftime('%Y-%m-%d')
    #market_data = load_market_data(start_date=start_date, end_date=end_date, file=market_data)
    market_data = load_market_data(start_date=start_date, end_date=end_date, file=adjust_market_data)
    # 使用 SubprocVecEnv 来并行化环境，提高训练效率。
    def make_env(rank, seed=0):
        def _init():
            env = FinancialTradingEnv(end_date=end_date, weeks=weeks, strategies_data=strategies_data, market_data=market_data)
            env = Monitor(env)
            env.seed(seed + rank)
            return env
        return _init
    # SubprocVecEnv会导致无法正常print
    #env = DummyVecEnv([lambda: env])
    # env = DummyVecEnv([make_env(i) for i in range(3)]) 
    env = SubprocVecEnv([make_env(i) for i in range(3)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=100.)
    env = VecMonitor(env)
    # 选择特征提取器
    if feature_extractor == 'cnn':
        features_extractor_class = CNNFeaturesExtractor
    elif feature_extractor == 'autoencoder':
        features_extractor_class = AutoEncoderFeaturesExtractor
    elif feature_extractor == 'vae':
        features_extractor_class = VAEFeaturesExtractor
    elif feature_extractor == 'fat_tailed_vae':
        features_extractor_class = VAEFeaturesExtractor
    else:
        raise ValueError("Invalid feature extractor. Choose 'cnn', 'autoencoder', 'vae', or 'fat_tailed_vae'.")

    # pi: 策略网络（Policy Network）决定采取什么行动。
    # vf: 价值网络（Value Function Network）评估这些行动的好坏，帮助策略网络改进。
    # 为策略网络和价值网络分别设置了两个隐藏层，每层128个节点。在任务中（处理150个策略的4类预测），这种结构应该能够提供足够的容量来学习复杂的策略选择模式。根据任务的具体复杂性，可能需要调整这些数值。例如，如果发现模型欠拟合，可以考虑增加层数或每层的节点数；如果出现过拟合，则可能需要减少网络复杂度或引入正则化技术。
    # 为 VAE 和 fat-tailed VAE 设置额外的参数
    features_extractor_kwargs = dict(features_dim=128)
    if feature_extractor in ['vae', 'fat_tailed_vae']:
        features_extractor_kwargs['use_fat_tailed'] = (feature_extractor == 'fat_tailed_vae')
    
    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=features_extractor_kwargs,
       net_arch=dict(pi=[1024,512,256,128,64], vf=[1024,512,256,128,64])  # Remove the outer list
    )

    # 创建PPO模型，使用多离散动作空间
    log_dir = "./tensorboard_logs"
    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func
    #model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device='cuda')
    # n_steps: 如果你的环境每次最多运行52步（52周），那么n_steps应该设置为52或更小。设置为52意味着每次更新前会收集整个周期的数据。
    # batch_size: 通常设置为n_steps的一个较小的因子。例如，你可以尝试16或32。
    # n_epochs:默认值通常是10。对于你的问题，可以从5开始尝试，然后根据性能调整。
    # ent_coef=0.01,
    n_steps = 52000
    learning_rate = 0.01
    batch_size = n_steps // 4
    n_epochs = 5
    gamma = 0.99
    vf_coef = 0.5
    ent_coef = 0.01
    gae_lambda = 0.90
    model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device='cuda', tensorboard_log=log_dir,n_steps = n_steps, batch_size = batch_size, n_epochs = n_epochs, learning_rate=learning_rate, gamma=gamma,vf_coef=vf_coef,ent_coef=ent_coef,gae_lambda=gae_lambda) 
    # 创建回调
    adversarial_callback = AdversarialTrainingCallback(adversarial_prob=adv_param[0], adversarial_std=adv_param[1])
    # eval_callback = EvalCallback(env, eval_freq=1000, n_eval_episodes=5, 
    #                              eval_fn=custom_eval_function,
    #                              best_model_save_path='./logs/',
    #                              log_path='./logs/', 
    #                              deterministic=True, render=False)
    # 配置 TensorBoard 日志
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    # 创建各类回调函数
    # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./model_checkpoints/')
    eval_callback = EvalCallback(env, best_model_save_path='./best_model/',
                                 log_path='./logs/', eval_freq=n_steps,
                                 deterministic=True, render=False)
    # 使用方法
    Custom_callback = CustomEvalCallback(
        eval_env=env,
        eval_freq=n_steps,
        n_eval_episodes=5,
        log_path="./logs/",
        best_model_save_path="./best_model/",
        deterministic=True,
        render=False
    )
    #loss_callback = LossGradientCallback(verbose=1, log_freq=100)
    #custom_callback = CustomCallback()
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    tb_log_dir = f"./tensorboard_logs/{end_date}_{weeks}weeks_{feature_extractor}_{start_time}"
    tb_callback = OptimizedTensorboardCallback(log_dir=tb_log_dir,log_freq=n_steps)
    optimized_callback = OptimizedTrainingCallback(eval_freq=n_steps, tensorboard_callback=tb_callback)
    # 组合回调
    if 'vae' in feature_extractor.lower():
        print(f"Using VAE model: {feature_extractor}")
        callbacks = [Custom_callback, eval_callback]
    else:
        print(f"Using non-VAE model: {feature_extractor}")
        callbacks = [Custom_callback, eval_callback]
# adversarial_callback,tb_callback,custom_callback,checkpoint_callback,optimized_callback，Custom_callback
    def check_data(env):
        result = env.reset()
        print("Checking initial observation:")
        
        obs, info = result if isinstance(result, tuple) else (result, None)
        
        for key, value in obs.items():
            print(f"\nChecking {key}:")
            print(f"Shape: {value.shape}")
            
            # 检查 NaN 值
            nan_mask = np.isnan(value)
            nan_count = np.sum(nan_mask)
            print(f"Contains NaN: {nan_count > 0}")
            print(f"Total NaN values: {nan_count}")
            
            if nan_count > 0:
                if key == 'strategies':
                    nan_strategies = np.any(nan_mask, axis=1)
                    print(f"Number of strategies with NaN values: {np.sum(nan_strategies)}")
                    
                    for i, strategy in enumerate(value):
                        if np.any(np.isnan(strategy)):
                            nan_features = np.where(np.isnan(strategy))[0]
                            print(f"Strategy {i} has NaN values in features: {nan_features}")
                else:
                    nan_indices = np.where(nan_mask)[0]
                    print(f"Indices of NaN values: {nan_indices}")
            
            # 检查 inf 值
            inf_mask = np.isinf(value)
            inf_count = np.sum(inf_mask)
            print(f"Contains inf: {inf_count > 0}")
            print(f"Total inf values: {inf_count}")
            
            if inf_count > 0:
                if key == 'strategies':
                    inf_strategies = np.any(inf_mask, axis=1)
                    print(f"Number of strategies with inf values: {np.sum(inf_strategies)}")
                    
                    for i, strategy in enumerate(value):
                        if np.any(np.isinf(strategy)):
                            inf_features = np.where(np.isinf(strategy))[0]
                            print(f"Strategy {i} has inf values in features: {inf_features}")
                else:
                    inf_indices = np.where(inf_mask)[0]
                    print(f"Indices of inf values: {inf_indices}")
            
            # 基本统计信息
            print(f"Min: {np.min(value)}, Max: {np.max(value)}")
            print(f"Mean: {np.mean(value)}, Std: {np.std(value)}")
            
            # 检查异常值 (使用 3 个标准差作为阈值)
            mean = np.mean(value)
            std = np.std(value)
            outliers_mask = np.abs(value - mean) > 3 * std
            outliers_count = np.sum(outliers_mask)
            print(f"Number of potential outliers (beyond 3 std): {outliers_count}")
            
            if outliers_count > 0 and outliers_count < 10:  # 只打印少量异常值
                outliers = value[outliers_mask]
                print(f"Outlier values: {outliers}")
        
        if info is not None:
            print("\nInfo:", info)
    
    # 运行检查
    check_data(env)
    # 训练模型
    model.learn(total_timesteps=1400040, callback=callbacks)
    # 构建保存模型的文件名，包含关键信息
    model_filename = f"ppo_model_nsteps_{n_steps}_lr_{learning_rate}_batchsize_{batch_size}_nepochs_{n_epochs}_gamma_{gamma}_vfcoef_{vf_coef}_entcoef_{ent_coef}_gaelambda_{gae_lambda}.zip"
    # 保存模型
    model.save(model_filename)
    all_true_labels = []
    all_predicted_labels = []
    all_portfolio_returns = []
    env = model.get_env()
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if isinstance(env, VecEnv):
            done = done[0]
            info = info[0]
        all_true_labels.append(info['true_label'])
        all_predicted_labels.append(info['predicted_label'])
        all_portfolio_returns.append(info['portfolio_return'])
    
    for name, param in model.policy.named_parameters():
        if param.grad is not None:
            pass
            print(f"{name} has grad")
            #print(f"{name}: Gradient norm = {param.grad.norm().item()}")
        else:
            print(f"{name} no grad")
            #print(f"{param}")
    
    # Create directory to save the model
    os.makedirs('models', exist_ok=True)
    
    # Save the model, filename includes end date, number of weeks, and adversarial training parameters
    model_filename = f"models/ppo_model_{end_date}_{weeks}weeks_{feature_extractor}_adv_prob{adv_param[0]}_adv_std{adv_param[1]}.zip"

    model.save(model_filename)
    
    # Save the training parameters
    params = {
    "end_date": end_date,
    "weeks": weeks,
    "feature_extractor": feature_extractor,
    "adversarial_prob": adv_param[0],
    "adversarial_std": adv_param[1]
}
    params_filename = f"models/params_{end_date}_{weeks}weeks_{feature_extractor}_adv_prob{adv_param[0]}_adv_std{adv_param[1]}.json"
    with open(params_filename, 'w') as f:
        json.dump(params, f)
    
    print(f"Model saved as {model_filename}")
    print(f"Parameters saved as {params_filename}")
    
    # 如果使用的是 VAE 或 FatTailedVAE，打印相关统计信息
    if 'vae' in feature_extractor.lower():
        print(f"Average KL divergence: {np.mean(custom_callback.kl_divergences)}")
        if feature_extractor == 'fat_tailed_vae':
            print(f"Final degrees of freedom: {model.policy.features_extractor.get_df().item()}")
            # 生成一些样本
            samples = model.policy.features_extractor.sample(10)
            print("Generated samples:", samples)

    return model, env

def evaluate_model(model, strategies_data: pd.DataFrame, market_data: pd.DataFrame, 
                   train_end_date: str, n_steps: int = 3):
    train_end_date = datetime.strptime(train_end_date, '%Y%m%d')
    # 创建一个新的环境，从训练结束日期后的第一个星期日开始
    start_date = train_end_date + timedelta(7)
    env = FinancialTradingEnv(end_date=start_date.strftime('%Y-%m-%d'), 
                              weeks=n_steps, 
                              strategies_data=strategies_data, 
                              market_data=market_data)
    
    obs = env.reset()
    done = False
    total_reward = 0
    step_rewards = []
    step_dates = []
    
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # 计算预测为正收益的策略的总收益
        predicted_positive = action == 1  # 1 表示预测为正收益
        true_returns = env.mixed_strategies['return']
        positive_returns = true_returns[predicted_positive].sum()
        
        total_reward += positive_returns
        step_rewards.append(positive_returns)
        step_dates.append(info['current_date'])
        
        print(f"Step {step + 1}, Date: {info['current_date']}, Reward: {positive_returns}")
        
        if done:
            break
    
    print(f"Total reward over {n_steps} weeks: {total_reward}")
    
    return {
        'total_reward': total_reward,
        'step_rewards': step_rewards,
        'step_dates': step_dates
    }


if __name__ == "__main__":
    logger.info("Starting the RL trading program")
    market_data = pd.read_csv('adjusted_df.csv')  # 请替换为实际的数据加载方法
    market_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    date_columns = market_data.select_dtypes(include=['datetime64', 'object']).columns
    numeric_columns = market_data.select_dtypes(include=['number']).columns
    non_numeric_columns = market_data.columns.difference(numeric_columns)    
    
    # 读取JSON文件
    config_path = "config.json"  # 替换为你的JSON文件的路径
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    
    end_dates = ['2024-03-31']
    weeks_options = [52]
    adversarial_params = [[0,0],[0.1,0.01],[0.1,0.05]] #第一个值为0代表不启动
    
    json_file_path = r"C:\Users\Administrator.DESKTOP-4H80TP4\Downloads\2021_10_08_2023_05_28_opt (1)"

    feature_extractors = ['cnn', 'autoencoder', 'vae', 'fat_tailed_vae']
    strategies_data, wrong_list = extract_strategies_data(json_file_path)

    #window_size (int): 滚动窗口大小，默认为30天。
    #method (str): 标准化方法，
    # 'robust': 一种基于统计量的标准化方法，它通过去掉中位数并按四分位距（IQR）缩放来缩放数据。特别适用于存在离群值的数据 
    # 'log', 
    # 'quantile': 将数据按其分布映射到目标分布（例如标准正态分布或均匀分布）的方法。它通过计算每个数据点的百分位数，然后根据目标分布的累积分布函数（CDF）将其转换。
    # 'maxabs': 将数据缩放到[-1, 1]之间，以最大绝对值为基准进行缩放。保留了数据的稀疏性（适合处理稀疏矩阵）
    # process_type (int): 
        # 1 - log_return_columns 中的变量保留原值和对数收益率，其他列只保留对数收益率
        # 2 - 不进行对数收益率处理，保留原值
        # 3 - 只留对数收益率，去除原值
    methods = ['robust', 'log', 'quantile', 'maxabs']
    process_types = [1, 2, 3]
    window_sizes = [30,60]

    bool_columns = ['1_funding_rate_regime', '7_funding_rate_regime', '30_funding_rate_regime']
    abs_mean_larger10 = ['1_quote_volume', '1_taker_buy_quote_volume', '1_avg_leverage',
           '1_net_flow', '1_sum_open_interest_close',
           '1_sum_open_interest_value_close', '7_quote_volume',
           '7_taker_buy_quote_volume', '7_volatility_change', '7_avg_leverage',
           '7_oi_std', '7_net_flow', '7_sum_open_interest_close',
           '7_sum_open_interest_value_close', '30_quote_volume',
           '30_taker_buy_quote_volume', '30_volatility_change', '30_avg_leverage',
           '30_oi_std', '30_net_flow', '30_sum_open_interest_close',
           '30_sum_open_interest_value_close', 'open', 'high', 'low', 'close',
           'volume', 'ETH-USD_Open', 'ETH-USD_High', 'ETH-USD_Low',
           'ETH-USD_Close', 'ETH-USD_Adj Close', 'ETH-USD_Volume', 'JPY=X_Open',
           'JPY=X_High', 'JPY=X_Low', 'JPY=X_Close', 'JPY=X_Adj Close',
           '^GSPC_Open', '^GSPC_High', '^GSPC_Low', '^GSPC_Close',
           '^GSPC_Adj Close', '^GSPC_Volume', '^IXIC_Open', '^IXIC_High',
           '^IXIC_Low', '^IXIC_Close', '^IXIC_Adj Close', '^IXIC_Volume',
           'GC=F_Open', 'GC=F_High', 'GC=F_Low', 'GC=F_Close', 'GC=F_Adj Close',
           'GC=F_Volume']
    end_date='2024-03-31'
    weeks = 52
    adv_param = [0.1,0.01]
    feature_extractor = 'autoencoder'
    adjust_market_data = rolling_window_standardize(df=market_data,bool_columns=bool_columns, abs_mean_larger10=abs_mean_larger10, window_size=30, method='robust', process_type=1)

    # for method in methods:
    #     for process_type in process_types:
    #         for window_size in window_sizes:
    #             adjust_market_data = rolling_window_standardize(df=market_data,bool_columns=bool_columns, abs_mean_larger10=abs_mean_larger10, window_size=window_size, method=method, process_type=process_type)
    #             for end_date in end_dates:
    #                 for weeks in weeks_options:
    #                     for feature_extractor in feature_extractors:
    #                         for adv_param in adversarial_params:
    #                             print(f"Training model for end date: {end_date}, weeks: {weeks}, feature_extractor: {feature_extractor}, adv_prob: {adv_param[0]}, adv_std: {adv_param[1]}")
                        #model, env = train_model(strategies_data=strategies_data, market_data=adjust_market_data, end_date=end_date, weeks=weeks, feature_extractor=feature_extractor, adv_param=adv_param)
                        #evaluate_model(model, env)
                        #print("\n")
        
    evaluation_result = evaluate_model(model, strategies_data, market_data, train_end_date='20240101', n_steps=3)
    # 打印详细结果
    print("\nDetailed Evaluation Results:")
    for date, reward in zip(evaluation_result['step_dates'], evaluation_result['step_rewards']):
        print(f"Date: {date}, Reward: {reward}")
    print(f"Total Reward: {evaluation_result['total_reward']}")
        
        
        
    