# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:17:00 2024

@author: Administrator
"""
import pandas as pd
import numpy as np
import copy
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
import logging
import json
import pickle
import os
from itertools import combinations
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 每次随机从所有策略里抽取100个（同日期），生成50个假的（这里先用伪代码替代，之后我会自己实现），action改为猜测这150个策略的分类（正收益，负收益，假收益）,奖励函数为三分类的f1值
class BaseTradingEnv(gym.Env, ABC):
# import gymnasium as gym
    """
    基础金融交易环境，处理策略和市场数据，并提供训练和预测模式的支持。

    参数:
        end_date (str): 环境的结束日期。
        weeks (int): 环境的持续周数。
        strategies_data (pd.DataFrame): 策略数据。
        market_data (pd.DataFrame): 市场数据。
        config_path (str): 配置文件路径。
        num_samples (int): 每个时间步的样本数量。
        is_training (bool): 是否处于训练模式。
    """
    def __init__(self, 
                 end_date: str, 
                 weeks: int, 
                 strategies_data: pd.DataFrame, 
                 market_data: pd.DataFrame, 
                 config_path: str = 'config.json', 
                 num_samples: int = 20, 
                 use_all_strategies: bool = False,
                 is_training: bool = True):
        super(BaseTradingEnv, self).__init__()
        """
        初始化 BaseTradingEnv 类。

        参数:
            end_date (str): 环境的结束日期。
            weeks (int): 环境的持续周数。
            strategies_data (pd.DataFrame): 策略数据。
            market_data (pd.DataFrame): 市场数据。
            config_path (str): 配置文件路径。
            num_samples (int): 每个时间步的样本数量。
            is_training (bool): 是否处于训练模式。
        """
        # 参数验证
        if not isinstance(end_date, str):
            raise TypeError("end_date must be a string")
        if not isinstance(weeks, int) or weeks <= 0:
            raise ValueError("weeks must be a positive integer")
        if not isinstance(strategies_data, pd.DataFrame):
            raise TypeError("strategies_data must be a pandas DataFrame")
        if not isinstance(market_data, pd.DataFrame):
            raise TypeError("market_data must be a pandas DataFrame")
        
        # 日期处理
        self.end_date = pd.to_datetime(end_date)
        self.start_date = self.end_date - pd.Timedelta(weeks=weeks-1)
        
        self.num_samples = num_samples
        self.is_training = is_training
        
        # 市场数据预处理
        market_data = market_data.copy()
        market_data['month'] = market_data.index.month
        market_data['is_month_start'] = market_data.index.is_month_start.astype(int)
        market_data['is_month_end'] = market_data.index.is_month_end.astype(int)
        market_data['is_quarter_start'] = market_data.index.is_quarter_start.astype(int)
        market_data['is_quarter_end'] = market_data.index.is_quarter_end.astype(int)
        self.market_data = market_data

        # 策略数据预处理
        strategies_data['date'] = pd.to_datetime(strategies_data['date'])
        self.strategies_data = strategies_data[
            (strategies_data['date'] >= self.start_date) & 
            (strategies_data['date'] <= self.end_date)
        ]
        self.max_strategies = self.strategies_data.groupby('date').size().max() 
        self.dates = np.sort(self.strategies_data['date'].unique())
        self.dates = pd.to_datetime(self.dates) 
        # 添加额外的历史数据
        extra_history_start = self.start_date - pd.Timedelta(days=7)
        extra_history_data = strategies_data[strategies_data['date'] == extra_history_start]        
        self.strategies_data = pd.concat([extra_history_data, self.strategies_data]).sort_values('date')
        
        # 数据预处理
        self.preprocess_strategies_data()
        # 如果使用虚假策略生成器，则生成虚假策略
        if self.use_fake_strategies():
            print("use generate_fake_strategies function")
            self.generate_fake_strategies()

        # 初始化其他参数
        self.mixed_strategies = None
        self.episode_count = 0
        self.episodes_per_epoch = 100
        self.np_random = None
        self.last_observation = self.current_observation = None
        self.last_info = {}
        
        self.current_date = None
        self.date_index = 0
        self.market_columns = self.market_data.columns
        market_obs_dim = len(self.market_columns) * 30  # 30天的历史数据

        current_strategy_dim = (
            3 +  # time_idx, target_idx, type_idx
            len(self.param_columns) +
            2  # 1 for target strategy flag, 1 for current/historical flag
        )
        historical_strategy_dim = current_strategy_dim + 1  # +1 for return
        
        self.use_all_strategies = use_all_strategies
    
        # 根据是否使用全量策略调整 self.total_strategies
        if self.use_all_strategies:
            self.total_strategies = self.max_strategies + 10  # 全量策略 + 历史策略
        else:
            self.total_strategies = 11  # 1 个当前策略 + 10 个历史策略

        # 定义观察空间
        self.observation_space = spaces.Dict({
            'market': spaces.Box(low=-np.inf, high=np.inf, shape=(market_obs_dim,), dtype=np.float32),
            'strategies': spaces.Box(low=-1000, high=np.inf, shape=(self.total_strategies, max(current_strategy_dim, historical_strategy_dim)), dtype=np.float32)
        })
        
        # 子类需要定义动作空间
        self.action_space = self.define_action_space()
        
        print(f"Market observation dimension: {market_obs_dim}")
        print(f"Current strategy dimension: {current_strategy_dim}")
        print(f"Historical strategy dimension: {historical_strategy_dim}")
        print(f"Total strategies in observation: {self.total_strategies}")
        
        # 奖励相关参数
        self.weeks = weeks
        self.episode_step = 0
        self.episode_portfolio_returns = []
        self.saved_recommended_strategies = {}
        
    @abstractmethod
    def define_action_space(self):
        """定义动作空间，子类需要实现"""
        pass

    @abstractmethod
    def calculate_reward(self, done: bool) -> float:
        """计算奖励，子类需要实现"""
        pass

    @abstractmethod
    def use_fake_strategies(self) -> bool:
        """是否使用虚假策略，子类需要实现"""
        pass
    
            
    @abstractmethod
    def process_action(self, action):
        """处理动作，根据动作调整策略或投资组合，子类需要实现"""
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def get_next_observation(self):
        return self._next_observation() 

    def analyze_param_types(self, strategies_data: pd.DataFrame) -> dict:
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

    def generate_fake_strategies(self, test_method=True):
        fake_strategies = []
        unique_time = self.strategies_data['time_idx'].unique()
        unique_target = self.strategies_data['target_idx'].unique()
        unique_type = self.strategies_data['type_idx'].unique()
        sam_num = self.max_strategies // 9  # 确保虚假策略占每次抽样的三分之一
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
                fake = pd.concat([
                    negative_strategies.sample(n=n_negative, replace=True) if n_negative > 0 else pd.DataFrame(),
                    positive_strategies.sample(n=n_positive, replace=True) if n_positive > 0 else pd.DataFrame(),
                    zero_strategies.sample(n=n_zero, replace=True) if n_zero > 0 else pd.DataFrame()
                ])
                
                # 补充策略数量
                if len(fake) < (sam_num * 3):
                    additional_needed = (sam_num * 3) - len(fake)
                    additional_strategies = real_strategies.sample(n=additional_needed, replace=True)
                    fake = pd.concat([fake, additional_strategies])
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print(f"Date: {date} (Type: {type(date)})")
                print(f"Real strategies DataFrame length: {len(real_strategies)}")
                print(f"Real strategies DataFrame: \n{real_strategies}")
                print(f"Index format in strategies_data: {self.strategies_data.index}")
                raise e  # 重新抛出异常以便进一步调试
            
            if test_method:
                fake_copy = copy.deepcopy(fake)
            
            # 50% 概率进行 Class Change，50% 概率进行 Param Change
            class_change_mask = np.random.rand(len(fake)) < 0.5
            param_change_mask = ~class_change_mask
            
            # Class Change
            class_change_strategies = fake[class_change_mask]
            for var in ['time_idx', 'target_idx', 'type_idx']:
                if var in class_change_strategies.columns:
                    new_values = np.random.choice(
                        locals()[f'unique_{var.split("_")[0]}'], 
                        size=len(class_change_strategies)
                    )
                    fake.loc[class_change_mask, var] = new_values
            
            # Param Change
            param_change_strategies = fake[param_change_mask]
            for col in self.param_columns:
                mask = (fake[col] != -999) & (np.random.rand(len(fake)) < 0.5)
                multiplier = np.random.choice([0.7, 0.8, 1.2, 1.3], size=mask.sum())
                fake.loc[mask, col] = fake.loc[mask, col] * multiplier.astype(int) if param_types[0][col] == int else fake.loc[mask, col] * multiplier
            
            fake['is_fake'] = True
            fake_strategies.append(fake)
            
            if test_method:
                comparison_result = fake.compare(fake_copy)
                if comparison_result.empty:
                    print(f"No changes were made to the fake DataFrame for date {date}.")
            
        self.fake_strategies = pd.concat(fake_strategies).fillna(-999)
        print('Fake strategies created successfully')

    def mix_strategies(self):
        date = self.dates[self.date_index]
        cur_fake = self.fake_strategies[self.fake_strategies['date'] == date]
        fake_len = len(cur_fake)
        real_strategies = self.strategies_data[self.strategies_data['date'] == date].copy()
        
        # 确保不超过总策略数
        num_real_needed = self.total_strategies - fake_len
        if num_real_needed < 0:
            raise ValueError(f"Number of fake strategies ({fake_len}) exceeds total_strategies ({self.total_strategies}) for date {date}.")
        
        real = real_strategies.sample(n=num_real_needed, replace=False)
        mixed = pd.concat([real, cur_fake]).sample(frac=1).reset_index(drop=True)
        
        logger.debug(f"Mixed strategies for date {date}: {len(mixed)} strategies.")
        return mixed


    def preprocess_strategies_data(self):
        self.strategies_data = self.strategies_data.copy()
        self.strategies_data = self.strategies_data.fillna(-999)
        self.param_columns = [col for col in self.strategies_data.columns if col.startswith('param_')]    
        
        # 对 strategies_data 按照 'time', 'target', 'type' 进行排序，以确保一致的顺序
        self.strategies_data = self.strategies_data.sort_values(by=['time', 'target', 'type']).reset_index(drop=True)
        
        # 确保分类变量的索引存在，可以根据排序后的顺序手动分配索引
        self.strategies_data['time_idx'] = self.strategies_data['time'].astype('category').cat.codes
        self.strategies_data['target_idx'] = self.strategies_data['target'].astype('category').cat.codes
        self.strategies_data['type_idx'] = self.strategies_data['type'].astype('category').cat.codes
        
        if 'is_fake' not in self.strategies_data.columns:
            self.strategies_data['is_fake'] = False 
            
        # 提取第一个日期的类别作为参考
        reference_date = self.dates[0]
        reference_strategies = self.strategies_data[self.strategies_data['date'] == reference_date]
        
        reference_time = reference_strategies['time_idx'].unique()
        reference_target = reference_strategies['target_idx'].unique()
        reference_type = reference_strategies['type_idx'].unique()
        # 定义类别属性以供其他方法使用
        self.time_categories = reference_time
        self.target_categories = reference_target
        self.type_categories = reference_type
        # 定义类别属性以供其他方法使用
        self.time_ori_categories = reference_strategies['time'].unique()
        self.target_ori_categories = reference_strategies['target'].unique()
        self.type_ori_categories = reference_strategies['type'].unique()
    
        # 创建索引到类别的映射字典
        self.time_idx_to_time = {idx: time for idx, time in enumerate(self.time_ori_categories)}
        self.target_idx_to_target = {idx: target for idx, target in enumerate(self.target_ori_categories)}
        self.type_idx_to_type = {idx: type_ for idx, type_ in enumerate(self.type_ori_categories)}
        
        columns_order = ['date', 'return', 'is_fake'] + sorted(self.param_columns) + ['time_idx', 'target_idx', 'type_idx']
        missing_columns = set(columns_order) - set(self.strategies_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in strategies_data: {missing_columns}")
        self.strategies_data = self.strategies_data[columns_order]
        # 验证所有其他日期的类别是否与参考一致
        for date in self.dates[1:]:
            strategies_on_date = self.strategies_data[self.strategies_data['date'] == date]
            
            current_time = strategies_on_date['time_idx'].unique()
            current_target = strategies_on_date['target_idx'].unique()
            current_type = strategies_on_date['type_idx'].unique()
            
            if not np.array_equal(current_time, reference_time):
                raise ValueError(f"Time categories mismatch on date {date}.\n"
                                 f"Expected: {reference_time}\nGot: {current_time}")
            
            if not np.array_equal(current_target, reference_target):
                raise ValueError(f"Target categories mismatch on date {date}.\n"
                                 f"Expected: {reference_target}\nGot: {current_target}")
            
            if not np.array_equal(current_type, reference_type):
                raise ValueError(f"Type categories mismatch on date {date}.\n"
                                 f"Expected: {reference_type}\nGot: {current_type}")
        
        print("Preprocessing and validation completed successfully.")


    def get_last_info(self):
        return self.last_info if hasattr(self, 'last_info') else {}
    

    def get_real_strategies(self, prediction_index=None):
        """
        获取真实策略，根据是否使用全量策略决定返回所有策略或部分策略。
    
        参数:
            prediction_index (int, optional): 预测模式下的索引，仅在非训练模式下使用。
    
        返回:
            pd.DataFrame: 真实策略的 DataFrame。
        """
        date = self.dates[self.date_index]
        all_strategies = self.strategies_data[self.strategies_data['date'] == date]
    
        if self.use_all_strategies:
            # 返回所有真实策略
            real = all_strategies.copy()
        else:
            if prediction_index is None:
                # 默认选择第一个策略作为预测目标
                predicted_strategy = all_strategies.iloc[0]
            else:
                # 根据 prediction_index 选择预测目标策略
                predicted_strategy = all_strategies.iloc[prediction_index]
    
            # 获取预测目标策略及其余策略
            other_strategies = all_strategies.drop(predicted_strategy.name)
            real = pd.concat([predicted_strategy.to_frame().T, other_strategies])
    
        return real

    
    def set_state(self, obs):
        self.current_observation = obs
        
    def _get_market_obs(self):
        if not isinstance(self.current_date, pd.Timestamp):
            raise TypeError("current_date must be a pandas Timestamp")
        end_date = self.current_date
        start_date = end_date - pd.Timedelta(days=29)
        market_data = self.market_data.loc[start_date:end_date]
        if len(market_data) < 30:
            padding = pd.DataFrame(0, index=pd.date_range(start=start_date, end=end_date, freq='D'), columns=market_data.columns)
            market_data = pd.concat([padding, market_data]).fillna(0)
            market_data = market_data.iloc[-30:]
        market_obs = market_data.values.flatten()
        return np.array(market_obs, dtype=np.float32)
                
    def _next_observation(self):
        current_strategy_obs = self._get_current_strategy_obs()
        historical_strategy_obs = self._get_historical_strategy_obs()
        strategy_obs = current_strategy_obs + historical_strategy_obs
        strategy_obs = self._pad_strategies(strategy_obs)
        market_obs = self._get_market_obs()
        return {
            'market': np.array(market_obs, dtype=np.float32),
            'strategies': np.array(strategy_obs, dtype=np.float32)
        }
    
    def _get_current_strategy_obs(self):
        if self.use_all_strategies:
            # 处理当前日期的所有策略
            strategies = self.mixed_strategies
            strategy_obs = []
            for _, strategy in strategies.iterrows():
                obs = [
                    strategy['time_idx'],
                    strategy['target_idx'],
                    strategy['type_idx'],
                    *[strategy[col] for col in self.param_columns],
                    1 if strategy['is_fake'] else 0,  # 标记为虚假策略
                    0  # 标记为当前或历史策略（根据需要调整）
                ]
                strategy_obs.append(obs)
            return strategy_obs
        else:
            # 原始行为：仅处理第一个策略
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
            return [current_strategy_obs]
    
    def _get_historical_strategy_obs(self):
        former_date = self.current_date - pd.Timedelta(days=7)
        former_strategies = self.strategies_data[self.strategies_data['date'] == former_date]
        top_ranking = former_strategies.nlargest(5, 'return')
        bottom_ranking = former_strategies.nsmallest(5, 'return')
        former_ranking = pd.concat([top_ranking, bottom_ranking]).sort_values(by='return', ascending=False)
        
        if len(former_ranking) != 10:
            logger.error("Historical ranking data is incomplete.")
            logger.error(f"Current date: {self.current_date}")
            logger.error(f"Former date: {former_date}")
            logger.error(f"Unique dates in strategies_data: {self.strategies_data['date'].unique()}")
            logger.error(f"Is former_date in strategies_data: {former_date in self.strategies_data['date'].values}")
            raise ValueError("Historical ranking data is incomplete.")
        
        historical_strategy_obs = []
        for _, strategy in former_ranking.iterrows():
            strategy_features = [
                strategy['time_idx'],
                strategy['target_idx'],
                strategy['type_idx'],
                *[strategy[col] for col in self.param_columns],
                strategy['return'],
                0,  # 不是目标策略
                0   # 标记为历史策略
            ]
            historical_strategy_obs.append(strategy_features)
        return historical_strategy_obs
    
    def _pad_strategies(self, strategy_obs):
        max_dim = max(len(obs) for obs in strategy_obs)
        padded_strategy_obs = [obs + [0] * (max_dim - len(obs)) for obs in strategy_obs]
        strategy_obs = np.array(padded_strategy_obs, dtype=np.float32)
        
        # 根据 self.total_strategies 进行填充或截断
        if len(strategy_obs) < self.total_strategies:
            padding = np.zeros((self.total_strategies - len(strategy_obs), strategy_obs.shape[1]), dtype=np.float32)
            strategy_obs = np.vstack([strategy_obs, padding])
            logger.debug(f"Padded strategies from {len(strategy_obs) - (self.total_strategies - len(strategy_obs))} to {self.total_strategies}.")
        else:
            strategy_obs = strategy_obs[:self.total_strategies]
            logger.debug(f"Truncated strategies to {self.total_strategies}.")
        
        return strategy_obs


    def get_last_observation(self):
        return self.last_observation

    def step(self, action, prediction_index=None):
        # 验证预测模式下是否提供了 prediction_index
        # 处理动作
        self.process_action(action)
        self.episode_step += 1
        # 检查是否达到终止条件
        done = False
        if self.is_training and self.date_index >= len(self.dates) - 1:
            done = True
        if self.episode_step >= self.weeks:
            done = True
    
        # 计算奖励
        reward = self.calculate_reward(done)
    
        # 准备下一个观察或最后的观察
        if not done:
            self.date_index += 1  # 仅在未终止时递增
            self.current_date = self.dates[self.date_index]
            next_obs = self._next_observation()
            self.last_observation = next_obs
        else:
            next_obs = self.last_observation  # 保持最后的观察
        # 准备信息字典
        portfolio_return = self.episode_portfolio_returns[-1] if self.episode_portfolio_returns else 0.0
        info = self.get_info(action, portfolio_return, done)
        self.last_info = info
        # 如果 episode 结束，不更新 mixed_strategies
        if not done:
            # 更新 mixed_strategies 为下一步做准备
            if self.is_training and self.use_fake_strategies() and not self.use_all_strategies:
                self.mixed_strategies = self.mix_strategies()
            elif self.use_all_strategies:
                self.mixed_strategies = self.get_real_strategies(prediction_index=None)
            else:
                new_prediction_index = prediction_index + 1 if prediction_index is not None else 0
                self.mixed_strategies = self.get_real_strategies(prediction_index=new_prediction_index)
        else:
            #print(f"Done! the end date is {self.end_date} and current date is {self.current_date}")
            logger.debug("Step: Episode done. Mixed strategies not updated.")
    
        return next_obs, reward, done, False, info

    def get_info(self, action: np.ndarray, portfolio_return: float, done: bool) -> dict:
        info = {
            'portfolio_return': portfolio_return,
            'current_date': self.current_date,
            'episode_step': self.episode_step
        }
        if done:
            if len(self.episode_portfolio_returns) > 0:
                total_portfolio_return = np.sum(self.episode_portfolio_returns)
                info['total_portfolio_return'] = total_portfolio_return
                info['sharpe_ratio'] = self.calculate_reward(done)
                info['recommended'] = self.saved_recommended_strategies
            # 如果不需要 f1_score，可以注释或移除以下行
            # info['f1_score'] = f1_score(self.episode_true_labels, self.episode_predicted_labels, average='macro', zero_division=1)
        return info

    def reset(self, seed=None, options=None):
        """
        重置环境状态，并进行必要的初始化。
        """
        # 不调用 super().reset()
        # 初始化环境状态
        self.selected_strategy_returns = []
        self.saved_recommended_strategies = {}
        logger.debug(f"Resetting environment. Episode count: {self.episode_count}")
        self.episode_count += 1
    
        self.date_index = 0
        self.current_date = self.dates[self.date_index]
        self.current_observation = None
        self.prediction_index = 0  # 初始化 prediction_index
        
        if self.is_training:
            if self.use_fake_strategies() and (self.episode_count % self.episodes_per_epoch == 20):
                self.generate_fake_strategies()

        if self.is_training and self.use_fake_strategies() and not self.use_all_strategies:
            self.mixed_strategies = self.mix_strategies()
            logger.debug("Using mixed strategies with fake strategies.")
        elif self.use_all_strategies:
            self.mixed_strategies = self.get_real_strategies(prediction_index=None)
            logger.debug("Using all real strategies.")
        else:
            self.mixed_strategies = self.get_real_strategies(prediction_index=0)
            logger.debug("Using real strategies with prediction_index=0.")
    
        # 获取初始观察
        observation = self._next_observation()
        self.last_observation = observation
        if not isinstance(observation, dict) or 'market' not in observation or 'strategies' not in observation:
            raise ValueError(f"Invalid observation returned by reset: {observation}")

        # 初始化 info
        info = {}
        self.episode_step = 0
        self.episode_portfolio_returns = []
        self.last_info = info  # 初始化 last_info

        return observation, info

    @abstractmethod
    def get_portfolio_return(self, strategy: pd.Series, action: int) -> float:
        """根据动作计算投资组合回报，子类需要实现"""
        pass

    def _get_true_label(self, strategy: pd.Series) -> int:
        return 0 if strategy['is_fake'] else (
            1 if strategy['return'] > 0 else (
            2 if strategy['return'] < 0 else 3))
    
    def _get_info(self, action: np.ndarray, portfolio_return: float, done: bool) -> dict:
        info = {
            'portfolio_return': portfolio_return,
            'current_date': self.current_date,
            'episode_step': self.episode_step
        }
        if done:
            if len(self.episode_portfolio_returns) > 0:
                total_portfolio_return = np.sum(self.episode_portfolio_returns)
                info['total_portfolio_return'] = total_portfolio_return
                info['sharpe_ratio'] = self.calculate_reward(done)
            info['f1_score'] = f1_score(self.episode_true_labels, self.episode_predicted_labels, average='macro', zero_division=1)
        return info
    
    def calculate_f1_score(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """
        处理稀有类别更健壮: 通过使用 LabelBinarizer，新的实现将每个类别转换为二进制格式，这样可以确保所有类别在计算 F1 分数时都被考虑，即使某些类别没有出现在 predictions 或 true_labels 中。
        支持多标签: 新的实现可以更好地支持多标签分类问题，因为它将每个类别都二值化处理，可以分别计算每个类别的 F1 分数，并通过 average='macro' 取平均。
        稳定性提升: 通过 binarization，即使某些类别在预测或真实标签中没有出现，f1_score 函数依然可以正常工作，避免潜在的异常。
        """
        lb = LabelBinarizer()
        lb.fit(range(4))  # 假设有 4 个类别
        true_labels_bin = lb.transform(true_labels)
        predictions_bin = lb.transform(predictions)
        logger.debug(f"true_labels_bin: {true_labels_bin}")
        logger.debug(f"predictions_bin: {predictions_bin}")
        return f1_score(true_labels_bin, predictions_bin, average='macro', zero_division=1)

    def get_attr(self, attr_name):
        """获取指定名称的属性值。"""
        return getattr(self, attr_name)

        
class ScoreTradingEnv(BaseTradingEnv):
    def __init__(self, 
                 end_date: str, 
                 weeks: int, 
                 strategies_data: pd.DataFrame, 
                 market_data: pd.DataFrame, 
                 config_path: str = 'config.json', 
                 num_samples: int = 20, 
                 is_training: bool = True,
                 top_k: int = 2):
        """
        初始化 ScoreTradingEnv 类。

        参数:
            top_k (int): 选择的前 k 个策略，用于奖励计算。
        """
        self.top_k = top_k
        super(ScoreTradingEnv, self).__init__(
            end_date=end_date,
            weeks=weeks,
            strategies_data=strategies_data,
            market_data=market_data,
            config_path=config_path,
            num_samples=num_samples,
            is_training=is_training,
            use_all_strategies=True  # 使用全量策略
        )
        
        # 用于存储选中策略的历史回报，以计算夏普比率
        self.selected_strategy_returns = []
        self.saved_recommended_strategies = {}
    # def define_action_space(self):
    #     """
    #     定义动作空间为对 'time_idx', 'target_idx', 'type_idx' 的权重分配。
    #     动作空间为一个向量，其长度为 N1 + N2 + N3，表示各细分类的权重。
    #     权重范围为 [0, 1]，并且所有权重之和为 1。
    #     """
    #     # 获取每个分类的细分类数量
    #     time_categories = self.time_categories  # 已在基类中排序并存储
    #     target_categories = self.target_categories
    #     type_categories = self.type_categories
        
    #     num_time = len(time_categories)
    #     num_target = len(target_categories)
    #     num_type = len(type_categories)
        
    #     # 定义动作空间为一个 Box，范围 [0,1]
    #     action_dim = num_time + num_target + num_type
    #     action_space = spaces.Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32)  # 使用 np.float32
    #     return action_space
    def define_action_space(self):
        # 动作空间是组合的索引
        self.strategy_combinations = list(combinations(range(self.max_strategies ), self.top_k))
        return spaces.Discrete(len(self.strategy_combinations))
    
    #def process_action(self, action: np.ndarray):
        
        """
        根据动作权重选择前 top_k 个策略，并设置为当前策略。
        归一化动作向量以确保权重之和为1。
        """
        # 归一化动作向量
        # action_sum = np.sum(action)
        # if action_sum > 0:
        #     normalized_action = action / action_sum
        # else:
        #     # 如果动作向量全为零，均匀分配权重
        #     normalized_action = np.ones_like(action) / len(action)
        
        # logger.debug(f"Action before normalization: {action}")
        # logger.debug(f"Action sum: {action_sum}")
        # logger.debug(f"Normalized action: {normalized_action}")
        
        # # 计算所有策略的得分
        # strategy_scores = self._compute_strategy_scores(normalized_action)
        # logger.debug(f"Strategy scores: {strategy_scores}")
        
        # # 选择得分最高的 top_k 策略
        # top_indices = np.argsort(strategy_scores)[-self.top_k:]
        # logger.debug(f"Top indices: {top_indices}")
        
        # self.selected_strategies = self.mixed_strategies.iloc[top_indices]
        # logger.debug(f"Selected strategies: \n{self.selected_strategies}")
    def process_action(self, action: int):
        selected_indices = self.strategy_combinations[action]
        self.selected_strategies = self.mixed_strategies.iloc[list(selected_indices)]
        
        # 计算组合的平均回报
        selected_returns = self.selected_strategies['return'].values
        portfolio_return = np.mean(selected_returns)  # 使用平均回报
        self.episode_portfolio_returns.append(portfolio_return)
        
        print(f"Selected returns: {selected_returns}")
        print(f"Portfolio return: {portfolio_return}")
        
        if not self.is_training:
            # 在预测模式下打印推荐策略和基于的日期
            date_str = self.current_date.strftime('%Y-%m-%d')
            print(f"推荐日期: {date_str}")
            print("推荐的交易策略：")
            # 使用映射字典将索引转换为实际值
            recommended_strategies = self.selected_strategies.copy()
            # 使用映射字典将索引转换为实际值
            recommended_strategies['time'] = recommended_strategies['time_idx'].map(self.time_idx_to_time)
            recommended_strategies['target'] = recommended_strategies['target_idx'].map(self.target_idx_to_target)
            recommended_strategies['type'] = recommended_strategies['type_idx'].map(self.type_idx_to_type)
            # 选择需要显示的列
            display_columns = ['time', 'target', 'type', 'return']
            strategies_str = recommended_strategies[display_columns].to_string(index=False)
            # 将推荐策略保存到字典中
            if date_str not in self.saved_recommended_strategies:
                self.saved_recommended_strategies[date_str] = []
            # 转换为字典列表
            strategies_dict = recommended_strategies[display_columns].to_dict(orient='records')
            self.saved_recommended_strategies[date_str].extend(strategies_dict)
            print(self.saved_recommended_strategies[date_str])

    
    def _compute_strategy_scores(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        计算每个策略的得分，用于选择 top_k 策略。
        每个策略的得分为其对应的 time_idx、target_idx 和 type_idx 在 normalized_action 中的权重之和。

        参数:
            normalized_action (np.ndarray): 归一化后的动作向量。

        返回:
            np.ndarray: 每个策略的得分。
        """
        num_time = len(self.time_categories)
        num_target = len(self.target_categories)
        num_type = len(self.type_categories)
        
        # 分割动作向量
        time_weights = normalized_action[:num_time]
        target_weights = normalized_action[num_time:num_time + num_target]
        type_weights = normalized_action[num_time + num_target:]
        
        # 提取策略的索引
        strategy_time_idx = self.mixed_strategies['time_idx'].values
        strategy_target_idx = self.mixed_strategies['target_idx'].values
        strategy_type_idx = self.mixed_strategies['type_idx'].values
        
        # 根据索引获取对应的权重
        strategy_time_weights = time_weights[strategy_time_idx]
        strategy_target_weights = target_weights[strategy_target_idx]
        strategy_type_weights = type_weights[strategy_type_idx]
        
        # 计算总得分
        scores = strategy_time_weights + strategy_target_weights + strategy_type_weights
        return scores
    
    def calculate_reward(self, done: bool) -> float:
        """
        计算奖励函数。无论是否为终止步，都基于整个 episode_portfolio_returns 计算夏普比率。
    
        参数:
            done (bool): 是否为终止步
    
        返回:
            float: 计算得到的奖励
        """
        reward = 0.0
        # if len(self.episode_portfolio_returns) > 4: #先储存一个月的信息再来计算，有利于平稳奖励value
        #     returns = np.array(self.episode_portfolio_returns)
        #     sharpe_ratio = self._calculate_sharpe_ratio(returns)
        #     reward = sharpe_ratio
        # else:
        #     reward = 0.0
        if done: #先储存一个月的信息再来计算，有利于平稳奖励value
            returns = np.array(self.episode_portfolio_returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            reward = sharpe_ratio
    
        return reward
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """
        计算夏普比率。

        参数:
            returns (np.ndarray): 投资组合的回报序列
            risk_free_rate (float): 无风险利率，默认为0.01

        返回:
            float: 计算得到的夏普比率
        """
        excess_returns = returns - risk_free_rate
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        if std_return == 0:
            return 0.0
        sharpe_ratio = mean_return / std_return
        return sharpe_ratio
    
    def use_fake_strategies(self) -> bool:
        """
        决定是否使用虚假策略。在本任务中，不再使用虚假策略。
        """
        return False  # 不使用虚假策略

    def get_portfolio_return(self, strategy: pd.Series, action: np.ndarray) -> float:
        """
        根据选择的策略计算投资组合的夏普比率。
        此方法在本实现中不再使用，因为夏普比率在 calculate_reward 中计算。
        可以根据需要进行调整或删除。
        """
        pass 
    
    def _get_info(self, action: np.ndarray, portfolio_return: float, done: bool) -> dict:
        info = {
            'portfolio_return': portfolio_return,
            'current_date': self.current_date,
            'episode_step': self.episode_step,
            'recommended':self.saved_recommended_strategies
        }
        if done:
            if len(self.episode_portfolio_returns) > 0:
                total_portfolio_return = np.sum(self.episode_portfolio_returns)
                info['total_portfolio_return'] = total_portfolio_return
                info['sharpe_ratio'] = self.calculate_reward(done)
        return info


class ClassifyTradingEnv(BaseTradingEnv):
    def __init__(self, *args, **kwargs):
        super(ClassifyTradingEnv, self).__init__(*args, **kwargs)
        # 可以在这里添加自定义的初始化逻辑

    def define_action_space(self):
        # 自定义动作空间
        # 例如：0 - 停止, 1 - 买入, 2 - 卖出, 3 - 持有
        return spaces.Discrete(4)  # 根据您的需求调整

    def calculate_reward(self, done: bool) -> float:
        if done:
            f1_reward = self.calculate_f1_score(np.array(self.episode_predicted_labels), np.array(self.episode_true_labels))
            portfolio_reward = np.sum(self.episode_portfolio_returns)
            return f1_reward + (0 * portfolio_reward)  # 调整权重
        else:
            # 训练过程中可以返回即时奖励
            return 0  # 或者其他逻辑

    def use_fake_strategies(self) -> bool:
        # 决定是否使用虚假策略
        return self.is_training  # 根据训练模式决定

    def get_portfolio_return(self, strategy: pd.Series, action: int) -> float:
        # 根据动作计算投资组合回报
        if action == 1:  # 假设1代表买入
            return strategy['return']
        elif action == 2:  # 假设2代表卖出
            return -strategy['return']
        elif action == 3:  # 假设3代表持有
            return 0
        else:  # 假设0代表不采取任何行动
            return 0

import unittest

class TestBaseTradingEnv(unittest.TestCase):
    def setUp(self):
        # 准备测试数据
        self.strategies_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'return': np.random.randn(100),
            'time': np.random.choice(['short', 'medium', 'long'], 100),
            'target': np.random.choice(['sqn', 'sharpe'], 100),
            'type': np.random.choice(['A', 'B'], 100),
            'param_alpha': np.random.rand(100),
            'param_beta': np.random.randint(1, 10, 100)
        })
        self.market_data = pd.DataFrame({
            'price': np.random.rand(100),
            'volume': np.random.randint(100, 1000, 100)
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='D'))
    
    def test_reset(self):
        env = ClassifyTradingEnv(
            end_date='2023-04-10',
            weeks=12,
            strategies_data=self.strategies_data,
            market_data=self.market_data,
            is_training=True
        )
        observation, info = env.reset()
        self.assertIsInstance(observation, dict)
        self.assertIn('market', observation)
        self.assertIn('strategies', observation)
    
    def test_step(self):
        env = ClassifyTradingEnv(
            end_date='2023-04-10',
            weeks=12,
            strategies_data=self.strategies_data,
            market_data=self.market_data,
            is_training=True
        )
        observation, info = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        self.assertIsInstance(next_obs, dict)
        self.assertIn('market', next_obs)
        self.assertIn('strategies', next_obs)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        

if __name__ == '__main__':
    unittest.main()
