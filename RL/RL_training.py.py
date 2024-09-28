#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: wlvh 124321452@qq.com
Date: 2024-09-23 08:37:29
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2024-09-24 14:50:20
FilePath: /trading/RL_selector/RL_select_model.py
Description: 
Copyright (c) 2024 by ${124321452@qq.com}, All Rights Reserved. 
'''
import gymnasium as gym  # 使用 gymnasium 替换 gym
from gymnasium import spaces  # 使用 gymnasium 的 spaces
import numpy as np
import copy
from torch.optim import AdamW
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
from typing import Tuple,Callable,Any,Dict
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import os
import json
from features_extractor import CNNFeaturesExtractor, AutoEncoderFeaturesExtractor, VAEFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from BaseTradingEnv import ScoreTradingEnv
from RL_data_process import extract_strategies_data, rolling_window_standardize
import logging
from logging.handlers import RotatingFileHandler
from stable_baselines3.common.vec_env import DummyVecEnv
import traceback
from functools import wraps
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from stable_baselines3.common.logger import configure
import warnings
import wandb
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

class WandbLoggingCallback(BaseCallback):
    """
    自定义回调函数，用于在训练过程中记录指标到 Weights & Biases (Wandb)。
    包含评估策略性能、记录训练损失、梯度信息以及实现早停和学习率调整。
    """

    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=50, total_timesteps=1400040, verbose=0):
        super(WandbLoggingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.total_timesteps = total_timesteps
        self.best_metrics = {
            'mean_reward': -np.inf,
            'total_portfolio_return': -np.inf,
            'sharpe_ratio': -np.inf
        }
        self.no_improvement_count = 0

    def _init_callback(self) -> None:
        """
        初始化 Wandb 记录，设置项目名称和配置参数。
        """
        # 如果使用了 VecNormalize，需要获取底层环境
        eval_env = self.eval_env
        if hasattr(eval_env, 'venv'):
            eval_env = eval_env.venv

        wandb.init(
            project="your_project_name",  # 请替换为您的项目名称
            config={
                "algorithm": "PPO",
                "env": "ScoreTradingEnv",
                "eval_freq": self.eval_freq,
                "n_eval_episodes": self.n_eval_episodes,
                "total_timesteps": self.total_timesteps,
                "learning_rate": self.model.learning_rate,
                "gamma": self.model.gamma,
                "vf_coef": self.model.vf_coef,
                "ent_coef": self.model.ent_coef,
                "gae_lambda": self.model.gae_lambda,
                "batch_size": self.model.batch_size,
                "n_steps": self.model.n_steps,
                "n_epochs": self.model.n_epochs,
                "net_arch": self.model.policy_kwargs.get('net_arch', 'default')
            }
        )
        if self.verbose > 0:
            print("WandbLoggingCallback: Wandb initialized.")

    def _on_rollout_end(self) -> None:
        """
        在每个 rollout 结束时调用，提取并记录损失函数和梯度信息。
        """
        # 从模型的 logger 获取损失信息
        if hasattr(self.model.logger, "name_to_value"):
            logger_data = self.model.logger.name_to_value
            policy_loss = logger_data.get('train/policy_loss', None)
            entropy_loss = logger_data.get('train/entropy_loss', None)
            value_loss = logger_data.get('train/value_loss', None)
            
            if policy_loss is not None and entropy_loss is not None and value_loss is not None:
                wandb.log({
                    "train/policy_loss": policy_loss,
                    "train/entropy_loss": entropy_loss,
                    "train/value_loss": value_loss
                }, step=self.num_timesteps)
                if self.verbose > 0:
                    print(f"WandbLoggingCallback: Losses - policy_loss: {policy_loss}, entropy_loss: {entropy_loss}, value_loss: {value_loss}")
            else:
                if self.verbose > 0:
                    print("WandbLoggingCallback: Loss information not found in logger.")
        else:
            if self.verbose > 0:
                print("WandbLoggingCallback: Logger does not have name_to_value attribute.")

        # 记录梯度信息
        self._log_gradients()

    def _on_step(self) -> bool:
        """
        在每个训练步骤中调用，执行评估和其他日志记录。
        """
        if self.num_timesteps % self.eval_freq == 0:
            eval_results = self.custom_eval_function(self.model, self.eval_env, self.n_eval_episodes)
            wandb.log(eval_results, step=self.num_timesteps)
            if self.verbose > 0:
                print(f"WandbLoggingCallback: Evaluation at step {self.num_timesteps} - {eval_results}")

            improved = False
            for metric in ['mean_reward', 'total_portfolio_return', 'sharpe_ratio']:
                if eval_results.get(metric, -np.inf) > self.best_metrics.get(metric, -np.inf):
                    self.best_metrics[metric] = eval_results[metric]
                    improved = True

            if improved:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= 50:
                if self.verbose > 0:
                    print("WandbLoggingCallback: Early stopping triggered.")
                return False  # 停止训练

            if self.no_improvement_count >= 3:
                # 减小学习率
                current_lr = self.model.policy.optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * 0.5, 1e-6)  # 例如，每次减半
                self.model.policy.optimizer.param_groups[0]['lr'] = new_lr
                wandb.log({"train/learning_rate": new_lr}, step=self.num_timesteps)
                if self.verbose > 0:
                    print(f"WandbLoggingCallback: Learning rate reduced to {new_lr}")

        # 继续记录训练指标
        self.log_training_metrics()
        return True  # 继续训练

    def log_training_metrics(self):
        """
        记录当前学习率和其他训练指标到 Wandb。
        """
        # 记录当前学习率
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        wandb.log({"train/learning_rate": current_lr}, step=self.num_timesteps)

        # 记录其他训练指标
        if hasattr(self.model.logger, "name_to_value"):
            for loss_name, loss_value in self.model.logger.name_to_value.items():
                if "loss" in loss_name:
                    wandb.log({f"train/{loss_name}": loss_value}, step=self.num_timesteps)

        # 记录投资组合回报
        last_info = self.get_last_info()
        portfolio_return = last_info.get('portfolio_return', 0)
        wandb.log({"environment/portfolio_return": portfolio_return}, step=self.num_timesteps)

    def _log_gradients(self):
        """
        记录每层的梯度范数和梯度分布到 Wandb。
        """
        gradients = []
        # 记录每层的梯度范数
        for name, param in self.model.policy.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                wandb.log({f"gradients/{name}": param_norm}, step=self.num_timesteps)
                if self.verbose > 1:
                    print(f"WandbLoggingCallback: Gradient norm for {name}: {param_norm}")
                gradients.extend(param.grad.data.cpu().numpy().flatten())

        # 记录总梯度范数
        total_norm = np.sqrt(sum([param.grad.data.norm(2).item() ** 2 for param in self.model.policy.parameters() if param.grad is not None]))
        wandb.log({"train/gradient_norm": total_norm}, step=self.num_timesteps)
        if self.verbose > 1:
            print(f"WandbLoggingCallback: Total Gradient Norm: {total_norm}")

        # 记录梯度分布（直方图）
        if gradients:
            hist, bins = np.histogram(gradients, bins=100)
            wandb.log({"gradients/distribution": wandb.Histogram(np_histogram=(hist, bins))}, step=self.num_timesteps)

    def get_last_info(self) -> Dict:
        """
        获取环境的最后信息。
        """
        if hasattr(self.eval_env, 'envs'):
            infos = self.eval_env.env_method('get_last_info')
            # 过滤 None 值并返回第一个有效信息
            for info in infos:
                if info:
                    return info
        elif hasattr(self.eval_env, 'get_last_info'):
            return self.eval_env.get_last_info()
        elif hasattr(self.eval_env, 'unwrapped') and hasattr(self.eval_env.unwrapped, 'last_info'):
            return self.eval_env.unwrapped.last_info
        return {}

    def custom_eval_function(self, model, eval_env, n_eval_episodes=50):
        """
        自定义评估函数，用于评估策略并计算相关指标，适应向量化环境。
        """
        print(f"Eval env type: {type(eval_env)}")
        
        # 处理 VecNormalize
        original_eval_env = eval_env
        if isinstance(eval_env, VecNormalize):
            eval_env = eval_env.venv
            eval_env.training = False
            eval_env.norm_reward = False
        
        # 评估策略的平均奖励和标准差
        try:
            mean_reward, std_reward = evaluate_policy(
                model, 
                original_eval_env,  # 使用原始环境，包括 VecNormalize
                n_eval_episodes=n_eval_episodes, 
                return_episode_rewards=False
            )
            print(f"Evaluate policy results: mean_reward={mean_reward}, std_reward={std_reward}")
        except Exception as e:
            print(f"Error in evaluate_policy: {e}")
            mean_reward, std_reward = 0, 0
    
        all_portfolio_returns = []
        is_vector_env = isinstance(eval_env, VecEnv)
        num_envs = eval_env.num_envs if is_vector_env else 1
        episodes_per_env = n_eval_episodes // num_envs
        remaining_episodes = n_eval_episodes % num_envs
    
        for env_idx in range(num_envs):
            episodes_to_run = episodes_per_env + (1 if env_idx < remaining_episodes else 0)
            for episode in range(episodes_to_run):
                obs = eval_env.reset() if is_vector_env else eval_env.reset()[0]
                done = False
                episode_portfolio_returns = []
    
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    if is_vector_env:
                        obs, _, dones, infos = eval_env.step(action)
                        done = dones[0]
                        info = infos[0]
                    else:
                        obs, _, terminated, truncated, info = eval_env.step(action)
                        done = terminated or truncated
    
                    if isinstance(info, dict):
                        episode_portfolio_returns.append(info.get('portfolio_return', 0))
                    else:
                        print(f"Warning: info is not a dict, but {type(info)}. Skipping portfolio_return.")
    
                all_portfolio_returns.extend(episode_portfolio_returns)
    
        total_portfolio_return = np.sum(all_portfolio_returns)
        returns = np.array(all_portfolio_returns)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
    
        # 恢复 VecNormalize 的原始设置
        if isinstance(original_eval_env, VecNormalize):
            original_eval_env.training = True
            original_eval_env.norm_reward = True
    
        return { 
            'mean_reward': mean_reward, 
            'std_reward': std_reward, 
            'total_portfolio_return': total_portfolio_return, 
            'sharpe_ratio': sharpe_ratio, 
        }


def train_model(strategies_data: pd.DataFrame, market_data: pd.DataFrame, end_date: str, weeks: int, 
                feature_extractor: str = 'cnn', adv_param: list = [0,0]):
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(weeks=weeks) - timedelta(days=30)).strftime('%Y-%m-%d')
    #market_data = load_market_data(start_date=start_date, end_date=end_date, file=market_data)
    market_data = load_market_data(start_date=start_date, end_date=end_date, file=adjust_market_data)
    # 使用 SubprocVecEnv 来并行化环境，提高训练效率。
    def make_env(rank, seed=0):
        def _init():
            env = ScoreTradingEnv(
                end_date=end_date, 
                weeks=weeks, 
                strategies_data=strategies_data, 
                market_data=market_data,
                num_samples=20,
                is_training=True,
                top_k=2
            )
            env = Monitor(env)
            env.seed(seed + rank)
            return env
        return _init

    # SubprocVecEnv会导致无法正常print
    num_envs = 5  # 根据您的CPU核心数量调整
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(
        vec_env, 
        norm_obs=False, 
        norm_reward=True, 
        clip_obs=5., #这个可以事后在样本外进行检测
        clip_reward=3.,
        training=True)

    env = ScoreTradingEnv(
        end_date=end_date, 
        weeks=weeks, 
        strategies_data=strategies_data, 
        market_data=market_data,
        num_samples=20,
        is_training=True,
        top_k=3)
    observation, info = env.reset()
    assert isinstance(observation, dict), "Observation should be a dict."
    assert 'market' in observation and 'strategies' in observation, "Observation dict should contain 'market' and 'strategies'."
    assert isinstance(info, dict), "Info should be a dict."
    print("Environment reset successful and returned correct observation and info.")
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
    net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])],
    ortho_init=True,  # 使用正交初始化
)

    def linear_schedule(initial_value: float):
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func
    # n_steps: 如果你的环境每次最多运行52步（52周），那么n_steps应该设置为52或更小。设置为52意味着每次更新前会收集整个周期的数据。
    # batch_size: 通常设置为n_steps的一个较小的因子。例如，你可以尝试16或32。
    # n_epochs:默认值通常是10。对于你的问题，可以从5开始尝试，然后根据性能调整。
    # ent_coef=0.01,
    n_steps = 520*4
    batch_size = 520*2
    n_epochs = 10
    gamma = 1
    vf_coef = 1
    ent_coef = 0.01
    gae_lambda = 1
    learning_rate = 0.001
    import math
    def cosine_annealing_schedule(initial_lr, min_lr=1e-5):
        def schedule(progress_remaining):
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress_remaining))
        return schedule
    #learning_rate = cosine_annealing_schedule(initial_lr=0.003, min_lr=1e-5)
    
    model = PPO(
        MultiInputActorCriticPolicy,
        vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device='cuda',
        learning_rate=learning_rate,
        n_steps=n_steps,
        clip_range=0.3,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        max_grad_norm=0.6,
        #clip_range_vf=None  # 禁用 value function 的裁剪
    )

    # 创建回调
    adversarial_callback = AdversarialTrainingCallback(adversarial_prob=adv_param[0], adversarial_std=adv_param[1])

    # 实例化回调
    wandb_callback = WandbLoggingCallback(
    eval_env=vec_env,
    eval_freq=n_steps,  # 根据需要调整
    n_eval_episodes=50,
    total_timesteps=14000000,  # 传递总步数
    verbose=1
    )
    
    callbacks = [wandb_callback]
    
    # 运行检查
    check_data(env)
    # 训练模型
    model.learn(total_timesteps=14000000, callback=callbacks)
    # 构建保存模型的文件名，包含关键信息
    model_filename = f"ppo_model_nsteps_{n_steps}_lr_{learning_rate}_batchsize_{batch_size}_nepochs_{n_epochs}_gamma_{gamma}_vfcoef_{vf_coef}_entcoef_{ent_coef}_gaelambda_{gae_lambda}.zip"
    
    model_name = "ppo_trading_model"
    run_id = wandb.run.id if wandb.run else "manual_run"
    # 保存模型
    model_path = f"./models/{model_name}_{run_id}"
    model.save(model_path)
    # 保存 VecNormalize 的统计数据
    vec_env.save(f"{model_path}_vec_normalize.pkl")
    # 重新创建向量化环境
    # 加载 PPO 模型
    model_path = r"C:\Users\Administrator.DESKTOP-4H80TP4\RBT\models\ppo_trading_model_idalfslp.zip"
    model = PPO.load(model_path, env=vec_env, device='cuda')  # 根据需要选择 'cuda' 或 'cpu'   
    # 设置 VecNormalize 为非训练模式
    # 定义单一环境创建函数
    def make_single_env():
        env = ScoreTradingEnv(
            end_date=end_date, 
            weeks=weeks, 
            strategies_data=strategies_data, 
            market_data=market_data,
            num_samples=20,
            is_training=False,  # 预测时设置为 False
            top_k=2  # 根据您的需求调整
        )
        env = Monitor(env)
        return env
    
    # 使用 DummyVecEnv 创建单一环境
    vec_env = DummyVecEnv([make_single_env])
    vec_env = VecMonitor(vec_env)

    # 加载保存的 VecNormalize 统计数据
    vec_norm_path = r"C:\Users\Administrator.DESKTOP-4H80TP4\RBT\models\ppo_trading_model_idalfslp_vec_normalize.pkl"
    vec_env = VecNormalize.load(vec_norm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False  # 预测时通常不需要归一化奖励
     # 重置环境，获取初始观察值
    obs = vec_env.reset()
    # 获取策略名称列表（假设 strategies_data 有 'strategy_name' 列）
    # 进行预测和交互
    done = False
    while not done:
        # 使用模型进行预测
        action, _states = model.predict(obs, deterministic=True)
        
        # 将动作传递给环境
        obs, rewards, done, info = vec_env.step(action)
        
        # 根据需要处理环境返回的信息
        print(f"Action: {action}, Reward: {rewards}, Done: {done}")
        
        # 如果需要，可以在这里添加更多的处理逻辑，例如记录日志、可视化等
    def get_attr(env, attr_name, indices=None):
        """使用 env_method 调用环境的 get_attr 方法。"""
        return env.env_method('get_attr', attr_name, indices=indices)

    
    # 获取环境实例
    env = vec_env.envs[0]
    
    def convert_timestamp(obj):
        if isinstance(obj, dict):
            return {key: convert_timestamp(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_timestamp(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')  # 使用 strftime 来格式化日期时间
        return obj
    
    converted_data = convert_timestamp(info[0]['recommended'])
    
    with open('all_recommended_strategies.json', 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)
    
    print("所有推荐的策略已保存到 'all_recommended_strategies.json'")    
    # 关闭环境
    vec_env.close()   
    
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
    key = '9400e057692977a1b3564041a6d635b1f84a8522'
    wandb.login(key=key)
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
        
        
        
    