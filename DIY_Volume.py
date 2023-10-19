#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Author: wlvh 124321452@qq.com
Date: 2023-06-16 13:58:55
LastEditors: wlvh 124321452@qq.com
LastEditTime: 2023-08-13 13:18:23
FilePath: /trading/DIY_Volume.py
Description: 
Copyright (c) 2023 by ${124321452@qq.com}, All Rights Reserved. 
'''

import numpy as np
import backtrader as bt
import pandas as pd
from datetime import datetime
import os
import statistics
import matplotlib.pyplot as plt
import time
import json
import logging
import optuna
import sys
from optuna.exceptions import TrialPruned
from optuna.samplers import TPESampler

class vwapDIY1(bt.Indicator):
    plotinfo = dict(subplot=False)

    alias = ('VWAP', 'VolumeWeightedAveragePrice','vwap',)
    lines = ('VWAP','typprice','cumprice', 'cumtypprice',)
    plotlines = dict(VWAP=dict(alpha=1.0, linestyle='-', linewidth=2.0, color = 'magenta'))

    params = (
        ('period', 14),  # Set the rolling window size here
    )

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        self.l.typprice[0] = ((self.data.close + self.data.high + self.data.low)/3) * self.data.volume
        self.l.cumtypprice[0] = sum(self.l.typprice.get(size=self.params.period))
        self.cumvol = sum(self.data.volume.get(size=self.params.period))+0.00001
        self.lines.VWAP[0] = self.l.cumtypprice[0] / self.cumvol

class vwapDIY(bt.Indicator):
    lines = ('emavwap',)
    params = (('period', 14),)

    def __init__(self):
        typprice = ((self.data.close + self.data.high + self.data.low)/3) * self.data.volume
        self.ema_typprice = bt.indicators.EMA(typprice, period=self.params.period)
        self.ema_volume = bt.indicators.EMA(self.data.volume, period=self.params.period)
        self.addminperiod(self.params.period)
        self.plotinfo.plot = True  # Ensure the indicator is plotted
        self.plotinfo.subplot = False  # Plot in the same window as price data
        self.plotinfo.plotvaluetags = True
    def next(self):
        if self.ema_volume[0] != 0:
            self.lines.emavwap[0] = self.ema_typprice[0] / self.ema_volume[0]
        else:
            self.lines.emavwap[0] = self.ema_typprice[0]


class VWAPStrategy(bt.Strategy):
    params = (
        ('buy_volume_window', 30),
        ('sell_volume_window', 30),
        ('buy_volume_multiplier', 2),
        ('sell_volume_multiplier', 2),
        ('buy_stop_loss', 0.02),    # 2% stop loss
        ('sell_stop_loss', 0.02),   # 2% stop loss
        ('buy_take_profit', 0.06),  # 6% profit taking
        ('sell_take_profit', 0.06),  # 6% profit taking
        ('buy_risk_per_trade', 0.5),  # risk 1% of capital per trade
        ('sell_risk_per_trade', 0.5),  # risk 1% of capital per trade
        ('printlog', False),
        ('buy_atr_period',14),
        ('buy_vwap_period',14),
        ('sell_vwap_period',14),
        ('sell_atr_period',14),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        if self.p.printlog:
            dt = dt or self.data.datetime[0]
            if isinstance(dt, float):
                dt = bt.num2date(dt)
            log_line = '%s, %s' % (dt.isoformat(), txt)
            print(log_line)
            with open('VWAPStrategy_log.txt', 'a') as f:
                f.write(log_line + '\n')

    def __init__(self):
        self.buy_avg_volume = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.p.buy_volume_window)
        self.sell_avg_volume = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.p.sell_volume_window)
        self.order = None  # To keep track of pending orders
        self.buyprice = 0
        self.sellprice = 0
        self.max_profit = 0  # Initialize max_profit
        self.buy_atr = bt.indicators.ATR(self.data, period=self.p.buy_atr_period)
        self.sell_atr = bt.indicators.ATR(self.data, period=self.p.sell_atr_period)
        self.buy_atr_sma = bt.indicators.SimpleMovingAverage(self.buy_atr, period=self.p.buy_atr_period)
        self.sell_atr_sma = bt.indicators.SimpleMovingAverage(self.sell_atr, period=self.p.sell_atr_period)
        self.sell_vwap = vwapDIY(self.data, period=self.params.sell_vwap_period) # get VWAP 
        self.buy_vwap = vwapDIY(self.data, period=self.params.buy_vwap_period) # get VWAP 

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self.order = order
            return
    
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Units: %.2f, Pnl: %.2f,current open position size: %.2f,Cost: %.2f, Comm %.2f' %
                (order.executed.price,
                order.executed.value/order.executed.price,
                order.executed.pnl,
                order.executed.psize,
                order.executed.pprice,
                order.executed.comm))
                self.buyprice = order.executed.price
            elif order.issell():
                self.sellprice = order.executed.price
                self.log('SELL EXECUTED, Price: %.2f, Units: %.2f, Pnl: %.2f,current open position size: %.2f,Cost: %.2f, Comm %.2f' %
                (order.executed.price,
                order.executed.value/order.executed.price,
                order.executed.pnl,
                order.executed.psize,
                order.executed.pprice,
                order.executed.comm))
    
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.log('Order Info: %s' % order.info)
            self.log('Order status: %s' % order.status)
    
        self.order = None

    def next(self):
        if self.order:
            return  # If there is a pending order, do nothing
    
        if self.position:  # In the market
            if self.data.close[0] > self.buy_vwap[0] and self.data.volume[0] > self.buy_avg_volume[0] * self.p.buy_volume_multiplier * (self.buy_atr[0] / self.buy_atr_sma[0]):
                if self.position.size < 0:  # if we have a short position
                    self.log(f'Close short and start buy, Value: {self.data.close[0]:.2f}')
                    self.close() 
                    size = self.broker.getvalue() * self.p.buy_risk_per_trade * (self.data.volume[0] / self.buy_avg_volume[0]) / self.data.close[0]
                    if size * self.data.close[0] > 2 * self.broker.getvalue():
                        size = 2 * self.broker.getvalue() / self.data.close[0]
                    self.log(f'BUY CREATE, Size: {size:.4f},Value: {self.data.close[0]:.2f}')
                    self.buy(size=size)
            elif self.data.close[0] < self.sell_vwap[0] and self.data.volume[0] > self.sell_avg_volume[0] * self.p.sell_volume_multiplier * (self.sell_atr[0] / self.sell_atr_sma[0]):
                if self.position.size > 0:  # if we have a long position
                    self.log(f'Close long and start sell, Value: {self.data.close[0]:.2f}')
                    self.close() 
                    size = self.broker.getvalue() * self.p.sell_risk_per_trade * (self.data.volume[0] / self.sell_avg_volume[0]) / self.data.close[0]
                    if size * self.data.close[0] > 2 * self.broker.getvalue():
                        size = 2 * self.broker.getvalue() / self.data.close[0]
                    self.log(f'SELL CREATE, Size: {size:.4f},Value: {self.data.close[0]:.2f}')
                    self.sell(size=size)
            if self.position.size > 0:  # long position
                profit = (self.data.close[0] - self.buyprice) / self.buyprice  # Calculate current profit
                if profit > self.max_profit:
                    self.max_profit = profit
                elif self.max_profit - profit >= self.p.buy_take_profit:
                    self.log(f'STOP PROFIT triggered: {self.data.close[0]:.2f}')
                    self.close()  # Stop Profit
                elif self.data.close[0] < self.buyprice * (1.0 - self.p.buy_stop_loss):
                    self.log(f'STOP LOSS triggered: {self.data.close[0]:.2f}')
                    self.close()  # Stop Loss
            else:  # short position
                profit = (self.sellprice - self.data.close[0]) / self.sellprice
                if profit > self.max_profit:
                    self.max_profit = profit
                elif self.max_profit - profit >= self.p.sell_take_profit:
                    self.log(f'STOP PROFIT triggered: {self.data.close[0]:.2f}')
                    self.close()  # Stop Profit
                elif self.data.close[0] > self.sellprice * (1.0 + self.p.sell_stop_loss):
                    self.log(f'STOP LOSS triggered: {self.data.close[0]:.2f}')
                    self.close()  # Stop Loss
        else: # Not in the market
            if self.data.close[0] > self.buy_vwap[0] and self.data.volume[0] > self.buy_avg_volume[0] * self.p.buy_volume_multiplier * (self.buy_atr[0] / self.buy_atr_sma[0]):
                size = self.broker.getvalue() * self.p.buy_risk_per_trade * (self.data.volume[0] / self.buy_avg_volume[0]) / self.data.close[0]
                if size * self.data.close[0] > 2 * self.broker.getvalue():
                    size = 2 * self.broker.getvalue() / self.data.close[0]
                self.log(f'BUY CREATE, Size: {size:.4f},Value: {self.data.close[0]:.2f}')
                self.buy(size=size)
            elif self.data.close[0] < self.sell_vwap[0] and self.data.volume[0] > self.sell_avg_volume[0] * self.p.sell_volume_multiplier * (self.sell_atr[0] / self.sell_atr_sma[0]):
                size = self.broker.getvalue() * self.p.sell_risk_per_trade * (self.data.volume[0] / self.sell_avg_volume[0]) / self.data.close[0]
                if size * self.data.close[0] > 2 * self.broker.getvalue():
                    size = 2 * self.broker.getvalue() / self.data.close[0]
                self.log(f'SELL CREATE, Size: {size:.4f},Value: {self.data.close[0]:.2f}')
                self.sell(size=size)

                

def main(df,sell_vwap_period, buy_vwap_period, sell_volume_window, buy_volume_window, sell_atr_period, buy_atr_period, buy_volume_multiplier, sell_volume_multiplier, buy_stop_loss, sell_stop_loss, buy_take_profit, sell_take_profit, buy_risk_per_trade, sell_risk_per_trade, printlog=True, para_opt=True, startcash=100000, com=0.0005, opt_target=None):
    
    sell_volume_window = round(sell_volume_window)
    buy_volume_window = round(buy_volume_window)
    buy_vwap_period = round(buy_vwap_period)
    sell_vwap_period = round(sell_vwap_period)
    sell_atr_period = round(sell_atr_period)
    buy_atr_period = round(buy_atr_period)
    num_days = len(pd.unique(df.index.date))
    print(f'our trade is {num_days} days.')
    print("1")
    start = time.time()
    print("2")
    cerebro = bt.Cerebro()
    print("3")
    cerebro.addstrategy(VWAPStrategy,
                        sell_atr_period=sell_atr_period,
                        buy_atr_period=buy_atr_period,
                        sell_volume_window=sell_volume_window,
                        buy_volume_window=buy_volume_window,
                        buy_volume_multiplier=buy_volume_multiplier,
                        sell_volume_multiplier=sell_volume_multiplier,
                        buy_stop_loss=buy_stop_loss,
                        sell_stop_loss=sell_stop_loss,
                        buy_vwap_period=buy_vwap_period,
                        sell_vwap_period=sell_vwap_period,
                        printlog=printlog,
                        buy_take_profit=buy_take_profit,
                        sell_take_profit=sell_take_profit,
                        buy_risk_per_trade=buy_risk_per_trade,
                        sell_risk_per_trade=sell_risk_per_trade)
    
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.setcash(startcash)
    cerebro.broker.setcommission(commission=com,leverage=3)
    
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annualreturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,timeframe=bt.TimeFrame.NoTimeFrame, _name='sharpe')    
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.GrossLeverage, _name='gross_leverage')
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='positions_value')
    cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='log_returns_rolling')
    cerebro.addanalyzer(bt.analyzers.PeriodStats, timeframe=bt.TimeFrame.NoTimeFrame, _name='period_stats')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A,timeframe=bt.TimeFrame.NoTimeFrame, _name='sharpe_A')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')

    # rest of the code remains same

    try:
        results = cerebro.run()
    except Exception as e:
        print(f"An error occurred during backtesting: {str(e)}")
        return -np.inf, None

    sharpe_ratio = results[0].analyzers.sharpe.get_analysis()
    drawdown = results[0].analyzers.drawdown.get_analysis()['max']['drawdown']
    sqn = results[0].analyzers.sqn.get_analysis()['sqn']
    final_value = cerebro.broker.getvalue()

    total_trades = results[0].analyzers.trades.get_analysis()['total']['total']
    trade_analysis = results[0].analyzers.trades.get_analysis()
    winning_trades = trade_analysis['won']['total'] if 'won' in trade_analysis else 0
    annual_return = results[0].analyzers.annualreturn.get_analysis()
    
    performances = {
        "trade_analysis":trade_analysis,
        'sharpe': sharpe_ratio,
        'value': final_value,
        'total_return': (final_value / startcash - 1.0) ,
        'sqn': sqn,
        'drawdown': -drawdown/100,
        'drawdown_value': (-drawdown/100) + (final_value / startcash - 1.0) ,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'annual_return': annual_return,
        'Win ratio': winning_trades/(total_trades+0.000001),
       # 'gross_leverage': results[0].analyzers.gross_leverage.get_analysis(),数据量太大了，先屏蔽掉。
        'positions_value': results[0].analyzers.positions_value.get_analysis(),
        'log_returns_rolling': results[0].analyzers.log_returns_rolling.get_analysis(),
        'period_stats': results[0].analyzers.period_stats.get_analysis(),
        'returns': results[0].analyzers.returns.get_analysis(),
        'sharpe_A': results[0].analyzers.sharpe_A.get_analysis(),
        'time_return': results[0].analyzers.time_return.get_analysis(),
        'vwr': results[0].analyzers.vwr.get_analysis()['vwr'],
        'win ratio value': (winning_trades/(total_trades+0.000001)) * final_value,
        'drawdown value': (1-(drawdown/100)) * final_value,
        'std_dev': statistics.stdev(list(results[0].analyzers.time_return.get_analysis().values())),
        'SharpeDIY':(final_value / startcash - 1.0)/(statistics.stdev(list(results[0].analyzers.time_return.get_analysis().values()))+0.000001),
        'sqn_drawdown': sqn * (1-(drawdown/100)),
        'drawdown_win_ratio': (1-(drawdown/100)) * winning_trades/(total_trades+0.000001),
        'drawdown_win_ratio_value': (1-(drawdown/100)) * winning_trades/(total_trades+0.000001) * final_value
    }
    num_days = num_days/2
    if performances['total_trades'] < num_days:
        performances['weighted_win_ratio'] = performances['total_trades'] * 1/num_days * performances['Win ratio']
        performances['weighted_sqn'] = performances['total_trades'] * 1/num_days * performances['sqn']
        performances['weighted_value'] = performances['total_trades'] * 1/num_days * performances['value']
        performances['weighted_drawdown'] = performances['total_trades'] * 1/num_days * (1-(drawdown/100))
        performances['weighted_SharpeDIY'] = performances['total_trades'] * 1/num_days * performances['SharpeDIY']
        performances['weighted_drawdown_value'] = performances['total_trades'] * 1/num_days * performances['drawdown value']
        performances['weighted_win_ratio_value'] = performances['total_trades'] * 1/num_days * performances['win ratio value']
        performances['weighted_vwr'] = performances['total_trades'] * 1/num_days * performances['vwr']
        performances['weighted_sqn_drawdown'] = performances['total_trades'] * 1/num_days * performances['sqn_drawdown']
        performances['weighted_drawdown_win_ratio'] = performances['total_trades'] * 1/num_days * performances['drawdown_win_ratio']
        performances['weighted_drawdown_win_ratio_value'] = performances['total_trades'] * 1/num_days * performances['drawdown_win_ratio_value']
    else:
        performances['weighted_win_ratio'] = performances['Win ratio']
        performances['weighted_sqn'] = performances['sqn']
        performances['weighted_value'] = performances['value']
        performances['weighted_drawdown'] = (1-(drawdown/100))
        performances['weighted_SharpeDIY'] = performances['SharpeDIY']
        performances['weighted_drawdown_value'] = performances['drawdown value']
        performances['weighted_win_ratio_value'] = performances['win ratio value']
        performances['weighted_vwr'] = performances['vwr']
        performances['weighted_sqn_drawdown'] = performances['sqn_drawdown']
        performances['weighted_drawdown_win_ratio'] = performances['drawdown_win_ratio'] 
        performances['weighted_drawdown_win_ratio_value'] = performances['drawdown_win_ratio_value']      
        
    end = time.time()
    total_time = end - start
    print(f"Cost time: {total_time}")

    if not para_opt:
        print('Drawdown:', drawdown)
        print('SQN:', sqn)
        print('Total Return: %.2f%%' % ((final_value / startcash - 1.0) * 100))
        print('Total trades:', total_trades)
        print('Winning trades:', winning_trades)
        
        # Create directory if not exist
        directory = "trading_pictures"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        strategy_name = VWAPStrategy.__name__
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        data_start_date = df.index[0].strftime("%Y-%m-%d")
        data_end_date = df.index[-1].strftime("%Y-%m-%d")
        filename = f"{current_time}_{strategy_name}_{data_start_date}_to_{data_end_date}_{opt_target}.png"
        filepath = os.path.join(directory, filename)
        
        plots = cerebro.plot(barup='green', bardown='red', figsize=(27, 15), dpi=600)

        
        # Double loop to iterate over strategies and figures
        for strategy_plots in plots:
            for i, plot in enumerate(strategy_plots):
                plot.set_size_inches(18, 10)
                plot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
                
                filepath = "trading_pictures/{}_{}_{}_{}_{}_{}.png".format(current_time, 'VWAPStrategy', df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"), opt_target, i)
                plot.savefig(filepath, dpi=100)
                plt.close(plot)
        
        return performances

    if para_opt:
        if opt_target is None:
            raise ValueError("Please specify an optimization target. You can use: 'sharpe', 'value', 'sqn', 'drawdown'")
        print(f'Now the optimize target is: {opt_target}')
        return performances, None 


              

              

'''
target = 'Win ratio'
buy_volume_window = opt_results[target][0]['buy_volume_window']
sell_volume_window = opt_results[target][0]['sell_volume_window']

buy_volume_multiplier = opt_results[target][0]['buy_volume_multiplier']
sell_volume_multiplier = opt_results[target][0]['sell_volume_multiplier']

buy_stop_loss = opt_results[target][0]['buy_stop_loss']
sell_stop_loss = opt_results[target][0]['sell_stop_loss']
buy_take_profit = opt_results[target][0]['buy_take_profit']
sell_take_profit = opt_results[target][0]['sell_take_profit']

buy_vwap_period = opt_results[target][0]['buy_vwap_period']
sell_vwap_period = opt_results[target][0]['sell_vwap_period']

buy_risk_per_trade = opt_results[target][0]['buy_risk_per_trade']
sell_risk_per_trade = opt_results[target][0]['sell_risk_per_trade']

buy_atr_period = opt_results[target][0]["buy_atr_period"]
sell_atr_period = opt_results[target][0]["sell_atr_period"]
opt_target='Win ratio'
performances, _ = main(sell_vwap_period, buy_vwap_period, sell_volume_window, 
                       buy_volume_window, sell_atr_period, buy_atr_period, 
                       buy_volume_multiplier, sell_volume_multiplier,
                       buy_stop_loss, sell_stop_loss, buy_take_profit,
                       sell_take_profit, buy_risk_per_trade, sell_risk_per_trade, 
                       printlog=True, df=df[-1000:], para_opt=True, startcash=100000, 
                       com=0.0005, opt_target=opt_target)
'''


class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold):
        self._threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study, trial):
        # Check if the current trial is pruned
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
            print(f'Trial for target was pruned. So far, {self._consequtive_pruned_count} trials have been pruned. The threshold is {self._threshold}.')
        else:
            self._consequtive_pruned_count = 0

        # Stop further trials if the threshold is exceeded
        if self._consequtive_pruned_count >= self._threshold:
            study.stop()



def optimize_and_run(targets, df, num_evals=100, printlog=True):
    opt_results = {}
    date_period = len(np.unique(pd.DatetimeIndex(df.index).date))
    end_date = df.index[-1]
    end_date = pd.to_datetime(end_date)
    for target in targets:
        print(f'Now the optimize target is: {target}')
        def make_main_opt(df, target):  
            def main_opt(trial):
                kwargs = {
                    'sell_volume_window': trial.suggest_int('sell_volume_window', 13, 50),
                    'buy_volume_window': trial.suggest_int('buy_volume_window', 13, 50),
                    'buy_volume_multiplier': trial.suggest_float('buy_volume_multiplier', 1.5, 12, step=0.1),
                    'sell_volume_multiplier': trial.suggest_float('sell_volume_multiplier', 1.5, 12, step=0.1),
                    "buy_atr_period": trial.suggest_int('buy_atr_period', 13, 50),
                    "sell_atr_period": trial.suggest_int('sell_atr_period', 13, 50),
                    'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.1, step=0.001),
                    'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.1, step=0.001),
                    'buy_take_profit': trial.suggest_float('buy_take_profit', 0.05, 0.10, step=0.001),
                    'sell_take_profit': trial.suggest_float('sell_take_profit', 0.05, 0.10, step=0.001),
                    "buy_risk_per_trade": trial.suggest_float('buy_risk_per_trade', 0.3, 1, step=0.01),
                    "sell_risk_per_trade": trial.suggest_float('sell_risk_per_trade', 0.3, 1, step=0.01),
                    "sell_vwap_period": trial.suggest_int('sell_vwap_period', 10, 30),
                    "buy_vwap_period": trial.suggest_int('buy_vwap_period', 10, 30),
                    'opt_target': target,
                    "printlog": printlog
                }
                performances, _ = main(**kwargs, df=df)
                #print(f'Now the optimize {target} is: {performances[target]}')
                if performances[target] is None:
                    print(f'No trade detected in {target}.')
                    return -100  # or any special value you prefer 
                if target == 'std_dev':
                    return -performances[target]  # return negative performance if target is 'std_dev'
                if target == 'drawdown' and performances[target] > -0.03:
                    return -100  # no trade no drawdown
                else:
                    return performances[target]

            print("opt_0")
            return main_opt  

        main_opt = make_main_opt(df, target)  
        start = time.time()
        old_date = end_date - pd.Timedelta(days=3)
        cur_study_name = f'example-study-{end_date}_{date_period}_{target}'
        old_study_name = f'example-study-{old_date}_{date_period}_{target}'
        try:
            old_study = optuna.load_study(study_name=old_study_name, storage='sqlite:///example.db') 
        except Exception as e:
            print(f"Caught exception: {e}")
            old_study = None
        if old_study is not None and len(old_study.trials) > 1:
            try:
                if old_study.best_trial.state == optuna.trial.TrialState.COMPLETE:
                    best_params = old_study.best_params
                    last_trial_params = old_study.trials[-1].params
                    # 将最优参数和最后一次trial的参数作为新的trial添加到新的study
                    new_study = optuna.create_study(study_name=cur_study_name, storage='sqlite:///example.db', direction='maximize', load_if_exists=True)
                    print(f"the old study best_params: {best_params}")
                    new_study.enqueue_trial(best_params)
                    new_study.enqueue_trial(last_trial_params)
                else:
                    raise ValueError("No complete trials found.")
            except ValueError:
                print(f'Encountered an issue while getting the best parameters from the old study for {cur_study_name}. Initializing a new study.')
                new_study = optuna.create_study(study_name=cur_study_name, storage='sqlite:///example.db', direction='maximize', load_if_exists=True)
        else:
            print(f'We did not find the best parameters for the old study {cur_study_name} and will initialize a new one.')
            new_study = optuna.create_study(study_name=cur_study_name, storage='sqlite:///example.db', direction='maximize', load_if_exists=True)
        # Add stream handler of stdout to show the messages to see Optuna works expectedly.
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))   
        print(f'Now optimizing target: {target} over {date_period} days.')
        new_study.optimize(main_opt, n_trials=num_evals, gc_after_trial=True)
        print("opt_1")
        optimal_pars = new_study.best_params
        details = new_study.best_value
        print("opt_2")
        opt_results[target] = (optimal_pars, details)
        print("opt_3")
        end = time.time()
        total_time = end - start
        print(f"Cost time: {total_time}")

    return opt_results

'''
def optimize_and_run(targets, df, num_evals=100, printlog=True, num_runs_per_trial=10):
    opt_results = {}
    
    for target in targets:
        print(f'Now the optimize target is: {target}')
        def make_main_opt(df, target):  
            def main_opt(trial):
                kwargs = {'sell_volume_window': trial.suggest_int('sell_volume_window', 13, 50),
                          'buy_volume_window': trial.suggest_int('buy_volume_window', 13, 50),
                          'buy_volume_multiplier': trial.suggest_float('buy_volume_multiplier', 1.5, 12, step=0.1),
                          'sell_volume_multiplier': trial.suggest_float('sell_volume_multiplier', 1.5, 12, step=0.1),
                          "buy_atr_period":trial.suggest_int('buy_atr_period', 13, 50),
                          "sell_atr_period":trial.suggest_int('sell_atr_period', 13, 50),
                          'buy_stop_loss': trial.suggest_float('buy_stop_loss', 0.01, 0.1, step=0.001),
                          'sell_stop_loss': trial.suggest_float('sell_stop_loss', 0.01, 0.1, step=0.001),
                          'buy_take_profit': trial.suggest_float('buy_take_profit', 0.05, 0.10, step=0.001),
                          'sell_take_profit': trial.suggest_float('sell_take_profit', 0.05, 0.10, step=0.001),
                          "buy_risk_per_trade":trial.suggest_float('buy_risk_per_trade', 0.3, 1, step=0.01),
                          "sell_risk_per_trade":trial.suggest_float('sell_risk_per_trade', 0.3, 1, step=0.01),
                          "sell_vwap_period":trial.suggest_int('sell_vwap_period', 10, 30),
                          "buy_vwap_period":trial.suggest_int('buy_vwap_period', 10, 30),
                          'opt_target': target,
                          "printlog": printlog}
                total_performance = 0
                step = 0  # 添加一个步骤数计数器
                for _ in range(num_runs_per_trial):
                    performances, _ = main(**kwargs, df=df)
                    performance = performances.get(target)
                    if performance is None:
                        print('detected not trade in {target}')
                        return -100  # or any special value you prefer 
                    total_performance += performance
                    step += 1  # 每次循环增加步骤数
                    trial.report(total_performance /(num_runs_per_trial+1), step)  # 使用步骤数计数器
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned() 
                return total_performance / num_runs_per_trial  # return the average performance after all runs
            return main_opt
        start = time.time()
        print(f'Now the optimize target is: {target}')
        print("opt_1")
        study = optuna.create_study(direction='maximize')
        print("opt_2")
        try:
            # Add StopWhenTrialKeepBeingPrunedCallback to the study
            callback = StopWhenTrialKeepBeingPrunedCallback(threshold=10) # replace 3 with your threshold
            study.optimize(make_main_opt(df, target), n_trials=num_evals, gc_after_trial=True, callbacks=[callback])
        except optuna.exceptions.TrialPruned:
            print(f'Trial for target {target} was pruned.')
        optimal_pars = study.best_params
        details = study.best_value
        opt_results[target] = (optimal_pars, details)
        print("opt_3")
        end = time.time()
        total_time = end - start
        print(f"Cost time: {total_time}")
    return opt_results
'''






'''
targets = ['Win ratio','sqn','value','drawdown','SharpeDIY']
#targets = ['Win ratio','sqn','value','drawdown','vwr','std_dev','SharpeDIY']
opt_results = optimize_and_run(targets,df=df[find_row_by_date(df, '2022-12-01 00:01:00'):find_row_by_date(df, '2023-01-01 00:01:00')],num_evals=1, printlog=False)
'''

def evaluate_opt_results(opt_results, df):
    # 保存每个目标的性能结果
    performance_results = {}

    # 计算市场同期的涨跌幅，波动率，最大涨幅，最大跌幅
    return_rate = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
    max_gain = (df['close'].max() - df['close'].iloc[0]) / df['close'].iloc[0]
    max_drop = (df['close'].min() - df['close'].iloc[0]) / df['close'].iloc[0]
    volatility = df['close'].pct_change().std()

    # 添加市场表现字典
    market_performance = {
        'return': return_rate,  # 同期市场涨跌幅
        'volatility': volatility,  # 波动率
        'max_gain': max_gain,  # 最大涨幅
        'max_drop': max_drop  # 最大跌幅
    }

    # 为每个优化目标循环
    for target in opt_results.keys():
        print(target)
        # 从opt_results获取优化参数
        sell_volume_window = opt_results[target][0]['sell_volume_window']
        buy_volume_window = opt_results[target][0]['buy_volume_window']
        buy_volume_multiplier = opt_results[target][0]['buy_volume_multiplier']
        sell_volume_multiplier = opt_results[target][0]['sell_volume_multiplier']
        buy_stop_loss = opt_results[target][0]['buy_stop_loss']
        sell_stop_loss = opt_results[target][0]['sell_stop_loss']
        buy_take_profit = opt_results[target][0]['buy_take_profit']
        sell_take_profit = opt_results[target][0]['sell_take_profit']
        sell_vwap_period = opt_results[target][0]['sell_vwap_period']
        buy_vwap_period = opt_results[target][0]['buy_vwap_period']
        buy_risk_per_trade = opt_results[target][0]['buy_risk_per_trade']
        sell_risk_per_trade = opt_results[target][0]['sell_risk_per_trade']
        buy_atr_period = opt_results[target][0]["buy_atr_period"]
        sell_atr_period = opt_results[target][0]["sell_atr_period"]


        # 使用参数运行main函数并获取性能
        #performances = main(df,sell_vwap_period, buy_vwap_period, sell_volume_window, 
        #               buy_volume_window, sell_atr_period, buy_atr_period, 
        #               buy_volume_multiplier, sell_volume_multiplier,
        #               buy_stop_loss, sell_stop_loss, buy_take_profit,
        #               sell_take_profit, buy_risk_per_trade, sell_risk_per_trade, 
        #               printlog=True, para_opt=False, startcash=100000, 
        #               com=0.0005)
        performances = main(df=df, sell_vwap_period=sell_vwap_period, buy_vwap_period=buy_vwap_period, sell_volume_window=sell_volume_window, 
                       buy_volume_window=buy_volume_window, sell_atr_period=sell_atr_period, buy_atr_period=buy_atr_period, 
                       buy_volume_multiplier=buy_volume_multiplier, sell_volume_multiplier=sell_volume_multiplier,
                       buy_stop_loss=buy_stop_loss, sell_stop_loss=sell_stop_loss, buy_take_profit=buy_take_profit,
                       sell_take_profit=sell_take_profit, buy_risk_per_trade=buy_risk_per_trade, sell_risk_per_trade=sell_risk_per_trade, 
                       printlog=False, para_opt=False, startcash=100000, 
                       com=0.0005, opt_target=target)

        
        
        # 将性能结果保存到字典
        data = {
    'sell_volume_window': opt_results[target][0]['sell_volume_window'],
    'buy_volume_window': opt_results[target][0]['buy_volume_window'],
    'buy_volume_multiplier': opt_results[target][0]['buy_volume_multiplier'],
    'sell_volume_multiplier': opt_results[target][0]['sell_volume_multiplier'],
    'buy_stop_loss': opt_results[target][0]['buy_stop_loss'],
    'sell_stop_loss': opt_results[target][0]['sell_stop_loss'],
    'buy_take_profit': opt_results[target][0]['buy_take_profit'],
    'sell_take_profit': opt_results[target][0]['sell_take_profit'],
    'sell_vwap_period': opt_results[target][0]['sell_vwap_period'],
    'buy_vwap_period': opt_results[target][0]['buy_vwap_period'],
    'buy_risk_per_trade': opt_results[target][0]['buy_risk_per_trade'],
    'sell_risk_per_trade': opt_results[target][0]['sell_risk_per_trade'],
    'buy_atr_period': opt_results[target][0]['buy_atr_period'],
    'sell_atr_period': opt_results[target][0]['sell_atr_period'],
}
        performance_results[target] = {}
        performance_results[target][0] = performances
        performance_results[target][1] = data

    # 在返回结果中添加市场表现
    performance_results['market_performance'] = market_performance

    return performance_results



def find_row_by_date(df, date_string):
    date = pd.to_datetime(date_string)
    return df.index.get_loc(date)
'''
# 使用示例
row = find_row_by_date(df, '2023-05-01 00:01:00')
print(row)
performance_results12 = evaluate_opt_results(opt_results, df=df[find_row_by_date(df, '2022-12-01 00:01:00'):find_row_by_date(df, '2023-02-01 00:01:00')])

performance_7days12 = evaluate_opt_results(opt_results, df=df[find_row_by_date(df, '2023-02-01 00:01:00'):find_row_by_date(df, '2023-02-08 00:01:00')])
performance_1month12 = evaluate_opt_results(opt_results, df=df[find_row_by_date(df, '2023-02-01 00:01:00'):find_row_by_date(df, '2023-03-01 00:01:00')])

performance_all = evaluate_opt_results(opt_results, df=df)
'''



def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }
'''
flattened_data = [flatten_dict(record) for record in performance_all]
df_P = pd.DataFrame(performance_all)
df_P.to_csv("Check.csv")
'''
def dict_to_csv(data, csv_filename):
    df = pd.json_normalize(data)
    df.to_csv(csv_filename, index=False)
#dict_to_csv(df_P, "Check1.csv")
'''
import json

# 你的原始字典
your_dict = opt_results

# 创建一个新的字典，只包含每个元组的第一个元素（也就是字典）
new_dict = {key: value[0] for key, value in your_dict.items()}

# 保存这个新字典到一个json文件
with open('my_dict.json', 'w') as f:
    json.dump(new_dict, f)
    
with open('/Users/lyuhongwang/Desktop/my_dict.json', 'r') as f:
    opt_results = json.load(f)

print(opt_results)
'''