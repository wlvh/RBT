# RBT

# Backtesting Library based on NumPy and Numba

---

## English

### Introduction

This repository houses a backtesting library that is built on top of `NumPy` and `Numba`. This library is designed to provide a highly efficient and versatile environment for trading strategy simulation and evaluation.

### Features

- **Switchable Strategies**: One of the unique aspects of this library is the ability to switch strategies during the backtesting period. This capability allows for a more dynamic and adaptive approach to market conditions.
  
- **State Preservation**: Information such as current positions, cost basis (for stop-loss), and peak profits (for take-profit) can be seamlessly transferred to the subsequent strategy period. This ensures continuity and can be vital for strategies that involve hedging, scaling, or other complex operations.

### Note

This library is currently made public for backtesting purposes only, as strategies built upon it have already proven to be profitable.

---

## 中文

### 简介

该仓库包含一个基于 `NumPy` 和 `Numba` 的回测库。该库旨在提供一个高效且多功能的环境，用于交易策略的模拟和评估。

### 特性

- **可切换策略**: 该库的一个独特之处在于可以在回测期间切换策略。这一功能允许更动态和适应性地应对市场条件。
  
- **状态保存**: 当前仓位、成本基础（用于止损）以及最高利润（用于止盈）等信息可以无缝转移到下一个策略期间。这确保了连续性，并且对于涉及对冲、规模调整或其他复杂操作的策略至关重要。

### 注意事项

由于基于该库构建的策略已经证明是盈利的，因此目前仅公开回测库。

