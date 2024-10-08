# 步骤1：数据收集与准备
收集详细的市场数据（如价格、成交量、技术指标）和交易策略数据（如策略参数、信号逻辑）。
标注每笔交易的结果（盈利与否）。

# 步骤2：初步微调
使用第一阶段的数据，让LLM输出根据市场数据和交易结果生成市场分析和策略评估（无监督）。
确保模型能够理解市场环境、交易策略与交易结果之间的关系。
prompt = 
"""

        市场数据：
        {json.dumps(market_analysis, ensure_ascii=False, indent=2)}
        交易策略数据:
        {json.dumps(strategy_signals, ensure_ascii=False, indent=2)}

        关键观察：
        1. 过去100分钟收益率为{overall_return*100:.2f}%，正收益比率为{positive_returns_ratio*100:.2f}%，价格振幅为{price_amplitude*100:.2f}%。
        2. 市场呈现{trend_direction}趋势，{'交易' if trend_direction != '横盘' else '波动'}由{majority_or_minority}决定。
        3. 交易量：{volume_change_direction}，交易金额加权价格相比当前价格{price_comparison}。
        4. 市场情绪：整体市场情绪偏{market_sentiment}。
        5. 过去7天表现：价格呈{price_trend}趋势，成交量呈{volume_trend}趋势。
        6. 波动性：成交价格波动率{price_volatility}，成交量波动率{volume_volatility}。
        7. 本次交易决策为{trade_decision}，选定的交易策略为{selected_strategy}。

        请基于所有提供的数据，包括市场数据、关键观察和交易策略数据，生成一份全面的市场分析和策略评估报告。您的报告应包含以下几个方面：

        1. 短期市场分析：基于近期100分钟的数据，分析当前市场状况和短期走势。请特别关注价格、交易量和市场情绪的变化。
        2. 中期趋势评估：结合过去7天的数据，评估中期市场趋势。考虑价格和交易量的变化模式，以及它们与短期趋势的关系。
        3. 市场情绪分析：评估当前的市场情绪，并基于提供的数据解释可能的原因。考虑各种指标如何影响整体市场氛围。
        4. 总体市场态势：基于您的短期和中期分析以及市场情绪评估，简要描述当前市场的整体状态和主要特征。请保持客观，仅基于数据得出结论。
        5. 策略评估：考虑当前的市场态势和给定的交易决策和选定的交易策略（{trade_decision}，{selected_strategy}），评估交易策略数据里包含的各类指标代表的市场状态。你的重点应放在交易策略数据里包含的各类指标的含义上，以及它们如何与市场数据和关键观察相互关联。

        请确保你的分析有说服力并且客观，不需要带有主观判断。报告应当简明扼要控制在400字以内，不需要输出除了1-5外的其他建议。当分析结束请标注**报告结束**
        **报告开始：**
        """
# 步骤3：生成交易理由
在第二阶段，结合前一阶段的输出（市场分析和策略评估），以及交易历史（是否盈利）训练模型生成交易理由（buy，sell or 不交易）。
确保模型能够根据综合信息解释交易决策的逻辑。
prompt = 
"""

        市场数据：
        {json.dumps(market_analysis, ensure_ascii=False, indent=2)}
        交易策略数据:
        {json.dumps(strategy_signals, ensure_ascii=False, indent=2)}
        市场分析和策略评估报告：
        {json.dumps(market_summary, ensure_ascii=False, indent=2)}

        关键观察：
        1. 过去100分钟收益率为{overall_return*100:.2f}%，正收益比率为{positive_returns_ratio*100:.2f}%，价格振幅为{price_amplitude*100:.2f}%。
        2. 市场呈现{trend_direction}趋势，{'交易' if trend_direction != '横盘' else '波动'}由{majority_or_minority}决定。
        3. 交易量：{volume_change_direction}，交易金额加权价格相比当前价格{price_comparison}。
        4. 市场情绪：整体市场情绪偏{market_sentiment}。
        5. 过去7天表现：价格呈{price_trend}趋势，成交量呈{volume_trend}趋势。
        6. 波动性：成交价格波动率{price_volatility}，成交量波动率{volume_volatility}。
        7. 本次交易决策为{trade_decision}，选定的交易策略为{selected_strategy}。

        请基于所有提供的数据，包括市场数据、关键观察和市场分析和策略评估报告，生成一份全面的交易理由报告。您的报告应包含以下几个方面：

        1. 交易评估：
        a) 市场状况分析：简要总结当前市场状态，包括短期和中期趋势。
        b) 策略适用性：评估选定策略（{selected_strategy}）在当前市场条件下的适用性。
        c) 风险评估：根据市场波动性和趋势，分析潜在风险。

        2. 交易理由：
        a) 决策依据：解释为什么选择这个交易决策（{trade_decision}），详细说明市场趋势、交易量变化、市场情绪等因素如何共同作用，支持此次交易决策。
        b) 策略选择理由：阐述为什么选定的策略（{selected_strategy}）是合适的，详细说明市场趋势、交易量变化、市场情绪等因素如何共同作用，支持此次选定的策略。
        c) 预期结果：基于市场状况和选定策略，简要说明预期的交易结果。

        3. 关键考虑因素：
        a) 列出支持这个交易决策的2-3个最重要的市场指标或观察结果。
        b) 指出可能影响交易结果的1-2个潜在风险因素。

        请确保您的分析客观、数据驱动，并且逻辑清晰。避免使用主观判断，专注于数据和策略信号提供的信息。报告应当简明扼要，控制在400字以内。当分析结束请标注**报告结束**
        **报告开始：**
        """
# 步骤4：综合微调（SFT）
将所有输入（市场数据、策略数据、分析、评估、理由）和输出（是否盈利）整合，进行监督微调。
优化模型在综合信息下生成准确交易信号的能力。
prompt = 
        """
        
        # A. 综合交易信息
        
        ## 1. Instruction:
        {instruction}
        
        ## 2. 市场数据：
        {market_analysis}
        
        ## 3. 交易策略数据：
        {strategy_signals}
        
        # B. 关键观察：
        1. 过去100分钟收益率为{overall_return*100:.2f}%，正收益比率为{positive_returns_ratio*100:.2f}%，价格振幅为{price_amplitude*100:.2f}%。
        2. 市场呈现{trend_direction}趋势，{'交易' if trend_direction != '横盘' else '波动'}由{majority_or_minority}决定。
        3. 交易量：{volume_change_direction}，交易金额加权价格相比当前价格{price_comparison}。
        4. 市场情绪：整体市场情绪偏{market_sentiment}。
        5. 过去7天表现：价格呈{price_trend}趋势，成交量呈{volume_trend}趋势。
        6. 波动性：成交价格波动率{price_volatility}，成交量波动率{volume_volatility}。
        
        # C. 市场分析和策略评估报告：
        {market_summary}  
        
        # D. 交易理由报告：
        {trade_analysis}
        
        # E.交易信号生成：
        基于以上所有信息，请生成一个交易信号（买入、卖出或不交易），并从交易策略数据中列出的交易策略里挑选交易策略。
        1. 交易决策： {trade_decision}
        2. 挑选策略： {selected_strategy}
        """
[交易策略清单] = for key in strategy_signals.keys():
                for sub_key in strategy_signals[key].keys():
                        交易策略清单.append(sub_key)
                
# 步骤5：PPO强化学习阶段
在这个阶段让模型自主生成市场分析，策略评估，是否交易以及交易理由。奖励模型可以有两个，一个是评估文字是否符合金融分析师标准（可以用SFT阶段的模型，或者bert或者其他小模型），一个是llm交易决策是否与真实交易结果一致。例如，文字符合标准，A=1，llm交易决策与真实交易结果一致，B=1。最终奖励为A*B。\
定义奖励函数：根据改进建议，设计更细化和丰富的奖励函数，如奖励 = α * A + β * B。\
训练RL模型：使用PPO算法，根据生成的交易决策和评估结果进行训练，优化模型生成高质量的交易信号。\
反馈循环：定期评估RL模型的表现，引入专家反馈，调整奖励函数和训练参数，确保模型的持续优化。\
