# 金融决策思维链数据集构建框架 / Financial Decision-Making Chain Dataset Construction Framework

[![GitHub License](https://img.shields.io/github/license/wlvh/RBT)](https://github.com/wlvh/RBT/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/wlvh/RBT)](https://github.com/wlvh/RBT/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/wlvh/RBT)](https://github.com/wlvh/RBT/network)
[![GitHub Issues](https://img.shields.io/github/issues/wlvh/RBT)](https://github.com/wlvh/RBT/issues)

## 项目概述 / Project Overview

本项目设计并实现了一个创新的**金融决策思维链数据集构建框架**，融合了多Agent系统、强化学习（RL）和大型语言模型（LLM），实现了无监督的高效思维链数据集生成。该框架旨在提升金融决策过程的多样性、深度和结构化水平，为金融分析和决策支持提供高质量的数据基础。

This project designs and implements an innovative **Financial Decision-Making Chain Dataset Construction Framework**, integrating multi-agent systems, Reinforcement Learning (RL), and Large Language Models (LLM) to achieve efficient, unsupervised generation of decision-making chain datasets. The framework aims to enhance the diversity, depth, and structuring of financial decision-making processes, providing high-quality data foundations for financial analysis and decision support.

## 主要功能 / Key Features

### A. 多样化Agent生态系统 / A. Diverse Agent Ecosystem

- **设计与训练多样化Agents**：构建并训练了超过400种具有不同风格和目标的智能Agent，每个Agent具备独特的决策逻辑和行为模式。
  
  **Design and Train Diverse Agents**: Developed and trained over 400 intelligent Agents with varying styles and objectives, each possessing unique decision-making logic and behavior patterns.

- **生成多角度决策日志**：通过多Agent协作，生成涵盖不同市场情景和决策路径的多层次决策日志，确保数据的全面性和多样性。
  
  **Generate Multi-Angle Decision Logs**: Through multi-Agent collaboration, generate multi-level decision logs covering various market scenarios and decision paths, ensuring data comprehensiveness and diversity.

- **定制化Agent行为**：根据金融决策的具体需求，定制Agent的行为策略超参数，实现对不同金融产品和市场条件的适应。
  
  **Customized Agent Behavior**: Tailor Agent behavior strategy hyperparameters based on specific financial decision-making requirements to adapt to different financial products and market conditions.

### B. 强化学习驱动的Agent筛选和组合机制 / B. RL-Driven Agent Selection and Combination Mechanism

- **多目标RL系统开发**：开发了基于多目标强化学习的Agent筛选和组合系统，优化多个目标如决策有效性、多样性和长期收益。
  
  **Development of Multi-Objective RL System**: Developed an Agent selection and combination system based on multi-objective Reinforcement Learning, optimizing multiple goals such as decision effectiveness, diversity, and long-term rewards.

- **奖励函数设计**：设计复杂的奖励函数，平衡短期和长期奖励，确保Agent的筛选过程既能实现高效决策又保持多样性。
  
  **Reward Function Design**: Designed complex reward functions to balance short-term and long-term rewards, ensuring the Agent selection process achieves efficient decision-making while maintaining diversity.

- **优化算法应用**：采用PPO强化学习算法，提高Agent筛选的准确性和效率。
  
  **Application of Optimization Algorithms**: Utilized Proximal Policy Optimization (PPO) reinforcement learning algorithms to enhance the accuracy and efficiency of Agent selection.

### C. LLM与RL融合的思维链生成器 / C. LLM and RL Integrated Decision-Making Chain Generator

- **集成大型语言模型**：将GPT-4 o1等先进LLM集成到思维链生成器中，提升自然语言理解和生成能力。
  
  **Integration of Large Language Models**: Integrated advanced LLMs such as GPT-4 o1 into the decision-making chain generator to enhance natural language understanding and generation capabilities.

- **数据模板设计**：创建结构化数据模板，确保生成的思维链具有一致的格式和高质量的内容。
  
  **Data Template Design**: Created structured data templates to ensure the generated decision-making chains have consistent formats and high-quality content.

- **RL优化路径**：使用强化学习优化Agent推荐路径，确保生成的思维链既符合逻辑又具备创新性。
  
  **RL Path Optimization**: Employed Reinforcement Learning to optimize Agent recommendation paths, ensuring the generated decision-making chains are both logical and innovative.

- **协同生成机制**：实现LLM与RL模块的无缝协作，通过中间层接口协调两者的工作流程，提升思维链生成的效率和质量。
  
  **Collaborative Generation Mechanism**: Achieved seamless collaboration between LLM and RL modules by coordinating their workflows through intermediate layer interfaces, enhancing the efficiency and quality of decision-making chain generation.

### D. 双轨模型优化策略 / D. Dual-Track Model Optimization Strategy

- **监督微调（SFT）**：利用标注数据对LLM进行监督微调，提升模型在金融决策推理中的准确性和可靠性。
  
  **Supervised Fine-Tuning (SFT)**: Utilized labeled data to perform supervised fine-tuning on LLMs, enhancing the accuracy and reliability of the model in financial decision-making reasoning.

- **近端策略优化（PPO）**：应用PPO算法优化Agent的决策策略，提高模型在复杂决策场景下的表现。
  
  **Proximal Policy Optimization (PPO)**: Applied PPO algorithms to optimize Agent decision strategies, improving model performance in complex decision-making scenarios.

- **协同训练流程**：设计了阶段性训练流程，使SFT和PPO策略互补，共同提升LLM的思维链推理能力。
  
  **Collaborative Training Process**: Designed a phased training process where SFT and PPO strategies complement each other, jointly enhancing the LLM's decision-making chain reasoning capabilities.

- **性能提升评估**：通过定量指标（如决策准确率、思维链一致性）评估优化策略的效果，验证模型性能的显著提升。
  
  **Performance Improvement Evaluation**: Assessed the effectiveness of optimization strategies using quantitative metrics (e.g., decision accuracy, chain consistency) to verify significant enhancements in model performance.

### E. 无监督的快速扩充高质量思维链数据集 / E. Unsupervised Rapid Expansion of High-Quality Decision-Making Chain Dataset

- **自动化数据生成流程**：构建了全自动化的数据生成流程，实现高效的思维链数据集扩充，无需人工干预。
  
  **Automated Data Generation Process**: Established a fully automated data generation process to efficiently expand the decision-making chain dataset without the need for manual intervention.

- **质量控制机制**：引入多层次过滤和验证机制，确保生成的思维链数据具备高质量和相关性，排除低质量或无关数据。
  
  **Quality Control Mechanism**: Introduced multi-level filtering and validation mechanisms to ensure the generated decision-making chain data maintains high quality and relevance, excluding low-quality or unrelated data.

- **数据集规模与覆盖面**：目前已构建超过1万条思维链数据，涵盖多种金融决策场景如风险评估和市场分析。
  
  **Dataset Scale and Coverage**: Currently constructed over 10,000 decision-making chain data entries, covering various financial decision-making scenarios such as risk assessment and market analysis.

### F. 未来计划 / F. Future Plans

- **强化学习作为LLM Agent的记忆与规划模块**：计划将强化学习技术应用于LLM Agent的记忆和规划功能，实现智能的营销策略和文案生成。
  
  **Reinforcement Learning as Memory and Planning Modules for LLM Agents**: Plan to apply Reinforcement Learning techniques to the memory and planning functions of LLM Agents, enabling intelligent generation of marketing strategies and copywriting.

- **自动化营销策略生成**：通过组合多源素材，自动生成针对不同市场需求的定制化营销策略和文案，提高营销效率和效果。
  
  **Automated Marketing Strategy Generation**: Automatically generate customized marketing strategies and copywriting tailored to different market demands by combining multi-source materials, enhancing marketing efficiency and effectiveness.

## 技术栈与工具 / Tech Stack and Tools

- **编程语言 / Programming Languages**: Python, Java
- **框架与库 / Frameworks and Libraries**: PyTorch, OpenAI GPT-4, Stable Baselines 3
- **工具与平台 / Tools and Platforms**: GitHub, Docker, Azure (Cloud Computing)
- **方法与技术 / Methods and Technologies**: 多Agent系统设计, 强化学习（PPO）, 监督微调 (SFT), 数据模板设计, 无监督学习  
  Multi-agent system design, Reinforcement Learning (PPO), Supervised Fine-Tuning (SFT), Data Template Design, Unsupervised Learning

## 安装与使用 / Installation and Usage

### 前提条件 / Prerequisites

- Python 3.8+
- Docker（可选，用于容器化部署） / Docker (optional, for containerized deployment)
- 云计算资源（如AWS或GCP） / Cloud computing resources (e.g., AWS or GCP)

### 安装步骤 / Installation Steps

1. 克隆仓库 / Clone the repository：
    ```bash
    git clone https://github.com/wlvh/RBT.git
    cd RBT/RL
    ```

2. 创建并激活虚拟环境 / Create and activate a virtual environment：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```

3. 安装依赖 / Install dependencies：
    ```bash
    pip install -r requirements.txt
    ```

4. 运行项目 / Run the project：
    ```bash
    python main.py
    ```

### 使用示例 / Usage Examples

详细的使用说明和示例代码请参考 [Wiki](https://github.com/wlvh/RBT/wiki) 或 [示例教程](https://github.com/wlvh/RBT/tree/main/RL/examples)。

For detailed usage instructions and example code, please refer to the [Wiki](https://github.com/wlvh/RBT/wiki) or [Example Tutorials](https://github.com/wlvh/RBT/tree/main/RL/examples).

## 项目链接 / Project Links

- **GitHub 仓库 / GitHub Repository**: [https://github.com/wlvh/RBT/tree/main/RL](https://github.com/wlvh/RBT/tree/main/RL)

## 成就与成果 / Achievements and Outcomes

- 成功生成并扩充了高质量的金融决策思维链数据集，显著提升了金融决策支持系统的性能。
  
  Successfully generated and expanded a high-quality financial decision-making chain dataset, significantly enhancing the performance of financial decision support systems.

- 通过双轨模型优化策略，提升LLM在复杂金融决策推理任务中的准确性和一致性。
  
  Enhanced the accuracy and consistency of LLMs in complex financial decision-making reasoning tasks through dual-track model optimization strategies.

- 实现了无监督的数据扩充方法，大幅减少了数据准备的时间和人力成本。
  
  Implemented an unsupervised data expansion method, greatly reducing the time and labor costs associated with data preparation.

## 贡献 / Contributions

欢迎贡献！请阅读 [贡献指南](https://github.com/wlvh/RBT/blob/main/CONTRIBUTING.md) 了解如何参与。

Welcome contributions! Please read the [Contributing Guide](https://github.com/wlvh/RBT/blob/main/CONTRIBUTING.md) to learn how to participate.

## 许可证 / License

本项目采用 [MIT 许可证](https://github.com/wlvh/RBT/blob/main/LICENSE)。

This project is licensed under the [MIT License](https://github.com/wlvh/RBT/blob/main/LICENSE).

## 联系方式 / Contact

如有任何问题或建议，请通过 [Issues](https://github.com/wlvh/RBT/issues) 联系我们。

For any questions or suggestions, please contact us through [Issues](https://github.com/wlvh/RBT/issues).

---
