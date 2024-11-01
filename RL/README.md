# 金融决策思维链数据集构建框架

[![GitHub License](https://img.shields.io/github/license/wlvh/RBT)](https://github.com/wlvh/RBT/blob/main/LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/wlvh/RBT)](https://github.com/wlvh/RBT/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/wlvh/RBT)](https://github.com/wlvh/RBT/network)
[![GitHub Issues](https://img.shields.io/github/issues/wlvh/RBT)](https://github.com/wlvh/RBT/issues)

## 项目概述

本项目设计并实现了一个创新的**金融决策思维链数据集构建框架**，融合了多Agent系统、强化学习（RL）和大型语言模型（LLM），实现了无监督的高效思维链数据集生成。该框架旨在提升金融决策过程的多样性、深度和结构化水平，为金融分析和决策支持提供高质量的数据基础。

## 主要功能

### A. 多样化Agent生态系统

- **设计与训练多样化Agents**：构建并训练了超过20种具有不同风格和目标的智能Agent，每个Agent具备独特的决策逻辑和行为模式。
- **生成多角度决策日志**：通过多Agent协作，生成涵盖不同市场情景和决策路径的多层次决策日志，确保数据的全面性和多样性。
- **定制化Agent行为**：根据金融决策的具体需求，定制Agent的行为策略，实现对不同金融产品和市场条件的适应。

### B. 强化学习驱动的Agent筛选和组合机制

- **多目标RL系统开发**：开发了基于多目标强化学习的Agent筛选和组合系统，优化多个目标如决策有效性、多样性和长期收益。
- **奖励函数设计**：设计复杂的奖励函数，平衡短期和长期奖励，确保Agent的筛选过程既能实现高效决策又保持多样性。
- **优化算法应用**：采用策略梯度和Q-learning等多种强化学习算法，提高Agent筛选的准确性和效率。

### C. LLM与RL融合的思维链生成器

- **集成大型语言模型**：将GPT-4等先进LLM集成到思维链生成器中，提升自然语言理解和生成能力。
- **数据模板设计**：创建结构化数据模板，确保生成的思维链具有一致的格式和高质量的内容。
- **RL优化路径**：使用强化学习优化Agent推荐路径，确保生成的思维链既符合逻辑又具备创新性。
- **协同生成机制**：实现LLM与RL模块的无缝协作，通过中间层接口协调两者的工作流程，提升思维链生成的效率和质量。

### D. 双轨模型优化策略

- **监督微调（SFT）**：利用标注数据对LLM进行监督微调，提升模型在金融决策推理中的准确性和可靠性。
- **近端策略优化（PPO）**：应用PPO算法优化Agent的决策策略，提高模型在复杂决策场景下的表现。
- **协同训练流程**：设计了阶段性训练流程，使SFT和PPO策略互补，共同提升LLM的思维链推理能力。
- **性能提升评估**：通过定量指标（如决策准确率、思维链一致性）评估优化策略的效果，验证模型性能的显著提升。

### E. 无监督的快速扩充高质量思维链数据集

- **自动化数据生成流程**：构建了全自动化的数据生成流程，实现高效的思维链数据集扩充，无需人工干预。
- **质量控制机制**：引入多层次过滤和验证机制，确保生成的思维链数据具备高质量和相关性，排除低质量或无关数据。
- **数据集规模与覆盖面**：目前已构建超过100万条思维链数据，涵盖多种金融决策场景如投资组合管理、风险评估和市场分析。
- **与现有数据集对比**：新构建的数据集在规模、质量和多样性上显著优于现有公开数据集，具备更高的实用价值和应用潜力。

### F. 未来计划

- **强化学习作为LLM Agent的记忆与规划模块**：计划将强化学习技术应用于LLM Agent的记忆和规划功能，实现智能的营销策略和文案生成。
- **自动化营销策略生成**：通过组合多源素材，自动生成针对不同市场需求的定制化营销策略和文案，提高营销效率和效果。

## 技术栈与工具

- **编程语言**：Python, Java
- **框架与库**：TensorFlow, PyTorch, OpenAI GPT-4, Stable Baselines
- **工具与平台**：GitHub, Docker, AWS/GCP（云计算）
- **方法与技术**：多Agent系统设计, 强化学习（策略梯度, Q-learning, PPO）, 监督微调 (SFT), 数据模板设计, 无监督学习

## 安装与使用

### 前提条件

- Python 3.8+
- Docker（可选，用于容器化部署）
- 云计算资源（如AWS或GCP）

### 安装步骤

1. 克隆仓库：
    ```bash
    git clone https://github.com/wlvh/RBT.git
    cd RBT/RL
    ```

2. 创建并激活虚拟环境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate  # Windows
    ```

3. 安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

4. 运行项目：
    ```bash
    python main.py
    ```

### 使用示例

详细的使用说明和示例代码请参考 [Wiki](https://github.com/wlvh/RBT/wiki) 或 [示例教程](https://github.com/wlvh/RBT/tree/main/RL/examples)。

## 项目链接

- **GitHub 仓库**: [https://github.com/wlvh/RBT/tree/main/RL](https://github.com/wlvh/RBT/tree/main/RL)

## 成就与成果

- 成功生成并扩充了高质量的金融决策思维链数据集，显著提升了金融决策支持系统的性能。
- 通过双轨模型优化策略，提升LLM在复杂金融决策推理任务中的准确性和一致性。
- 实现了无监督的数据扩充方法，大幅减少了数据准备的时间和人力成本。

## 贡献

欢迎贡献！请阅读 [贡献指南](https://github.com/wlvh/RBT/blob/main/CONTRIBUTING.md) 了解如何参与。

## 许可证

本项目采用 [MIT 许可证](https://github.com/wlvh/RBT/blob/main/LICENSE)。

## 联系方式

如有任何问题或建议，请通过 [Issues](https://github.com/wlvh/RBT/issues) 联系我们。

---
