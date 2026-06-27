# 四代龙珠 (Loong-pearl) - 连续神经场机

> 📚 **龙珠进化全貌**: 本系统为四代龙珠，是实验前沿项目。
> 完整进化历程: [`一代龙神.md`](../一代龙神.md) → [`二代龙珠.md`](../二代龙珠.md) → [`三代龙珠.md`](../三代龙珠.md) → **四代龙珠 (本项目)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-orange.svg)](https://pytorch.org/)

## 项目简介

四代龙珠是一个基于**连续神经场理论**的认知系统，实现了完全去模块化的认知架构。该项目探索了一种全新的认知计算范式，突破了传统模块化架构的局限。

### 核心特点

| 特性 | 说明 |
|------|------|
| 🌊 **无模块边界** | 所有认知活动都是同一个场的自组织波动 |
| ✨ **语义涌现** | 语义原子从场的自组织中自然涌现，无需人工定义 |
| 🔄 **连续学习** | 场的运行、学习、演化是同一个持续过程 |
| 🧠 **内在驱动** | 场自带好奇心，基于信息熵梯度的探索机制 |

### 与传统架构对比

| 传统架构 | 四代龙珠 |
|---------|---------|
| 模块化设计（感知→认知→决策） | 统一连续场 |
| 离线训练 + 在线推理 | 持续演化学习 |
| 人工定义符号系统 | 语义原子自然涌现 |
| 外部任务驱动 | 内在好奇心驱动 |

## 核心架构

系统由七个核心组件构成，形成完整的认知场：

### 1. 液态时间常数网络 (Liquid Time Constant Network)

核心场状态演化引擎，实现连续时间动力学。

```python
# 配置参数
field_dim: 4096        # 场状态维度
hidden_dim: 4096       # 隐藏层维度
tau_max: 1.0          # 最大时间常数
dt: 0.1               # 时间步长(秒)
```

**架构细节**：
- 2层MLP结构：4096 → 4096 → 4096
- GELU激活函数
- 可学习的时间常数向量，每个维度独立演化速率
- 状态裁剪防止数值爆炸

### 2. 语义原子管理 (Semantic Atom Manager)

从语料中自动涌现语义原子，实现无监督语义发现。

```python
# 配置参数
atom_dim: 128          # 原子向量维度
initial_atoms: 5000    # 初始原子数量
pmi_window_size: 5     # PMI窗口大小
pmi_threshold: 2.0     # PMI阈值
```

**核心算法**：
- PMI (Pointwise Mutual Information) 统计计算
- Louvain社区检测算法聚类
- 动态分裂/合并机制
- 成语/词组特殊处理

### 3. Hebbian学习 (Hebbian Learning)

实现"同时激活→连接增强"的生物可塑性学习规则。

```python
# 配置参数
learning_rate: 1.0e-5     # 学习率
decay_rate: 1.0e-6        # 权重衰减率
activation_threshold: 0.7  # 激活阈值
max_density: 0.05         # 最大连接密度
```

**学习规则**：
- Hebbian增强：`Δw = lr * pre * post`
- 自然衰减：`w *= (1 - decay_rate)`
- 稀疏矩阵存储，初始密度0.1%
- 自动剪枝过小权重

### 4. 好奇心驱动 (Curiosity Drive)

基于信息熵的内在探索机制，实现自主学习。

```python
# 配置参数
entropy_low_threshold: 2.0   # 低熵阈值
entropy_high_threshold: 5.0  # 高熵阈值
exploration_noise_std: 0.05  # 探索噪声
```

**驱动机制**：
- 实时监控场状态信息熵
- 局部熵异常检测
- 自动探索未知区域
- 学习率自适应调整

### 5. 场接口 (Field Interface)

输入输出的薄层投影，实现文本与场状态的转换。

```python
# 配置参数
projection_top_k: 5           # Top-K投影
projection_update_interval: 10 # 更新间隔
```

**转换流程**：
- 输入：文本 → 分词 → 语义原子 → 场区域扰动
- 输出：激活模式 → Top-K原子 → 文本解码

### 6. 守护进程 (Field Guardian)

系统主循环管理器，协调所有组件运行。

```python
# 配置参数
checkpoint_interval: 1000    # 检查点间隔
atom_evolve_interval: 2000   # 原子演化间隔
max_steps: 100000           # 最大运行步数
```

**职责**：
- 持续运行循环管理
- 自动检查点保存/恢复
- 输入处理与演化调度
- 搜索驱动桥接

### 7. 搜索驱动 (Search Driver)

好奇心触发时的外部知识获取机制。

**工作流程**：
1. 好奇心检测到未知区域
2. 提取关键词
3. 调用搜索API/知识库
4. 结果注入场中学习

## 知识注入

### ✓ 已导入知识数据

语料库 `data/corpus.txt` 已从 Loong-agent 项目导入：

| 指标 | 数值 |
|------|------|
| 总行数 | 4,559,886 行 |
| 文件大小 | 541 MB |
| 四字词 | ~100,000 条 |
| 成语 | ~50,000 条 |
| 中英词典 | ~200,000 条 |
| 汉字字典 | ~20,000 条 |
| 维基百科 | ~4,000,000 条 |

详细说明见：
- `DATA_IMPORT.md` - 数据导入流程
- `KNOWLEDGE_INJECTION.md` - 知识注入指南

### 自定义搜索驱动

```python
def my_search(keywords):
    """自定义搜索回调函数"""
    # 示例：调用搜索API
    results = search_api.search(keywords)
    return results

# 设置搜索回调
guardian.set_search_callback(my_search)
```

### 添加自定义知识

```python
# 方式1：直接添加语料
with open('data/corpus.txt', 'a', encoding='utf-8') as f:
    f.write("新的知识内容\n")

# 方式2：通过场接口注入
interface.inject_knowledge("新知识文本", field)
```

## 安装

### 环境要求

- Python 3.12+
- CUDA 11.8+ (推荐，支持GPU加速)
- 8GB+ RAM
- 5GB+ 磁盘空间

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/your-username/Loong-pearl.git
cd Loong-pearl

# 安装依赖
pip install -r requirements.txt

# 验证安装
python verify.py
```

### 依赖包

核心依赖（见 `requirements.txt`）：
- `torch==2.1.0` - 深度学习框架
- `numpy==1.24.3` - 数值计算
- `scipy==1.11.3` - 科学计算
- `scikit-learn==1.3.2` - 机器学习工具
- `jieba==0.42.1` - 中文分词
- `python-louvain==0.16` - 社区检测
- `networkx==3.2` - 图算法
- `pyyaml==6.0.1` - 配置解析

## 快速开始

### 1. 准备语料

创建 `data/corpus.txt` 文件，每行一段文本：

```text
龙飞凤舞
龙腾虎跃
春暖花开
...
```

### 2. 运行测试

```bash
# 核心组件测试
python tests/test_core.py

# 语言生成器测试
python test_language_generator.py

# 优化系统测试
python test_optimized_system.py
```

### 3. 训练模型

**完整训练**（推荐）：
```bash
python train.py --corpus data/corpus.txt
```

**分阶段训练**：
```bash
# 阶段1：验证液态网络
python train.py --phase 1

# 阶段2：语义原子生成与Hebbian学习
python train.py --phase 2 --corpus data/corpus.txt

# 阶段3：输入输出测试
python train.py --phase 3
```

### 4. 交互使用

```python
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.semantic_atoms import SemanticAtomManager
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian

# 初始化系统（使用默认配置）
field = LiquidTimeConstantNetwork(field_dim=4096)
hebbian = HebbianUpdater(field_dim=4096)
curiosity = CuriosityDrive(field_dim=4096)
semantic_atoms = SemanticAtomManager(field_dim=4096)
interface = FieldInterface(field_dim=4096)

# 创建守护进程
guardian = FieldGuardian(
    field=field,
    hebbian_updater=hebbian,
    curiosity_drive=curiosity,
    semantic_atoms=semantic_atoms,
    field_interface=interface
)

# 处理输入
question = "什么是龙"
answer = guardian.process_input(question)
print(answer)

# 持续运行（后台演化）
guardian.run(max_steps=10000)
```

### 5. 性能基准

```bash
# 运行性能测试
python benchmark.py

# 输出示例：
# - 单步演化时间: 0.12ms
# - 内存占用: 75MB
# - 吞吐量: 8333 steps/s
```

## 项目结构

```
Loong-pearl/
├── src/                           # 源代码
│   ├── core/                      # 核心组件
│   │   ├── liquid_time_constant.py    # 液态时间常数网络
│   │   ├── hebbian_learning.py        # Hebbian学习
│   │   ├── curiosity_drive.py         # 好奇心驱动
│   │   ├── semantic_atoms.py          # 语义原子管理
│   │   ├── field_interface.py         # 场接口
│   │   └── field_guardian.py          # 守护进程
│   ├── utils/                         # 工具函数
│   └── data/                          # 数据处理
├── tests/                         # 测试套件
│   └── test_core.py                   # 核心组件测试
├── data/                          # 数据目录
│   ├── corpus.txt                     # 训练语料
│   └── raw/                           # 原始知识数据
├── checkpoints/                   # 检查点存储
├── logs/                          # 日志文件
├── docs/                          # 文档
├── experiments/                   # 实验脚本
├── config.yaml                    # 配置文件
├── train.py                       # 训练脚本
├── run.py                         # 运行脚本
├── benchmark.py                   # 性能测试
├── verify.py                      # 安装验证
└── requirements.txt               # 依赖包
```

### 关键文件说明

| 文件 | 用途 |
|------|------|
| `config.yaml` | 全局配置参数 |
| `train.py` | 模型训练入口 |
| `run.py` | 交互运行入口 |
| `benchmark.py` | 性能基准测试 |
| `verify.py` | 环境验证脚本 |

## 配置参数

完整配置见 `config.yaml`，核心参数说明：

### 场状态参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `field_dim` | 4096 | 场状态维度 |
| `hidden_dim` | 4096 | 隐藏层维度 |
| `tau_max` | 1.0 | 最大时间常数 |
| `dt` | 0.1 | 时间步长(秒) |
| `noise_std` | 0.01 | 噪声标准差 |
| `state_clip` | 5.0 | 状态裁剪阈值 |

### 语义原子参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `atom_dim` | 128 | 原子向量维度 |
| `initial_atoms` | 5000 | 初始原子数量 |
| `pmi_window_size` | 5 | PMI窗口大小 |
| `pmi_threshold` | 2.0 | PMI阈值 |
| `max_idiom_atoms` | 1000 | 最大成语原子数 |

### 学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-5 | Hebbian学习率 |
| `decay_rate` | 1e-6 | 权重衰减率 |
| `activation_threshold` | 0.7 | 激活阈值 |
| `max_density` | 0.05 | 最大连接密度 |

### 好奇心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `entropy_low_threshold` | 2.0 | 低熵阈值 |
| `entropy_high_threshold` | 5.0 | 高熵阈值 |
| `exploration_noise_std` | 0.05 | 探索噪声强度 |

## 性能指标

### 资源占用

| 指标 | 数值 | 说明 |
|------|------|------|
| 显存占用 | ~75MB | 场状态8KB + 网络参数67MB |
| 内存占用 | ~200MB | 语义原子 + Hebbian矩阵 |
| 单步计算 | ~0.1ms | 稀疏矩阵乘法 |
| 检查点大小 | ~100MB | 完整状态保存 |

### 吞吐量

| 硬件配置 | 单步耗时 | 吞吐量 |
|---------|---------|--------|
| RTX 3060 12GB | 0.12ms | 8333 steps/s |
| RTX 4090 24GB | 0.08ms | 12500 steps/s |
| CPU Only | 1.5ms | 667 steps/s |

### 硬件要求

- **最低配置**：RTX 3060 12GB / 16GB RAM
- **推荐配置**：RTX 4090 24GB / 32GB RAM
- **CPU模式**：支持，但性能降低10倍

## 设计原则

### 核心理念

1. **连续性优先**
   - 场持续演化，无离线训练阶段
   - 学习与推理统一为同一过程
   - 实时响应，实时学习

2. **自组织涌现**
   - 语义原子自然涌现，无需人工定义
   - 知识结构自发生成
   - 适应性强，泛化能力高

3. **内在驱动**
   - 好奇心驱动探索，不依赖外部指令
   - 主动学习未知区域
   - 持续自我优化

4. **退化设计**
   - 输入输出退化为薄层皮肤
   - 核心认知在场中完成
   - 接口简洁，易于扩展

### 理论基础

- **连续神经场理论**：将认知建模为连续场动力学
- **Hebbian可塑性**：生物启发的学习规则
- **信息熵驱动**：内在动机的形式化
- **复杂系统理论**：自组织与涌现机制

## 开发路线

### 已完成 ✓

- [x] **阶段一**：液态时间常数网络实现
  - 连续时间动力学
  - 可学习时间常数
  - 状态稳定性保证

- [x] **阶段二**：语义原子生成与Hebbian学习
  - PMI统计与聚类
  - 稀疏Hebbian矩阵
  - 动态演化机制

- [x] **阶段三**：输入输出接口测试
  - 文本-场转换
  - 语义映射
  - 问答测试

- [x] **阶段四**：知识注入系统
  - 大规模语料导入
  - 成语/词组处理
  - 搜索驱动集成

### 进行中 🚧

- [ ] **性能优化**
  - GPU加速优化
  - 内存使用优化
  - 混合精度训练

- [ ] **语言生成器**
  - 流式生成
  - 多样性控制
  - 质量评估

### 计划中 📋

- [ ] **大规模训练**
  - 百万级语料训练
  - 分布式训练支持
  - 增量学习机制

- [ ] **应用部署**
  - API服务封装
  - Web界面开发
  - 模型压缩部署

- [ ] **理论研究**
  - 涌现机制分析
  - 理论边界探索
  - 论文撰写

## 常见问题

### Q: 为什么选择连续场而非模块化架构？

A: 连续场架构具有以下优势：
- 无需人工定义模块边界
- 语义自然涌现，泛化能力强
- 学习与推理统一，无迁移损失
- 内在驱动，持续自我优化

### Q: 语义原子如何涌现？

A: 通过三步流程：
1. PMI统计计算词共现模式
2. Louvain聚类发现语义社区
3. 动态演化调整原子结构

### Q: 系统需要多少训练数据？

A: 
- 最小：10,000行语料即可运行
- 推荐：100,000行以上效果较好
- 当前：4,559,886行（已导入）

### Q: 如何调整好奇心强度？

A: 修改 `config.yaml` 中的参数：
```yaml
entropy_low_threshold: 2.0   # 降低=更敏感
entropy_high_threshold: 5.0  # 提高=更探索
exploration_noise_std: 0.05  # 增大=更随机
```

### Q: 支持哪些语言？

A: 当前主要支持中文，英文支持有限。计划扩展多语言支持。

## 相关文档

- [快速开始指南](QUICKSTART.md) - 5分钟快速上手
- [数据导入说明](DATA_IMPORT.md) - 语料导入流程
- [知识注入指南](KNOWLEDGE_INJECTION.md) - 知识系统使用
- [性能优化分析](PERFORMANCE_OPTIMIZATION.md) - 优化策略
- [语言生成器说明](LANGUAGE_GENERATOR_README.md) - 生成器使用
- [深度优化报告](DEEP_OPTIMIZATION.md) - 高级优化技术

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 开发流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 使用 Python 3.12+ 特性
- 遵循 PEP 8 编码规范
- 添加类型注解
- 编写单元测试

## 致谢

本项目受以下研究启发：
- 连续神经场理论
- Hebbian学习理论
- 内在动机研究
- 复杂系统与涌现

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 引用

如果您在研究中使用本项目，请引用：

```bibtex
@misc{loong-pearl2024,
  title={Loong-pearl: 连续神经场认知系统},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-username/Loong-pearl}
}
```

---

**Made with ❤️ by the Loong-pearl Team**