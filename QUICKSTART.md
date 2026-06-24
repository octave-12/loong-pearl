# 四代龙珠 - 快速开始指南

## 项目已实现完成 ✓

### 核心架构实现

1. **液态时间常数网络** (`src/core/liquid_time_constant.py`)
   - 4096维场状态向量
   - 2层MLP (4096→4096→4096) + GELU激活
   - 可学习时间常数向量τ
   - 连续演化: h(t+1) = h(t) + f(h,u)*τ*dt + η

2. **Hebbian连续学习** (`src/core/hebbian_learning.py`)
   - 稀疏矩阵存储 (初始密度0.1%)
   - 同时激活→连接增强
   - 久不激活→自然衰减
   - 休眠神经元加速衰减

3. **好奇心驱动** (`src/core/curiosity_drive.py`)
   - 信息熵监控 (64桶统计)
   - 局部熵异常检测
   - 自动探索机制
   - 学习率自适应

4. **语义原子管理** (`src/core/semantic_atoms.py`)
   - PMI字对提取
   - Louvain社区聚类
   - 动态分裂/合并
   - 5000-10000个原子

5. **场接口** (`src/core/field_interface.py`)
   - 文本→扰动编码
   - 激活→文本解码
   - 退化投影层设计

6. **守护进程** (`src/core/field_guardian.py`)
   - 持续演化循环
   - 输入处理管理
   - 自动检查点保存

### 安装依赖

```bash
# 基础依赖
pip install numpy scipy scikit-learn tqdm jieba python-louvain

# PyTorch (CPU版本，避免DLL问题)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 或GPU版本
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 使用方式

#### 1. 演示模式
```bash
python run.py --mode demo
```
使用简化配置(1024维)快速演示系统功能

#### 2. 交互模式
```bash
python run.py --mode interactive
```
完整4096维系统，支持:
- 文本问答交互
- 持续演化模式
- 检查点保存/加载

#### 3. 完整训练
```bash
# 三阶段训练
python train.py --corpus data/corpus.txt

# 分阶段训练
python train.py --phase 1  # 液态网络验证
python train.py --phase 2  # 语义原子生成
python train.py --phase 3  # 接口测试
```

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| field_dim | 4096 | 场状态维度 |
| atom_dim | 128 | 语义原子维度 |
| learning_rate | 1e-5 | Hebbian学习率 |
| decay_rate | 1e-6 | 权重衰减率 |
| dt | 0.1s | 时间步长 |
| tau_max | 1.0 | 最大时间常数 |

### 性能指标

- **显存**: ~75MB (状态8KB + 参数67MB)
- **计算**: ~0.1ms/步
- **存储**: ~100MB/检查点
- **硬件**: RTX 3060 12GB可运行

### 项目结构

```
Loong-pearl/
├── src/core/              # 核心实现
│   ├── liquid_time_constant.py   # 液态网络
│   ├── hebbian_learning.py       # Hebbian学习
│   ├── curiosity_drive.py        # 好奇心驱动
│   ├── semantic_atoms.py         # 语义原子
│   ├── field_interface.py        # 输入输出接口
│   └── field_guardian.py         # 守护进程
├── tests/                 # 测试
├── data/corpus.txt        # 语料
├── train.py              # 训练脚本
├── run.py                # 运行脚本
└── verify.py             # 验证脚本
```

### 设计亮点

1. **完全去模块化**: 无解义器、策应器等独立模块
2. **语义涌现**: 不使用人工定义的汉字锚点
3. **连续统一**: 训练与推理是同一过程
4. **内在驱动**: 基于信息熵梯度的自探索
5. **退化设计**: 输入输出退化为薄层皮肤

### 下一步

1. 准备大规模中文语料
2. 运行长期演化训练
3. 观察语义盆地形成
4. 测试问答质量提升
5. 部署实际应用

---

**项目状态**: ✓ 核心实现完成，可开始训练使用