# 语言生成模块 - 功能说明

## 文件位置
`src/core/language_generator.py`

## 核心功能

### 1. LanguageGenerator 类

基于语义原子的文本生成器，整合了以下核心能力：
- 语义原子激活模式
- PMI关联指导字符选择
- 知识检索增强生成
- 多种采样策略

### 2. 主要方法

#### `generate(prompt, max_length=100, mode='continue')`
通用生成接口，支持三种模式：
- `continue`: 续写模式
- `qa`: 问答模式
- `creative`: 创作模式

#### `generate_with_knowledge(prompt, knowledge_query)`
知识增强生成，先检索相关知识再生成

#### `sample_next_char(context, candidates)`
核心采样方法，支持：
- 温度采样 (temperature)
- Top-k 采样
- Nucleus 采样 (top-p)

#### `beam_search(prompt, beam_width=5)`
束搜索生成，返回多个候选结果

### 3. 核心算法

#### 语义原子激活
```python
def _get_candidate_chars(context, knowledge_context):
    # 1. 从上下文字符的语义原子获取候选
    # 2. 从知识上下文提取字符
    # 3. 添加常用字符
```

#### PMI关联增强
```python
def _apply_pmi_association(context, candidates, scores):
    # 使用预计算的PMI字对增强字符关联
    # 高PMI字对获得更高分数
```

#### 场激活增强
```python
def _apply_field_activation(candidates, scores):
    # 基于场状态的槽位激活度
    # 激活区域的字符获得加成
```

#### 温度采样
```python
def _sample_with_temperature(candidates, scores, temperature, top_k, top_p):
    # 1. 温度缩放: scores / temperature
    # 2. Top-k 过滤
    # 3. Nucleus (top-p) 采样
    # 4. 多项式采样
```

### 4. 生成模式详解

#### 续写模式 (continue)
```
输入: "春风得意"
过程: 
  1. 注入提示到场中
  2. 基于上下文采样字符
  3. 应用PMI关联
  4. 应用场激活
  5. 温度采样
输出: "春风得意马蹄疾..."
```

#### 问答模式 (qa)
```
输入: "什么是量子？"
过程:
  1. 检索知识库
  2. 注入知识上下文
  3. 生成回答
输出: "量子是能量的最小单位..."
```

#### 创作模式 (creative)
```
输入: "春天"
过程:
  1. 动态调整温度（周期性变化）
  2. 应用创造性增强（稀有性奖励）
  3. 生成创意文本
输出: "春风拂面花香浓..."
```

### 5. 知识检索增强

```python
def _retrieve_knowledge(query):
    # 1. 搜索维基百科
    wiki_results = knowledge_manager.search_wiki(query)
    
    # 2. 查询概念图谱
    concept_relations = knowledge_manager.get_concept_relations(query)
    
    # 3. 整合知识上下文
    return knowledge_context
```

### 6. 采样策略

#### 温度采样
- 低温度 (0.3-0.7): 更确定性，保守
- 中温度 (0.8-1.0): 平衡
- 高温度 (1.2-1.5): 更随机，创造性

#### Top-k 采样
保留概率最高的k个候选

#### Nucleus 采样 (Top-p)
保留累积概率达到p的最小候选集

## 使用示例

### 基础使用
```python
from src.core import LanguageGenerator, SemanticAtomManager
from src.data import UnifiedKnowledgeManager

# 初始化
semantic_atoms = SemanticAtomManager(field_dim=512)
knowledge_manager = UnifiedKnowledgeManager()

generator = LanguageGenerator(
    semantic_atoms=semantic_atoms,
    knowledge_manager=knowledge_manager,
    field_dim=512
)

# 续写
result = generator.generate("春风得意", max_length=20, mode='continue')

# 问答
answer = generator.generate("什么是量子？", max_length=30, mode='qa')

# 创作
creation = generator.generate("春天", max_length=30, mode='creative')
```

### 知识增强生成
```python
result = generator.generate_with_knowledge(
    prompt="量子力学研究",
    knowledge_query="量子",
    max_length=30
)
```

### 束搜索
```python
results = generator.beam_search(
    prompt="人工智能",
    beam_width=5,
    max_length=20
)

for text, score in results:
    print(f"{text} (分数: {score:.3f})")
```

### 不同采样策略
```python
# 保守生成
result = generator.generate(prompt, temperature=0.5, top_k=20, top_p=0.8)

# 创造性生成
result = generator.generate(prompt, temperature=1.3, top_k=100, top_p=0.95)
```

## 技术特点

1. **语义原子驱动**: 基于聚类字符的语义单元进行生成
2. **PMI关联**: 利用点互信息指导字符选择
3. **场激活**: 动态场状态影响生成过程
4. **知识增强**: 整合维基百科、概念图谱等知识源
5. **多样采样**: 支持温度、top-k、nucleus等多种采样策略
6. **束搜索**: 提供更优的生成质量

## 性能优化

- PMI缓存: 避免重复计算
- 字符频率缓存: 加速创造性增强
- 场状态复用: 减少重复注入
- 向量化计算: 使用PyTorch加速

## 测试验证

运行验证脚本：
```bash
python verify_language_generator.py
```

完整测试（需安装依赖）：
```bash
pip install -r requirements.txt
python test_language_generator.py
```

## 文件统计

- 总行数: 682
- 代码行数: 541
- 文档字符串: 32
- 文件大小: 23,748 字节

## 依赖模块

- `torch`: 深度学习框架
- `numpy`: 数值计算
- `jieba`: 中文分词
- `SemanticAtomManager`: 语义原子管理
- `UnifiedKnowledgeManager`: 知识管理
- `FieldInterface`: 场接口