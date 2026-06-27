# 深度推理引擎使用指南

## 概述

深度推理引擎实现了多种推理能力，支持逻辑推理和因果推理，可整合概念图谱进行知识增强推理。

## 核心功能

### 1. 演绎推理 (Deductive Reasoning)
从一般到特殊的推理过程。

**示例：苏格拉底三段论**
```python
from src.core.reasoning_engine import ReasoningEngine

engine = ReasoningEngine()

result = engine.deductive_reasoning(
    premises=["所有人都会死亡", "苏格拉底是人"],
    conclusion="苏格拉底会死亡"
)

print(f"推理有效: {result['valid']}")  # True
print(f"置信度: {result['confidence']}")  # 0.90
```

### 2. 归纳推理 (Inductive Reasoning)
从特殊到一般的推理过程。

**示例：乌鸦颜色归纳**
```python
result = engine.inductive_reasoning([
    "乌鸦1是黑色的",
    "乌鸦2是黑色的",
    "乌鸦3是黑色的",
    "乌鸦4是黑色的"
])

print(f"归纳结论: {result['generalization']}")  # 所有乌鸦都是黑色的
print(f"支持例子数: {result['support_count']}")  # 4
```

### 3. 类比推理 (Analogical Reasoning)
基于相似性进行推理。

**示例：水与空气类比**
```python
result = engine.analogical_reasoning(
    source={'entity': '水', 'attributes': ['流动性', '可蒸发', '无色']},
    target={'entity': '空气', 'attributes': ['流动性', '无色']}
)

print(f"推断属性: {result['inferred_attributes']}")  # ['可蒸发']
print(f"相似度: {result['similarity']}")  # 0.67
```

### 4. 因果推理 (Causal Reasoning)
基于概念图谱识别因果关系。

**示例：下雨导致地面湿**
```python
# 提供概念图谱
triples = [
    ("下雨", "CAUSES", "地面湿", 0.90),
    ("地面湿", "CAUSES", "摩擦力降低", 0.85),
    ("摩擦力降低", "CAUSES", "路面滑", 0.88),
]

engine = ReasoningEngine(concept_graph=triples)

result = engine.causal_reasoning("下雨", "地面湿")
print(f"存在因果关系: {result['is_causal']}")  # True
print(f"因果链: {result['causal_chain']}")
```

### 5. 多步推理 (Multi-step Reasoning)
链式推理，逐步推导。

**示例：为什么下雨后路面会滑？**
```python
result = engine.multi_step_reasoning("为什么下雨后路面会滑？", steps=5)

print(f"答案: {result['answer']}")
print(f"推理步数: {result['steps_used']}")
for step in result['reasoning_chain']:
    print(f"  {step}")
```

## 与知识库集成

### 使用 UnifiedKnowledgeManager

```python
from src.data.unified_knowledge_manager import UnifiedKnowledgeManager
from src.core.reasoning_engine import ReasoningEngine

# 加载知识库
km = UnifiedKnowledgeManager(data_dir="data/raw")

# 创建推理引擎（自动加载概念图谱）
engine = ReasoningEngine(knowledge_manager=km)

# 进行推理
result = engine.multi_step_reasoning("什么是人？", steps=3)
```

### 直接提供概念图谱

```python
triples = [
    ("人", "IS_A", "生物", 0.95),
    ("苏格拉底", "IS_A", "人", 0.99),
    ("病毒", "CAUSES", "疾病", 0.95),
]

engine = ReasoningEngine(concept_graph=triples)
```

## 推理链管理器

用于构建复杂的推理链：

```python
from src.core.reasoning_engine import ReasoningChain

chain = ReasoningChain()

chain.add_step(
    premise="所有人都会死亡",
    conclusion="苏格拉底会死亡",
    rule="演绎推理",
    confidence=0.9
)

chain.add_step(
    premise="苏格拉底会死亡",
    conclusion="苏格拉底终将死亡",
    rule="等价转换",
    confidence=0.95
)

print(f"最终结论: {chain.get_final_conclusion()}")
print(f"综合置信度: {chain.confidence}")  # 0.9 * 0.95 = 0.855
```

## 概念图谱格式

概念图谱使用三元组格式：`(主语, 关系, 宾语, 置信度)`

### 常用关系类型

- **分类关系**: `IS_A`, `INSTANCE_OF`, `属于`
- **因果关系**: `CAUSES`, `LEADS_TO`, `RESULTS_IN`, `导致`
- **属性关系**: `HAS_PROPERTY`, `HAS_ATTRIBUTE`
- **部分关系**: `PART_OF`, `HAS_PART`
- **症状关系**: `HAS_SYMPTOM`, `INDICATES`

### 示例

```python
concept_graph = [
    # 分类关系
    ("苏格拉底", "IS_A", "人", 0.99),
    ("人", "IS_A", "生物", 0.95),
    
    # 因果关系
    ("下雨", "CAUSES", "地面湿", 0.90),
    ("病毒", "CAUSES", "疾病", 0.95),
    
    # 属性关系
    ("乌鸦", "HAS_PROPERTY", "黑色", 0.92),
    ("水", "HAS_PROPERTY", "流动性", 0.95),
    
    # 部分关系
    ("轮子", "PART_OF", "汽车", 0.99),
]
```

## 置信度计算

- **演绎推理**: 基于规则匹配，固定置信度 0.9
- **归纳推理**: 基于支持例子数量，`min(0.95, 0.5 + count * 0.05)`
- **类比推理**: 基于相似度，`similarity * 0.8`
- **因果推理**: 直接使用概念图谱中的置信度
- **多步推理**: 链式置信度，逐步相乘

## 性能优化

推理引擎使用索引加速查询：

```python
# 自动构建索引
engine = ReasoningEngine(concept_graph=triples)

# 索引结构
engine._relation_index  # {关系: [(主语, 宾语, 置信度), ...]}
engine._concept_index   # {概念: [(关系, 相关概念, 置信度), ...]}
```

## 应用场景

1. **知识问答**: 基于概念图谱回答问题
2. **因果分析**: 分析事件因果关系链
3. **智能推理**: 多步推理解决复杂问题
4. **知识发现**: 归纳发现潜在规律
5. **类比迁移**: 基于相似性进行知识迁移

## 测试

运行测试套件：

```bash
python tests/test_reasoning_standalone.py
```

## 接口参考

### ReasoningEngine

```python
class ReasoningEngine:
    def __init__(self, knowledge_manager=None, concept_graph=None)
    
    def deductive_reasoning(self, premises: List[str], conclusion: str) -> Dict
    def inductive_reasoning(self, examples: List[str]) -> Dict
    def analogical_reasoning(self, source: Dict, target: Dict) -> Dict
    def causal_reasoning(self, cause: str, effect: str) -> Dict
    def multi_step_reasoning(self, question: str, steps: int = 5) -> Dict
```

### ReasoningChain

```python
class ReasoningChain:
    def add_step(self, premise: str, conclusion: str, rule: str, confidence: float)
    def get_final_conclusion(self) -> str
    def to_dict(self) -> Dict
```