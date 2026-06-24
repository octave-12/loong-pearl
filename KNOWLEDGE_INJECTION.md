# 四代龙珠 - 知识注入指南

## 一、语料数据准备（必需）

### 1. 语料作用
- 计算PMI字对，生成初始语义原子
- 提供共现模式，让Hebbian学习形成语义连接
- 语料规模直接影响语义原子质量和系统表现

### 2. 推荐语料源

| 语料类型 | 获取方式 | 预计规模 |
|---------|---------|---------|
| 中文维基百科 | https://dumps.wikimedia.org/zhwiki/ | 100万+词条 |
| 全唐诗 | GitHub: chinese-poetry | 4.8万首 |
| 宋词 | GitHub: chinese-poetry | 2万首 |
| 成语词典 | GitHub: chinese-xinhua | 3万+ |
| 现代散文 | 自行收集 | 建议10万+行 |
| 对话语料 | 开源对话数据集 | 建议50万+行 |

### 3. 语料格式

每行一段文本（不需要分词，系统会自动用jieba分词）：

```
龙飞凤舞
龙腾虎跃
春暖花开
...
```

### 4. 语料合并脚本

```python
import os

corpus_dir = "data/raw/"
output_file = "data/corpus.txt"

with open(output_file, 'w', encoding='utf-8') as out:
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 5:  # 过滤太短的行
                        out.write(line + '\n')

print(f"语料已合并到 {output_file}")
```

### 5. 语料规模建议

| 规模 | 语义原子数 | 效果 |
|-----|-----------|------|
| 1万行 | ~500 | 基础演示 |
| 10万行 | ~2000 | 初步可用 |
| 100万行 | ~5000 | 推荐 |
| 1000万行+ | ~10000 | 最佳 |

---

## 二、搜索驱动配置（可选）

### 1. 搜索驱动作用

当好奇心驱动标记"值得探索"的原子时，自动触发外部搜索，将结果注入场中。

### 2. 简单示例：本地知识库搜索

```python
import json

# 加载本地知识库
with open('data/knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)  # {"关键词": "相关知识文本", ...}

def local_search_callback(keywords: list) -> list:
    results = []
    for kw in keywords:
        if kw in knowledge_base:
            results.append(knowledge_base[kw])
    return results

guardian.set_search_callback(local_search_callback)
```

### 3. 进阶示例：调用搜索API

```python
import requests

def web_search_callback(keywords: list) -> list:
    results = []
    for kw in keywords[:3]:  # 限制搜索次数
        try:
            # 示例：调用必应搜索API
            response = requests.get(
                "https://api.bing.microsoft.com/v7.0/search",
                headers={"Ocp-Apim-Subscription-Key": "YOUR_API_KEY"},
                params={"q": kw, "count": 3}
            )
            data = response.json()
            for item in data.get("webPages", {}).get("value", []):
                results.append(item["snippet"])
        except Exception as e:
            print(f"搜索失败: {e}")
    return results

guardian.set_search_callback(web_search_callback)
```

### 4. 知识库格式

```json
{
  "龙": "龙是中国传统文化中的神兽，象征着权力、尊贵和吉祥...",
  "春天": "春天是四季之一，万物复苏，生机勃勃...",
  "水": "水是生命之源，在中国文化中象征智慧和柔韧..."
}
```

---

## 三、预训练嵌入（可选）

### 1. 作用

用预训练字嵌入初始化语义原子，可加速初期收敛，但非必需。
系统默认使用随机初始化，通过Hebbian演化自动调优。

### 2. 推荐嵌入

| 嵌入 | 维度 | 获取方式 |
|-----|------|---------|
| BAAI/bge-large-zh | 1024 | HuggingFace |
| Tencent AI Lab | 200 | 官方开源 |
| 字向量 | 300 | GitHub: wordvectors |

### 3. 使用示例

```python
import numpy as np

# 加载预训练字嵌入
char_embeddings = {}
with open('data/char_vectors.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        char = parts[0]
        vector = np.array([float(x) for x in parts[1:]])
        char_embeddings[char] = vector

# 修改 SemanticAtomManager 初始化
def initialize_atoms_from_clusters_with_embeddings(self, clusters, char_embeddings):
    for cluster in clusters:
        # 使用簇内字嵌入的平均值
        embeddings = [char_embeddings.get(c) for c in cluster if c in char_embeddings]
        if embeddings:
            atom_embedding = np.mean(embeddings, axis=0)
        else:
            atom_embedding = np.random.randn(self.atom_dim) * 0.1
        # ... 创建原子
```

---

## 四、完整初始化流程

```python
from src.utils.config import Config

# 1. 加载配置
config = Config.get("config.yaml")

# 2. 创建系统组件
field = config.create_field()
hebbian = config.create_hebbian()
curiosity = config.create_curiosity()
semantic_atoms = config.create_semantic_atoms()
interface = config.create_interface()

# 3. 加载语料并生成语义原子
corpus = []
with open("data/corpus.txt", 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f if line.strip()]

pmi_pairs = semantic_atoms.compute_pmi(corpus, window_size=5, min_count=10, pmi_threshold=2.0)
clusters = semantic_atoms.cluster_characters(pmi_pairs)
semantic_atoms.initialize_atoms_from_clusters(clusters)
print(f"生成 {semantic_atoms.get_num_atoms()} 个语义原子")

# 4. 创建守护进程
guardian = config.create_guardian(field, hebbian, curiosity, semantic_atoms, interface)

# 5. 设置搜索回调（可选）
def my_search(keywords):
    return [f"关于{kw}的知识..." for kw in keywords]
guardian.set_search_callback(my_search)

# 6. 启动系统
guardian.run(max_steps=100000)
```

---

## 五、知识注入最佳实践

### 1. 语料质量 > 数量
- 优先使用高质量、语义丰富的文本
- 避免重复、噪声过多的数据

### 2. 领域适配
- 如果系统用于特定领域（如医学、法律），添加领域语料
- 领域语料权重可高于通用语料

### 3. 持续注入
- 系统支持运行时动态添加语料
- 通过搜索驱动持续注入新知识

### 4. 检查点管理
- 定期保存检查点
- 不同语料配置使用不同检查点目录

---

## 六、当前系统状态

- **语料**: `data/corpus.txt` (196行成语) - **需要扩充**
- **语义原子**: 将从PMI统计中自动生成
- **搜索驱动**: 已实现，需要设置回调函数
- **预训练嵌入**: 未使用（使用随机初始化，符合需求）

**建议下一步**：
1. 准备大规模中文语料（100万行+）
2. 替换 `data/corpus.txt`
3. 运行 `python train.py` 训练
4. 可选：设置搜索回调增强知识获取