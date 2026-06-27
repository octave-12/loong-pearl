"""数据使用情况分析"""
import os
from pathlib import Path
import json

print("=" * 70)
print("数据使用情况分析")
print("=" * 70)

# 1. Loong-pearl当前使用的数据
print("\n[1] Loong-pearl 当前使用的数据")
print("-" * 70)

pearl_data = {
    'data/corpus.txt': {
        'desc': '语料库',
        'size': '542 MB',
        'lines': '456万行',
        'used': True,
        'usage': 'PMI计算、语义原子初始化'
    },
    'data/raw/zhwiki.db': {
        'desc': '维基百科数据库',
        'size': '4.02 GB',
        'used': False,
        'usage': '未使用'
    },
    'data/raw/cedict_parsed.json': {
        'desc': '汉英词典',
        'size': '19 MB',
        'used': False,
        'usage': '未使用'
    },
    'data/raw/idioms.json': {
        'desc': '成语词典',
        'size': '461 KB',
        'used': False,
        'usage': '未使用'
    },
    'data/raw/dict_unihan.json': {
        'desc': 'Unicode汉字数据',
        'size': '2.7 MB',
        'used': False,
        'usage': '未使用'
    },
    'data/raw/dict_decompose.json': {
        'desc': '汉字分解数据',
        'size': '2 MB',
        'used': False,
        'usage': '未使用'
    },
    'data/raw/four_char_words.txt': {
        'desc': '四字词',
        'size': '243 KB',
        'used': False,
        'usage': '未使用'
    },
    'data/raw/gb2312_level1.txt': {
        'desc': 'GB2312一级字库',
        'size': '11 KB',
        'used': False,
        'usage': '未使用'
    },
    'data/raw/directed_pairs.json': {
        'desc': '有向词对',
        'size': '332 KB',
        'used': False,
        'usage': '未使用'
    }
}

for path, info in pearl_data.items():
    status = '✅' if info['used'] else '❌'
    print(f"{status} {path}")
    print(f"   描述: {info['desc']}")
    print(f"   大小: {info['size']}")
    print(f"   用途: {info['usage']}")

# 2. Loong-agent的数据
print("\n\n[2] Loong-agent 可用的数据")
print("-" * 70)

agent_data = {
    'dicts/idioms.json': {
        'desc': '成语词典',
        'size': '~500 KB',
        'type': '语言知识',
        'priority': '高'
    },
    'dicts/cedict_parsed.json': {
        'desc': '汉英词典',
        'size': '~19 MB',
        'type': '语言知识',
        'priority': '高'
    },
    'dicts/dict_unihan.json': {
        'desc': 'Unicode汉字数据',
        'size': '~2.7 MB',
        'type': '语言知识',
        'priority': '中'
    },
    'dicts/dict_decompose.json': {
        'desc': '汉字分解数据',
        'size': '~2 MB',
        'type': '语言知识',
        'priority': '中'
    },
    'wordlists/four_char_words.txt': {
        'desc': '四字词表',
        'size': '~243 KB',
        'type': '语言知识',
        'priority': '中'
    },
    'wordlists/gb2312_level1.txt': {
        'desc': 'GB2312一级字库',
        'size': '~11 KB',
        'type': '语言知识',
        'priority': '低'
    },
    'wikipedia/zhwiki.db': {
        'desc': '维基百科数据库',
        'size': '~4 GB',
        'type': '世界知识',
        'priority': '高'
    },
    'models/dragon_field_patterns.pt': {
        'desc': '龙场模式',
        'size': '~数MB',
        'type': '训练模型',
        'priority': '高'
    },
    'models/concept_graph.db': {
        'desc': '概念图谱',
        'size': '~数MB',
        'type': '知识图谱',
        'priority': '高'
    },
    'models/energy_landscape_1024d.pt': {
        'desc': '能量景观模型',
        'size': '~数MB',
        'type': '训练模型',
        'priority': '高'
    },
    'models/directed_pairs.json': {
        'desc': '有向词对',
        'size': '~332 KB',
        'type': '语言知识',
        'priority': '中'
    }
}

for path, info in agent_data.items():
    print(f"📁 {path}")
    print(f"   描述: {info['desc']}")
    print(f"   大小: {info['size']}")
    print(f"   类型: {info['type']}")
    print(f"   优先级: {info['priority']}")

# 3. 数据整合建议
print("\n\n[3] 数据整合建议")
print("-" * 70)

suggestions = """
优先级排序：

【高优先级 - 立即整合】
1. ✅ 成语词典 (idioms.json)
   - 用途: 语义原子初始化、文化知识注入
   - 方法: 加载成语及其释义，作为高权重语义原子
   
2. ✅ 概念图谱 (concept_graph.db)
   - 用途: 结构化知识、关系推理
   - 方法: 转換为语义原子的连接关系
   
3. ✅ 维基百科 (zhwiki.db)
   - 用途: 世界知识、实体关系
   - 方法: 提取摘要、建立实体-概念映射
   
4. ✅ 汉英词典 (cedict_parsed.json)
   - 用途: 跨语言知识、词义消歧
   - 方法: 建立中英词对映射

【中优先级 - 逐步整合】
5. ⚠️ 汉字分解 (dict_decompose.json)
   - 用途: 字形结构知识
   - 方法: 建立偏旁部首关系
   
6. ⚠️ 四字词表 (four_char_words.txt)
   - 用途: 固定搭配、成语识别
   - 方法: 作为PMI高权重词对
   
7. ⚠️ 有向词对 (directed_pairs.json)
   - 用途: 语义方向、因果关系
   - 方法: 建立有向语义关系

【低优先级 - 长期规划】
8. ⏳ Unicode汉字数据 (dict_unihan.json)
   - 用途: 字符属性、读音信息
   
9. ⏳ GB2312字库 (gb2312_level1.txt)
   - 用途: 常用字范围限定
"""

print(suggestions)

# 4. 缺少的知识数据
print("\n[4] 系统缺少的知识数据")
print("-" * 70)

missing_data = """
当前缺失的重要知识数据：

【语言知识】
❌ 同义词/反义词词典
   - 用途: 语义相似度、对立关系
   
❌ 词性标注数据
   - 用途: 语法结构、词类信息
   
❌ 依存句法数据
   - 用途: 句法结构、依存关系
   
❌ 词向量预训练模型
   - 用途: 语义嵌入初始化
   - 建议: 使用word2vec/fasttext中文模型

【世界知识】
❌ 常识知识库 (如CN-DBpedia)
   - 用途: 实体属性、常识推理
   
❌ 时事新闻数据
   - 用途: 时效性知识、热点追踪
   
❌ 领域知识库 (医疗、法律、金融等)
   - 用途: 专业领域知识

【对话知识】
❌ 对话语料库
   - 用途: 对话模式学习
   - 建议: 使用开源对话数据集
   
❌ 问答对数据
   - 用途: QA能力训练
   - 建议: 使用百度知道、知乎问答等

【多模态知识】
❌ 图像-文本对数据
   - 用途: 视觉语义对齐
   
❌ 语音-文本对数据
   - 用途: 语音识别/合成

【时序知识】
❌ 时间序列事件
   - 用途: 时序推理、因果关系
   
❌ 历史事件数据
   - 用途: 历史知识、时序推理
"""

print(missing_data)

# 5. 数据使用统计
print("\n[5] 数据使用统计")
print("-" * 70)

stats = """
┌──────────────────────────────────────────────────────────┐
│ 数据类型       │ Loong-pearl │ Loong-agent │ 整合建议  │
├──────────────────────────────────────────────────────────┤
│ 语料库         │ ✅ 542MB    │ ❌          │ 已使用    │
│ 维基百科       │ ❌ 4GB      │ ✅ 4GB      │ 需整合    │
│ 成语词典       │ ❌          │ ✅ 500KB    │ 需整合    │
│ 汉英词典       │ ❌          │ ✅ 19MB     │ 需整合    │
│ 概念图谱       │ ❌          │ ✅ 数MB     │ 需整合    │
│ 训练模型       │ ❌          │ ✅ 数个     │ 可复用    │
│ 汉字数据       │ ❌          │ ✅ 5MB      │ 可选      │
│ 词表           │ ❌          │ ✅ 250KB    │ 可选      │
├──────────────────────────────────────────────────────────┤
│ 总计           │ 542MB       │ ~4.1GB      │ 潜力巨大  │
└──────────────────────────────────────────────────────────┘

数据利用率:
- Loong-pearl: 542MB / 4.6GB = 11.8%
- Loong-agent数据: 未充分利用
- 整合后潜在数据量: ~5GB
"""

print(stats)

# 6. 实施建议
print("\n[6] 实施建议")
print("-" * 70)

implementation = """
分阶段实施计划：

【Phase 1: 基础知识整合】(1-2天)
1. 复制Loong-agent的dicts和wordlists到Loong-pearl
2. 修改semantic_atoms.py加载成语词典
3. 测试成语知识对语义原子的影响

【Phase 2: 结构化知识整合】(3-5天)
1. 整合概念图谱(concept_graph.db)
2. 建立实体-概念-语义原子映射
3. 实现知识图谱查询接口

【Phase 3: 世界知识整合】(1周)
1. 整合维基百科数据
2. 建立实体识别和链接
3. 实现知识检索增强

【Phase 4: 预训练模型复用】(2-3天)
1. 加载dragon_field_patterns.pt
2. 分析与当前模型的兼容性
3. 决定是复用还是重新训练

【Phase 5: 缺失数据补充】(持续)
1. 收集对话语料
2. 构建问答对数据
3. 持续更新知识库
"""

print(implementation)

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)