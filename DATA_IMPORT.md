# 知识数据导入完成

## 数据源统计

从 Loong-agent 项目复制以下知识数据：

| 数据源 | 文件 | 大小 | 说明 |
|--------|------|------|------|
| 四字词 | `data/raw/four_char_words.txt` | 243 KB | 19,124个四字词 |
| 成语词典 | `data/raw/idioms.json` | 461 KB | 成语列表 |
| 中英词典 | `data/raw/cedict_parsed.json` | 19 MB | CC-CEDICT词典 |
| 汉字字典 | `data/raw/dict_unihan.json` | 2.8 MB | Unihan汉字定义 |
| 维基百科 | `data/raw/zhwiki.db` | 4.0 GB | 中文维基百科数据库 |

## 生成语料库

运行 `scripts/generate_corpus.py` 和 `scripts/extract_wiki.py` 生成：

**`data/corpus.txt`**
- **总行数**: 4,559,886 行
- **文件大小**: 541 MB
- **内容来源**:
  - 四字词: 19,124 条
  - 成语: 23,174 条
  - 中英词典词条: 193,670 条
  - 汉字定义: 22,670 条
  - 维基百科句子: 4,301,248 条（来自10万篇文章）

## 语料质量

语料包含丰富的中文语义信息：
- 成语、四字词：提供固定搭配模式
- 词典定义：提供语义解释
- 维基百科：提供百科知识和自然语言句子

## 下一步

语料库已就绪，可以开始训练：

```bash
# 训练（会自动使用 data/corpus.txt）
python train.py

# 或交互模式
python run.py --mode interactive
```

## 预期效果

- **语义原子数**: 预计生成 3,000-8,000 个
- **PMI字对**: 预计提取 10万+ 高PMI字对
- **训练时间**: 首次PMI计算可能需要几分钟

## 注意事项

1. **显存需求**: 4096维场 + Hebbian稀疏矩阵，建议8GB+显存
2. **训练时长**: 建议运行10万步以上让语义盆地形成
3. **检查点**: 每1000步自动保存，可随时恢复

## 文件结构

```
data/
├── corpus.txt          # 主语料库 (541 MB, 456万行)
├── raw/                # 原始知识数据
│   ├── four_char_words.txt
│   ├── idioms.json
│   ├── cedict_parsed.json
│   ├── dict_unihan.json
│   └── zhwiki.db
```