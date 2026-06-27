"""
知识源加载器 - 从 data/raw/ 目录加载各种知识数据文件
支持：汉字拆解、成语词典、基础字符集、预计算PMI字对、CEDICT等
"""
import json
import os
import logging
from typing import List, Tuple, Dict, Optional


class KnowledgeLoader:
    """统一知识源加载接口"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self._cache = {}
        self._logger = logging.getLogger("KnowledgeLoader")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.WARNING)

    def _load_json(self, filename: str) -> Optional[object]:
        """加载JSON文件（带缓存）"""
        if filename in self._cache:
            return self._cache[filename]
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            self._logger.debug(f"知识源文件不存在: {filepath}")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cache[filename] = data
            self._logger.info(f"成功加载知识源: {filename} (条目数: {len(data) if isinstance(data, (list, dict)) else 'N/A'})")
            return data
        except json.JSONDecodeError as e:
            self._logger.warning(f"JSON解析失败: {filepath}, 错误: {e}")
            return None
        except Exception as e:
            self._logger.warning(f"加载知识源失败: {filepath}, 错误: {e}")
            return None

    def _load_text(self, filename: str) -> Optional[str]:
        """加载文本文件"""
        if filename in self._cache:
            return self._cache[filename]
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            self._logger.debug(f"知识源文件不存在: {filepath}")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.read()
            self._cache[filename] = data
            self._logger.info(f"成功加载知识源: {filename} (大小: {len(data)} 字符)")
            return data
        except Exception as e:
            self._logger.warning(f"加载知识源失败: {filepath}, 错误: {e}")
            return None

    def load_directed_pairs(self) -> List[Tuple[str, str, float]]:
        """加载预计算的有向字对（PMI统计结果）
        格式: [[char_a, char_b, score], ...]
        可直接注入 semantic_atoms.compute_pmi 的结果中作为先验知识
        """
        data = self._load_json("directed_pairs.json")
        if data is None:
            return []
        return [(item[0], item[1], float(item[2])) for item in data if len(item) >= 3]

    def load_dict_decompose(self) -> Dict[str, dict]:
        """加载汉字拆解字典
        格式: {character: {radical, strokes, components, ...}}
        可用于增强语义原子的字符特征表示
        """
        data = self._load_json("dict_decompose.json")
        return data if data else {}

    def load_cedict(self) -> Dict[str, dict]:
        """加载CEDICT中文-英文词典
        格式: {word: {pinyin, english, ...}}
        可用于词级语义关联
        """
        data = self._load_json("cedict_parsed.json")
        return data if data else {}

    def load_idioms(self) -> list:
        """加载成语词典
        可能是字符串列表["龙飞凤舞", ...]或字典列表
        """
        data = self._load_json("idioms.json")
        if data is None:
            return []
        return data if isinstance(data, list) else []

    def load_unihan(self) -> Dict[str, dict]:
        """加载Unihan汉字统一数据库
        格式: {character: {definition, readings, ...}}
        汉字的完整属性信息
        """
        data = self._load_json("dict_unihan.json")
        return data if data else {}

    def load_gb2312_level1(self) -> List[str]:
        """加载GB2312一级常用汉字表（3755字）
        最常用的汉字集合，可用于初始化字符覆盖
        """
        text = self._load_text("gb2312_level1.txt")
        if text is None:
            return []
        # 文件内容是单行，每个字符一个汉字
        return list(text.strip())

    def load_four_char_words(self) -> List[str]:
        """加载四字词语表
        四字格是中文中重要的语义单元
        """
        text = self._load_text("four_char_words.txt")
        if text is None:
            return []
        return [line.strip() for line in text.strip().split('\n') if line.strip()]

    def get_enhanced_pmi_pairs(
        self,
        corpus_pairs: List[Tuple[str, str, float]],
        min_score: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """合并语料PMI对与预计算directed_pairs
        预计算的directed_pairs作为先验知识补充，
        确保即使语料量不足也能捕获基本的字间关联
        """
        # 用dict去重，取最大分数
        pair_map = {}
        for char_a, char_b, score in corpus_pairs:
            key = (char_a, char_b)
            if key not in pair_map or score > pair_map[key]:
                pair_map[key] = score

        # 注入预计算的directed_pairs
        directed = self.load_directed_pairs()
        injected = 0
        for char_a, char_b, score in directed:
            if score < min_score:
                continue
            key = (char_a, char_b)
            # 预计算对的分数作为底线：如果语料中没有，则注入
            if key not in pair_map:
                pair_map[key] = score * 0.8  # 略微降低权重，语料优先
                injected += 1

        result = [(k[0], k[1], v) for k, v in pair_map.items()]
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def get_idiom_atoms(self) -> List[List[str]]:
        """从成语词典中提取成语作为预定义的语义原子簇
        每个成语的字符构成一个原子簇
        """
        idioms = self.load_idioms()
        clusters = []
        for idiom in idioms:
            # 支持字符串和字典两种格式
            if isinstance(idiom, str):
                word = idiom
            elif isinstance(idiom, dict):
                word = idiom.get('word', idiom.get('idiom', ''))
            else:
                continue
            if len(word) >= 2:
                clusters.append(list(word))
        return clusters

    def get_common_chars(self) -> List[str]:
        """获取常用汉字列表（GB2312一级 + 四字词语中的字符）
        用于确保语义原子覆盖基本字符集
        """
        chars = set()

        # GB2312一级汉字
        gb_chars = self.load_gb2312_level1()
        chars.update(gb_chars)

        # 四字词语中的字符
        four_char = self.load_four_char_words()
        for word in four_char:
            chars.update(list(word))

        return sorted(chars)

    def get_available_sources(self) -> Dict[str, bool]:
        """检查哪些知识源可用"""
        sources = {
            "directed_pairs": os.path.exists(os.path.join(self.data_dir, "directed_pairs.json")),
            "dict_decompose": os.path.exists(os.path.join(self.data_dir, "dict_decompose.json")),
            "cedict": os.path.exists(os.path.join(self.data_dir, "cedict_parsed.json")),
            "idioms": os.path.exists(os.path.join(self.data_dir, "idioms.json")),
            "unihan": os.path.exists(os.path.join(self.data_dir, "dict_unihan.json")),
            "gb2312_level1": os.path.exists(os.path.join(self.data_dir, "gb2312_level1.txt")),
            "four_char_words": os.path.exists(os.path.join(self.data_dir, "four_char_words.txt")),
        }
        return sources
