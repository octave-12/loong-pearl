"""
统一知识管理器 - 整合所有知识数据源
支持：成语词典、概念图谱、维基百科、汉英词典、汉字拆解、Unihan
"""
import json
import sqlite3
import os
import logging
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import gzip


class UnifiedKnowledgeManager:
    """统一知识数据管理接口"""

    def __init__(self, data_dir: str = "data/raw", concept_graph_path: str = None, wiki_db_path: str = None):
        self.data_dir = data_dir
        self.concept_graph_path = concept_graph_path or os.path.join(data_dir, "concept_graph.db")
        self.wiki_db_path = wiki_db_path or os.path.join(data_dir, "zhwiki.db")
        self._cache = {}
        self._logger = logging.getLogger("UnifiedKnowledgeManager")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _load_json(self, filename: str) -> Optional[object]:
        """加载JSON文件（带缓存）"""
        if filename in self._cache:
            return self._cache[filename]
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            self._logger.debug(f"文件不存在: {filepath}")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._cache[filename] = data
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            self._logger.info(f"加载 {filename}: {len(data) if isinstance(data, (list, dict)) else 'N/A'} 条, {size_mb:.1f}MB")
            return data
        except Exception as e:
            self._logger.warning(f"加载失败: {filepath}, 错误: {e}")
            return None

    def _load_text(self, filename: str) -> Optional[str]:
        """加载文本文件"""
        if filename in self._cache:
            return self._cache[filename]
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.read()
            self._cache[filename] = data
            return data
        except Exception as e:
            self._logger.warning(f"加载失败: {filepath}, 错误: {e}")
            return None

    def load_idioms(self) -> List[str]:
        """加载成语词典"""
        data = self._load_json("idioms.json")
        if data is None:
            return []
        return data if isinstance(data, list) else []

    def load_cedict(self) -> Dict[str, dict]:
        """加载汉英词典"""
        return self._load_json("cedict_parsed.json") or {}

    def load_dict_decompose(self) -> Dict[str, dict]:
        """加载汉字拆解字典"""
        return self._load_json("dict_decompose.json") or {}

    def load_unihan(self) -> Dict[str, dict]:
        """加载Unihan汉字数据库"""
        return self._load_json("dict_unihan.json") or {}

    def load_gb2312_level1(self) -> List[str]:
        """加载GB2312一级汉字表"""
        text = self._load_text("gb2312_level1.txt")
        if text is None:
            return []
        return list(text.strip())

    def load_four_char_words(self) -> List[str]:
        """加载四字词语表"""
        text = self._load_text("four_char_words.txt")
        if text is None:
            return []
        return [line.strip() for line in text.strip().split('\n') if line.strip()]

    def load_directed_pairs(self) -> List[Tuple[str, str, float]]:
        """加载预计算PMI字对"""
        data = self._load_json("directed_pairs.json")
        if data is None:
            return []
        return [(item[0], item[1], float(item[2])) for item in data if len(item) >= 3]

    def load_concept_graph(self, limit: int = None) -> List[Tuple[str, str, str, float]]:
        """加载概念图谱三元组
        
        Args:
            limit: 限制加载数量（None表示全部）
        
        Returns:
            [(主语, 关系, 宾语, 置信度), ...]
        """
        if "concept_graph" in self._cache:
            return self._cache["concept_graph"][:limit] if limit else self._cache["concept_graph"]
        
        if not os.path.exists(self.concept_graph_path):
            self._logger.warning(f"概念图谱不存在: {self.concept_graph_path}")
            return []
        
        try:
            conn = sqlite3.connect(self.concept_graph_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM triples")
            total = cursor.fetchone()[0]
            self._logger.info(f"概念图谱总数: {total} 个三元组")
            
            if limit:
                cursor.execute(f"SELECT s, r, o, c FROM triples LIMIT {limit}")
            else:
                cursor.execute("SELECT s, r, o, c FROM triples")
            
            triples = [(row[0], row[1], row[2], float(row[3])) for row in cursor.fetchall()]
            conn.close()
            
            self._cache["concept_graph"] = triples
            self._logger.info(f"加载概念图谱: {len(triples)} 个三元组")
            return triples
        except Exception as e:
            self._logger.warning(f"加载概念图谱失败: {e}")
            return []

    def get_concept_relations(self, concept: str) -> Dict[str, List[Tuple[str, float]]]:
        """获取概念的所有关系
        
        Returns:
            {关系类型: [(相关概念, 置信度), ...]}
        """
        triples = self.load_concept_graph()
        relations = defaultdict(list)
        
        for s, r, o, c in triples:
            if s == concept:
                relations[r].append((o, c))
            elif o == concept:
                inverse_r = self._get_inverse_relation(r)
                relations[inverse_r].append((s, c))
        
        return dict(relations)

    def _get_inverse_relation(self, relation: str) -> str:
        """获取逆关系"""
        inverse_map = {
            "PART_OF": "HAS_PART",
            "HAS_PART": "PART_OF",
            "IS_A": "SUBTYPE_OF",
            "SUBTYPE_OF": "IS_A",
            "HAS": "PART_OF",
        }
        return inverse_map.get(relation, f"INVERSE_{relation}")

    def get_idiom_atoms(self, max_idioms: int = None) -> List[List[str]]:
        """获取成语原子簇"""
        idioms = self.load_idioms()
        if max_idioms:
            idioms = idioms[:max_idioms]
        
        clusters = []
        for idiom in idioms:
            if isinstance(idiom, str) and len(idiom) >= 2:
                clusters.append(list(idiom))
        return clusters

    def get_enhanced_pmi_pairs(
        self,
        corpus_pairs: List[Tuple[str, str, float]],
        inject_concepts: bool = True,
        inject_wiki: bool = True,
        min_score: float = 0.3,
        concept_limit: int = 500000,
        wiki_limit: int = 10000
    ) -> List[Tuple[str, str, float]]:
        """增强PMI字对：注入预计算字对 + 概念关联 + 维基百科
        
        Args:
            corpus_pairs: 语料计算的PMI字对
            inject_concepts: 是否注入概念图谱关联
            inject_wiki: 是否注入维基百科关联
            min_score: 最小分数阈值
            concept_limit: 概念图谱加载限制
            wiki_limit: 维基百科加载限制
        """
        pair_map = {}
        for char_a, char_b, score in corpus_pairs:
            key = (char_a, char_b)
            if key not in pair_map or score > pair_map[key]:
                pair_map[key] = score

        directed = self.load_directed_pairs()
        injected_directed = 0
        for char_a, char_b, score in directed:
            if score < min_score:
                continue
            key = (char_a, char_b)
            if key not in pair_map:
                pair_map[key] = score * 0.8
                injected_directed += 1
        
        if inject_concepts:
            concepts = self.load_concept_graph(limit=concept_limit)
            injected_concepts = 0
            injected_entity_pairs = 0
            
            for s, r, o, c in concepts:
                if c < min_score:
                    continue
                
                if len(s) == 1 and len(o) == 1:
                    key = (s, o)
                    if key not in pair_map:
                        pair_map[key] = c * 0.5
                        injected_concepts += 1
                
                elif len(s) >= 2 and len(o) >= 2:
                    for char_s in s:
                        for char_o in o:
                            if len(char_s) == 1 and len(char_o) == 1:
                                key = (char_s, char_o)
                                if key not in pair_map:
                                    pair_map[key] = c * 0.3
                                    injected_entity_pairs += 1
            
            self._logger.info(f"注入概念关联: {injected_concepts} 个字对, {injected_entity_pairs} 个实体字对")
        
        if inject_wiki:
            articles = self.load_wiki_articles(limit=wiki_limit)
            injected_wiki = 0
            
            for title, text in articles:
                title_chars = [ch for ch in title if len(ch) == 1]
                if len(title_chars) >= 2:
                    for i in range(len(title_chars) - 1):
                        for j in range(i + 1, min(i + 3, len(title_chars))):
                            key = (title_chars[i], title_chars[j])
                            if key not in pair_map:
                                pair_map[key] = 0.4
                                injected_wiki += 1
            
            self._logger.info(f"注入维基百科: {injected_wiki} 个字对")
        
        self._logger.info(f"注入预计算字对: {injected_directed} 个")
        
        result = [(k[0], k[1], v) for k, v in pair_map.items()]
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def get_char_decomposition(self, char: str) -> Optional[dict]:
        """获取汉字拆解信息"""
        decompose = self.load_dict_decompose()
        return decompose.get(char)

    def get_word_definition(self, word: str) -> Optional[dict]:
        """获取词语定义（从汉英词典）"""
        cedict = self.load_cedict()
        return cedict.get(word)

    def get_all_chars(self) -> Set[str]:
        """获取所有已知字符集合"""
        chars = set()
        
        gb_chars = self.load_gb2312_level1()
        chars.update(gb_chars)
        
        four_char = self.load_four_char_words()
        for word in four_char:
            chars.update(list(word))
        
        idioms = self.load_idioms()
        for idiom in idioms[:5000]:
            if isinstance(idiom, str):
                chars.update(list(idiom))
        
        cedict = self.load_cedict()
        for word in list(cedict.keys())[:5000]:
            chars.update(list(word))
        
        return chars

    def get_knowledge_stats(self) -> Dict[str, int]:
        """获取知识库统计信息"""
        stats = {}
        
        idioms = self.load_idioms()
        stats["成语"] = len(idioms)
        
        cedict = self.load_cedict()
        stats["汉英词典"] = len(cedict)
        
        decompose = self.load_dict_decompose()
        stats["汉字拆解"] = len(decompose)
        
        unihan = self.load_unihan()
        stats["Unihan"] = len(unihan)
        
        gb = self.load_gb2312_level1()
        stats["GB2312一级"] = len(gb)
        
        four_char = self.load_four_char_words()
        stats["四字词语"] = len(four_char)
        
        directed = self.load_directed_pairs()
        stats["预计算PMI"] = len(directed)
        
        if os.path.exists(self.concept_graph_path):
            try:
                conn = sqlite3.connect(self.concept_graph_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM triples")
                stats["概念图谱"] = cursor.fetchone()[0]
                conn.close()
            except:
                stats["概念图谱"] = 0
        
        if os.path.exists(self.wiki_db_path):
            try:
                conn = sqlite3.connect(self.wiki_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM articles")
                stats["维基百科"] = cursor.fetchone()[0]
                conn.close()
            except:
                stats["维基百科"] = 0
        
        return stats
    
    def load_wiki_articles(self, limit: int = None, offset: int = 0) -> List[Tuple[str, str]]:
        """加载维基百科文章
        
        Args:
            limit: 限制加载数量
            offset: 偏移量
        
        Returns:
            [(标题, 正文), ...]
        """
        cache_key = f"wiki_articles_{offset}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not os.path.exists(self.wiki_db_path):
            self._logger.warning(f"维基百科数据库不存在: {self.wiki_db_path}")
            return []
        
        try:
            conn = sqlite3.connect(self.wiki_db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM articles")
            total = cursor.fetchone()[0]
            self._logger.info(f"维基百科总数: {total} 篇文章")
            
            if limit:
                cursor.execute(f"SELECT title, text FROM articles LIMIT {limit} OFFSET {offset}")
            else:
                cursor.execute(f"SELECT title, text FROM articles LIMIT 10000 OFFSET {offset}")
            
            articles = [(row[0], row[1]) for row in cursor.fetchall()]
            conn.close()
            
            self._cache[cache_key] = articles
            self._logger.info(f"加载维基百科: {len(articles)} 篇文章")
            return articles
        except Exception as e:
            self._logger.warning(f"加载维基百科失败: {e}")
            return []
    
    def search_wiki(self, keyword: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """搜索维基百科
        
        Args:
            keyword: 搜索关键词
            limit: 返回数量
        
        Returns:
            [(标题, 摘要, 相关度), ...]
        """
        if not os.path.exists(self.wiki_db_path):
            return []
        
        try:
            conn = sqlite3.connect(self.wiki_db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT title, text FROM articles 
                WHERE title LIKE '%{keyword}%' OR text LIKE '%{keyword}%'
                LIMIT {limit * 10}
            """)
            
            results = []
            for title, text in cursor.fetchall():
                if keyword in title:
                    score = 1.0
                elif keyword in text:
                    score = 0.5
                else:
                    score = 0.0
                
                summary = text[:200] if len(text) > 200 else text
                results.append((title, summary, score))
            
            conn.close()
            
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:limit]
        except Exception as e:
            self._logger.warning(f"搜索维基百科失败: {e}")
            return []
    
    def get_wiki_summary(self, title: str) -> Optional[str]:
        """获取维基百科文章摘要"""
        if not os.path.exists(self.wiki_db_path):
            return None
        
        try:
            conn = sqlite3.connect(self.wiki_db_path)
            cursor = conn.cursor()
            cursor.execute(f"SELECT text FROM articles WHERE title = '{title}' LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            
            if row:
                text = row[0]
                return text[:500] if len(text) > 500 else text
            return None
        except:
            return None
    
    def extract_wiki_entities(self, article_limit: int = 1000) -> Dict[str, int]:
        """从维基百科提取实体（标题词频）
        
        Args:
            article_limit: 处理的文章数量
        
        Returns:
            {实体: 频次}
        """
        articles = self.load_wiki_articles(limit=article_limit)
        entity_freq = defaultdict(int)
        
        for title, text in articles:
            words = title.split()
            for word in words:
                if len(word) >= 2:
                    entity_freq[word] += 1
        
        return dict(sorted(entity_freq.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    km = UnifiedKnowledgeManager()
    stats = km.get_knowledge_stats()
    print("\n=== 知识库统计 ===")
    for name, count in stats.items():
        print(f"{name}: {count:,}")
    
    print("\n=== 成语样本 ===")
    idioms = km.load_idioms()
    print(f"总数: {len(idioms)}")
    print(f"前10个: {idioms[:10]}")
    
    print("\n=== 概念图谱样本 ===")
    triples = km.load_concept_graph(limit=10)
    for s, r, o, c in triples:
        print(f"  {s} --[{r}]--> {o} (置信度: {c})")
    
    print("\n=== 汉英词典样本 ===")
    cedict = km.load_cedict()
    sample_words = list(cedict.keys())[:5]
    for word in sample_words:
        print(f"  {word}: {cedict[word]}")