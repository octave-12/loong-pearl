"""
语料迭代器 - 支持大语料分批加载，避免内存溢出
"""
from typing import Iterator, List
import os


class CorpusIterator:
    """语料文件迭代器，支持分批加载"""
    
    def __init__(
        self,
        corpus_path: str,
        batch_size: int = 10000,
        max_lines: int = None,
        encoding: str = 'utf-8'
    ):
        self.corpus_path = corpus_path
        self.batch_size = batch_size
        self.max_lines = max_lines
        self.encoding = encoding
        
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"语料文件不存在: {corpus_path}")
        
        self._total_lines = None
    
    def __iter__(self) -> Iterator[List[str]]:
        """迭代返回批次语料"""
        batch = []
        total_read = 0
        
        with open(self.corpus_path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                batch.append(line)
                
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                
                total_read += 1
                if self.max_lines and total_read >= self.max_lines:
                    break
        
        if batch:
            yield batch
    
    def count_lines(self) -> int:
        """统计语料总行数（惰性计算）"""
        if self._total_lines is not None:
            return self._total_lines
        
        count = 0
        with open(self.corpus_path, 'r', encoding=self.encoding) as f:
            for _ in f:
                count += 1
        
        self._total_lines = count
        return count
    
    def get_file_size(self) -> int:
        """获取文件大小（字节）"""
        return os.path.getsize(self.corpus_path)
    
    def get_stats(self) -> dict:
        """获取语料统计信息"""
        return {
            "path": self.corpus_path,
            "total_lines": self.count_lines(),
            "file_size_mb": self.get_file_size() / 1024 / 1024,
            "batch_size": self.batch_size,
            "num_batches": (self.count_lines() + self.batch_size - 1) // self.batch_size
        }


class StreamingPMICalculator:
    """流式PMI计算器，支持大语料分批处理，使用numpy向量化计数"""
    
    def __init__(
        self,
        window_size: int = 5,
        min_count: int = 10,
        pmi_threshold: float = 2.0
    ):
        self.window_size = window_size
        self.min_count = min_count
        self.pmi_threshold = pmi_threshold
        
        self.char_counts = {}
        self.pair_counts = {}
        self.total_chars = 0
        self.total_pairs = 0
        self._char_to_id = {}
        self._id_to_char = None
    
    def process_batch(self, texts: List[str]):
        """向量化处理一批文本，用numpy替代Python双重循环"""
        import numpy as np
        
        all_ids = []
        text_boundaries = [0]
        new_chars_added = False
        
        for text in texts:
            for ch in text:
                if ch not in self._char_to_id:
                    self._char_to_id[ch] = len(self._char_to_id)
                    new_chars_added = True
                all_ids.append(self._char_to_id[ch])
            text_boundaries.append(len(all_ids))
        
        if not all_ids:
            return
        
        # 如果有新字符加入，失效反向缓存
        if new_chars_added:
            self._id_to_char = None
        
        ids = np.array(all_ids, dtype=np.int32)
        n_vocab = len(self._char_to_id)
        
        # 字符计数
        unique_ids, counts = np.unique(ids, return_counts=True)
        for cid, count in zip(unique_ids, counts):
            ch = self._char_to_id_inv(int(cid))
            self.char_counts[ch] = self.char_counts.get(ch, 0) + int(count)
        
        self.total_chars += len(ids)
        
        # 向量化字对计数
        for offset in range(1, self.window_size):
            pairs_i = ids[:-offset]
            pairs_j = ids[offset:]
            
            positions = np.arange(len(pairs_i))
            valid = np.ones(len(pairs_i), dtype=bool)
            for boundary in text_boundaries[1:-1]:
                valid &= (positions < boundary - offset) | (positions >= boundary)
            
            if not valid.any():
                continue
            
            valid_i = pairs_i[valid]
            valid_j = pairs_j[valid]
            
            linear_keys = valid_i.astype(np.int64) * n_vocab + valid_j.astype(np.int64)
            unique_keys, pair_cnts = np.unique(linear_keys, return_counts=True)
            
            for key, count in zip(unique_keys, pair_cnts):
                key = int(key)
                ci = key // n_vocab
                cj = key % n_vocab
                pair = (self._char_to_id_inv(ci), self._char_to_id_inv(cj))
                self.pair_counts[pair] = self.pair_counts.get(pair, 0) + int(count)
            
            self.total_pairs += int(valid.sum())
    
    def _char_to_id_inv(self, cid: int) -> str:
        """反向查找ID对应的字符（带缓存）"""
        if self._id_to_char is None:
            self._id_to_char = {v: k for k, v in self._char_to_id.items()}
        return self._id_to_char.get(cid, '?')
    
    def compute_pmi(self) -> List[tuple]:
        """计算PMI并返回高分字对"""
        import numpy as np
        
        if self.total_pairs == 0:
            return []
        
        high_pmi_pairs = []
        
        for (char_a, char_b), count in self.pair_counts.items():
            if count < self.min_count:
                continue
            
            p_a = self.char_counts[char_a] / self.total_chars
            p_b = self.char_counts[char_b] / self.total_chars
            p_ab = count / self.total_pairs
            
            pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)
            
            if pmi > self.pmi_threshold:
                high_pmi_pairs.append((char_a, char_b, pmi))
        
        return high_pmi_pairs
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_chars": self.total_chars,
            "total_pairs": self.total_pairs,
            "unique_chars": len(self.char_counts),
            "unique_pairs": len(self.pair_counts)
        }