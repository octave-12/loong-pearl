"""
性能优化工具集：预取缓存、内存池、混合存储格式
"""
import torch
import numpy as np
from typing import Dict, Optional, Any
from functools import lru_cache
from threading import Thread, Queue
from queue import Empty


class TensorMemoryPool:
    """张量内存池，减少频繁分配开销"""
    
    def __init__(self, field_dim: int = 4096, atom_dim: int = 128, device: str = 'cpu'):
        self.field_dim = field_dim
        self.atom_dim = atom_dim
        self.device = device
        
        self._pool: Dict[str, torch.Tensor] = {}
        self._initialize_pool()
    
    def _initialize_pool(self):
        """预分配常用张量"""
        self._pool = {
            'field_state': torch.zeros(self.field_dim, device=self.device),
            'perturbation': torch.zeros(self.field_dim, device=self.device),
            'noise': torch.zeros(self.field_dim, device=self.device),
            'atom_embedding': torch.zeros(self.atom_dim, device=self.device),
            'region_activation': torch.zeros(self.atom_dim, device=self.device),
        }
    
    def get(self, name: str, clone: bool = True) -> torch.Tensor:
        """获取池中张量"""
        if name not in self._pool:
            raise KeyError(f"张量 '{name}' 不在内存池中")
        
        if clone:
            return self._pool[name].clone()
        return self._pool[name]
    
    def get_zero(self, name: str) -> torch.Tensor:
        """获取归零的张量"""
        tensor = self.get(name, clone=True)
        tensor.zero_()
        return tensor
    
    def put(self, name: str, tensor: torch.Tensor):
        """将张量放回池中（可选）"""
        if name in self._pool:
            self._pool[name].copy_(tensor)
    
    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        total_memory = sum(t.numel() * t.element_size() for t in self._pool.values())
        return {
            "num_tensors": len(self._pool),
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / 1024 / 1024,
            "device": self.device
        }


class PrefetchIterator:
    """预取迭代器，IO与计算重叠"""
    
    def __init__(self, iterator, prefetch_size: int = 2, daemon: bool = True):
        self.iterator = iterator
        self.prefetch_size = prefetch_size
        self.queue = Queue(maxsize=prefetch_size)
        self.stop_flag = False
        
        self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=daemon)
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """预取工作线程"""
        try:
            for item in self.iterator:
                if self.stop_flag:
                    break
                self.queue.put(item)
        except Exception:
            pass
        finally:
            self.queue.put(None)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item
    
    def stop(self):
        """停止预取"""
        self.stop_flag = True


class CachedAtomLookup:
    """带缓存的原子查找"""
    
    def __init__(self, semantic_atoms, max_cache_size: int = 10000):
        self.semantic_atoms = semantic_atoms
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, int] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def find_atom(self, char: str) -> int:
        """查找字符对应的原子ID（带缓存）"""
        if char in self._cache:
            self._cache_hits += 1
            return self._cache[char]
        
        self._cache_misses += 1
        atom_id = self.semantic_atoms.find_atom_for_char(char)
        
        if len(self._cache) < self.max_cache_size:
            self._cache[char] = atom_id
        
        return atom_id
    
    def invalidate(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate
        }


class HybridSparseMatrix:
    """混合稀疏矩阵，自动切换COO/CSR格式"""
    
    def __init__(self, field_dim: int = 4096, device: str = 'cpu'):
        self.field_dim = field_dim
        self.device = device
        self.matrix = None
        self._current_format = None
    
    def initialize(self, density: float = 0.001):
        """初始化稀疏矩阵"""
        num_nonzero = int(self.field_dim * self.field_dim * density)
        
        indices = torch.randperm(self.field_dim * self.field_dim, device=self.device)[:num_nonzero]
        row_indices = indices // self.field_dim
        col_indices = indices % self.field_dim
        values = torch.randn(num_nonzero, device=self.device) * 0.01
        
        self.matrix = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            values,
            size=(self.field_dim, self.field_dim)
        ).coalesce()
        self._current_format = 'coo'
    
    def ensure_format(self, target_format: str):
        """确保矩阵为目标格式"""
        if self._current_format == target_format:
            return
        
        if target_format == 'csr':
            if hasattr(self.matrix, 'is_sparse_csr') and not self.matrix.is_sparse_csr:
                self.matrix = self.matrix.to_sparse_csr()
                self._current_format = 'csr'
        elif target_format == 'coo':
            if hasattr(self.matrix, 'is_sparse_csr') and self.matrix.is_sparse_csr:
                self.matrix = self.matrix.to_sparse_coo().coalesce()
                self._current_format = 'coo'
    
    def update(self, h: torch.Tensor, learning_rate: float):
        """更新矩阵（COO格式更高效）"""
        self.ensure_format('coo')
        
        activated = (h.abs() > 0.7).nonzero(as_tuple=True)[0]
        if len(activated) > 0:
            active_values = h[activated]
            outer = torch.outer(active_values, active_values)
            delta_values = (learning_rate * outer).reshape(-1)
            
            mask = delta_values.abs() > 1e-8
            if mask.any():
                pairs = torch.cartesian_prod(activated, activated)
                delta_w = torch.sparse_coo_tensor(
                    pairs[mask].t(),
                    delta_values[mask],
                    size=(self.field_dim, self.field_dim),
                    device=self.device
                )
                self.matrix = (self.matrix + delta_w).coalesce()
    
    def matmul(self, x: torch.Tensor) -> torch.Tensor:
        """矩阵乘法（CSR格式更高效）"""
        self.ensure_format('csr')
        return torch.sparse.mm(self.matrix, x.unsqueeze(1)).squeeze(1)
    
    def decay(self, factor: float):
        """衰减（原地操作）"""
        if self._current_format == 'csr':
            vals = self.matrix.values()
            self.matrix = torch.sparse_csr_tensor(
                self.matrix.crow_indices(),
                self.matrix.col_indices(),
                vals * factor,
                size=(self.field_dim, self.field_dim),
                device=self.device
            )
        else:
            vals = self.matrix.values()
            self.matrix = torch.sparse_coo_tensor(
                self.matrix.indices(),
                vals * factor,
                size=(self.field_dim, self.field_dim),
                device=self.device
            ).coalesce()
    
    def nnz(self) -> int:
        """非零元素数"""
        return self.matrix._nnz() if hasattr(self.matrix, '_nnz') else 0
    
    def get_format(self) -> str:
        """获取当前格式"""
        return self._current_format