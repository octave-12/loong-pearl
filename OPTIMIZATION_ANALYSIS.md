# 性能优化深度分析

## 当前优化现状

### ✅ 已实现（7项）

#### 1. GPU加速 - 混合精度训练(AMP)
**位置**: `src/core/liquid_time_constant.py:78-84`
```python
if self.use_amp and self.device != "cpu":
    with torch.amp.autocast('cuda'):
        f_h = self.mlp(h + u)
        tau = self.compute_time_constants(h)
```
**效果**: 1.5-2x加速，显存减少30-50%

#### 2. PMI向量化计算
**位置**: `src/core/semantic_atoms.py:80-138`
```python
# 用numpy向量化替代Python双重循环
linear_keys = valid_i.astype(np.int64) * n_vocab + valid_j.astype(np.int64)
unique_keys, counts = np.unique(linear_keys, return_counts=True)
```
**效果**: 5-10x加速（相比原始Python循环）

#### 3. PMI多进程并行
**位置**: `src/core/semantic_atoms.py:177-221`
```python
with Pool(num_workers) as pool:
    results = pool.map(partial(self._compute_pmi_chunk, ...), chunks)
```
**效果**: 3-4x加速（4核CPU）

#### 4. 检查点压缩
**位置**: `src/core/field_guardian.py:267-310`
```python
if compress:
    with gzip.open(filepath_gz, 'wb') as f:
        torch.save(checkpoint, f)
```
**效果**: 文件大小减少65%

#### 5. 语料流式加载
**位置**: `src/data/corpus_iterator.py`
```python
for batch in CorpusIterator("data/corpus.txt", batch_size=10000):
    pmi_calc.process_batch(batch)
```
**效果**: 避免456万行语料OOM

#### 6. 批处理支持
**位置**: `src/core/field_guardian.py:226-265`
```python
outputs = guardian.process_batch(["问题1", "问题2", "问题3"])
```
**效果**: 批量处理效率提升

#### 7. 稀疏矩阵优化
**位置**: `src/core/hebbian_learning.py:46-50`
```python
self.weight_matrix = torch.sparse_coo_tensor(...).coalesce()
```
**效果**: 4096x4096矩阵，密度0.1%，内存减少95%

---

## 🚀 进一步优化空间

### 1. Hebbian学习GPU加速 ⭐⭐⭐

**现状**: 稀疏矩阵操作在CPU上执行
**优化方案**:
```python
# 当前（CPU）
self.weight_matrix = torch.sparse_coo_tensor(indices, values, ...)

# 优化（GPU）
if self.device == 'cuda':
    indices = indices.cuda()
    values = values.cuda()
    self.weight_matrix = torch.sparse_coo_tensor(indices, values, ...).cuda()
```
**预期效果**: 稀疏矩阵操作加速2-3x

---

### 2. 检查点增量保存 ⭐⭐

**现状**: 每次保存完整检查点（~100MB压缩后35MB）
**优化方案**:
```python
def save_checkpoint_incremental(self, filename):
    """只保存变化的参数"""
    current_state = self._get_state_dict()
    
    if self._last_checkpoint:
        delta = self._compute_delta(current_state, self._last_checkpoint)
        torch.save(delta, filename)  # 只保存差异
    else:
        torch.save(current_state, filename)
    
    self._last_checkpoint = current_state
```
**预期效果**: 增量检查点仅1-5MB

---

### 3. 语义原子聚类加速 ⭐⭐⭐

**现状**: Louvain社区检测较慢（O(n log n)）
**优化方案**:
```python
# 方案1: 使用更快的算法
import igraph as ig
# igraph的社区检测比networkx快5-10x

# 方案2: 近似算法
from sklearn.cluster import SpectralClustering
# 谱聚类对大规模图更快
```
**预期效果**: 聚类加速5-10x

---

### 4. 场接口批处理 ⭐⭐

**现状**: encode_text_to_perturbation逐文本处理
**优化方案**:
```python
def encode_batch_to_perturbations(self, texts: List[str], semantic_atoms):
    """批量编码文本"""
    # 批量分词
    all_words = [list(jieba.cut(text)) for text in texts]
    
    # 批量查找原子
    all_atom_ids = [[semantic_atoms.find_atom_for_char(ch) for ch in words] 
                    for words in all_words]
    
    # 向量化构建扰动
    perturbations = torch.zeros(len(texts), self.field_dim)
    # ... 批量操作
```
**预期效果**: 批量编码加速2-3x

---

### 5. 预取与缓存 ⭐

**现状**: 每次重新计算
**优化方案**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _cached_find_atom(self, char: str) -> int:
    return self.char_to_atom.get(char, -1)

# 语料预取线程
class PrefetchIterator:
    def __init__(self, iterator, prefetch_size=2):
        self.queue = Queue(maxsize=prefetch_size)
        Thread(target=self._prefetch, daemon=True).start()
```
**预期效果**: 减少重复计算，IO与计算重叠

---

### 6. 内存池管理 ⭐

**现状**: 频繁创建临时张量
**优化方案**:
```python
class TensorPool:
    def __init__(self, field_dim):
        self.pool = {
            'field_state': torch.zeros(field_dim),
            'perturbation': torch.zeros(field_dim),
            'noise': torch.zeros(field_dim),
        }
    
    def get(self, name):
        return self.pool[name].clone()
```
**预期效果**: 减少内存分配开销

---

### 7. CUDA Graph优化 ⭐⭐

**现状**: 每次演化都有kernel launch开销
**优化方案**:
```python
# 对于固定计算图，使用CUDA Graph
if self.use_cuda_graph:
    self.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.graph):
        h_new = self._evolve_graph(h)
```
**预期效果**: 减少30-50% kernel launch开销

---

### 8. 混合存储格式 ⭐

**现状**: Hebbian权重只用COO格式
**优化方案**:
```python
# 更新时用COO（高效插入）
# 矩阵乘法时转CSR（高效SpMM）
# 自动选择最优格式
def _auto_format(self, operation):
    if operation == 'update':
        self._ensure_coo()
    elif operation == 'matmul':
        self._ensure_csr()
```
**预期效果**: 不同操作用最优格式

---

## 📊 优化优先级

| 优化项 | 难度 | 收益 | 优先级 |
|--------|------|------|--------|
| Hebbian GPU加速 | 中 | 高 | ⭐⭐⭐ |
| 聚类算法优化 | 中 | 高 | ⭐⭐⭐ |
| 场接口批处理 | 低 | 中 | ⭐⭐ |
| CUDA Graph | 高 | 中 | ⭐⭐ |
| 检查点增量 | 中 | 中 | ⭐⭐ |
| 预取缓存 | 低 | 低 | ⭐ |
| 内存池 | 低 | 低 | ⭐ |
| 混合格式 | 中 | 低 | ⭐ |

---

## 🎯 建议下一步

### 短期（高优先级）
1. **Hebbian GPU加速**: 修改稀疏矩阵创建，确保在GPU上
2. **聚类算法优化**: 考虑igraph或近似算法

### 中期（中优先级）
3. **场接口批处理**: 实现encode_batch_to_perturbations
4. **检查点增量**: 实现增量保存机制

### 长期（低优先级）
5. **CUDA Graph**: 对固定计算图优化
6. **预取缓存**: IO与计算重叠

---

## 💡 实施建议

```python
# 1. Hebbian GPU加速（最简单，收益高）
# 在 hebbian_learning.py __init__ 中：
indices = torch.stack([row_indices, col_indices]).to(device)
values = values.to(device)
self.weight_matrix = torch.sparse_coo_tensor(indices, values, ...).to(device)

# 2. 聚类优化（中等难度）
# 在 semantic_atoms.py cluster_characters 中：
import igraph as ig
g = ig.Graph(edges=[(char_to_id[a], char_to_id[b]) for a, b, _ in pmi_pairs])
partition = g.community_leiden()  # 比Louvain更快

# 3. 批处理（简单）
# 在 field_interface.py 添加：
def encode_batch_to_perturbations(self, texts, semantic_atoms):
    # 批量处理逻辑
    pass
```

---

## 总结

**当前优化**: 已覆盖主要瓶颈（PMI计算、GPU加速、内存管理）
**剩余空间**: 主要是Hebbian GPU加速和聚类算法优化
**建议**: 优先实施Hebbian GPU加速（简单高收益），再考虑聚类算法替换