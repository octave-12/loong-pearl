import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import jieba
import community as community_louvain
import networkx as nx
from multiprocessing import Pool, cpu_count
from functools import partial

from src.data.knowledge_loader import KnowledgeLoader


class SemanticAtom:
    def __init__(
        self,
        atom_id: int,
        characters: List[str],
        embedding: np.ndarray,
        field_region: Tuple[int, int]
    ):
        self.atom_id = atom_id
        self.characters = characters
        self.embedding = embedding
        self.field_region = field_region
        self.activation_count = 0
        self.last_activation = 0
        self.raw_activation_count = 0

    def to_dict(self) -> dict:
        return {
            "atom_id": self.atom_id,
            "characters": self.characters,
            "embedding": self.embedding.tolist(),
            "field_region": list(self.field_region),
            "activation_count": self.activation_count,
            "last_activation": self.last_activation,
            "raw_activation_count": self.raw_activation_count
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticAtom":
        atom = cls(
            atom_id=d["atom_id"],
            characters=d["characters"],
            embedding=np.array(d["embedding"]),
            field_region=tuple(d["field_region"])
        )
        atom.activation_count = d.get("activation_count", 0)
        atom.last_activation = d.get("last_activation", 0)
        atom.raw_activation_count = d.get("raw_activation_count", 0)
        return atom


class SemanticAtomManager:
    def __init__(
        self,
        field_dim: int = 4096,
        atom_dim: int = 128,
        initial_atoms: int = 5000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        knowledge_loader: KnowledgeLoader = None
    ):
        self.field_dim = field_dim
        self.atom_dim = atom_dim
        self.initial_atoms = initial_atoms
        self.device = device

        self.atoms: Dict[int, SemanticAtom] = {}
        self.char_to_atom: Dict[str, int] = {}
        self.next_atom_id = 0

        self.atom_embeddings = None
        self.atom_regions = []

        self._slot_atoms: Dict[int, List[int]] = defaultdict(list)
        self.knowledge = knowledge_loader or KnowledgeLoader()

    @staticmethod
    def _fast_count_pairs(texts: List[str], window_size: int):
        """向量化字对计数，用numpy替代Python双重循环"""
        all_chars = []
        char_to_id = {}
        text_boundaries = [0]  # 记录每条文本的字符起始位置

        for text in texts:
            for ch in text:
                if ch not in char_to_id:
                    char_to_id[ch] = len(char_to_id)
                all_chars.append(char_to_id[ch])
            text_boundaries.append(len(all_chars))

        if not all_chars:
            return {}, {}, 0, 0

        id_to_char = {v: k for k, v in char_to_id.items()}
        all_ids = np.array(all_chars, dtype=np.int32)
        n_vocab = len(char_to_id)

        char_counts = {}
        for cid, count in zip(*np.unique(all_ids, return_counts=True)):
            char_counts[id_to_char[int(cid)]] = int(count)

        total_chars = len(all_ids)
        pair_counts = {}
        total_pairs = 0

        # 向量化生成窗口内字对：每个偏移量一次性处理全文本
        for offset in range(1, window_size):
            pairs_i = all_ids[:-offset]
            pairs_j = all_ids[offset:]

            # 过滤跨文本边界的字对
            positions = np.arange(len(pairs_i))
            valid = np.ones(len(pairs_i), dtype=bool)
            for boundary in text_boundaries[1:-1]:
                valid &= (positions < boundary - offset) | (positions >= boundary)

            if not valid.any():
                continue

            valid_i = pairs_i[valid]
            valid_j = pairs_j[valid]

            # 用线性化索引进行快速计数
            linear_keys = valid_i.astype(np.int64) * n_vocab + valid_j.astype(np.int64)
            unique_keys, counts = np.unique(linear_keys, return_counts=True)

            for key, count in zip(unique_keys, counts):
                key = int(key)
                ci = key // n_vocab
                cj = key % n_vocab
                pair = (id_to_char[ci], id_to_char[cj])
                pair_counts[pair] = pair_counts.get(pair, 0) + int(count)

            total_pairs += int(valid.sum())

        return char_counts, pair_counts, total_chars, total_pairs

    def compute_pmi(
        self,
        corpus: List[str],
        window_size: int = 5,
        min_count: int = 10,
        pmi_threshold: float = 2.0,
        num_workers: int = None
    ) -> List[Tuple[str, str, float]]:
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
        
        if num_workers > 1 and len(corpus) > 1000:
            return self._compute_pmi_parallel(corpus, window_size, min_count, pmi_threshold, num_workers)
        
        char_counts, pair_counts, total_chars, total_pairs = self._fast_count_pairs(
            corpus, window_size
        )

        if total_pairs == 0:
            return []

        high_pmi_pairs = []
        for (char_a, char_b), count in pair_counts.items():
            if count < min_count:
                continue

            p_a = char_counts[char_a] / total_chars
            p_b = char_counts[char_b] / total_chars
            p_ab = count / total_pairs

            pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)

            if pmi > pmi_threshold:
                high_pmi_pairs.append((char_a, char_b, pmi))

        return high_pmi_pairs

    def _compute_pmi_parallel(
        self,
        corpus: List[str],
        window_size: int,
        min_count: int,
        pmi_threshold: float,
        num_workers: int
    ) -> List[Tuple[str, str, float]]:
        chunk_size = max(1, len(corpus) // num_workers)
        chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]
        
        with Pool(num_workers) as pool:
            results = pool.map(
                partial(self._compute_pmi_chunk, window_size=window_size),
                chunks
            )
        
        char_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        total_chars = 0
        total_pairs = 0
        
        for chunk_char_counts, chunk_pair_counts, chunk_total_chars, chunk_total_pairs in results:
            total_chars += chunk_total_chars
            total_pairs += chunk_total_pairs
            for char, count in chunk_char_counts.items():
                char_counts[char] += count
            for pair, count in chunk_pair_counts.items():
                pair_counts[pair] += count
        
        high_pmi_pairs = []
        for (char_a, char_b), count in pair_counts.items():
            if count < min_count:
                continue

            p_a = char_counts[char_a] / total_chars
            p_b = char_counts[char_b] / total_chars
            p_ab = count / total_pairs

            pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)

            if pmi > pmi_threshold:
                high_pmi_pairs.append((char_a, char_b, pmi))

        return high_pmi_pairs

    @staticmethod
    def _compute_pmi_chunk(texts: List[str], window_size: int):
        """用向量化方法计算分片的字对统计"""
        return SemanticAtomManager._fast_count_pairs(texts, window_size)

    def compute_pmi_streaming(
        self,
        corpus_path: str,
        batch_size: int = 10000,
        max_lines: int = None,
        window_size: int = 5,
        min_count: int = 10,
        pmi_threshold: float = 2.0
    ) -> List[Tuple[str, str, float]]:
        """流式计算PMI，支持大语料文件
        
        Args:
            corpus_path: 语料文件路径
            batch_size: 批次大小
            max_lines: 最大读取行数（None表示全部）
            window_size: PMI窗口大小
            min_count: 最小出现次数
            pmi_threshold: PMI阈值
        
        Returns:
            高PMI字对列表
        """
        from src.data.corpus_iterator import CorpusIterator, StreamingPMICalculator
        
        corpus_iter = CorpusIterator(corpus_path, batch_size=batch_size, max_lines=max_lines)
        pmi_calc = StreamingPMICalculator(window_size, min_count, pmi_threshold)
        
        for batch in corpus_iter:
            pmi_calc.process_batch(batch)
        
        return pmi_calc.compute_pmi()

    def cluster_characters(
        self,
        pmi_pairs: List[Tuple[str, str, float]],
        use_knowledge: bool = True,
        use_igraph: bool = False
    ) -> List[List[str]]:
        """基于PMI字对进行字符聚类，可选择注入知识源先验
        
        Args:
            pmi_pairs: PMI字对列表
            use_knowledge: 是否使用知识增强
            use_igraph: 是否使用igraph（Leiden算法，更快更好，需安装python-igraph）
                       默认False使用Louvain算法以保持向后兼容
        
        Returns:
            聚类结果列表
        """
        if use_knowledge and self.knowledge:
            pmi_pairs = self.knowledge.get_enhanced_pmi_pairs(pmi_pairs)

        if len(pmi_pairs) == 0:
            return []

        if use_igraph:
            try:
                import igraph as ig
                return self._cluster_with_igraph(pmi_pairs)
            except ImportError:
                pass

        G = nx.Graph()

        for char_a, char_b, pmi in pmi_pairs:
            G.add_edge(char_a, char_b, weight=pmi)

        if len(G.nodes()) == 0:
            return []

        partition = community_louvain.best_partition(G)

        clusters = defaultdict(list)
        for char, cluster_id in partition.items():
            clusters[cluster_id].append(char)

        return list(clusters.values())

    def _cluster_with_igraph(self, pmi_pairs: List[Tuple[str, str, float]]) -> List[List[str]]:
        """使用igraph进行快速聚类（比networkx快5-10x）"""
        import igraph as ig
        
        char_to_id = {}
        edges = []
        weights = []
        
        for char_a, char_b, pmi in pmi_pairs:
            if char_a not in char_to_id:
                char_to_id[char_a] = len(char_to_id)
            if char_b not in char_to_id:
                char_to_id[char_b] = len(char_to_id)
            
            edges.append((char_to_id[char_a], char_to_id[char_b]))
            weights.append(pmi)
        
        if len(edges) == 0:
            return []
        
        id_to_char = {v: k for k, v in char_to_id.items()}
        
        g = ig.Graph(n=len(char_to_id), edges=edges, edge_attrs={'weight': weights})
        
        partition = g.community_leiden(weights='weight', resolution_parameter=1.0)
        
        clusters = []
        for community in partition:
            cluster = [id_to_char[node_id] for node_id in community]
            clusters.append(cluster)
        
        return clusters

    def _allocate_region(self, atom_id: int) -> Tuple[int, int]:
        num_slots = self.field_dim // self.atom_dim

        h = atom_id
        h = ((h ^ (h >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
        h = ((h ^ (h >> 16)) * 0x45d9f3b) & 0xFFFFFFFF
        h = (h ^ (h >> 16)) & 0xFFFFFFFF
        base_slot = h % num_slots

        self._slot_atoms[base_slot].append(atom_id)

        start_idx = base_slot * self.atom_dim
        end_idx = start_idx + self.atom_dim
        return (start_idx, end_idx)

    def initialize_atoms_from_clusters(
        self,
        clusters: List[List[str]],
        field_dim: int = 4096,
        inject_idiom_atoms: bool = True,
        max_idiom_atoms: int = 1000
    ):
        self.atoms.clear()
        self.char_to_atom.clear()
        self.next_atom_id = 0
        self._slot_atoms.clear()

        all_clusters = list(clusters)

        if inject_idiom_atoms and self.knowledge:
            idiom_clusters = self.knowledge.get_idiom_atoms()
            if idiom_clusters:
                all_clusters.extend(idiom_clusters[:min(max_idiom_atoms, len(idiom_clusters))])

        num_atoms = min(len(all_clusters), self.initial_atoms)

        for i, cluster in enumerate(all_clusters[:num_atoms]):
            embedding = np.random.randn(self.atom_dim) * 0.1

            field_region = self._allocate_region(self.next_atom_id)

            atom = SemanticAtom(
                atom_id=self.next_atom_id,
                characters=cluster,
                embedding=embedding,
                field_region=field_region
            )

            self.atoms[self.next_atom_id] = atom
            for char in cluster:
                self.char_to_atom[char] = self.next_atom_id

            self.next_atom_id += 1

        self._build_atom_matrices()

    def _build_atom_matrices(self):
        if len(self.atoms) == 0:
            return

        embeddings = []
        regions = []

        for atom_id in sorted(self.atoms.keys()):
            atom = self.atoms[atom_id]
            embeddings.append(atom.embedding)
            regions.append(atom.field_region)

        self.atom_embeddings = torch.tensor(
            np.array(embeddings),
            dtype=torch.float32,
            device=self.device
        )
        self.atom_regions = regions

    def find_atom_for_char(self, char: str) -> int:
        return self.char_to_atom.get(char, -1)

    def get_atom_region(self, atom_id: int) -> Tuple[int, int]:
        if atom_id in self.atoms:
            return self.atoms[atom_id].field_region
        return (0, self.atom_dim)

    def update_atom_activation(self, atom_id: int, step: int):
        if atom_id in self.atoms:
            self.atoms[atom_id].activation_count += 1
            self.atoms[atom_id].raw_activation_count += 1
            self.atoms[atom_id].last_activation = step

    def get_exploration_keywords(self, exploration_atom_ids: List[int]) -> List[str]:
        keywords = []
        for atom_id in exploration_atom_ids:
            if atom_id in self.atoms:
                atom = self.atoms[atom_id]
                if len(atom.characters) > 0:
                    keywords.append(atom.characters[0])
        return keywords

    def inject_search_result(self, text: str, field, field_interface, strength: float = 0.3):
        perturbations = field_interface.encode_text_to_perturbation(text, self)
        for perturbation, _ in perturbations:
            h = field.get_state()
            h = field_interface.inject_perturbation(h, perturbation, strength=strength)
            field.set_state(h)

    def split_atom(self, atom_id: int) -> List[int]:
        if atom_id not in self.atoms:
            return []

        atom = self.atoms[atom_id]
        chars = atom.characters

        if len(chars) < 2:
            return [atom_id]

        mid = len(chars) // 2
        chars_a = chars[:mid]
        chars_b = chars[mid:]

        del self.atoms[atom_id]
        for char in chars:
            if char in self.char_to_atom:
                del self.char_to_atom[char]

        new_ids = []
        for new_chars in [chars_a, chars_b]:
            new_embedding = np.random.randn(self.atom_dim) * 0.1
            field_region = self._allocate_region(self.next_atom_id)

            new_atom = SemanticAtom(
                atom_id=self.next_atom_id,
                characters=new_chars,
                embedding=new_embedding,
                field_region=field_region
            )

            self.atoms[self.next_atom_id] = new_atom
            for char in new_chars:
                self.char_to_atom[char] = self.next_atom_id

            new_ids.append(self.next_atom_id)
            self.next_atom_id += 1

        self._build_atom_matrices()
        return new_ids

    def merge_atoms(self, atom_ids: List[int]) -> int:
        valid_ids = [aid for aid in atom_ids if aid in self.atoms]
        if len(valid_ids) < 2:
            return valid_ids[0] if valid_ids else -1

        merged_chars = []
        merged_embedding = np.zeros(self.atom_dim)

        for aid in valid_ids:
            atom = self.atoms[aid]
            merged_chars.extend(atom.characters)
            merged_embedding += atom.embedding
            del self.atoms[aid]

        merged_embedding /= len(valid_ids)

        for char in merged_chars:
            if char in self.char_to_atom:
                del self.char_to_atom[char]

        field_region = self._allocate_region(self.next_atom_id)

        new_atom = SemanticAtom(
            atom_id=self.next_atom_id,
            characters=merged_chars,
            embedding=merged_embedding,
            field_region=field_region
        )

        self.atoms[self.next_atom_id] = new_atom
        for char in merged_chars:
            self.char_to_atom[char] = self.next_atom_id

        new_id = self.next_atom_id
        self.next_atom_id += 1

        self._build_atom_matrices()
        return new_id

    def evolve_atoms(self, current_step: int, split_threshold: int = 50, merge_inactive_steps: int = 5000):
        """批量处理原子分裂/合并，避免逐个操作的开销"""
        atoms_to_split = []
        atoms_to_merge = []

        # 一次性遍历收集需要操作的原子
        for atom_id, atom in self.atoms.items():
            if atom.raw_activation_count > split_threshold and len(atom.characters) > 3:
                atoms_to_split.append(atom_id)
            elif atom.raw_activation_count == 0 and current_step - atom.last_activation > merge_inactive_steps:
                atoms_to_merge.append(atom_id)

        # 批量分裂（注意分裂会修改atoms字典，但新ID不会与待分裂列表冲突）
        split_count = 0
        for atom_id in atoms_to_split:
            if atom_id in self.atoms:  # 确保还没被删除
                result = self.split_atom(atom_id)
                if result and len(result) > 1:
                    split_count += 1

        # 批量合并: 按区域邻近度分组，而非全部合并成一个
        merge_count = 0
        if len(atoms_to_merge) >= 2:
            # 按2个一组进行合并，避免大规模合并导致语义混乱
            for i in range(0, len(atoms_to_merge) - 1, 2):
                aid_a = atoms_to_merge[i]
                aid_b = atoms_to_merge[i + 1]
                if aid_a in self.atoms and aid_b in self.atoms:
                    self.merge_atoms([aid_a, aid_b])
                    merge_count += 1

        if split_count > 0 or merge_count > 0:
            # 只在有变更时才重建矩阵
            pass  # split_atom和merge_atoms内部已调用_build_atom_matrices

    def get_all_regions(self) -> List[Tuple[int, int]]:
        return self.atom_regions

    def get_num_atoms(self) -> int:
        return len(self.atoms)

    def get_atoms_state(self) -> dict:
        return {
            "atoms": {str(k): v.to_dict() for k, v in self.atoms.items()},
            "char_to_atom": self.char_to_atom,
            "next_atom_id": self.next_atom_id
        }

    def load_atoms_state(self, state: dict):
        self.atoms.clear()
        self.char_to_atom.clear()
        self._slot_atoms.clear()

        for k, v in state["atoms"].items():
            atom = SemanticAtom.from_dict(v)
            self.atoms[int(k)] = atom

        self.char_to_atom = state["char_to_atom"]
        self.next_atom_id = state["next_atom_id"]

        for atom_id, atom in self.atoms.items():
            start_idx = atom.field_region[0]
            slot = start_idx // self.atom_dim
            self._slot_atoms[slot].append(atom_id)

        self._build_atom_matrices()
