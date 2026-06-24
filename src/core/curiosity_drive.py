import torch
from typing import List, Tuple, Optional
from collections import deque
import math


class CuriosityDrive:
    def __init__(
        self,
        field_dim: int = 4096,
        num_bins: int = 64,
        entropy_low_threshold: float = 2.0,
        entropy_high_threshold: float = 5.0,
        exploration_noise_std: float = 0.05,
        max_history: int = 10000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.field_dim = field_dim
        self.num_bins = num_bins
        self.entropy_low_threshold = entropy_low_threshold
        self.entropy_high_threshold = entropy_high_threshold
        self.exploration_noise_std = exploration_noise_std
        self.max_history = max_history
        self.device = device

        self.entropy_history = deque(maxlen=max_history)
        self.exploration_regions = []

    def compute_entropy(self, h: torch.Tensor) -> float:
        """纯GPU熵计算，避免.cpu().numpy()传输"""
        h_abs = h.abs().float()
        h_sum = h_abs.sum() + 1e-10
        h_normalized = h_abs / h_sum

        bin_size = self.field_dim // self.num_bins
        remainder = self.field_dim % self.num_bins

        if remainder == 0:
            binned = h_normalized[:self.num_bins * bin_size].reshape(self.num_bins, bin_size).sum(dim=1)
        else:
            main_part = h_normalized[:(self.num_bins - 1) * bin_size].reshape(self.num_bins - 1, bin_size).sum(dim=1)
            last_bin = h_normalized[(self.num_bins - 1) * bin_size:].sum().unsqueeze(0)
            binned = torch.cat([main_part, last_bin])

        # 纯PyTorch计算熵，避免GPU→CPU传输
        probs = binned.clamp(min=1e-10)
        entropy = -(probs * torch.log(probs)).sum()

        return entropy.item()

    def compute_local_entropy(
        self,
        h: torch.Tensor,
        semantic_atoms: List[Tuple[int, int]]
    ) -> List[Tuple[int, float]]:
        """向量化局部熵计算，批量处理所有原子区域"""
        if len(semantic_atoms) == 0:
            return []

        h_abs = h.abs().float()
        num_atoms = len(semantic_atoms)

        # 预计算所有区域的起止索引和大小
        starts = torch.tensor([s for s, e in semantic_atoms], device=h.device)
        ends = torch.tensor([e for s, e in semantic_atoms], device=h.device)
        sizes = ends - starts
        valid_mask = sizes > 0

        if not valid_mask.any():
            return []

        # 用segment操作批量计算每个区域的归一化概率和熵
        local_entropies = []
        # 按区域大小分组：相同大小的区域可以完全向量化
        size_groups = {}
        for idx in range(num_atoms):
            if not valid_mask[idx]:
                continue
            s = sizes[idx].item()
            if s not in size_groups:
                size_groups[s] = []
            size_groups[s].append(idx)

        for region_size, indices in size_groups.items():
            # 提取所有相同大小区域的子向量，一次性计算
            region_starts = starts[indices]
            region_matrix = torch.stack([
                h_abs[s:s + region_size] for s in region_starts
            ])  # (num_regions, region_size)

            region_sums = region_matrix.sum(dim=1, keepdim=True) + 1e-10
            region_probs = region_matrix / region_sums
            region_probs = region_probs.clamp(min=1e-10)

            # 批量计算熵
            entropies = -(region_probs * torch.log(region_probs)).sum(dim=1)
            max_entropy = math.log(region_size) if region_size > 1 else 1.0
            normalized_entropies = entropies / max_entropy

            for i, idx in enumerate(indices):
                local_entropies.append((idx, normalized_entropies[i].item()))

        return local_entropies

    def detect_entropy_anomaly(
        self,
        h: torch.Tensor,
        semantic_atoms: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[float, List[int], Optional[torch.Tensor], str]:
        global_entropy = self.compute_entropy(h)

        self.entropy_history.append(global_entropy)  # deque自动截断

        exploration_atoms = []

        if global_entropy < self.entropy_low_threshold:
            noise = torch.randn(self.field_dim, device=self.device) * self.exploration_noise_std
            return global_entropy, [], noise, "low_entropy"

        elif global_entropy > self.entropy_high_threshold:
            return global_entropy, [], None, "high_entropy"

        else:
            if semantic_atoms is not None and len(semantic_atoms) > 0:
                local_entropies = self.compute_local_entropy(h, semantic_atoms)

                if len(local_entropies) > 0:
                    # 纯PyTorch统计，避免numpy转换开销
                    entropy_tensor = torch.tensor([e for _, e in local_entropies], device=h.device)
                    mean_entropy = entropy_tensor.mean()
                    std_entropy = entropy_tensor.std() if len(entropy_tensor) > 1 else torch.tensor(0.0, device=h.device)

                    if std_entropy > 1e-6:
                        threshold = mean_entropy + 2 * std_entropy
                        for atom_id, local_e in local_entropies:
                            if local_e > threshold:
                                exploration_atoms.append(atom_id)

            return global_entropy, exploration_atoms, None, "normal"

    def adapt_learning_rate(
        self,
        base_learning_rate: float,
        input_frequency: float,
        entropy_state: str = "normal"
    ) -> float:
        lr = base_learning_rate

        if entropy_state == "high_entropy":
            lr = base_learning_rate * 0.1
        elif entropy_state == "low_entropy":
            lr = base_learning_rate * 1.5

        if input_frequency > 0.5:
            lr *= 2.0
        elif input_frequency < 0.1:
            lr *= 0.5

        return lr

    def get_entropy_history(self) -> List[float]:
        return list(self.entropy_history)

    def get_exploration_regions(self) -> List[int]:
        return self.exploration_regions
