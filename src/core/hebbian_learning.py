import torch
from typing import Optional

import warnings
try:
    torch.sparse.check_sparse_tensor_invariants(True)
except (AttributeError, TypeError):
    pass
warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")


class HebbianUpdater:
    def __init__(
        self,
        field_dim: int = 4096,
        learning_rate: float = 1e-5,
        decay_rate: float = 1e-6,
        activation_threshold: float = 0.7,
        dormancy_threshold: int = 1000,
        weight_prune_threshold: float = 1e-4,
        max_density: float = 0.05,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.field_dim = field_dim
        self.learning_rate = learning_rate
        self.base_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.dormancy_threshold = dormancy_threshold
        self.weight_prune_threshold = weight_prune_threshold
        self.max_density = max_density
        self.device = device

        initial_density = 0.001
        num_nonzero = int(field_dim * field_dim * initial_density)

        indices = torch.randperm(field_dim * field_dim, device=device)[:num_nonzero]
        row_indices = indices // field_dim
        col_indices = indices % field_dim

        values = torch.randn(num_nonzero, device=device) * 0.01

        self.weight_matrix = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            values,
            size=(field_dim, field_dim)
        ).coalesce()

        # 不再初始化时转CSR，保持COO格式以便高效更新
        # CSR仅在检查点保存时按需转换

        self.activation_counts = torch.zeros(field_dim, device=device)
        self.last_activation_time = torch.zeros(field_dim, dtype=torch.long, device=device)
        self.current_step = 0

    def _ensure_sparse_csr(self):
        """将COO转为CSR格式（仅用于存储/检查点，日常更新保持COO）"""
        if self.weight_matrix.is_sparse and not self.weight_matrix.is_sparse_csr:
            try:
                self.weight_matrix = self.weight_matrix.to_sparse_csr()
            except Exception:
                self.weight_matrix = self.weight_matrix.coalesce()

    def _ensure_coo(self):
        """确保权重矩阵为COO格式（更新操作需要）"""
        if hasattr(self.weight_matrix, 'is_sparse_csr') and self.weight_matrix.is_sparse_csr:
            self.weight_matrix = self.weight_matrix.to_sparse_coo().coalesce()
        elif self.weight_matrix.is_sparse:
            self.weight_matrix = self.weight_matrix.coalesce()
        else:
            self.weight_matrix = self.weight_matrix.to_sparse_coo().coalesce()

    def _nnz(self) -> int:
        """获取非零元素数量"""
        if hasattr(self.weight_matrix, '_nnz'):
            return self.weight_matrix._nnz()
        if self.weight_matrix.is_sparse:
            return self.weight_matrix._nnz()
        if hasattr(self.weight_matrix, 'is_sparse_csr') and self.weight_matrix.is_sparse_csr:
            return self.weight_matrix._nnz()
        return (self.weight_matrix != 0).sum().item()

    def _prune_small_weights(self):
        self._ensure_coo()
        indices = self.weight_matrix.indices()
        values = self.weight_matrix.values()

        mask = values.abs() > self.weight_prune_threshold
        if not mask.all():
            self.weight_matrix = torch.sparse_coo_tensor(
                indices[:, mask],
                values[mask],
                size=(self.field_dim, self.field_dim),
                device=self.device
            ).coalesce()
            self._ensure_sparse_csr()

    def _enforce_max_density(self):
        self._ensure_coo()
        total_possible = self.field_dim * self.field_dim
        current_nnz = self._nnz()
        current_density = current_nnz / total_possible
        if current_density > self.max_density:
            values = self.weight_matrix.values()
            k = int(total_possible * self.max_density)
            if k < current_nnz:
                topk_values, topk_idx = torch.topk(values.abs(), k)
                indices = self.weight_matrix.indices()
                self.weight_matrix = torch.sparse_coo_tensor(
                    indices[:, topk_idx],
                    values[topk_idx],
                    size=(self.field_dim, self.field_dim),
                    device=self.device
                ).coalesce()
                self._ensure_sparse_csr()

    def update(
        self,
        h: torch.Tensor,
        weight_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if weight_matrix is not None:
            self.weight_matrix = weight_matrix

        self.current_step += 1

        activated = (h.abs() > self.activation_threshold).float()
        self.activation_counts = self.activation_counts * 0.999 + activated
        activated_indices = (activated > 0).nonzero(as_tuple=True)[0]
        self.last_activation_time[activated_indices] = self.current_step

        if len(activated_indices) > 0:
            self._ensure_coo()

            active_values = h[activated_indices] - self.activation_threshold
            n_active = len(activated_indices)

            # 向量化Hebbian外积: 用torch.outer替代repeat_interleave+repeat
            outer = torch.outer(active_values, active_values)
            delta_values = (self.learning_rate * outer).reshape(-1)

            # 过滤微小更新
            mask = delta_values.abs() > 1e-8
            if mask.any():
                # 批量构建索引对，避免创建多个中间张量
                pairs = torch.cartesian_prod(activated_indices, activated_indices)
                delta_w = torch.sparse_coo_tensor(
                    pairs[mask].t(),
                    delta_values[mask],
                    size=(self.field_dim, self.field_dim),
                    device=self.device
                )
                self.weight_matrix = (self.weight_matrix + delta_w).coalesce()

        # 衰减: 就地修改values，避免每步重建稀疏张量
        decay_factor = 1 - self.decay_rate
        if hasattr(self.weight_matrix, 'is_sparse_csr') and self.weight_matrix.is_sparse_csr:
            self.weight_matrix.values().mul_(decay_factor)
        elif self.weight_matrix.is_sparse:
            if not self.weight_matrix.is_coalesced:
                self.weight_matrix = self.weight_matrix.coalesce()
            self.weight_matrix.values().mul_(decay_factor)

        dormant_neurons = (self.current_step - self.last_activation_time) > self.dormancy_threshold
        if dormant_neurons.any():
            self._ensure_coo()
            accelerated_decay = 1 - self.decay_rate * 10
            dormant_indices = dormant_neurons.nonzero(as_tuple=True)[0]

            indices = self.weight_matrix.indices()

            dormant_lookup = torch.zeros(self.field_dim, dtype=torch.bool, device=self.device)
            dormant_lookup[dormant_indices] = True

            row_mask = dormant_lookup[indices[0]]
            col_mask = dormant_lookup[indices[1]]
            mask = row_mask | col_mask

            # 就地修改受影响的values，避免clone整个张量
            if mask.any():
                self.weight_matrix.values()[mask].mul_(accelerated_decay)

        if self.current_step % 100 == 0:
            self._prune_small_weights()

        if self.current_step % 500 == 0:
            self._enforce_max_density()

        # 每200步合并一次重复索引（coalesce），不再频繁转CSR
        if self.current_step % 200 == 0:
            if self.weight_matrix.is_sparse and not self.weight_matrix.is_sparse_csr:
                self.weight_matrix = self.weight_matrix.coalesce()

        return self.weight_matrix

    def get_weight_matrix(self) -> torch.Tensor:
        return self.weight_matrix

    def get_activation_stats(self) -> dict:
        return {
            "activation_counts": self.activation_counts.cpu().numpy(),
            "last_activation_time": self.last_activation_time.cpu().numpy(),
            "current_step": self.current_step,
            "num_nonzero_weights": self._nnz()
        }
