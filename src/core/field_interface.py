import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import jieba


class FieldInterface:
    def __init__(
        self,
        field_dim: int = 4096,
        atom_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.field_dim = field_dim
        self.atom_dim = atom_dim
        self.device = device

        self.input_projection = torch.randn(field_dim, atom_dim, device=device) * 0.01
        self.output_projection = torch.randn(atom_dim, field_dim, device=device) * 0.01

        self.char_embeddings = {}
        self._cached_atom_embeddings = None
        self._cached_atom_ids = None

    def invalidate_cache(self):
        self._cached_atom_embeddings = None
        self._cached_atom_ids = None

    def _get_atom_embeddings_matrix(self, semantic_atoms) -> Optional[torch.Tensor]:
        if (self._cached_atom_embeddings is not None and
                self._cached_atom_ids == set(semantic_atoms.atoms.keys())):
            return self._cached_atom_embeddings

        if len(semantic_atoms.atoms) == 0:
            return None

        # 优先复用SemanticAtomManager预构建的GPU张量，避免重复从numpy转换
        if (hasattr(semantic_atoms, 'atom_embeddings') and
                semantic_atoms.atom_embeddings is not None and
                semantic_atoms.atom_embeddings.device == self.input_projection.device):
            self._cached_atom_embeddings = semantic_atoms.atom_embeddings
            self._cached_atom_ids = set(semantic_atoms.atoms.keys())
            return self._cached_atom_embeddings

        embeddings = []
        for atom_id in sorted(semantic_atoms.atoms.keys()):
            atom = semantic_atoms.atoms[atom_id]
            embeddings.append(atom.embedding)

        self._cached_atom_embeddings = torch.tensor(
            np.array(embeddings),
            dtype=torch.float32,
            device=self.device
        )
        self._cached_atom_ids = set(semantic_atoms.atoms.keys())
        return self._cached_atom_embeddings

    def _ensure_char_embedding(self, char: str, semantic_atoms, atom_id: int):
        if char not in self.char_embeddings:
            atom = semantic_atoms.atoms.get(atom_id)
            if atom and hasattr(atom, 'embedding') and atom.embedding is not None:
                # 基于atom embedding + 字符特异性扰动，确保同atom内不同字符有不同嵌入
                char_hash = hash(char) & 0xFFFFFFFF
                rng = torch.Generator(device=self.device).manual_seed(char_hash)
                char_noise = torch.randn(self.atom_dim, generator=rng, device=self.device) * 0.05
                char_emb = torch.tensor(atom.embedding, device=self.device, dtype=torch.float32) + char_noise
            else:
                char_emb = torch.randn(self.atom_dim, device=self.device) * 0.1
            self.char_embeddings[char] = char_emb
        return self.char_embeddings[char]

    def encode_text_to_perturbation(
        self,
        text: str,
        semantic_atoms,
        word_interval: float = 0.1
    ) -> List[Tuple[torch.Tensor, float]]:
        words = list(jieba.cut(text))
        perturbations = []

        for i, word in enumerate(words):
            perturbation = torch.zeros(self.field_dim, device=self.device)
            
            chars = list(word)
            if not chars:
                continue
            
            atom_ids = [semantic_atoms.find_atom_for_char(ch) for ch in chars]
            
            valid_chars = []
            valid_atom_ids = []
            valid_regions = []
            
            for char, atom_id in zip(chars, atom_ids):
                if atom_id >= 0:
                    start_idx, end_idx = semantic_atoms.get_atom_region(atom_id)
                    valid_chars.append(char)
                    valid_atom_ids.append(atom_id)
                    valid_regions.append((start_idx, end_idx))
            
            if not valid_chars:
                continue
            
            char_embeddings = []
            for char, atom_id in zip(valid_chars, valid_atom_ids):
                char_emb = self._ensure_char_embedding(char, semantic_atoms, atom_id)
                char_embeddings.append(char_emb)
            
            if char_embeddings:
                char_emb_matrix = torch.stack(char_embeddings)
                projected = torch.mm(char_emb_matrix, self.input_projection.t())
                
                for j, (start_idx, end_idx) in enumerate(valid_regions):
                    region_size = end_idx - start_idx
                    perturbation[start_idx:end_idx] += projected[j, :region_size]

            if perturbation.abs().sum() > 1e-6:
                perturbation = perturbation / (perturbation.abs().max() + 1e-6) * 0.5
                perturbations.append((perturbation, i * word_interval))

        return perturbations

    def encode_batch_to_perturbations(
        self,
        texts: List[str],
        semantic_atoms,
        word_interval: float = 0.1
    ) -> List[List[Tuple[torch.Tensor, float]]]:
        """批量编码多个文本到扰动（向量化版本）
        
        Args:
            texts: 文本列表
            semantic_atoms: 语义原子管理器
            word_interval: 词间间隔
        
        Returns:
            每个文本的扰动列表
        """
        all_words = [list(jieba.cut(text)) for text in texts]
        
        all_perturbations = []
        
        for text_idx, words in enumerate(all_words):
            perturbations = []
            
            for word_idx, word in enumerate(words):
                perturbation = torch.zeros(self.field_dim, device=self.device)
                
                chars = list(word)
                if not chars:
                    continue
                
                atom_ids = [semantic_atoms.find_atom_for_char(ch) for ch in chars]
                
                valid_chars = []
                valid_atom_ids = []
                valid_regions = []
                
                for char, atom_id in zip(chars, atom_ids):
                    if atom_id >= 0:
                        start_idx, end_idx = semantic_atoms.get_atom_region(atom_id)
                        valid_chars.append(char)
                        valid_atom_ids.append(atom_id)
                        valid_regions.append((start_idx, end_idx))
                
                if not valid_chars:
                    continue
                
                char_embeddings = []
                for char, atom_id in zip(valid_chars, valid_atom_ids):
                    char_emb = self._ensure_char_embedding(char, semantic_atoms, atom_id)
                    char_embeddings.append(char_emb)
                
                if char_embeddings:
                    char_emb_matrix = torch.stack(char_embeddings)
                    projected = torch.mm(char_emb_matrix, self.input_projection.t())
                    
                    for j, (start_idx, end_idx) in enumerate(valid_regions):
                        region_size = end_idx - start_idx
                        perturbation[start_idx:end_idx] += projected[j, :region_size]
                
                if perturbation.abs().sum() > 1e-6:
                    perturbation = perturbation / (perturbation.abs().max() + 1e-6) * 0.5
                    perturbations.append((perturbation, word_idx * word_interval))
            
            all_perturbations.append(perturbations)
        
        return all_perturbations

    def _compute_slot_means(self, h: torch.Tensor) -> torch.Tensor:
        """向量化计算所有槽位的平均激活度，单次操作替代逐原子循环"""
        h_abs = h.abs()
        num_slots = self.field_dim // self.atom_dim
        remainder = self.field_dim % self.atom_dim
        if remainder == 0:
            return h_abs[:num_slots * self.atom_dim].reshape(num_slots, self.atom_dim).mean(dim=1)
        else:
            main = h_abs[:(num_slots - 1) * self.atom_dim].reshape(num_slots - 1, self.atom_dim).mean(dim=1)
            last = h_abs[(num_slots - 1) * self.atom_dim:].mean().unsqueeze(0)
            return torch.cat([main, last])

    def decode_activation_to_text(
        self,
        h: torch.Tensor,
        semantic_atoms,
        top_k: int = 10,
        activation_threshold: float = 0.3
    ) -> str:
        if len(semantic_atoms.atoms) == 0:
            return ""

        # 向量化计算所有槽位的平均激活度
        slot_means = self._compute_slot_means(h)

        atom_emb_matrix = self._get_atom_embeddings_matrix(semantic_atoms)
        sorted_ids = sorted(semantic_atoms.atoms.keys())

        if atom_emb_matrix is not None:
            # 确保嵌入矩阵在正确设备上
            if atom_emb_matrix.device != h.device:
                atom_emb_matrix = atom_emb_matrix.to(h.device)
                self._cached_atom_embeddings = atom_emb_matrix

            projected = torch.mv(self.output_projection, h)

            projected_norm = F.normalize(projected.unsqueeze(0), dim=1)
            emb_norm = F.normalize(atom_emb_matrix, dim=1)
            cos_sims = (projected_norm @ emb_norm.T).squeeze(0)

            # 向量化收集所有原子的区域激活度
            slot_indices = torch.tensor([
                semantic_atoms.atoms[aid].field_region[0] // self.atom_dim
                for aid in sorted_ids
            ], device=h.device)
            region_activations = slot_means[slot_indices]

            # 向量化计算组合分数
            cos_sims_clamped = cos_sims.clamp(min=0)
            combined_scores = 0.6 * cos_sims_clamped + 0.4 * region_activations

            # 过滤 + 排序
            threshold = activation_threshold * 0.3
            valid_mask = combined_scores > threshold
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]

            atom_scores = []
            for idx in valid_indices:
                i = idx.item()
                atom_id = sorted_ids[i]
                atom = semantic_atoms.atoms[atom_id]
                atom_scores.append((atom_id, combined_scores[i].item(), atom.characters))
        else:
            # 回退路径：仅基于区域激活度
            atom_scores = []
            for atom_id in sorted_ids:
                atom = semantic_atoms.atoms[atom_id]
                slot = atom.field_region[0] // self.atom_dim
                if slot < len(slot_means):
                    act = slot_means[slot].item()
                else:
                    act = 0.0
                if act > activation_threshold:
                    atom_scores.append((atom_id, act, atom.characters))

        atom_scores.sort(key=lambda x: x[1], reverse=True)

        output_chars = []
        used_chars = set()
        for atom_id, score, chars in atom_scores[:top_k * 2]:
            if len(chars) > 0:
                best_char = self._select_best_char(chars, atom_id, semantic_atoms)
                if best_char not in used_chars:
                    output_chars.append(best_char)
                    used_chars.add(best_char)
            if len(output_chars) >= top_k:
                break

        return "".join(output_chars)

    def _select_best_char(self, chars, atom_id, semantic_atoms):
        if len(chars) == 1:
            return chars[0]

        atom = semantic_atoms.atoms.get(atom_id)
        if not atom or not hasattr(atom, 'embedding') or atom.embedding is None:
            return chars[0]

        atom_emb = torch.tensor(atom.embedding, device=self.device, dtype=torch.float32)
        
        cached_chars = [ch for ch in chars if ch in self.char_embeddings]
        
        if not cached_chars:
            return chars[0]
        
        if len(cached_chars) == 1:
            return cached_chars[0]
        
        char_emb_matrix = torch.stack([self.char_embeddings[ch] for ch in cached_chars])
        
        atom_emb_norm = F.normalize(atom_emb.unsqueeze(0), dim=1)
        char_emb_norm = F.normalize(char_emb_matrix, dim=1)
        
        similarities = (atom_emb_norm @ char_emb_norm.T).squeeze(0)
        
        best_idx = similarities.argmax().item()
        return cached_chars[best_idx]

    def inject_perturbation(
        self,
        h: torch.Tensor,
        perturbation: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        return h + perturbation * strength

    def evolve_projections(
        self,
        h: torch.Tensor,
        semantic_atoms,
        learning_rate: float = 1e-6,
        gradient_clip: float = 1.0,
        top_k: int = 5
    ):
        """向量化更新top-k最活跃原子的投影层"""
        if len(semantic_atoms.atoms) == 0:
            return

        # 向量化计算所有槽位激活度
        slot_means = self._compute_slot_means(h)

        # 收集每个原子的槽位索引和atom_id
        atom_slot_map = []
        for atom_id, atom in semantic_atoms.atoms.items():
            slot = atom.field_region[0] // self.atom_dim
            if slot < len(slot_means):
                atom_slot_map.append((atom_id, slot))

        if not atom_slot_map:
            return

        # 用slot_means直接取激活度，避免再次遍历
        slot_indices = torch.tensor([s for _, s in atom_slot_map], device=h.device)
        activations = slot_means[slot_indices]

        # top-k最活跃原子
        if len(activations) > top_k:
            topk_vals, topk_idx = torch.topk(activations, top_k)
        else:
            topk_vals = activations
            topk_idx = torch.arange(len(activations), device=h.device)

        if topk_vals.numel() == 0 or topk_vals[0] < 1e-6:
            return

        # 批量更新 input_projection
        for i in range(topk_vals.numel()):
            act = topk_vals[i].item()
            if act < 1e-6:
                break
            idx = topk_idx[i].item()
            atom_id, slot = atom_slot_map[idx]
            atom = semantic_atoms.atoms.get(atom_id)
            if atom is None:
                continue

            atom_emb = torch.tensor(atom.embedding, device=self.device, dtype=torch.float32)
            start_idx, end_idx = atom.field_region
            h_region = h[start_idx:end_idx]
            projected = self.input_projection[start_idx:end_idx, :] @ atom_emb
            residual = h_region - projected
            scaled_lr = learning_rate * min(act, 1.0)
            delta_in = scaled_lr * torch.outer(residual, atom_emb)
            self.input_projection[start_idx:end_idx, :] += delta_in

        self.input_projection = torch.clamp(self.input_projection, -gradient_clip, gradient_clip)

        # 批量更新 output_projection
        projected_h = self.output_projection @ h
        delta_out = torch.zeros_like(self.output_projection)
        total_weight = 0.0
        for i in range(topk_vals.numel()):
            act = topk_vals[i].item()
            if act < 1e-6:
                break
            idx = topk_idx[i].item()
            atom_id, slot = atom_slot_map[idx]
            atom = semantic_atoms.atoms.get(atom_id)
            if atom is None:
                continue
            atom_emb = torch.tensor(atom.embedding, device=self.device, dtype=torch.float32)
            residual = atom_emb - projected_h
            delta_out += act * torch.outer(residual, h)
            total_weight += act

        if total_weight > 1e-6:
            self.output_projection += learning_rate * delta_out / total_weight
        self.output_projection = torch.clamp(self.output_projection, -gradient_clip, gradient_clip)
