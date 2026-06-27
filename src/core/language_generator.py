"""
语言生成模块 - 基于语义原子的文本生成
支持续写、问答、创作等多种生成模式
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import jieba
import logging

from src.core.semantic_atoms import SemanticAtomManager
from src.data.unified_knowledge_manager import UnifiedKnowledgeManager
from src.core.field_interface import FieldInterface


class LanguageGenerator:
    """基于语义原子的语言生成器"""
    
    def __init__(
        self,
        semantic_atoms: SemanticAtomManager,
        knowledge_manager: UnifiedKnowledgeManager,
        field_dim: int = 512,
        atom_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self._logger = logging.getLogger("LanguageGenerator")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        
        self.semantic_atoms = semantic_atoms
        self.knowledge_manager = knowledge_manager
        self.field_dim = field_dim
        self.atom_dim = atom_dim
        self.device = device
        
        self.field_interface = FieldInterface(field_dim, atom_dim, device)
        
        self.field_state = torch.zeros(field_dim, device=device)
        
        self.pmi_cache = {}
        self.char_freq_cache = {}
        self.pmi_index = {}
        self.idiom_set = set()
        self.common_phrases = {}
        
        self._build_pmi_index()
        self._load_linguistic_resources()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        mode: str = 'continue',
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """生成文本
        
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            mode: 生成模式 ('continue', 'qa', 'creative')
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
        
        Returns:
            生成的文本
        """
        if mode == 'continue':
            return self._generate_continue(prompt, max_length, temperature, top_k, top_p)
        elif mode == 'qa':
            return self._generate_qa(prompt, max_length, temperature, top_k, top_p)
        elif mode == 'creative':
            return self._generate_creative(prompt, max_length, temperature, top_k, top_p)
        else:
            raise ValueError(f"不支持的生成模式: {mode}")
    
    def _generate_continue(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> str:
        """续写模式生成 - 添加连贯性验证"""
        self._reset_field_state()
        
        self._inject_prompt(prompt)
        
        generated = prompt
        context_chars = list(prompt)
        
        for _ in range(max_length):
            next_char = self.sample_next_char(
                context_chars,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            if next_char is None or next_char == '<EOS>':
                break
            
            if len(generated) > 0 and not self._is_coherent_combination(generated[-1], next_char):
                retry_char = self.sample_next_char(
                    context_chars,
                    temperature=temperature * 0.5,
                    top_k=top_k // 2,
                    top_p=top_p * 0.9
                )
                if retry_char and self._is_coherent_combination(generated[-1], retry_char):
                    next_char = retry_char
            
            generated += next_char
            context_chars.append(next_char)
            
            if len(context_chars) > 20:
                context_chars = context_chars[-20:]
            
            self._update_field_for_char(next_char)
        
        coherence_score = self.evaluate_coherence(generated)
        self._logger.info(f"生成连贯性分数: {coherence_score:.2f}")
        
        return generated
    
    def _generate_qa(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> str:
        """问答模式生成 - 增强知识检索"""
        knowledge_context = self._retrieve_knowledge(prompt)
        
        self._reset_field_state()
        
        if knowledge_context:
            self._inject_knowledge_context(knowledge_context)
        
        self._inject_prompt(prompt)
        
        qa_prompt = prompt + "答案是："
        generated = qa_prompt
        context_chars = list(qa_prompt)
        
        for _ in range(max_length):
            next_char = self.sample_next_char(
                context_chars,
                temperature=temperature * 0.7,
                top_k=top_k,
                top_p=top_p,
                knowledge_context=knowledge_context
            )
            
            if next_char is None or next_char in ['。', '！', '？', '\n']:
                if next_char:
                    generated += next_char
                break
            
            if len(generated) > 0 and not self._is_coherent_combination(generated[-1], next_char):
                retry_char = self.sample_next_char(
                    context_chars,
                    temperature=temperature * 0.4,
                    top_k=top_k // 2,
                    top_p=top_p * 0.85,
                    knowledge_context=knowledge_context
                )
                if retry_char and self._is_coherent_combination(generated[-1], retry_char):
                    next_char = retry_char
            
            generated += next_char
            context_chars.append(next_char)
            
            if len(context_chars) > 30:
                context_chars = context_chars[-30:]
        
        coherence_score = self.evaluate_coherence(generated)
        self._logger.info(f"问答生成连贯性分数: {coherence_score:.2f}")
        
        return generated
    
    def _generate_creative(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> str:
        """创作模式生成"""
        self._reset_field_state()
        
        self._inject_prompt(prompt)
        
        generated = prompt
        context_chars = list(prompt)
        
        creative_temperature = temperature * 1.2
        
        for i in range(max_length):
            current_temp = creative_temperature * (1.0 + 0.3 * np.sin(i / 10.0))
            
            next_char = self.sample_next_char(
                context_chars,
                temperature=current_temp,
                top_k=top_k,
                top_p=top_p,
                use_creativity_boost=True
            )
            
            if next_char is None:
                break
            
            generated += next_char
            context_chars.append(next_char)
            
            if len(context_chars) > 25:
                context_chars = context_chars[-25:]
            
            self._update_field_for_char(next_char)
        
        return generated
    
    def generate_with_knowledge(
        self,
        prompt: str,
        knowledge_query: str,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> str:
        """知识增强生成
        
        Args:
            prompt: 输入提示
            knowledge_query: 知识检索查询
            max_length: 最大生成长度
            temperature: 采样温度
        
        Returns:
            生成的文本
        """
        knowledge_context = self._retrieve_knowledge(knowledge_query)
        
        if not knowledge_context:
            return self.generate(prompt, max_length, mode='continue', temperature=temperature)
        
        self._reset_field_state()
        self._inject_knowledge_context(knowledge_context)
        self._inject_prompt(prompt)
        
        generated = prompt
        context_chars = list(prompt)
        
        for _ in range(max_length):
            next_char = self.sample_next_char(
                context_chars,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                knowledge_context=knowledge_context
            )
            
            if next_char is None or next_char == '<EOS>':
                break
            
            generated += next_char
            context_chars.append(next_char)
            
            if len(context_chars) > 20:
                context_chars = context_chars[-20:]
        
        return generated
    
    def sample_next_char(
        self,
        context: List[str],
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        knowledge_context: str = None,
        use_creativity_boost: bool = False
    ) -> Optional[str]:
        """采样下一个字符
        
        Args:
            context: 上下文字符列表
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            knowledge_context: 知识上下文
            use_creativity_boost: 是否使用创造性增强
        
        Returns:
            采样的字符
        """
        candidates = self._get_candidate_chars(context, knowledge_context)
        
        if not candidates:
            return None
        
        scores = self._compute_char_scores(context, candidates, knowledge_context)
        
        if use_creativity_boost:
            scores = self._apply_creativity_boost(scores, candidates)
        
        scores = self._apply_pmi_association(context, candidates, scores)
        
        scores = self._apply_field_activation(candidates, scores)
        
        sampled_char = self._sample_with_temperature(
            candidates, scores, temperature, top_k, top_p
        )
        
        return sampled_char
    
    def _get_candidate_chars(
        self,
        context: List[str],
        knowledge_context: str = None
    ) -> List[str]:
        """获取候选字符集合 - 优化选择策略"""
        candidates = set()
        pmi_candidates = set()
        
        if len(context) > 0:
            for j in range(min(3, len(context))):
                context_char = context[-(j+1)]
                if context_char in self.pmi_index:
                    top_pmi_chars = sorted(
                        self.pmi_index[context_char].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:20]
                    pmi_candidates.update([char for char, _ in top_pmi_chars])
        
        if len(context) > 0:
            last_char = context[-1]
            atom_id = self.semantic_atoms.find_atom_for_char(last_char)
            if atom_id >= 0:
                atom = self.semantic_atoms.atoms.get(atom_id)
                if atom:
                    candidates.update(atom.characters)
        
        if len(context) >= 2:
            for i in range(max(0, len(context) - 3), len(context)):
                char = context[i]
                atom_id = self.semantic_atoms.find_atom_for_char(char)
                if atom_id >= 0:
                    atom = self.semantic_atoms.atoms.get(atom_id)
                    if atom:
                        candidates.update(atom.characters)
        
        if knowledge_context:
            knowledge_chars = set(knowledge_context[:200])
            candidates.update(knowledge_chars)
        
        if len(context) >= 2:
            recent_text = ''.join(context[-4:])
            for idiom in list(self.idiom_set)[:1000]:
                if idiom.startswith(recent_text) and len(idiom) > len(recent_text):
                    next_char = idiom[len(recent_text)]
                    candidates.add(next_char)
        
        common_chars = self._get_common_chars()
        
        final_candidates = candidates.union(pmi_candidates)
        final_candidates.update(common_chars)
        
        filtered_candidates = self._filter_incoherent_chars(
            context, list(final_candidates)
        )
        
        return filtered_candidates
    
    def _filter_incoherent_chars(
        self,
        context: List[str],
        candidates: List[str]
    ) -> List[str]:
        """过滤不连贯的字符组合"""
        if len(context) == 0:
            return candidates
        
        filtered = []
        last_char = context[-1]
        
        for candidate in candidates:
            if self._is_coherent_combination(last_char, candidate):
                filtered.append(candidate)
        
        if len(filtered) < 10:
            filtered = candidates
        
        return filtered
    
    def _is_coherent_combination(self, char_a: str, char_b: str) -> bool:
        """检查字符组合是否连贯"""
        if char_a in '，。！？、：；' and char_b in '，。！？、：；':
            return False
        
        if char_a in '（《' and char_b in '）》':
            return False
        
        if char_a in '）》' and char_b in '（《':
            return False
        
        return True
    
    def _get_common_chars(self) -> set:
        """获取常用字符集合"""
        common = set('，。！？、：；""''（）【】《》')
        common.update(set('的是不在有个和这中大为上以人们到'))
        common.update(set('他她说会对去能你我也看就那'))
        return common
    
    def _build_pmi_index(self):
        """构建PMI索引，加速查询"""
        try:
            directed_pairs = self.knowledge_manager.load_directed_pairs()
            for a, b, score in directed_pairs[:50000]:
                if a not in self.pmi_index:
                    self.pmi_index[a] = {}
                self.pmi_index[a][b] = score
            self._logger.info(f"构建PMI索引完成，包含 {len(self.pmi_index)} 个字符")
        except Exception as e:
            self._logger.warning(f"构建PMI索引失败: {e}")
    
    def _load_linguistic_resources(self):
        """加载语言资源（成语、常用短语）"""
        try:
            idioms = self.knowledge_manager.load_idioms()
            self.idiom_set = set(idioms)
            self._logger.info(f"加载成语 {len(self.idiom_set)} 个")
        except Exception as e:
            self._logger.warning(f"加载成语失败: {e}")
        
        try:
            four_char_words = self.knowledge_manager.load_four_char_words()
            for word in four_char_words:
                if len(word) >= 2:
                    for i in range(len(word) - 1):
                        bigram = word[i:i+2]
                        self.common_phrases[bigram] = self.common_phrases.get(bigram, 0) + 1
            self._logger.info(f"加载常用短语 {len(self.common_phrases)} 个")
        except Exception as e:
            self._logger.warning(f"加载常用短语失败: {e}")
    
    def _compute_char_scores(
        self,
        context: List[str],
        candidates: List[str],
        knowledge_context: str = None
    ) -> torch.Tensor:
        """计算候选字符分数"""
        scores = torch.zeros(len(candidates), device=self.device)
        
        if len(context) > 0:
            context_embedding = self._get_context_embedding(context)
            
            for i, char in enumerate(candidates):
                char_embedding = self._get_char_embedding(char)
                
                similarity = F.cosine_similarity(
                    context_embedding.unsqueeze(0),
                    char_embedding.unsqueeze(0)
                )
                scores[i] = similarity.item()
        
        if knowledge_context:
            knowledge_embedding = self._get_text_embedding(knowledge_context[:100])
            for i, char in enumerate(candidates):
                char_embedding = self._get_char_embedding(char)
                knowledge_sim = F.cosine_similarity(
                    knowledge_embedding.unsqueeze(0),
                    char_embedding.unsqueeze(0)
                )
                scores[i] += 0.3 * knowledge_sim.item()
        
        return scores
    
    def _apply_pmi_association(
        self,
        context: List[str],
        candidates: List[str],
        scores: torch.Tensor
    ) -> torch.Tensor:
        """应用PMI关联增强 - 充分利用PMI信息"""
        if len(context) == 0:
            return scores
        
        for i, candidate in enumerate(candidates):
            total_pmi = 0.0
            weight_sum = 0.0
            
            for j in range(min(5, len(context))):
                context_char = context[-(j+1)]
                weight = 1.0 / (j + 1)
                pmi_score = self._get_pmi_score(context_char, candidate)
                total_pmi += weight * pmi_score
                weight_sum += weight
            
            if weight_sum > 0:
                avg_pmi = total_pmi / weight_sum
                scores[i] += 0.5 * avg_pmi
            
            if len(context) >= 1:
                bigram = context[-1] + candidate
                if bigram in self.common_phrases:
                    phrase_freq = self.common_phrases[bigram]
                    scores[i] += 0.3 * min(1.0, phrase_freq / 10.0)
        
        return scores
    
    def _get_pmi_score(self, char_a: str, char_b: str) -> float:
        """获取字符对的PMI分数 - 使用索引加速"""
        cache_key = (char_a, char_b)
        if cache_key in self.pmi_cache:
            return self.pmi_cache[cache_key]
        
        pmi_score = 0.0
        if char_a in self.pmi_index and char_b in self.pmi_index[char_a]:
            pmi_score = self.pmi_index[char_a][char_b]
        
        self.pmi_cache[cache_key] = pmi_score
        return pmi_score
    
    def _apply_field_activation(
        self,
        candidates: List[str],
        scores: torch.Tensor
    ) -> torch.Tensor:
        """应用场激活增强"""
        slot_means = self.field_interface._compute_slot_means(self.field_state)
        
        for i, char in enumerate(candidates):
            atom_id = self.semantic_atoms.find_atom_for_char(char)
            if atom_id >= 0:
                atom = self.semantic_atoms.atoms.get(atom_id)
                if atom:
                    slot = atom.field_region[0] // self.atom_dim
                    if slot < len(slot_means):
                        field_activation = slot_means[slot].item()
                        scores[i] += 0.15 * field_activation
        
        return scores
    
    def _apply_creativity_boost(
        self,
        scores: torch.Tensor,
        candidates: List[str]
    ) -> torch.Tensor:
        """应用创造性增强"""
        noise = torch.randn_like(scores) * 0.1
        scores = scores + noise
        
        rarity_bonus = torch.zeros_like(scores)
        for i, char in enumerate(candidates):
            freq = self._get_char_frequency(char)
            if freq > 0:
                rarity_bonus[i] = 0.1 * (1.0 / (1.0 + freq))
        
        scores = scores + rarity_bonus
        
        return scores
    
    def _get_char_frequency(self, char: str) -> int:
        """获取字符频率"""
        if char in self.char_freq_cache:
            return self.char_freq_cache[char]
        
        freq = 0
        for atom_id, atom in self.semantic_atoms.atoms.items():
            if char in atom.characters:
                freq += atom.activation_count
        
        self.char_freq_cache[char] = freq
        return freq
    
    def _sample_with_temperature(
        self,
        candidates: List[str],
        scores: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Optional[str]:
        """温度采样 - 添加连贯性约束"""
        if temperature <= 0:
            best_idx = scores.argmax().item()
            return candidates[best_idx]
        
        scores_scaled = scores / temperature
        
        if top_k > 0 and top_k < len(candidates):
            top_k_scores, top_k_indices = torch.topk(scores_scaled, min(top_k, len(candidates)))
            filtered_candidates = [candidates[idx.item()] for idx in top_k_indices]
            filtered_scores = top_k_scores
        else:
            filtered_candidates = candidates
            filtered_scores = scores_scaled
        
        probs = F.softmax(filtered_scores, dim=0)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            cutoff_idx = (cumulative_probs > top_p).nonzero(as_tuple=True)[0]
            if len(cutoff_idx) > 0:
                cutoff_idx = cutoff_idx[0].item()
            else:
                cutoff_idx = len(sorted_probs) - 1
            
            kept_indices = sorted_indices[:cutoff_idx + 1]
            kept_probs = sorted_probs[:cutoff_idx + 1]
            kept_probs = kept_probs / kept_probs.sum()
            
            final_candidates = [filtered_candidates[idx.item()] for idx in kept_indices]
            final_probs = kept_probs
        else:
            final_candidates = filtered_candidates
            final_probs = probs
        
        sampled_idx = torch.multinomial(final_probs, 1).item()
        return final_candidates[sampled_idx]
    
    def _reset_field_state(self):
        """重置场状态"""
        self.field_state = torch.zeros(self.field_dim, device=self.device)
    
    def _inject_prompt(self, prompt: str):
        """注入提示文本到场中"""
        perturbations = self.field_interface.encode_text_to_perturbation(
            prompt, self.semantic_atoms
        )
        
        for perturbation, _ in perturbations:
            self.field_state = self.field_interface.inject_perturbation(
                self.field_state, perturbation, strength=0.5
            )
    
    def _update_field_for_char(self, char: str):
        """根据字符更新场状态"""
        atom_id = self.semantic_atoms.find_atom_for_char(char)
        if atom_id >= 0:
            self.semantic_atoms.update_atom_activation(atom_id, 0)
            
            start_idx, end_idx = self.semantic_atoms.get_atom_region(atom_id)
            
            perturbation = torch.zeros(self.field_dim, device=self.device)
            perturbation[start_idx:end_idx] = torch.randn(
                end_idx - start_idx, device=self.device
            ) * 0.1
            
            self.field_state = self.field_interface.inject_perturbation(
                self.field_state, perturbation, strength=0.3
            )
    
    def _retrieve_knowledge(self, query: str) -> Optional[str]:
        """检索知识"""
        wiki_results = self.knowledge_manager.search_wiki(query, limit=3)
        
        if wiki_results:
            knowledge_text = ""
            for title, summary, score in wiki_results:
                knowledge_text += f"{title}：{summary}。"
            return knowledge_text
        
        concept_relations = self.knowledge_manager.get_concept_relations(query)
        if concept_relations:
            knowledge_text = ""
            for relation, related_concepts in concept_relations.items():
                concepts_str = "、".join([c for c, _ in related_concepts[:3]])
                knowledge_text += f"{query}{relation}{concepts_str}。"
            return knowledge_text
        
        return None
    
    def _inject_knowledge_context(self, knowledge_context: str):
        """注入知识上下文到场中"""
        perturbations = self.field_interface.encode_text_to_perturbation(
            knowledge_context, self.semantic_atoms
        )
        
        for perturbation, _ in perturbations:
            self.field_state = self.field_interface.inject_perturbation(
                self.field_state, perturbation, strength=0.3
            )
    
    def _get_context_embedding(self, context: List[str]) -> torch.Tensor:
        """获取上下文嵌入"""
        embeddings = []
        for char in context[-5:]:
            emb = self._get_char_embedding(char)
            embeddings.append(emb)
        
        if not embeddings:
            return torch.zeros(self.atom_dim, device=self.device)
        
        context_emb = torch.stack(embeddings).mean(dim=0)
        return context_emb
    
    def _get_char_embedding(self, char: str) -> torch.Tensor:
        """获取字符嵌入"""
        atom_id = self.semantic_atoms.find_atom_for_char(char)
        
        if atom_id >= 0:
            atom = self.semantic_atoms.atoms.get(atom_id)
            if atom and hasattr(atom, 'embedding'):
                return torch.tensor(atom.embedding, device=self.device, dtype=torch.float32)
        
        return torch.randn(self.atom_dim, device=self.device) * 0.1
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """获取文本嵌入"""
        chars = list(text)
        embeddings = [self._get_char_embedding(ch) for ch in chars[:20]]
        
        if not embeddings:
            return torch.zeros(self.atom_dim, device=self.device)
        
        return torch.stack(embeddings).mean(dim=0)
    
    def beam_search(
        self,
        prompt: str,
        beam_width: int = 5,
        max_length: int = 50,
        length_penalty: float = 1.0
    ) -> List[Tuple[str, float]]:
        """束搜索生成
        
        Args:
            prompt: 输入提示
            beam_width: 束宽度
            max_length: 最大生成长度
            length_penalty: 长度惩罚
        
        Returns:
            [(生成的文本, 分数), ...]
        """
        self._reset_field_state()
        self._inject_prompt(prompt)
        
        beams = [(prompt, 0.0, list(prompt))]
        
        for _ in range(max_length):
            candidates = []
            
            for text, score, context in beams:
                next_chars = self._get_candidate_chars(context)
                
                if not next_chars:
                    candidates.append((text, score, context))
                    continue
                
                char_scores = self._compute_char_scores(context, next_chars)
                char_scores = self._apply_pmi_association(context, next_chars, char_scores)
                
                log_probs = F.log_softmax(char_scores, dim=0)
                
                top_k = min(beam_width, len(next_chars))
                top_scores, top_indices = torch.topk(log_probs, top_k)
                
                for i in range(top_k):
                    idx = top_indices[i].item()
                    char = next_chars[idx]
                    new_score = score + top_scores[i].item()
                    new_text = text + char
                    new_context = context + [char]
                    if len(new_context) > 20:
                        new_context = new_context[-20:]
                    
                    candidates.append((new_text, new_score, new_context))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            all_end = all(
                len(context) > 0 and context[-1] in ['。', '！', '？', '\n']
                for _, _, context in beams
            )
            if all_end:
                break
        
        results = []
        for text, score, _ in beams:
            length = len(text) - len(prompt)
            adjusted_score = score / (length ** length_penalty) if length > 0 else score
            results.append((text, adjusted_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_generation_stats(self) -> Dict[str, any]:
        """获取生成统计信息"""
        return {
            "field_dim": self.field_dim,
            "atom_dim": self.atom_dim,
            "num_atoms": len(self.semantic_atoms.atoms),
            "pmi_cache_size": len(self.pmi_cache),
            "char_freq_cache_size": len(self.char_freq_cache),
            "pmi_index_size": len(self.pmi_index),
            "idiom_count": len(self.idiom_set),
            "phrase_count": len(self.common_phrases),
            "device": self.device
        }
    
    def evaluate_coherence(self, text: str) -> float:
        """评估生成文本的连贯性
        
        Args:
            text: 待评估的文本
        
        Returns:
            连贯性分数 (0-1)
        """
        if len(text) < 2:
            return 0.0
        
        scores = []
        
        for i in range(len(text) - 1):
            char_a = text[i]
            char_b = text[i+1]
            
            pmi_score = self._get_pmi_score(char_a, char_b)
            scores.append(min(1.0, pmi_score / 5.0))
            
            bigram = char_a + char_b
            if bigram in self.common_phrases:
                scores.append(0.8)
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def detect_idiom_usage(self, text: str) -> List[str]:
        """检测文本中使用的成语
        
        Args:
            text: 待检测的文本
        
        Returns:
            检测到的成语列表
        """
        detected = []
        for idiom in self.idiom_set:
            if idiom in text:
                detected.append(idiom)
        return detected
    
    def generate_response(self, intent, entities, context, action):
        """生成对话响应
        
        Args:
            intent: 意图类型
            entities: 提取的实体
            context: 对话上下文
            action: 对话动作
        
        Returns:
            响应文本
        """
        if hasattr(intent, 'value'):
            intent_str = intent.value
        else:
            intent_str = str(intent)
        
        keywords = entities.get('keywords', [])
        if keywords:
            keyword = keywords[0]
            wiki_results = self.knowledge_manager.search_wiki(keyword, limit=1)
            if wiki_results:
                title, summary, _ = wiki_results[0]
                return f"关于「{keyword}」，这是一个值得深入探讨的话题。{summary[:100]}"
        
        return f"我理解您的问题，让我为您解答。"