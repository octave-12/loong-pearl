
import numpy as np
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import json
import re
from datetime import datetime


@dataclass
class MemoryItem:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    importance: float = 0.0
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "importance": self.importance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().timestamp()),
            importance=data.get("importance", 0.0),
        )


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[str, MemoryItem] = {}
        self.access_order: List[str] = []
    
    def get(self, key: str) -> Optional[MemoryItem]:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, item: MemoryItem):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = item
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()
    
    def __len__(self):
        return len(self.cache)


class SimpleEmbedding:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab: Dict[str, np.ndarray] = {}
    
    def _hash_text(self, text: str) -> np.ndarray:
        words = re.findall(r'\w+', text.lower())
        embedding = np.zeros(self.dim, dtype=np.float32)
        
        for word in words:
            if word not in self.vocab:
                word_hash = hashlib.md5(word.encode()).hexdigest()
                np.random.seed(int(word_hash, 16) % (2**32))
                self.vocab[word] = np.random.randn(self.dim).astype(np.float32)
                self.vocab[word] /= (np.linalg.norm(self.vocab[word]) + 1e-8)
            embedding += self.vocab[word]
        
        if len(words) > 0:
            embedding /= len(words)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm
        
        return embedding
    
    def encode(self, text: str) -> np.ndarray:
        return self._hash_text(text)
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))


class ContextMemory:
    def __init__(
        self,
        short_term_size: int = 1000,
        long_term_size: int = 10000,
        embedding_dim: int = 128,
        compression_threshold: int = 500,
        device: str = "cpu",
        max_sessions: int = 100,
        max_history_per_session: int = 50
    ):
        self.short_term_size = short_term_size
        self.long_term_size = long_term_size
        self.embedding_dim = embedding_dim
        self.compression_threshold = compression_threshold
        self.device = device
        self.max_sessions = max_sessions
        self.max_history_per_session = max_history_per_session
        
        self.short_term_window: deque = deque(maxlen=short_term_size)
        self.short_term_cache = LRUCache(capacity=short_term_size)
        
        self.long_term_memory: List[MemoryItem] = []
        self.long_term_index: Dict[str, int] = {}
        
        self.embedding_model = SimpleEmbedding(dim=embedding_dim)
        
        self.compression_cache: Dict[str, str] = {}
        
        self.sessions: Dict[str, Any] = {}
        
        self.stats = {
            "short_term_adds": 0,
            "long_term_adds": 0,
            "retrievals": 0,
            "compressions": 0,
            "sessions_created": 0,
            "sessions_evicted": 0,
        }
    
    def _generate_key(self, text: str, metadata: Dict[str, Any]) -> str:
        content = f"{text}_{json.dumps(metadata, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_to_short_term(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if metadata is None:
            metadata = {}
        
        item = MemoryItem(
            content=text,
            metadata=metadata,
            embedding=self.embedding_model.encode(text)
        )
        
        key = self._generate_key(text, metadata)
        
        self.short_term_window.append(item)
        self.short_term_cache.put(key, item)
        
        self.stats["short_term_adds"] += 1
        
        return key
    
    def add_to_long_term(
        self,
        key: str,
        value: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        if metadata is None:
            metadata = {}
        
        if len(self.long_term_memory) >= self.long_term_size:
            self._evict_long_term()
        
        item = MemoryItem(
            content=value,
            metadata={**metadata, "key": key},
            importance=importance,
            embedding=self.embedding_model.encode(value)
        )
        
        if key in self.long_term_index:
            idx = self.long_term_index[key]
            self.long_term_memory[idx] = item
        else:
            idx = len(self.long_term_memory)
            self.long_term_memory.append(item)
            self.long_term_index[key] = idx
        
        self.stats["long_term_adds"] += 1
        
        return True
    
    def _evict_long_term(self):
        if not self.long_term_memory:
            return
        
        sorted_indices = sorted(
            range(len(self.long_term_memory)),
            key=lambda i: self.long_term_memory[i].importance
        )
        
        num_to_evict = max(1, len(self.long_term_memory) // 10)
        indices_to_remove = set(sorted_indices[:num_to_evict])
        
        new_memory = []
        new_index = {}
        for idx, item in enumerate(self.long_term_memory):
            if idx not in indices_to_remove:
                new_idx = len(new_memory)
                new_memory.append(item)
                key = item.metadata.get("key", f"item_{idx}")
                new_index[key] = new_idx
        
        self.long_term_memory = new_memory
        self.long_term_index = new_index
    
    def retrieve_relevant(
        self,
        query: str,
        k: int = 5,
        include_short_term: bool = True,
        include_long_term: bool = True
    ) -> List[Tuple[MemoryItem, float]]:
        query_embedding = self.embedding_model.encode(query)
        
        candidates: List[Tuple[MemoryItem, float]] = []
        
        if include_short_term:
            for item in self.short_term_window:
                if item.embedding is not None:
                    similarity = self.embedding_model.similarity(query_embedding, item.embedding)
                    candidates.append((item, similarity))
        
        if include_long_term:
            for item in self.long_term_memory:
                if item.embedding is not None:
                    similarity = self.embedding_model.similarity(query_embedding, item.embedding)
                    importance_weight = 1.0 + item.importance
                    weighted_similarity = similarity * importance_weight
                    candidates.append((item, weighted_similarity))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        self.stats["retrievals"] += 1
        
        return candidates[:k]
    
    def compress_memory(self, force: bool = False) -> Dict[str, Any]:
        total_items = len(self.short_term_window) + len(self.long_term_memory)
        
        if not force and total_items < self.compression_threshold:
            return {"compressed": False, "reason": "below_threshold"}
        
        compressed_items = []
        
        for item in self.short_term_window:
            compressed_content = self._compress_text(item.content)
            compressed_items.append({
                "original_length": len(item.content),
                "compressed_length": len(compressed_content),
                "compression_ratio": len(compressed_content) / max(len(item.content), 1),
                "key_info": self._extract_key_info(item.content),
            })
        
        self.stats["compressions"] += 1
        
        return {
            "compressed": True,
            "total_items": total_items,
            "compression_stats": {
                "short_term": len(self.short_term_window),
                "long_term": len(self.long_term_memory),
                "items_processed": len(compressed_items),
            },
            "average_compression_ratio": np.mean([c["compression_ratio"] for c in compressed_items]) if compressed_items else 0.0,
        }
    
    def _compress_text(self, text: str) -> str:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.compression_cache:
            return self.compression_cache[cache_key]
        
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            compressed = text
        else:
            compressed = '。'.join(sentences[:2] + sentences[-1:])
        
        self.compression_cache[cache_key] = compressed
        return compressed
    
    def _extract_key_info(self, text: str) -> List[str]:
        key_info = []
        
        sentences = re.split(r'[。！？.!?]', text)
        for sentence in sentences:
            words = re.findall(r'\w+', sentence)
            if len(words) >= 3:
                key_info.append(sentence.strip())
        
        return key_info[:5]
    
    def get_context_window(self, size: int = 10) -> List[MemoryItem]:
        items = list(self.short_term_window)
        return items[-size:] if len(items) > size else items
    
    def get_conversation_history(self, max_turns: int = 5) -> List[Dict[str, Any]]:
        history = []
        items = self.get_context_window(size=max_turns * 2)
        
        for item in items:
            role = item.metadata.get("role", "unknown")
            history.append({
                "role": role,
                "content": item.content,
                "timestamp": item.timestamp,
            })
        
        return history
    
    def update_importance(self, key: str, new_importance: float) -> bool:
        if key not in self.long_term_index:
            return False
        
        idx = self.long_term_index[key]
        self.long_term_memory[idx].importance = new_importance
        return True
    
    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            "short_term_size": len(self.short_term_window),
            "short_term_capacity": self.short_term_size,
            "long_term_size": len(self.long_term_memory),
            "long_term_capacity": self.long_term_size,
            "compression_cache_size": len(self.compression_cache),
            "stats": self.stats.copy(),
            "utilization": {
                "short_term": len(self.short_term_window) / self.short_term_size,
                "long_term": len(self.long_term_memory) / self.long_term_size,
            },
        }
    
    def clear_short_term(self):
        self.short_term_window.clear()
        self.short_term_cache.clear()
    
    def clear_long_term(self):
        self.long_term_memory.clear()
        self.long_term_index.clear()
    
    def clear_all(self):
        self.clear_short_term()
        self.clear_long_term()
        self.compression_cache.clear()
    
    def save_state(self, filepath: str):
        state = {
            "short_term": [item.to_dict() for item in self.short_term_window],
            "long_term": [item.to_dict() for item in self.long_term_memory],
            "stats": self.stats,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load_state(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.clear_all()
        
        for item_data in state.get("short_term", []):
            item = MemoryItem.from_dict(item_data)
            item.embedding = self.embedding_model.encode(item.content)
            self.short_term_window.append(item)
        
        for item_data in state.get("long_term", []):
            item = MemoryItem.from_dict(item_data)
            item.embedding = self.embedding_model.encode(item.content)
            key = item.metadata.get("key", f"item_{len(self.long_term_memory)}")
            idx = len(self.long_term_memory)
            self.long_term_memory.append(item)
            self.long_term_index[key] = idx
        
        self.stats = state.get("stats", self.stats)
    
    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """获取或创建会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话上下文字典，包含:
            - session_id: 会话ID
            - history: 对话历史列表
            - current_intent: 当前意图
            - current_entities: 当前实体
            - topic_stack: 话题栈
            - referenced_entities: 引用实体
            - last_topic: 最后话题
            - state: 会话状态
            - turn_count: 对话轮数
            - created_at: 创建时间
            - updated_at: 更新时间
        """
        if session_id not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                self._evict_oldest_session()
            
            self.sessions[session_id] = {
                "session_id": session_id,
                "history": [],
                "current_intent": None,
                "current_entities": {},
                "topic_stack": [],
                "referenced_entities": {},
                "last_topic": None,
                "state": "init",
                "turn_count": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            self.stats["sessions_created"] += 1
        
        return self.sessions[session_id]
    
    def _evict_oldest_session(self):
        """驱逐最旧的会话（LRU策略）"""
        if not self.sessions:
            return
        
        oldest_id = min(
            self.sessions.keys(),
            key=lambda sid: self.sessions[sid]["updated_at"]
        )
        del self.sessions[oldest_id]
        self.stats["sessions_evicted"] += 1
    
    def update_session(self, session_id: str, context: Dict[str, Any]):
        """更新会话
        
        Args:
            session_id: 会话ID
            context: 会话上下文
        """
        if len(context.get("history", [])) > self.max_history_per_session:
            context["history"] = context["history"][-self.max_history_per_session:]
        
        context["updated_at"] = datetime.now().isoformat()
        self.sessions[session_id] = context
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话（不创建）
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话上下文，不存在返回None
        """
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功删除
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID
        
        Returns:
            会话ID列表
        """
        return list(self.sessions.keys())
    
    def get_session_count(self) -> int:
        """获取会话数量
        
        Returns:
            活跃会话数量
        """
        return len(self.sessions)
    
    def add_turn_to_session(
        self,
        session_id: str,
        user_input: str,
        system_response: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ):
        """添加一轮对话到会话
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            system_response: 系统回复
            intent: 意图类型
            entities: 提取的实体
        """
        context = self.get_or_create_session(session_id)
        
        turn = {
            "turn": context["turn_count"],
            "user": user_input,
            "system": system_response,
            "intent": intent,
            "entities": entities or {},
            "timestamp": datetime.now().isoformat()
        }
        
        context["history"].append(turn)
        context["turn_count"] += 1
        context["updated_at"] = datetime.now().isoformat()
        
        self.update_session(session_id, context)
    
    def get_relevant_context(
        self,
        session_id: str,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """获取相关上下文
        
        Args:
            session_id: 会话ID
            query: 查询文本
            top_k: 返回top-k个相关对话
            
        Returns:
            相关对话列表
        """
        context = self.sessions.get(session_id)
        if not context:
            return []
        
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_turns = []
        for turn in context["history"]:
            turn_text = turn["user"] + " " + turn["system"]
            turn_words = set(re.findall(r'\w+', turn_text.lower()))
            overlap = len(query_words & turn_words)
            if overlap > 0:
                scored_turns.append((turn, overlap))
        
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        return [turn for turn, _ in scored_turns[:top_k]]