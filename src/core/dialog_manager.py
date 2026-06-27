"""
多轮对话管理器 - 实现对话状态跟踪、意图识别、上下文理解
支持：状态机、意图识别、实体提取、对话策略、上下文管理
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from enum import Enum
import logging
import jieba
import re
from datetime import datetime


class IntentType(Enum):
    """意图类型枚举"""
    QA = "qa"
    CHITCHAT = "chitchat"
    TASK = "task"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"


class DialogState(Enum):
    """对话状态枚举"""
    INIT = "init"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    CLARIFYING = "clarifying"
    COMPLETED = "completed"
    ERROR = "error"


class DialogContext:
    """对话上下文"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []
        self.current_intent: Optional[IntentType] = None
        self.current_entities: Dict[str, Any] = {}
        self.topic_stack: List[str] = []
        self.referenced_entities: Dict[str, Any] = {}
        self.last_topic: Optional[str] = None
        self.state = DialogState.INIT
        self.turn_count = 0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def add_turn(self, user_input: str, system_response: str, 
                 intent: IntentType, entities: Dict[str, Any]):
        """添加一轮对话"""
        self.history.append({
            "turn": self.turn_count,
            "user": user_input,
            "system": system_response,
            "intent": intent.value,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        })
        self.turn_count += 1
        self.updated_at = datetime.now()
    
    def get_last_n_turns(self, n: int = 3) -> List[Dict[str, Any]]:
        """获取最近n轮对话"""
        return self.history[-n:] if self.history else []
    
    def update_topic(self, topic: str):
        """更新话题"""
        if topic and topic != self.last_topic:
            if self.last_topic:
                self.topic_stack.append(self.last_topic)
            self.last_topic = topic
            self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """序列化"""
        return {
            "session_id": self.session_id,
            "history": self.history,
            "current_intent": self.current_intent.value if self.current_intent else None,
            "current_entities": self.current_entities,
            "topic_stack": self.topic_stack,
            "referenced_entities": self.referenced_entities,
            "last_topic": self.last_topic,
            "state": self.state.value,
            "turn_count": self.turn_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class IntentDetector:
    """意图识别器"""
    
    def __init__(self, semantic_atoms=None, knowledge_manager=None):
        self.semantic_atoms = semantic_atoms
        self.knowledge_manager = knowledge_manager
        self._logger = logging.getLogger("IntentDetector")
        
        self.intent_keywords = {
            IntentType.QA: [
                "什么是", "是什么", "为什么", "怎么", "如何", "哪些", "哪个",
                "请问", "解释", "介绍", "定义", "原理", "机制", "特点",
                "what", "why", "how", "which", "explain", "describe"
            ],
            IntentType.CHITCHAT: [
                "你好", "您好", "谢谢", "再见", "好的", "嗯", "哦",
                "哈哈", "呵呵", "觉得", "认为", "感觉", "喜欢",
                "hello", "hi", "thanks", "bye", "good"
            ],
            IntentType.TASK: [
                "帮我", "请帮我", "帮我做", "帮我查", "帮我找",
                "创建", "删除", "修改", "更新", "添加", "设置",
                "help me", "create", "delete", "update", "add", "set"
            ],
            IntentType.CLARIFICATION: [
                "你是说", "是指", "意思是", "具体是", "更详细",
                "再说一遍", "没听懂", "不明白", "什么意思",
                "clarify", "mean", "detail"
            ],
            IntentType.FEEDBACK: [
                "很好", "不错", "太棒了", "有问题", "不对", "错误",
                "满意", "不满意", "建议", "反馈",
                "good", "bad", "wrong", "feedback"
            ]
        }
        
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """构建关键词索引"""
        self.keyword_to_intent = {}
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                self.keyword_to_intent[keyword] = intent
    
    def detect_intent(self, text: str, context: DialogContext = None) -> Tuple[IntentType, float]:
        """识别意图
        
        Returns:
            (意图类型, 置信度)
        """
        scores = defaultdict(float)
        
        for keyword, intent in self.keyword_to_intent.items():
            if keyword in text.lower():
                scores[intent] += 1.0 / len(keyword)
        
        if context and context.last_topic:
            if any(q in text for q in ["它", "这个", "那", "其"]):
                scores[IntentType.QA] += 0.5
        
        if text.endswith("？") or text.endswith("?"):
            scores[IntentType.QA] += 0.3
        
        if len(text) < 10 and any(ch in text for ch in ["好", "嗯", "哦", "是", "对"]):
            scores[IntentType.CHITCHAT] += 0.4
        
        if not scores:
            return (IntentType.UNKNOWN, 0.0)
        
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_intents[0]
        
        confidence = min(1.0, best_score / 2.0)
        
        return (best_intent, confidence)
    
    def detect_intent_semantic(self, text: str, context: DialogContext = None) -> Tuple[IntentType, float]:
        """基于语义的意图识别（使用语义原子）"""
        if self.semantic_atoms is None:
            return self.detect_intent(text, context)
        
        text_embedding = self._get_text_embedding(text)
        
        intent_embeddings = {}
        for intent, keywords in self.intent_keywords.items():
            keyword_embs = []
            for keyword in keywords:
                emb = self._get_text_embedding(keyword)
                if emb is not None:
                    keyword_embs.append(emb)
            if keyword_embs:
                intent_embeddings[intent] = np.mean(keyword_embs, axis=0)
        
        if not intent_embeddings or text_embedding is None:
            return self.detect_intent(text, context)
        
        similarities = {}
        for intent, intent_emb in intent_embeddings.items():
            sim = np.dot(text_embedding, intent_emb) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(intent_emb) + 1e-10
            )
            similarities[intent] = sim
        
        sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_sim = sorted_intents[0]
        
        keyword_intent, keyword_conf = self.detect_intent(text, context)
        
        if keyword_conf > 0.5:
            return (keyword_intent, keyword_conf)
        else:
            confidence = (best_sim + 1) / 2
            return (best_intent, confidence)
    
    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本嵌入（基于语义原子）"""
        if self.semantic_atoms is None:
            return None
        
        embeddings = []
        for char in text:
            atom_id = self.semantic_atoms.find_atom_for_char(char)
            if atom_id >= 0:
                atom = self.semantic_atoms.atoms.get(atom_id)
                if atom:
                    embeddings.append(atom.embedding)
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)


class EntityExtractor:
    """实体提取器"""
    
    def __init__(self, knowledge_manager=None):
        self.knowledge_manager = knowledge_manager
        self._logger = logging.getLogger("EntityExtractor")
        
        self.entity_patterns = {
            "time": [
                r"\d{4}年\d{1,2}月\d{1,2}日",
                r"\d{1,2}月\d{1,2}日",
                r"\d{1,2}点\d{1,2}分",
                r"今天|明天|昨天|后天|前天",
                r"上午|下午|晚上|中午|早上"
            ],
            "number": [
                r"\d+",
                r"一|二|三|四|五|六|七|八|九|十",
                r"百|千|万|亿"
            ],
            "location": [
                r"[北上广深重武成杭南西苏][市京州]",
                r"[\u4e00-\u9fa5]{2,}(省|市|县|区|镇|村)"
            ]
        }
    
    def extract_entities(self, text: str, context: DialogContext = None) -> Dict[str, Any]:
        """提取实体"""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text)
                matches.extend(found)
            if matches:
                entities[entity_type] = matches
        
        words = list(jieba.cut(text))
        entities["keywords"] = [w for w in words if len(w) >= 2 and w not in ["什么", "怎么", "如何", "为什么"]]
        
        if context and context.referenced_entities:
            pronouns = ["它", "这个", "那个", "其"]
            for pronoun in pronouns:
                if pronoun in text:
                    entities["referenced"] = context.referenced_entities
                    break
        
        if self.knowledge_manager:
            entities["knowledge"] = self._extract_knowledge_entities(text)
        
        return entities
    
    def _extract_knowledge_entities(self, text: str) -> Dict[str, Any]:
        """从知识库提取实体"""
        knowledge_entities = {}
        
        if self.knowledge_manager:
            idioms = self.knowledge_manager.load_idioms()
            found_idioms = [idiom for idiom in idioms if idiom in text]
            if found_idioms:
                knowledge_entities["idioms"] = found_idioms
        
        return knowledge_entities
    
    def update_referenced_entities(self, entities: Dict[str, Any], context: DialogContext):
        """更新引用实体"""
        if "keywords" in entities:
            for keyword in entities["keywords"]:
                if len(keyword) >= 2:
                    context.referenced_entities[keyword] = {
                        "text": keyword,
                        "turn": context.turn_count
                    }


class DialogPolicy:
    """对话策略"""
    
    def __init__(self):
        self._logger = logging.getLogger("DialogPolicy")
        
        self.policy_rules = {
            (IntentType.QA, DialogState.WAITING_INPUT): self._qa_policy,
            (IntentType.CHITCHAT, DialogState.WAITING_INPUT): self._chitchat_policy,
            (IntentType.TASK, DialogState.WAITING_INPUT): self._task_policy,
            (IntentType.CLARIFICATION, DialogState.WAITING_INPUT): self._clarification_policy,
            (IntentType.FEEDBACK, DialogState.WAITING_INPUT): self._feedback_policy,
        }
    
    def select_action(self, intent: IntentType, state: DialogState, 
                     context: DialogContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """选择对话动作"""
        key = (intent, state)
        
        if key in self.policy_rules:
            return self.policy_rules[key](context, entities)
        
        return {
            "action": "default",
            "response_type": "text",
            "requires_clarification": False
        }
    
    def _qa_policy(self, context: DialogContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """问答策略"""
        return {
            "action": "answer",
            "response_type": "qa",
            "requires_clarification": False,
            "update_topic": True,
            "store_reference": True
        }
    
    def _chitchat_policy(self, context: DialogContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """闲聊策略"""
        return {
            "action": "chat",
            "response_type": "chitchat",
            "requires_clarification": False,
            "update_topic": False,
            "store_reference": False
        }
    
    def _task_policy(self, context: DialogContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """任务策略"""
        required_slots = ["action", "target"]
        missing_slots = [slot for slot in required_slots if slot not in entities]
        
        if missing_slots:
            return {
                "action": "request_slot",
                "response_type": "clarification",
                "requires_clarification": True,
                "missing_slots": missing_slots,
                "update_topic": False,
                "store_reference": False
            }
        
        return {
            "action": "execute_task",
            "response_type": "task",
            "requires_clarification": False,
            "update_topic": True,
            "store_reference": True
        }
    
    def _clarification_policy(self, context: DialogContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """澄清策略"""
        return {
            "action": "clarify",
            "response_type": "clarification",
            "requires_clarification": False,
            "update_topic": False,
            "store_reference": False
        }
    
    def _feedback_policy(self, context: DialogContext, entities: Dict[str, Any]) -> Dict[str, Any]:
        """反馈策略"""
        return {
            "action": "acknowledge",
            "response_type": "feedback",
            "requires_clarification": False,
            "update_topic": False,
            "store_reference": False
        }


class LanguageGenerator:
    """语言生成器（简化版）"""
    
    def __init__(self, knowledge_manager=None, field_interface=None, semantic_atoms=None):
        self.knowledge_manager = knowledge_manager
        self.field_interface = field_interface
        self.semantic_atoms = semantic_atoms
        self._logger = logging.getLogger("LanguageGenerator")
    
    def generate_response(self, intent: IntentType, entities: Dict[str, Any],
                         context: DialogContext, action: Dict[str, Any]) -> str:
        """生成回复"""
        if intent == IntentType.QA:
            return self._generate_qa_response(entities, context)
        elif intent == IntentType.CHITCHAT:
            return self._generate_chitchat_response(entities, context)
        elif intent == IntentType.TASK:
            return self._generate_task_response(entities, context, action)
        elif intent == IntentType.CLARIFICATION:
            return self._generate_clarification_response(entities, context)
        elif intent == IntentType.FEEDBACK:
            return self._generate_feedback_response(entities, context)
        else:
            return self._generate_default_response(entities, context)
    
    def _generate_qa_response(self, entities: Dict[str, Any], context: DialogContext) -> str:
        """生成问答回复"""
        keywords = entities.get("keywords", [])
        
        if not keywords:
            return "请问您想了解什么内容？"
        
        main_keyword = keywords[0]
        
        if self.knowledge_manager:
            wiki_results = self.knowledge_manager.search_wiki(main_keyword, limit=3)
            if wiki_results:
                title, summary, score = wiki_results[0]
                if score > 0.5:
                    return f"{title}：{summary}"
            
            relations = self.knowledge_manager.get_concept_relations(main_keyword)
            if relations:
                response_parts = [f"关于「{main_keyword}」的相关信息："]
                for relation, related_concepts in list(relations.items())[:3]:
                    concepts_str = "、".join([c for c, _ in related_concepts[:3]])
                    response_parts.append(f"  {relation}：{concepts_str}")
                return "\n".join(response_parts)
        
        if context.last_topic:
            return f"关于{context.last_topic}，{main_keyword}是一个重要方面。具体来说..."
        
        return f"关于「{main_keyword}」，这是一个值得深入探讨的话题。"
    
    def _generate_chitchat_response(self, entities: Dict[str, Any], context: DialogContext) -> str:
        """生成闲聊回复"""
        text = " ".join(entities.get("keywords", []))
        
        if any(word in text for word in ["你好", "您好", "hello", "hi"]):
            return "你好！很高兴和你交流。有什么我可以帮助你的吗？"
        elif any(word in text for word in ["谢谢", "thanks"]):
            return "不客气！如果还有其他问题，随时可以问我。"
        elif any(word in text for word in ["再见", "bye"]):
            return "再见！期待下次和你交流。"
        else:
            return "嗯，我明白了。请继续说。"
    
    def _generate_task_response(self, entities: Dict[str, Any], 
                               context: DialogContext, action: Dict[str, Any]) -> str:
        """生成任务回复"""
        if action.get("requires_clarification"):
            missing = action.get("missing_slots", [])
            return f"为了帮您完成任务，还需要更多信息：{', '.join(missing)}"
        
        return "好的，我正在为您处理这个任务。"
    
    def _generate_clarification_response(self, entities: Dict[str, Any], 
                                        context: DialogContext) -> str:
        """生成澄清回复"""
        last_turns = context.get_last_n_turns(1)
        if last_turns:
            last_response = last_turns[0]["system"]
            return f"让我再解释一下：{last_response}"
        return "请问您需要我澄清哪部分内容？"
    
    def _generate_feedback_response(self, entities: Dict[str, Any], 
                                   context: DialogContext) -> str:
        """生成反馈回复"""
        return "感谢您的反馈！我会继续改进。"
    
    def _generate_default_response(self, entities: Dict[str, Any], 
                                  context: DialogContext) -> str:
        """生成默认回复"""
        keywords = entities.get("keywords", [])
        if keywords:
            return f"关于「{keywords[0]}」，我正在思考..."
        return "我理解了。请继续。"


class SessionManager:
    """会话管理器 - 管理多轮对话会话"""
    
    def __init__(self, max_sessions: int = 100, max_history_per_session: int = 50):
        self.sessions: Dict[str, DialogContext] = {}
        self.max_sessions = max_sessions
        self.max_history_per_session = max_history_per_session
        self._logger = logging.getLogger("SessionManager")
    
    def get_or_create_session(self, session_id: str) -> DialogContext:
        """获取或创建会话"""
        if session_id not in self.sessions:
            if len(self.sessions) >= self.max_sessions:
                self._evict_oldest_session()
            self.sessions[session_id] = DialogContext(session_id)
            self._logger.info(f"创建新会话: {session_id}")
        return self.sessions[session_id]
    
    def _evict_oldest_session(self):
        """驱逐最旧的会话"""
        if not self.sessions:
            return
        oldest_id = min(self.sessions.keys(), 
                       key=lambda sid: self.sessions[sid].updated_at)
        del self.sessions[oldest_id]
        self._logger.info(f"驱逐旧会话: {oldest_id}")
    
    def update_session(self, session_id: str, context: DialogContext):
        """更新会话"""
        if len(context.history) > self.max_history_per_session:
            context.history = context.history[-self.max_history_per_session:]
        self.sessions[session_id] = context
    
    def get_relevant_context(self, session_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """获取相关上下文"""
        context = self.sessions.get(session_id)
        if not context:
            return []
        
        query_words = set(jieba.cut(query))
        
        scored_turns = []
        for turn in context.history:
            turn_words = set(jieba.cut(turn["user"] + " " + turn["system"]))
            overlap = len(query_words & turn_words)
            if overlap > 0:
                scored_turns.append((turn, overlap))
        
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        return [turn for turn, _ in scored_turns[:top_k]]


class DialogManager:
    """多轮对话管理器"""
    
    def __init__(
        self,
        context_memory=None,
        language_generator: LanguageGenerator = None,
        semantic_atoms=None,
        knowledge_manager=None,
        field_interface=None
    ):
        if context_memory is None:
            from src.core.context_memory import ContextMemory as UnifiedContextMemory
            self.context_memory = UnifiedContextMemory()
            self.session_manager = SessionManager()
            self._use_unified_memory = True
        else:
            self.context_memory = context_memory
            if hasattr(context_memory, 'get_or_create_session'):
                test_session = context_memory.get_or_create_session("__test__")
                if isinstance(test_session, dict):
                    self._use_unified_memory = True
                    self.session_manager = None
                else:
                    self._use_unified_memory = False
                    self.session_manager = context_memory
            else:
                self._use_unified_memory = False
                self.session_manager = SessionManager()
        
        self.semantic_atoms = semantic_atoms
        self.knowledge_manager = knowledge_manager
        self.field_interface = field_interface
        
        self.intent_detector = IntentDetector(semantic_atoms, knowledge_manager)
        self.entity_extractor = EntityExtractor(knowledge_manager)
        self.dialog_policy = DialogPolicy()
        self.language_generator = language_generator or LanguageGenerator(
            knowledge_manager, field_interface, semantic_atoms
        )
        
        self._logger = logging.getLogger("DialogManager")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
    
    def process_input(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """处理用户输入
        
        Args:
            user_input: 用户输入文本
            session_id: 会话ID
        
        Returns:
            {
                "response": 系统回复,
                "intent": 意图类型,
                "entities": 提取的实体,
                "state": 对话状态,
                "context": 对话上下文
            }
        """
        context = self._get_session(session_id)
        
        if self._use_unified_memory:
            context["state"] = "processing"
            intent, confidence = self.detect_intent(user_input, self._dict_to_dialog_context(context))
            entities = self.extract_entities(user_input, self._dict_to_dialog_context(context))
            
            self._update_state_dict(context, intent, entities)
            
            action = self.dialog_policy.select_action(
                intent, 
                DialogState.PROCESSING, 
                self._dict_to_dialog_context(context), 
                entities
            )
            
            response = self.generate_response(intent, entities, self._dict_to_dialog_context(context), action)
            
            self._add_turn_to_dict(context, user_input, response, intent, entities)
            
            if action.get("update_topic"):
                keywords = entities.get("keywords", [])
                if keywords:
                    self._update_topic_dict(context, keywords[0])
            
            if action.get("store_reference"):
                self._update_referenced_entities_dict(context, entities)
            
            context["state"] = "waiting_input"
            self._update_session(session_id, context)
            
            self._logger.info(f"会话 {session_id} - 意图: {intent.value}, 置信度: {confidence:.2f}")
            
            return {
                "response": response,
                "intent": intent.value,
                "confidence": confidence,
                "entities": entities,
                "state": context["state"],
                "action": action["action"],
                "turn_count": context["turn_count"]
            }
        else:
            context.state = DialogState.PROCESSING
            
            intent, confidence = self.detect_intent(user_input, context)
            entities = self.extract_entities(user_input, context)
            
            self.update_state(intent, entities, context)
            
            action = self.dialog_policy.select_action(intent, context.state, context, entities)
            
            response = self.generate_response(intent, entities, context, action)
            
            context.add_turn(user_input, response, intent, entities)
            
            if action.get("update_topic"):
                keywords = entities.get("keywords", [])
                if keywords:
                    context.update_topic(keywords[0])
            
            if action.get("store_reference"):
                self.entity_extractor.update_referenced_entities(entities, context)
            
            context.state = DialogState.WAITING_INPUT
            self._update_session(session_id, context)
            
            self._logger.info(f"会话 {session_id} - 意图: {intent.value}, 置信度: {confidence:.2f}")
            
            return {
                "response": response,
                "intent": intent.value,
                "confidence": confidence,
                "entities": entities,
                "state": context.state.value,
                "action": action["action"],
                "turn_count": context.turn_count
            }
    
    def _get_session(self, session_id: str):
        """获取会话（统一接口）"""
        if self._use_unified_memory:
            return self.context_memory.get_or_create_session(session_id)
        else:
            return self.session_manager.get_or_create_session(session_id)
    
    def _update_session(self, session_id: str, context):
        """更新会话（统一接口）"""
        if self._use_unified_memory:
            self.context_memory.update_session(session_id, context)
        else:
            self.session_manager.update_session(session_id, context)
    
    def _dict_to_dialog_context(self, context_dict: Dict[str, Any]) -> DialogContext:
        """将字典转换为DialogContext对象"""
        context = DialogContext(context_dict["session_id"])
        context.history = context_dict.get("history", [])
        context.current_intent = IntentType(context_dict["current_intent"]) if context_dict.get("current_intent") else None
        context.current_entities = context_dict.get("current_entities", {})
        context.topic_stack = context_dict.get("topic_stack", [])
        context.referenced_entities = context_dict.get("referenced_entities", {})
        context.last_topic = context_dict.get("last_topic")
        context.state = DialogState(context_dict.get("state", "init"))
        context.turn_count = context_dict.get("turn_count", 0)
        return context
    
    def _update_state_dict(self, context_dict: Dict[str, Any], intent: IntentType, entities: Dict[str, Any]):
        """更新状态（字典版本）"""
        context_dict["current_intent"] = intent.value
        context_dict["current_entities"] = entities
    
    def _add_turn_to_dict(self, context_dict: Dict[str, Any], user_input: str, 
                         response: str, intent: IntentType, entities: Dict[str, Any]):
        """添加对话轮（字典版本）"""
        turn = {
            "turn": context_dict["turn_count"],
            "user": user_input,
            "system": response,
            "intent": intent.value,
            "entities": entities,
            "timestamp": datetime.now().isoformat()
        }
        context_dict["history"].append(turn)
        context_dict["turn_count"] += 1
        context_dict["updated_at"] = datetime.now().isoformat()
    
    def _update_topic_dict(self, context_dict: Dict[str, Any], topic: str):
        """更新话题（字典版本）"""
        if topic and topic != context_dict.get("last_topic"):
            if context_dict.get("last_topic"):
                context_dict["topic_stack"].append(context_dict["last_topic"])
            context_dict["last_topic"] = topic
            context_dict["updated_at"] = datetime.now().isoformat()
    
    def _update_referenced_entities_dict(self, context_dict: Dict[str, Any], entities: Dict[str, Any]):
        """更新引用实体（字典版本）"""
        if "keywords" in entities:
            for keyword in entities["keywords"]:
                if len(keyword) >= 2:
                    context_dict["referenced_entities"][keyword] = {
                        "text": keyword,
                        "turn": context_dict["turn_count"]
                    }
    
    def detect_intent(self, text: str, context: DialogContext = None) -> Tuple[IntentType, float]:
        """意图识别"""
        if self.semantic_atoms:
            return self.intent_detector.detect_intent_semantic(text, context)
        return self.intent_detector.detect_intent(text, context)
    
    def extract_entities(self, text: str, context: DialogContext = None) -> Dict[str, Any]:
        """实体提取"""
        return self.entity_extractor.extract_entities(text, context)
    
    def update_state(self, intent: IntentType, entities: Dict[str, Any], 
                    context: DialogContext):
        """更新对话状态"""
        context.current_intent = intent
        context.current_entities = entities
    
    def generate_response(self, intent: IntentType, entities: Dict[str, Any],
                         context: DialogContext, action: Dict[str, Any]) -> str:
        """生成回复"""
        return self.language_generator.generate_response(intent, entities, context, action)
    
    def manage_dialog(self, session_id: str):
        """管理对话（获取当前对话状态）"""
        if self._use_unified_memory:
            return self.context_memory.get_or_create_session(session_id)
        else:
            return self.session_manager.get_or_create_session(session_id)
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """获取上下文摘要"""
        context = self._get_session(session_id)
        
        if self._use_unified_memory:
            return {
                "session_id": session_id,
                "turn_count": context.get("turn_count", 0),
                "current_intent": context.get("current_intent"),
                "last_topic": context.get("last_topic"),
                "referenced_entities": list(context.get("referenced_entities", {}).keys()),
                "state": context.get("state")
            }
        else:
            return {
                "session_id": session_id,
                "turn_count": context.turn_count,
                "current_intent": context.current_intent.value if context.current_intent else None,
                "last_topic": context.last_topic,
                "referenced_entities": list(context.referenced_entities.keys()),
                "state": context.state.value
            }
    
    def reset_session(self, session_id: str):
        """重置会话"""
        if self._use_unified_memory:
            if session_id in self.context_memory.sessions:
                del self.context_memory.sessions[session_id]
                self._logger.info(f"重置会话: {session_id}")
        else:
            if session_id in self.session_manager.sessions:
                del self.session_manager.sessions[session_id]
                self._logger.info(f"重置会话: {session_id}")
    
    @property
    def sessions(self):
        """访问会话字典"""
        if self._use_unified_memory:
            return self.context_memory.sessions
        else:
            return self.session_manager.sessions


def create_dialog_manager(
    semantic_atoms=None,
    knowledge_manager=None,
    field_interface=None
) -> DialogManager:
    """创建对话管理器（工厂函数）"""
    return DialogManager(
        semantic_atoms=semantic_atoms,
        knowledge_manager=knowledge_manager,
        field_interface=field_interface
    )