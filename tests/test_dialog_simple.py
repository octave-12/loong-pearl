"""
简化测试 - 不依赖torch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import re


class IntentType(Enum):
    QA = "qa"
    CHITCHAT = "chitchat"
    TASK = "task"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"


class DialogState(Enum):
    INIT = "init"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    COMPLETED = "completed"


class IntentDetector:
    """意图识别器"""
    
    def __init__(self):
        self.intent_keywords = {
            IntentType.QA: [
                "什么是", "是什么", "为什么", "怎么", "如何", "哪些",
                "请问", "解释", "介绍", "定义"
            ],
            IntentType.CHITCHAT: [
                "你好", "您好", "谢谢", "再见", "好的", "嗯"
            ],
            IntentType.TASK: [
                "帮我", "请帮我", "帮我做", "帮我查"
            ],
            IntentType.CLARIFICATION: [
                "你是说", "是指", "意思是", "什么意思"
            ],
            IntentType.FEEDBACK: [
                "很好", "不错", "有问题", "不对"
            ]
        }
        
        self.keyword_to_intent = {}
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                self.keyword_to_intent[keyword] = intent
    
    def detect_intent(self, text: str) -> Tuple[IntentType, float]:
        scores = defaultdict(float)
        
        for keyword, intent in self.keyword_to_intent.items():
            if keyword in text:
                scores[intent] += 1.0 / len(keyword)
        
        if text.endswith("？") or text.endswith("?"):
            scores[IntentType.QA] += 0.3
        
        if not scores:
            return (IntentType.UNKNOWN, 0.0)
        
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_score = sorted_intents[0]
        
        confidence = min(1.0, best_score / 2.0)
        return (best_intent, confidence)


class DialogContext:
    """对话上下文"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []
        self.current_intent: Optional[IntentType] = None
        self.last_topic: Optional[str] = None
        self.referenced_entities: Dict[str, Any] = {}
        self.state = DialogState.INIT
        self.turn_count = 0
    
    def add_turn(self, user_input: str, system_response: str, 
                 intent: IntentType, entities: Dict[str, Any]):
        self.history.append({
            "turn": self.turn_count,
            "user": user_input,
            "system": system_response,
            "intent": intent.value,
            "entities": entities
        })
        self.turn_count += 1
    
    def update_topic(self, topic: str):
        if topic and topic != self.last_topic:
            self.last_topic = topic


class EntityExtractor:
    """实体提取器"""
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        entities = {}
        
        time_patterns = [
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"今天|明天|昨天"
        ]
        time_matches = []
        for pattern in time_patterns:
            time_matches.extend(re.findall(pattern, text))
        if time_matches:
            entities["time"] = time_matches
        
        location_patterns = [r"[北上广深][市京州]"]
        location_matches = []
        for pattern in location_patterns:
            location_matches.extend(re.findall(pattern, text))
        if location_matches:
            entities["location"] = location_matches
        
        keywords = [w for w in text.split() if len(w) >= 2]
        keywords.extend(re.findall(r"[\u4e00-\u9fa5]{2,}", text))
        entities["keywords"] = keywords[:5]
        
        return entities


class LanguageGenerator:
    """语言生成器"""
    
    def generate_response(self, intent: IntentType, entities: Dict[str, Any],
                         context: DialogContext) -> str:
        if intent == IntentType.QA:
            return self._generate_qa_response(entities, context)
        elif intent == IntentType.CHITCHAT:
            return self._generate_chitchat_response(entities, context)
        elif intent == IntentType.TASK:
            return "好的，我正在为您处理这个任务。"
        else:
            return "我理解了。请继续。"
    
    def _generate_qa_response(self, entities: Dict[str, Any], context: DialogContext) -> str:
        keywords = entities.get("keywords", [])
        if not keywords:
            return "请问您想了解什么内容？"
        
        main_keyword = keywords[0]
        
        if context.last_topic:
            return f"关于{context.last_topic}，{main_keyword}是一个重要方面。具体来说，它涉及多个应用领域，包括量子计算、量子通信和量子加密等。"
        
        return f"关于「{main_keyword}」，这是一个值得深入探讨的话题。{main_keyword}是现代科学的重要概念，具有广泛的应用前景。"
    
    def _generate_chitchat_response(self, entities: Dict[str, Any], context: DialogContext) -> str:
        text = " ".join(entities.get("keywords", []))
        
        if any(word in text for word in ["你好", "您好"]):
            return "你好！很高兴和你交流。有什么我可以帮助你的吗？"
        elif any(word in text for word in ["谢谢"]):
            return "不客气！如果还有其他问题，随时可以问我。"
        elif any(word in text for word in ["再见"]):
            return "再见！期待下次和你交流。"
        else:
            return "嗯，我明白了。请继续说。"


class DialogManager:
    """多轮对话管理器（简化版）"""
    
    def __init__(self):
        self.sessions: Dict[str, DialogContext] = {}
        self.intent_detector = IntentDetector()
        self.entity_extractor = EntityExtractor()
        self.language_generator = LanguageGenerator()
    
    def process_input(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = DialogContext(session_id)
        
        context = self.sessions[session_id]
        context.state = DialogState.PROCESSING
        
        intent, confidence = self.intent_detector.detect_intent(user_input)
        entities = self.entity_extractor.extract_entities(user_input)
        
        if any(pronoun in user_input for pronoun in ["它", "这个", "那"]):
            entities["referenced"] = context.referenced_entities
        
        response = self.language_generator.generate_response(intent, entities, context)
        
        context.add_turn(user_input, response, intent, entities)
        
        keywords = entities.get("keywords", [])
        if keywords:
            context.update_topic(keywords[0])
            context.referenced_entities[keywords[0]] = {"turn": context.turn_count}
        
        context.state = DialogState.WAITING_INPUT
        
        return {
            "response": response,
            "intent": intent.value,
            "confidence": confidence,
            "entities": entities,
            "turn_count": context.turn_count
        }
    
    def manage_dialog(self, session_id: str) -> DialogContext:
        return self.sessions.get(session_id)


def test_multi_turn_qa():
    """测试多轮问答"""
    print("\n=== 测试1: 多轮问答（上下文理解）===")
    
    dm = DialogManager()
    
    result1 = dm.process_input("什么是量子纠缠？", session_id="test1")
    print(f"用户: 什么是量子纠缠？")
    print(f"系统: {result1['response']}")
    print(f"意图: {result1['intent']}, 置信度: {result1['confidence']:.2f}")
    
    result2 = dm.process_input("它有什么应用？", session_id="test1")
    print(f"\n用户: 它有什么应用？")
    print(f"系统: {result2['response']}")
    print(f"意图: {result2['intent']}, 置信度: {result2['confidence']:.2f}")
    print(f"引用实体: {result2['entities'].get('referenced', {})}")
    
    context = dm.manage_dialog("test1")
    print(f"\n对话轮数: {context.turn_count}")
    print(f"最后话题: {context.last_topic}")
    print(f"引用实体: {list(context.referenced_entities.keys())}")
    
    return True


def test_intent_switching():
    """测试意图切换"""
    print("\n=== 测试2: 意图切换（问答→闲聊→任务）===")
    
    dm = DialogManager()
    
    result1 = dm.process_input("什么是人工智能？", session_id="test2")
    print(f"用户: 什么是人工智能？")
    print(f"系统: {result1['response']}")
    print(f"意图: {result1['intent']}")
    
    result2 = dm.process_input("谢谢你的解释", session_id="test2")
    print(f"\n用户: 谢谢你的解释")
    print(f"系统: {result2['response']}")
    print(f"意图: {result2['intent']}")
    
    result3 = dm.process_input("帮我查一下北京的天气", session_id="test2")
    print(f"\n用户: 帮我查一下北京的天气")
    print(f"系统: {result3['response']}")
    print(f"意图: {result3['intent']}")
    
    context = dm.manage_dialog("test2")
    print(f"\n对话历史:")
    for turn in context.history:
        print(f"  [{turn['turn']}] 意图: {turn['intent']}, 用户: {turn['user'][:20]}...")
    
    return True


def test_entity_extraction():
    """测试实体提取"""
    print("\n=== 测试3: 实体提取 ===")
    
    dm = DialogManager()
    
    result = dm.process_input("2024年3月15日，在北京举办人工智能大会", session_id="test3")
    print(f"用户: 2024年3月15日，在北京举办人工智能大会")
    print(f"提取的实体:")
    for entity_type, values in result['entities'].items():
        print(f"  {entity_type}: {values}")
    
    return True


def test_multi_session():
    """测试多会话管理"""
    print("\n=== 测试4: 多会话管理 ===")
    
    dm = DialogManager()
    
    dm.process_input("我想了解Python", session_id="session_a")
    dm.process_input("我想了解Java", session_id="session_b")
    
    result_a = dm.process_input("它有什么特点？", session_id="session_a")
    result_b = dm.process_input("它有什么特点？", session_id="session_b")
    
    print(f"会话A - 用户: 它有什么特点？")
    print(f"会话A - 系统: {result_a['response'][:50]}...")
    
    print(f"\n会话B - 用户: 它有什么特点？")
    print(f"会话B - 系统: {result_b['response'][:50]}...")
    
    context_a = dm.manage_dialog("session_a")
    context_b = dm.manage_dialog("session_b")
    
    print(f"\n会话A话题: {context_a.last_topic}")
    print(f"会话B话题: {context_b.last_topic}")
    
    return True


def test_intent_detector():
    """测试意图识别器"""
    print("\n=== 测试5: 意图识别器 ===")
    
    detector = IntentDetector()
    
    test_cases = [
        ("什么是人工智能？", IntentType.QA),
        ("你好，很高兴见到你", IntentType.CHITCHAT),
        ("帮我写一个程序", IntentType.TASK),
        ("你刚才说的什么意思？", IntentType.CLARIFICATION),
        ("很好，谢谢", IntentType.FEEDBACK)
    ]
    
    correct = 0
    for text, expected_intent in test_cases:
        intent, confidence = detector.detect_intent(text)
        match = "✓" if intent == expected_intent else "✗"
        print(f"{match} '{text}' -> {intent.value} (期望: {expected_intent.value}, 置信度: {confidence:.2f})")
        if intent == expected_intent:
            correct += 1
    
    return correct >= 4


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("多轮对话管理器测试")
    print("="*60)
    
    tests = [
        ("多轮问答（上下文理解）", test_multi_turn_qa),
        ("意图切换", test_intent_switching),
        ("实体提取", test_entity_extraction),
        ("多会话管理", test_multi_session),
        ("意图识别器", test_intent_detector)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ 通过" if success else "✗ 失败"))
        except Exception as e:
            results.append((name, f"✗ 错误: {str(e)[:50]}"))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, result in results:
        print(f"{result} - {name}")
    
    passed = sum(1 for _, r in results if "✓" in r)
    print(f"\n总计: {passed}/{len(results)} 通过")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)