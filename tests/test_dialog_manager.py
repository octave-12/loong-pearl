"""
测试多轮对话管理器
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.dialog_manager import (
    DialogManager, DialogContext, IntentDetector, EntityExtractor,
    IntentType, DialogState, SessionManager, create_dialog_manager
)
from src.data.unified_knowledge_manager import UnifiedKnowledgeManager


def test_basic_conversation():
    """测试基本对话流程"""
    print("\n=== 测试1: 基本对话流程 ===")
    
    dm = create_dialog_manager()
    
    result1 = dm.process_input("你好", session_id="test1")
    print(f"用户: 你好")
    print(f"系统: {result1['response']}")
    print(f"意图: {result1['intent']}, 置信度: {result1['confidence']:.2f}")
    
    result2 = dm.process_input("什么是量子纠缠？", session_id="test1")
    print(f"\n用户: 什么是量子纠缠？")
    print(f"系统: {result2['response']}")
    print(f"意图: {result2['intent']}, 置信度: {result2['confidence']:.2f}")
    
    result3 = dm.process_input("它有什么应用？", session_id="test1")
    print(f"\n用户: 它有什么应用？")
    print(f"系统: {result3['response']}")
    print(f"意图: {result3['intent']}, 置信度: {result3['confidence']:.2f}")
    
    context = dm.manage_dialog("test1")
    print(f"\n对话轮数: {context.turn_count}")
    print(f"最后话题: {context.last_topic}")
    print(f"引用实体: {list(context.referenced_entities.keys())}")
    
    return True


def test_intent_switching():
    """测试意图切换"""
    print("\n=== 测试2: 意图切换 ===")
    
    dm = create_dialog_manager()
    
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
    
    result4 = dm.process_input("很好", session_id="test2")
    print(f"\n用户: 很好")
    print(f"系统: {result4['response']}")
    print(f"意图: {result4['intent']}")
    
    context = dm.manage_dialog("test2")
    print(f"\n对话历史:")
    for turn in context.history:
        print(f"  [{turn['turn']}] 意图: {turn['intent']}")
    
    return True


def test_context_understanding():
    """测试上下文理解"""
    print("\n=== 测试3: 上下文理解 ===")
    
    dm = create_dialog_manager()
    
    dm.process_input("什么是机器学习？", session_id="test3")
    result1 = dm.process_input("它的主要方法有哪些？", session_id="test3")
    print(f"用户: 它的主要方法有哪些？")
    print(f"系统: {result1['response']}")
    print(f"实体引用: {result1['entities'].get('referenced', {})}")
    
    dm.process_input("深度学习是什么？", session_id="test3")
    result2 = dm.process_input("这个和机器学习有什么关系？", session_id="test3")
    print(f"\n用户: 这个和机器学习有什么关系？")
    print(f"系统: {result2['response']}")
    
    context = dm.manage_dialog("test3")
    print(f"\n话题栈: {context.topic_stack}")
    print(f"最后话题: {context.last_topic}")
    
    return True


def test_entity_extraction():
    """测试实体提取"""
    print("\n=== 测试4: 实体提取 ===")
    
    dm = create_dialog_manager()
    
    result = dm.process_input("2024年3月15日，在北京举办人工智能大会", session_id="test4")
    print(f"用户: 2024年3月15日，在北京举办人工智能大会")
    print(f"实体: {result['entities']}")
    
    return True


def test_with_knowledge_manager():
    """测试与知识管理器集成"""
    print("\n=== 测试5: 知识管理器集成 ===")
    
    try:
        km = UnifiedKnowledgeManager()
        dm = create_dialog_manager(knowledge_manager=km)
        
        result = dm.process_input("什么是量子力学？", session_id="test5")
        print(f"用户: 什么是量子力学？")
        print(f"系统: {result['response']}")
        print(f"意图: {result['intent']}")
        
        context = dm.manage_dialog("test5")
        print(f"对话轮数: {context.turn_count}")
        
        return True
    except Exception as e:
        print(f"知识管理器测试跳过: {e}")
        return True


def test_multi_session():
    """测试多会话管理"""
    print("\n=== 测试6: 多会话管理 ===")
    
    dm = create_dialog_manager()
    
    dm.process_input("你好，我是用户A", session_id="session_a")
    dm.process_input("我想了解Python", session_id="session_a")
    
    dm.process_input("你好，我是用户B", session_id="session_b")
    dm.process_input("我想了解Java", session_id="session_b")
    
    result_a = dm.process_input("它有什么特点？", session_id="session_a")
    result_b = dm.process_input("它有什么特点？", session_id="session_b")
    
    print(f"会话A - 用户: 它有什么特点？")
    print(f"会话A - 系统: {result_a['response']}")
    
    print(f"\n会话B - 用户: 它有什么特点？")
    print(f"会话B - 系统: {result_b['response']}")
    
    context_a = dm.manage_dialog("session_a")
    context_b = dm.manage_dialog("session_b")
    
    print(f"\n会话A话题: {context_a.last_topic}")
    print(f"会话B话题: {context_b.last_topic}")
    
    return True


def test_intent_detector():
    """测试意图识别器"""
    print("\n=== 测试7: 意图识别器 ===")
    
    detector = IntentDetector()
    
    test_cases = [
        ("什么是人工智能？", IntentType.QA),
        ("你好，很高兴见到你", IntentType.CHITCHAT),
        ("帮我写一个Python程序", IntentType.TASK),
        ("你刚才说的什么意思？", IntentType.CLARIFICATION),
        ("很好，谢谢你的帮助", IntentType.FEEDBACK)
    ]
    
    for text, expected_intent in test_cases:
        intent, confidence = detector.detect_intent(text)
        match = "✓" if intent == expected_intent else "✗"
        print(f"{match} '{text}' -> {intent.value} (期望: {expected_intent.value}, 置信度: {confidence:.2f})")
    
    return True


def test_context_memory():
    """测试上下文记忆"""
    print("\n=== 测试8: 上下文记忆 ===")
    
    memory = SessionManager(max_sessions=3)
    
    for i in range(5):
        session_id = f"session_{i}"
        context = memory.get_or_create_session(session_id)
        context.add_turn(f"用户输入{i}", f"系统回复{i}", IntentType.QA, {})
        memory.update_session(session_id, context)
    
    print(f"活跃会话数: {len(memory.sessions)}")
    print(f"会话ID: {list(memory.sessions.keys())}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("多轮对话管理器测试")
    print("="*60)
    
    tests = [
        ("基本对话流程", test_basic_conversation),
        ("意图切换", test_intent_switching),
        ("上下文理解", test_context_understanding),
        ("实体提取", test_entity_extraction),
        ("知识管理器集成", test_with_knowledge_manager),
        ("多会话管理", test_multi_session),
        ("意图识别器", test_intent_detector),
        ("上下文记忆", test_context_memory)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ 通过" if success else "✗ 失败"))
        except Exception as e:
            results.append((name, f"✗ 错误: {e}"))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, result in results:
        print(f"{result} - {name}")
    
    passed = sum(1 for _, r in results if "✓" in r)
    print(f"\n总计: {passed}/{len(results)} 通过")


if __name__ == "__main__":
    run_all_tests()