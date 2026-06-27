"""
测试对话接口统一性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

def import_module_direct(module_path):
    """直接导入模块，避免通过__init__.py"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_file = os.path.join(base_path, *module_path.split(".")) + ".py"
    
    spec = importlib.util.spec_from_file_location(
        module_path.replace(".", "_"), 
        module_file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_context_memory_session_interface():
    """测试ContextMemory会话接口"""
    print("\n=== 测试1: ContextMemory会话接口 ===")
    
    context_memory_module = import_module_direct("src.core.context_memory")
    ContextMemory = context_memory_module.ContextMemory
    
    memory = ContextMemory(max_sessions=5)
    
    session1 = memory.get_or_create_session("session_1")
    print(f"✓ 创建会话: {session1['session_id']}")
    print(f"  - 类型: {type(session1)}")
    print(f"  - 包含字段: {list(session1.keys())}")
    
    session1_again = memory.get_or_create_session("session_1")
    print(f"✓ 获取现有会话: {session1_again['session_id']}")
    print(f"  - 是同一对象: {session1 is session1_again}")
    
    session2 = memory.get_or_create_session("session_2")
    print(f"✓ 创建第二个会话: {session2['session_id']}")
    
    all_sessions = memory.get_all_sessions()
    print(f"✓ 所有会话: {all_sessions}")
    print(f"✓ 会话数量: {memory.get_session_count()}")
    
    memory.add_turn_to_session(
        "session_1",
        "你好",
        "你好！有什么可以帮助你的？",
        "chitchat",
        {"keywords": ["你好"]}
    )
    print(f"✓ 添加对话轮")
    
    session1_updated = memory.get_session("session_1")
    print(f"  - 对话历史长度: {len(session1_updated['history'])}")
    print(f"  - 对话轮数: {session1_updated['turn_count']}")
    
    success = memory.delete_session("session_2")
    print(f"✓ 删除会话: {success}")
    print(f"  - 剩余会话: {memory.get_all_sessions()}")
    
    return True


def test_session_manager_interface():
    """测试SessionManager接口"""
    print("\n=== 测试2: SessionManager接口 ===")
    
    dialog_manager_module = import_module_direct("src.core.dialog_manager")
    SessionManager = dialog_manager_module.SessionManager
    DialogContext = dialog_manager_module.DialogContext
    IntentType = dialog_manager_module.IntentType
    
    manager = SessionManager(max_sessions=3)
    
    session1 = manager.get_or_create_session("session_1")
    print(f"✓ 创建会话: {session1.session_id}")
    print(f"  - 类型: {type(session1)}")
    print(f"  - 状态: {session1.state}")
    
    session1.add_turn("你好", "你好！", IntentType.CHITCHAT, {})
    print(f"✓ 添加对话轮")
    print(f"  - 对话历史长度: {len(session1.history)}")
    print(f"  - 对话轮数: {session1.turn_count}")
    
    manager.update_session("session_1", session1)
    print(f"✓ 更新会话")
    
    relevant = manager.get_relevant_context("session_1", "你好", top_k=1)
    print(f"✓ 获取相关上下文: {len(relevant)} 条")
    
    return True


def test_dialog_manager_with_unified_memory():
    """测试DialogManager使用统一记忆接口"""
    print("\n=== 测试3: DialogManager使用统一记忆 ===")
    
    dialog_manager_module = import_module_direct("src.core.dialog_manager")
    create_dialog_manager = dialog_manager_module.create_dialog_manager
    
    dm = create_dialog_manager()
    
    result1 = dm.process_input("你好", session_id="test1")
    print(f"✓ 对话1: 你好")
    print(f"  - 回复: {result1['response'][:50]}...")
    print(f"  - 意图: {result1['intent']}")
    print(f"  - 状态: {result1['state']}")
    
    result2 = dm.process_input("什么是人工智能？", session_id="test1")
    print(f"✓ 对话2: 什么是人工智能？")
    print(f"  - 回复: {result2['response'][:50]}...")
    print(f"  - 意图: {result2['intent']}")
    
    result3 = dm.process_input("它有什么应用？", session_id="test1")
    print(f"✓ 对话3: 它有什么应用？")
    print(f"  - 回复: {result3['response'][:50]}...")
    print(f"  - 意图: {result3['intent']}")
    
    context = dm.manage_dialog("test1")
    print(f"✓ 获取会话上下文")
    print(f"  - 类型: {type(context)}")
    if isinstance(context, dict):
        print(f"  - 对话轮数: {context['turn_count']}")
        print(f"  - 最后话题: {context.get('last_topic')}")
    else:
        print(f"  - 对话轮数: {context.turn_count}")
        print(f"  - 最后话题: {context.last_topic}")
    
    summary = dm.get_context_summary("test1")
    print(f"✓ 上下文摘要: {summary}")
    
    return True


def test_multi_session_management():
    """测试多会话管理"""
    print("\n=== 测试4: 多会话管理 ===")
    
    dialog_manager_module = import_module_direct("src.core.dialog_manager")
    create_dialog_manager = dialog_manager_module.create_dialog_manager
    
    dm = create_dialog_manager()
    
    dm.process_input("你好，我是用户A", session_id="user_a")
    dm.process_input("我想了解Python", session_id="user_a")
    
    dm.process_input("你好，我是用户B", session_id="user_b")
    dm.process_input("我想了解Java", session_id="user_b")
    
    result_a = dm.process_input("它有什么特点？", session_id="user_a")
    result_b = dm.process_input("它有什么特点？", session_id="user_b")
    
    print(f"✓ 用户A对话:")
    print(f"  - 回复: {result_a['response'][:50]}...")
    
    print(f"✓ 用户B对话:")
    print(f"  - 回复: {result_b['response'][:50]}...")
    
    all_sessions = list(dm.sessions.keys())
    print(f"✓ 所有会话: {all_sessions}")
    
    dm.reset_session("user_a")
    print(f"✓ 重置会话user_a")
    print(f"  - 剩余会话: {list(dm.sessions.keys())}")
    
    return True


def test_session_eviction():
    """测试会话驱逐"""
    print("\n=== 测试5: 会话驱逐（LRU） ===")
    
    context_memory_module = import_module_direct("src.core.context_memory")
    ContextMemory = context_memory_module.ContextMemory
    
    memory = ContextMemory(max_sessions=3)
    
    for i in range(5):
        session_id = f"session_{i}"
        memory.get_or_create_session(session_id)
        print(f"  创建会话 {session_id}, 当前会话: {memory.get_all_sessions()}")
    
    final_sessions = memory.get_all_sessions()
    print(f"✓ 最终保留会话: {final_sessions}")
    print(f"  - 会话数量: {len(final_sessions)} (最大: 3)")
    
    stats = memory.get_memory_stats()
    print(f"✓ 统计信息:")
    print(f"  - 创建会话数: {stats['stats']['sessions_created']}")
    print(f"  - 驱逐会话数: {stats['stats']['sessions_evicted']}")
    
    return True


def test_intent_detection():
    """测试意图识别"""
    print("\n=== 测试6: 意图识别 ===")
    
    dialog_manager_module = import_module_direct("src.core.dialog_manager")
    IntentDetector = dialog_manager_module.IntentDetector
    IntentType = dialog_manager_module.IntentType
    
    detector = IntentDetector()
    
    test_cases = [
        ("什么是人工智能？", IntentType.QA),
        ("你好，很高兴见到你", IntentType.CHITCHAT),
        ("帮我写一个Python程序", IntentType.TASK),
        ("你刚才说的什么意思？", IntentType.CLARIFICATION),
        ("很好，谢谢你的帮助", IntentType.FEEDBACK)
    ]
    
    correct = 0
    for text, expected_intent in test_cases:
        intent, confidence = detector.detect_intent(text)
        match = "✓" if intent == expected_intent else "✗"
        print(f"  {match} '{text}' -> {intent.value} (期望: {expected_intent.value}, 置信度: {confidence:.2f})")
        if intent == expected_intent:
            correct += 1
    
    accuracy = correct / len(test_cases) * 100
    print(f"✓ 意图识别准确率: {accuracy:.1f}%")
    
    return accuracy >= 80


def test_entity_extraction():
    """测试实体提取"""
    print("\n=== 测试7: 实体提取 ===")
    
    dialog_manager_module = import_module_direct("src.core.dialog_manager")
    EntityExtractor = dialog_manager_module.EntityExtractor
    
    extractor = EntityExtractor()
    
    text = "2024年3月15日，在北京举办人工智能大会"
    entities = extractor.extract_entities(text)
    
    print(f"✓ 输入文本: {text}")
    print(f"  - 时间实体: {entities.get('time', [])}")
    print(f"  - 地点实体: {entities.get('location', [])}")
    print(f"  - 关键词: {entities.get('keywords', [])}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("对话接口统一性测试")
    print("="*60)
    
    tests = [
        ("ContextMemory会话接口", test_context_memory_session_interface),
        ("SessionManager接口", test_session_manager_interface),
        ("DialogManager统一记忆", test_dialog_manager_with_unified_memory),
        ("多会话管理", test_multi_session_management),
        ("会话驱逐", test_session_eviction),
        ("意图识别", test_intent_detection),
        ("实体提取", test_entity_extraction),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ 通过" if success else "✗ 失败"))
        except Exception as e:
            results.append((name, f"✗ 错误: {e}"))
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, result in results:
        print(f"{result} - {name}")
    
    passed = sum(1 for _, r in results if "✓" in r)
    total = len(results)
    success_rate = passed / total * 100
    
    print(f"\n总计: {passed}/{total} 通过")
    print(f"成功率: {success_rate:.1f}%")
    
    return success_rate >= 90


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)