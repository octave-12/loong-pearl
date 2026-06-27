import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.context_memory import ContextMemory, MemoryItem
import time


def test_short_term_memory():
    print("\n=== 测试短期记忆 ===")
    
    memory = ContextMemory(short_term_size=100, long_term_size=1000)
    
    conversations = [
        ("你好，我是AI助手", {"role": "assistant"}),
        ("我想了解机器学习", {"role": "user"}),
        ("机器学习是AI的核心技术之一", {"role": "assistant"}),
        ("能详细解释一下吗？", {"role": "user"}),
        ("机器学习包括监督学习、无监督学习和强化学习", {"role": "assistant"}),
    ]
    
    for text, metadata in conversations:
        key = memory.add_to_short_term(text, metadata)
        print(f"添加短期记忆: {text[:30]}... -> key: {key[:8]}")
    
    context = memory.get_context_window(size=3)
    print(f"\n最近{len(context)}轮对话:")
    for i, item in enumerate(context, 1):
        role = item.metadata.get("role", "unknown")
        print(f"  {i}. [{role}]: {item.content[:40]}...")
    
    history = memory.get_conversation_history(max_turns=3)
    print(f"\n对话历史({len(history)}条):")
    for h in history:
        print(f"  [{h['role']}]: {h['content'][:40]}...")
    
    stats = memory.get_memory_stats()
    print(f"\n短期记忆统计:")
    print(f"  当前大小: {stats['short_term_size']}/{stats['short_term_capacity']}")
    print(f"  利用率: {stats['utilization']['short_term']:.2%}")
    
    return True


def test_long_term_memory():
    print("\n=== 测试长期记忆 ===")
    
    memory = ContextMemory(short_term_size=100, long_term_size=100)
    
    knowledge_base = [
        ("python_basics", "Python是一种高级编程语言，具有简洁易读的语法", 0.8),
        ("machine_learning", "机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习", 0.9),
        ("deep_learning", "深度学习是机器学习的子领域，使用多层神经网络进行学习", 0.85),
        ("neural_network", "神经网络是模拟人脑结构的计算模型，由神经元和连接组成", 0.7),
        ("nlp_basics", "自然语言处理是让计算机理解和生成人类语言的技术", 0.75),
    ]
    
    for key, value, importance in knowledge_base:
        memory.add_to_long_term(key, value, importance)
        print(f"添加长期记忆: {key} (重要性: {importance})")
    
    queries = [
        "什么是机器学习？",
        "Python有什么特点？",
        "深度学习和神经网络的关系",
    ]
    
    print("\n检索测试:")
    for query in queries:
        results = memory.retrieve_relevant(query, k=3, include_short_term=False)
        print(f"\n查询: {query}")
        for item, score in results:
            print(f"  相似度={score:.4f}: {item.content[:50]}...")
    
    stats = memory.get_memory_stats()
    print(f"\n长期记忆统计:")
    print(f"  当前大小: {stats['long_term_size']}/{stats['long_term_capacity']}")
    print(f"  利用率: {stats['utilization']['long_term']:.2%}")
    
    return True


def test_memory_compression():
    print("\n=== 测试记忆压缩 ===")
    
    memory = ContextMemory(short_term_size=100, long_term_size=100, compression_threshold=5)
    
    long_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。",
        "机器学习是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构使之不断改善自身的性能。",
        "深度学习是机器学习领域中一个新的研究方向，它被引入机器学习使其更接近于其最初的目标——人工智能。深度学习是学习样本数据的内在规律和表示层次，这些学习过程中获得的信息对诸如文字、图像和声音等数据的解释有很大的帮助。",
    ]
    
    for i, text in enumerate(long_texts):
        memory.add_to_short_term(text, {"role": "assistant", "turn": i})
        print(f"添加长文本 {i+1}: {len(text)} 字符")
    
    result = memory.compress_memory(force=True)
    print(f"\n压缩结果:")
    print(f"  是否压缩: {result['compressed']}")
    print(f"  处理项目数: {result['compression_stats']['items_processed']}")
    print(f"  平均压缩率: {result['average_compression_ratio']:.2%}")
    
    return True


def test_retrieval_augmentation():
    print("\n=== 测试检索增强 ===")
    
    memory = ContextMemory(short_term_size=100, long_term_size=100)
    
    memory.add_to_short_term("用户询问Python编程", {"role": "user"})
    memory.add_to_short_term("Python是一种高级编程语言", {"role": "assistant"})
    
    memory.add_to_long_term("python_features", "Python支持多种编程范式：面向对象、命令式、函数式编程", 0.8)
    memory.add_to_long_term("python_applications", "Python广泛应用于Web开发、数据科学、人工智能等领域", 0.9)
    memory.add_to_long_term("java_features", "Java是一种面向对象的编程语言，具有跨平台特性", 0.7)
    
    query = "Python有哪些应用领域？"
    results = memory.retrieve_relevant(query, k=5, include_short_term=True, include_long_term=True)
    
    print(f"查询: {query}")
    print(f"\n检索结果({len(results)}条):")
    for i, (item, score) in enumerate(results, 1):
        source = "长期记忆" if "key" in item.metadata else "短期记忆"
        print(f"  {i}. [{source}] 相似度={score:.4f}")
        print(f"     {item.content[:60]}...")
    
    return True


def test_memory_update():
    print("\n=== 测试记忆更新 ===")
    
    memory = ContextMemory(short_term_size=100, long_term_size=100)
    
    memory.add_to_long_term("topic1", "初始内容：机器学习基础", 0.5)
    memory.add_to_long_term("topic2", "深度学习进阶", 0.6)
    
    print("更新前:")
    results = memory.retrieve_relevant("机器学习", k=2, include_short_term=False)
    for item, _ in results:
        print(f"  {item.metadata.get('key')}: 重要性={item.importance}")
    
    memory.update_importance("topic1", 0.9)
    
    print("\n更新后:")
    results = memory.retrieve_relevant("机器学习", k=2, include_short_term=False)
    for item, _ in results:
        print(f"  {item.metadata.get('key')}: 重要性={item.importance}")
    
    return True


def test_state_persistence():
    print("\n=== 测试状态持久化 ===")
    
    memory = ContextMemory(short_term_size=100, long_term_size=100)
    
    memory.add_to_short_term("测试对话1", {"role": "user"})
    memory.add_to_short_term("测试对话2", {"role": "assistant"})
    memory.add_to_long_term("key1", "长期记忆内容", 0.8)
    
    filepath = "test_memory_state.json"
    memory.save_state(filepath)
    print(f"状态已保存到: {filepath}")
    
    new_memory = ContextMemory(short_term_size=100, long_term_size=100)
    new_memory.load_state(filepath)
    print(f"状态已加载")
    
    old_stats = memory.get_memory_stats()
    new_stats = new_memory.get_memory_stats()
    
    print(f"\n状态对比:")
    print(f"  短期记忆: {old_stats['short_term_size']} -> {new_stats['short_term_size']}")
    print(f"  长期记忆: {old_stats['long_term_size']} -> {new_stats['long_term_size']}")
    
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"\n临时文件已清理: {filepath}")
    
    return True


def run_all_tests():
    print("\n" + "=" * 60)
    print("上下文记忆模块测试")
    print("=" * 60)
    
    tests = [
        ("短期记忆测试", test_short_term_memory),
        ("长期记忆测试", test_long_term_memory),
        ("记忆压缩测试", test_memory_compression),
        ("检索增强测试", test_retrieval_augmentation),
        ("记忆更新测试", test_memory_update),
        ("状态持久化测试", test_state_persistence),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ 通过" if success else "✗ 失败"))
        except Exception as e:
            results.append((name, f"✗ 错误: {str(e)}"))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, result in results:
        print(f"{name}: {result}")
    
    passed = sum(1 for _, r in results if "✓" in r)
    print(f"\n总计: {passed}/{len(results)} 通过")


if __name__ == "__main__":
    run_all_tests()