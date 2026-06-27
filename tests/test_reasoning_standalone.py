"""
独立测试脚本 - 不依赖外部库
直接测试推理引擎的核心功能
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "reasoning_engine", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                 "src", "core", "reasoning_engine.py")
)
reasoning_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reasoning_module)

ReasoningEngine = reasoning_module.ReasoningEngine
ReasoningChain = reasoning_module.ReasoningChain


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_deductive():
    print_section("测试1: 演绎推理 (从一般到特殊)")
    
    triples = [
        ("人", "IS_A", "生物", 0.95),
        ("苏格拉底", "IS_A", "人", 0.99),
        ("生物", "都会", "死亡", 0.98),
    ]
    
    engine = ReasoningEngine(concept_graph=triples)
    
    print("\n前提:")
    print("  1. 所有人都会死亡")
    print("  2. 苏格拉底是人")
    print("\n结论: 苏格拉底会死亡")
    
    result = engine.deductive_reasoning(
        premises=["所有人都会死亡", "苏格拉底是人"],
        conclusion="苏格拉底会死亡"
    )
    
    print(f"\n结果:")
    print(f"  推理有效: {result['valid']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"\n推理链:")
    for step in result['reasoning_chain']:
        print(f"  → {step}")
    
    return result['valid'] and result['confidence'] > 0


def test_inductive():
    print_section("测试2: 归纳推理 (从特殊到一般)")
    
    triples = [("乌鸦", "HAS_PROPERTY", "黑色", 0.92)]
    engine = ReasoningEngine(concept_graph=triples)
    
    examples = [
        "乌鸦1是黑色的",
        "乌鸦2是黑色的",
        "乌鸦3是黑色的",
        "乌鸦4是黑色的",
        "乌鸦5是黑色的"
    ]
    
    print("\n观察例子:")
    for ex in examples:
        print(f"  • {ex}")
    
    result = engine.inductive_reasoning(examples)
    
    print(f"\n归纳结论: {result['generalization']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"支持例子数: {result['support_count']}")
    
    return result['generalization'] is not None and result['support_count'] == 5


def test_analogical():
    print_section("测试3: 类比推理 (基于相似性)")
    
    engine = ReasoningEngine()
    
    source = {'entity': '水', 'attributes': ['流动性', '可蒸发', '无色', '透明']}
    target = {'entity': '空气', 'attributes': ['流动性', '无色', '透明']}
    
    print(f"\n源域: {source['entity']}")
    print(f"  属性: {', '.join(source['attributes'])}")
    print(f"\n目标域: {target['entity']}")
    print(f"  已知属性: {', '.join(target['attributes'])}")
    
    result = engine.analogical_reasoning(source, target)
    
    print(f"\n推断结果:")
    print(f"  推断属性: {', '.join(result['inferred_attributes'])}")
    print(f"  相似度: {result['similarity']:.2f}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"\n推理说明:")
    print(f"  {result['reasoning']}")
    
    return len(result['inferred_attributes']) > 0 and result['similarity'] > 0


def test_causal():
    print_section("测试4: 因果推理 (因果关系识别)")
    
    triples = [
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
        ("下雨", "CAUSES", "空气湿润", 0.75),
    ]
    
    engine = ReasoningEngine(concept_graph=triples)
    
    print("\n因果链测试:")
    print("  原因: 下雨")
    print("  结果: 地面湿")
    
    result = engine.causal_reasoning("下雨", "地面湿")
    
    print(f"\n结果:")
    print(f"  存在因果关系: {result['is_causal']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"\n因果链:")
    for chain in result['causal_chain']:
        print(f"  → {chain}")
    
    print("\n间接因果链测试:")
    print("  原因: 下雨")
    print("  结果: 路面滑")
    
    result2 = engine.causal_reasoning("下雨", "路面滑")
    
    print(f"\n结果:")
    print(f"  存在因果关系: {result2['is_causal']}")
    print(f"  置信度: {result2['confidence']:.2f}")
    print(f"\n因果链:")
    for chain in result2['causal_chain']:
        print(f"  → {chain}")
    
    return result['is_causal'] and result2['is_causal']


def test_multi_step():
    print_section("测试5: 多步推理 (链式推理)")
    
    triples = [
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
        ("路面滑", "CAUSES", "容易摔倒", 0.82),
    ]
    
    engine = ReasoningEngine(concept_graph=triples)
    
    question = "为什么下雨后路面会滑？"
    print(f"\n问题: {question}")
    
    result = engine.multi_step_reasoning(question, steps=5)
    
    print(f"\n答案: {result['answer']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"使用步数: {result['steps_used']}")
    print(f"\n推理链:")
    for step in result['reasoning_chain']:
        print(f"  → {step}")
    
    return result['steps_used'] > 0


def test_chain_manager():
    print_section("测试6: 推理链管理器")
    
    chain = ReasoningChain()
    
    print("\n构建推理链:")
    
    chain.add_step(
        premise="所有人都会死亡",
        conclusion="苏格拉底会死亡",
        rule="演绎推理",
        confidence=0.9
    )
    print(f"  步骤1: 所有人都会死亡 → 苏格拉底会死亡 (置信度: 0.90)")
    
    chain.add_step(
        premise="苏格拉底会死亡",
        conclusion="苏格拉底终将死亡",
        rule="等价转换",
        confidence=0.95
    )
    print(f"  步骤2: 苏格拉底会死亡 → 苏格拉底终将死亡 (置信度: 0.95)")
    
    print(f"\n最终结论: {chain.get_final_conclusion()}")
    print(f"综合置信度: {chain.confidence:.4f}")
    print(f"推理步骤数: {len(chain.steps)}")
    
    chain_dict = chain.to_dict()
    print(f"\n导出字典:")
    print(f"  最终结论: {chain_dict['final_conclusion']}")
    print(f"  综合置信度: {chain_dict['confidence']:.4f}")
    
    return chain.confidence == 0.9 * 0.95


def test_real_world():
    print_section("测试7: 真实场景应用")
    
    triples = [
        ("病毒", "CAUSES", "疾病", 0.95),
        ("流感病毒", "IS_A", "病毒", 0.99),
        ("流感病毒", "CAUSES", "流感", 0.98),
        ("流感", "HAS_SYMPTOM", "发烧", 0.90),
        ("流感", "HAS_SYMPTOM", "咳嗽", 0.88),
        ("发烧", "INDICATES", "体温升高", 0.95),
        ("体温升高", "REQUIRES", "降温措施", 0.85),
    ]
    
    engine = ReasoningEngine(concept_graph=triples)
    
    print("\n场景1: 医疗诊断推理")
    print("  问题: 为什么流感病毒感染会导致发烧？")
    
    result = engine.multi_step_reasoning("为什么流感病毒会导致发烧？", steps=5)
    
    print(f"\n  答案: {result['answer']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"  推理步数: {result['steps_used']}")
    
    print("\n场景2: 因果关系验证")
    print("  验证: 流感病毒 → 流感")
    
    result2 = engine.causal_reasoning("流感病毒", "流感")
    
    print(f"\n  存在因果关系: {result2['is_causal']}")
    print(f"  置信度: {result2['confidence']:.2f}")
    if result2['causal_chain']:
        print(f"  因果链: {result2['causal_chain'][0]}")
    
    return result['steps_used'] > 0 and result2['is_causal']


def main():
    print("\n" + "=" * 70)
    print("  深度推理引擎 - 完整测试套件")
    print("=" * 70)
    
    tests = [
        ("演绎推理", test_deductive),
        ("归纳推理", test_inductive),
        ("类比推理", test_analogical),
        ("因果推理", test_causal),
        ("多步推理", test_multi_step),
        ("推理链管理器", test_chain_manager),
        ("真实场景应用", test_real_world),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    print_section("测试结果汇总")
    
    passed_count = 0
    for name, passed, error in results:
        if passed:
            print(f"  ✓ {name}: 通过")
            passed_count += 1
        else:
            print(f"  ✗ {name}: 失败")
            if error:
                print(f"    错误: {error}")
    
    print(f"\n总计: {passed_count}/{len(tests)} 测试通过")
    
    if passed_count == len(tests):
        print("\n🎉 所有测试通过！推理引擎工作正常。")
    else:
        print(f"\n⚠️  {len(tests) - passed_count} 个测试失败，请检查。")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()