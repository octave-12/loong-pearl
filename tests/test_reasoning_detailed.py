"""
详细推理引擎测试 - 诊断问题
"""
import sys
import os

reasoning_engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'core')
sys.path.insert(0, reasoning_engine_path)

from reasoning_engine import ReasoningEngine


def test_deductive_socrates():
    """经典三段论：苏格拉底会死"""
    print("\n【测试1】经典三段论：苏格拉底会死")
    print("-" * 60)
    
    test_triples = [
        ("人", "IS_A", "生物", 0.95),
        ("苏格拉底", "IS_A", "人", 0.99),
        ("生物", "都会", "死亡", 0.98),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    test_cases = [
        {
            'name': '标准格式',
            'premises': ["所有人都会死亡", "苏格拉底是人"],
            'conclusion': "苏格拉底会死亡"
        },
        {
            'name': '变体1：会死',
            'premises': ["所有人都会死", "苏格拉底是人"],
            'conclusion': "苏格拉底会死"
        },
        {
            'name': '变体2：都要死',
            'premises': ["所有人都要死", "苏格拉底是人"],
            'conclusion': "苏格拉底要死"
        },
        {
            'name': '变体3：是会死的',
            'premises': ["所有人都是会死的", "苏格拉底是人"],
            'conclusion': "苏格拉底是会死的"
        },
    ]
    
    results = []
    for case in test_cases:
        result = engine.deductive_reasoning(
            premises=case['premises'],
            conclusion=case['conclusion']
        )
        
        print(f"\n{case['name']}:")
        print(f"  前提: {case['premises']}")
        print(f"  结论: {case['conclusion']}")
        print(f"  推理有效: {result['valid']}")
        print(f"  置信度: {result['confidence']:.2f}")
        
        results.append({
            'name': case['name'],
            'valid': result['valid'],
            'confidence': result['confidence']
        })
    
    return results


def test_causal_chain():
    """因果推理链测试"""
    print("\n\n【测试2】因果推理链")
    print("-" * 60)
    
    test_triples = [
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
        ("路面滑", "CAUSES", "交通事故风险增加", 0.75),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    test_cases = [
        ("下雨", "地面湿", "直接因果"),
        ("下雨", "路面滑", "间接因果（2步）"),
        ("下雨", "交通事故风险增加", "间接因果（3步）"),
        ("地面湿", "路面滑", "间接因果（1步）"),
    ]
    
    results = []
    for cause, effect, desc in test_cases:
        result = engine.causal_reasoning(cause, effect)
        
        print(f"\n{desc}: {cause} → {effect}")
        print(f"  存在因果关系: {result['is_causal']}")
        print(f"  置信度: {result['confidence']:.2f}")
        if result['causal_chain']:
            print(f"  因果链: {' → '.join(result['causal_chain'])}")
        
        results.append({
            'cause': cause,
            'effect': effect,
            'is_causal': result['is_causal'],
            'confidence': result['confidence']
        })
    
    return results


def test_confidence_propagation():
    """置信度传递测试"""
    print("\n\n【测试3】置信度传递")
    print("-" * 60)
    
    test_triples = [
        ("A", "CAUSES", "B", 0.9),
        ("B", "CAUSES", "C", 0.8),
        ("C", "CAUSES", "D", 0.7),
        ("D", "CAUSES", "E", 0.6),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.multi_step_reasoning("为什么A会导致E？", steps=5)
    
    print(f"答案: {result['answer']}")
    print(f"综合置信度: {result['confidence']:.4f}")
    print(f"预期置信度: {0.9 * 0.8 * 0.7 * 0.6:.4f}")
    print(f"使用步数: {result['steps_used']}")
    print("推理链:")
    for step in result['reasoning_chain']:
        print(f"  {step}")
    
    return {
        'confidence': result['confidence'],
        'expected': 0.9 * 0.8 * 0.7 * 0.6,
        'steps': result['steps_used']
    }


def calculate_accuracy(results_list):
    """计算准确率"""
    total = len(results_list)
    correct = sum(1 for r in results_list if r.get('valid', False) or r.get('is_causal', False))
    
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    print("=" * 60)
    print("推理引擎详细诊断测试")
    print("=" * 60)
    
    deductive_results = test_deductive_socrates()
    causal_results = test_causal_chain()
    confidence_result = test_confidence_propagation()
    
    print("\n\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    print("\n演绎推理准确率:")
    deductive_accuracy = sum(1 for r in deductive_results if r['valid']) / len(deductive_results)
    print(f"  {deductive_accuracy:.1%} ({sum(1 for r in deductive_results if r['valid'])}/{len(deductive_results)})")
    
    print("\n因果推理准确率:")
    causal_accuracy = sum(1 for r in causal_results if r['is_causal']) / len(causal_results)
    print(f"  {causal_accuracy:.1%} ({sum(1 for r in causal_results if r['is_causal'])}/{len(causal_results)})")
    
    print("\n置信度传递:")
    print(f"  实际: {confidence_result['confidence']:.4f}")
    print(f"  预期: {confidence_result['expected']:.4f}")
    print(f"  误差: {abs(confidence_result['confidence'] - confidence_result['expected']):.4f}")
    
    overall_accuracy = (deductive_accuracy + causal_accuracy) / 2
    print(f"\n总体准确率: {overall_accuracy:.1%}")
    
    if overall_accuracy < 0.8:
        print("\n⚠ 警告：准确率低于80%，需要优化推理逻辑")
    else:
        print("\n✓ 准确率达标")