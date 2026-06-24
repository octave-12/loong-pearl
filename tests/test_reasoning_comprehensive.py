"""
推理引擎综合测试 - 验证所有要求
"""
import sys
import os

reasoning_engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'core')
sys.path.insert(0, reasoning_engine_path)

from reasoning_engine import ReasoningEngine


def test_deductive_reasoning_requirements():
    """测试演绎推理要求"""
    print("\n【要求1】演绎推理：所有人会死 + 苏格拉底是人 → 苏格拉底会死 (True)")
    print("-" * 60)
    
    test_triples = [
        ("人", "IS_A", "生物", 0.95),
        ("苏格拉底", "IS_A", "人", 0.99),
        ("生物", "都会", "死亡", 0.98),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.deductive_reasoning(
        premises=["所有人都会死", "苏格拉底是人"],
        conclusion="苏格拉底会死"
    )
    
    print(f"✓ 推理有效: {result['valid']}")
    print(f"✓ 置信度: {result['confidence']:.2f}")
    print("推理链:")
    for step in result['reasoning_chain']:
        print(f"  {step}")
    
    assert result['valid'] == True, "演绎推理应该返回True"
    assert result['confidence'] > 0.8, "置信度应该大于0.8"
    
    return True


def test_causal_reasoning_requirements():
    """测试因果推理要求"""
    print("\n\n【要求2】因果推理：下雨 → 地面湿 (高置信度)")
    print("-" * 60)
    
    test_triples = [
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.causal_reasoning("下雨", "地面湿")
    
    print(f"✓ 存在因果关系: {result['is_causal']}")
    print(f"✓ 置信度: {result['confidence']:.2f} (高置信度)")
    print("因果链:")
    for chain in result['causal_chain']:
        print(f"  {chain}")
    
    assert result['is_causal'] == True, "应该识别出因果关系"
    assert result['confidence'] > 0.8, "置信度应该大于0.8（高置信度）"
    
    print("\n间接因果链测试：下雨 → 路面滑")
    result2 = engine.causal_reasoning("下雨", "路面滑")
    print(f"✓ 存在间接因果关系: {result2['is_causal']}")
    print(f"✓ 置信度: {result2['confidence']:.2f}")
    print(f"✓ 因果链长度: {len(result2['causal_chain'])} 步")
    
    assert result2['is_causal'] == True, "应该识别出间接因果关系"
    
    return True


def test_accuracy_requirements():
    """测试准确率要求"""
    print("\n\n【要求3】推理准确率 > 80%")
    print("-" * 60)
    
    test_triples = [
        ("人", "IS_A", "生物", 0.95),
        ("苏格拉底", "IS_A", "人", 0.99),
        ("柏拉图", "IS_A", "人", 0.99),
        ("生物", "都会", "死亡", 0.98),
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    test_cases = [
        ("演绎", ["所有人都会死", "苏格拉底是人"], "苏格拉底会死"),
        ("演绎", ["所有人都会死", "柏拉图是人"], "柏拉图会死"),
        ("因果", "下雨", "地面湿"),
        ("因果", "下雨", "路面滑"),
        ("因果", "地面湿", "路面滑"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        if case[0] == "演绎":
            result = engine.deductive_reasoning(case[1], case[2])
            is_correct = result['valid']
            print(f"测试{i}: 演绎推理 - {case[2]} -> {'✓' if is_correct else '✗'}")
        else:
            result = engine.causal_reasoning(case[1], case[2])
            is_correct = result['is_causal']
            print(f"测试{i}: 因果推理 - {case[1]}→{case[2]} -> {'✓' if is_correct else '✗'}")
        
        if is_correct:
            correct += 1
    
    accuracy = correct / total
    print(f"\n✓ 推理准确率: {accuracy:.1%} ({correct}/{total})")
    
    assert accuracy > 0.8, f"推理准确率应该大于80%，实际为{accuracy:.1%}"
    
    return accuracy


def test_confidence_propagation():
    """测试置信度传递"""
    print("\n\n【附加测试】置信度传递和不确定性处理")
    print("-" * 60)
    
    test_triples = [
        ("A", "CAUSES", "B", 0.9),
        ("B", "CAUSES", "C", 0.8),
        ("C", "CAUSES", "D", 0.7),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.causal_reasoning("A", "D")
    
    expected_confidence = 0.9 * 0.8 * 0.7
    print(f"✓ 实际置信度: {result['confidence']:.4f}")
    print(f"✓ 预期置信度: {expected_confidence:.4f} (0.9 × 0.8 × 0.7)")
    print(f"✓ 置信度传递误差: {abs(result['confidence'] - expected_confidence):.4f}")
    
    assert abs(result['confidence'] - expected_confidence) < 0.01, "置信度传递应该正确"
    
    return True


def test_enhanced_features():
    """测试增强功能"""
    print("\n\n【增强功能测试】")
    print("-" * 60)
    
    test_triples = [
        ("人", "IS_A", "生物", 0.95),
        ("苏格拉底", "IS_A", "人", 0.99),
        ("生物", "都会", "死亡", 0.98),
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    print("\n1. 三段论推理验证:")
    result = engine.deductive_reasoning(
        premises=["所有人都会死", "苏格拉底是人"],
        conclusion="苏格拉底会死"
    )
    print(f"   - 前提和结论的逻辑关系验证: {'✓ 有效' if result['valid'] else '✗ 无效'}")
    print(f"   - 支持证据数量: {len(result['support'])}")
    
    print("\n2. 因果链推理:")
    result = engine.causal_reasoning("下雨", "路面滑")
    print(f"   - 间接因果关系识别: {'✓ 成功' if result['is_causal'] else '✗ 失败'}")
    print(f"   - 因果链长度: {len(result['causal_chain'])} 步")
    print(f"   - 因果强度: {result['confidence']:.2f}")
    
    print("\n3. 多步推理:")
    result = engine.multi_step_reasoning("为什么下雨后路面会滑？", steps=5)
    print(f"   - 推理答案: {result['answer']}")
    print(f"   - 使用步数: {result['steps_used']}")
    print(f"   - 综合置信度: {result['confidence']:.4f}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("推理引擎综合测试 - 验证所有要求")
    print("=" * 60)
    
    try:
        test_deductive_reasoning_requirements()
        test_causal_reasoning_requirements()
        accuracy = test_accuracy_requirements()
        test_confidence_propagation()
        test_enhanced_features()
        
        print("\n\n" + "=" * 60)
        print("✓ 所有测试通过")
        print(f"✓ 推理准确率: {accuracy:.1%} > 80%")
        print("=" * 60)
        
        print("\n【优化总结】")
        print("1. ✓ 演绎推理：正确实现三段论推理，验证前提和结论的逻辑关系")
        print("2. ✓ 因果推理：基于概念图谱的因果链，支持间接因果关系，计算因果强度")
        print("3. ✓ 置信度优化：基于证据强度计算置信度，多步推理的置信度正确传递")
        print("4. ✓ 推理准确率 > 80%")
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)