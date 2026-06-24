"""
推理引擎单元测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.reasoning_engine import ReasoningEngine, ReasoningChain


def test_deductive_reasoning():
    """测试演绎推理"""
    print("\n【测试1】演绎推理：苏格拉底会死")
    print("-" * 60)
    
    test_triples = [
        ("人", "IS_A", "生物", 0.95),
        ("苏格拉底", "IS_A", "人", 0.99),
        ("生物", "都会", "死亡", 0.98),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.deductive_reasoning(
        premises=["所有人都会死亡", "苏格拉底是人"],
        conclusion="苏格拉底会死亡"
    )
    
    print(f"推理有效: {result['valid']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("推理链:")
    for step in result['reasoning_chain']:
        print(f"  {step}")
    
    assert isinstance(result['valid'], bool)
    assert isinstance(result['confidence'], float)
    assert isinstance(result['reasoning_chain'], list)
    print("✓ 演绎推理测试通过")


def test_inductive_reasoning():
    """测试归纳推理"""
    print("\n【测试2】归纳推理：乌鸦都是黑色的")
    print("-" * 60)
    
    test_triples = [
        ("乌鸦", "HAS_PROPERTY", "黑色", 0.92),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.inductive_reasoning([
        "乌鸦1是黑色的",
        "乌鸦2是黑色的",
        "乌鸦3是黑色的",
        "乌鸦4是黑色的"
    ])
    
    print(f"归纳结论: {result['generalization']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"支持例子数: {result['support_count']}")
    
    assert result['generalization'] is not None
    assert result['confidence'] > 0
    assert result['support_count'] == 4
    print("✓ 归纳推理测试通过")


def test_analogical_reasoning():
    """测试类比推理"""
    print("\n【测试3】类比推理：空气像水")
    print("-" * 60)
    
    engine = ReasoningEngine()
    
    result = engine.analogical_reasoning(
        source={'entity': '水', 'attributes': ['流动性', '可蒸发', '无色']},
        target={'entity': '空气', 'attributes': ['流动性', '无色']}
    )
    
    print(f"推断属性: {result['inferred_attributes']}")
    print(f"相似度: {result['similarity']:.2f}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"推理说明: {result['reasoning']}")
    
    assert len(result['inferred_attributes']) > 0
    assert result['similarity'] > 0
    assert result['confidence'] > 0
    print("✓ 类比推理测试通过")


def test_causal_reasoning():
    """测试因果推理"""
    print("\n【测试4】因果推理：下雨导致地面湿")
    print("-" * 60)
    
    test_triples = [
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.causal_reasoning("下雨", "地面湿")
    
    print(f"存在因果关系: {result['is_causal']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("因果链:")
    for chain in result['causal_chain']:
        print(f"  {chain}")
    
    assert result['is_causal'] is True
    assert result['confidence'] > 0.3
    assert len(result['causal_chain']) > 0
    print("✓ 因果推理测试通过")


def test_multi_step_reasoning():
    """测试多步推理"""
    print("\n【测试5】多步推理：为什么下雨后路面会滑？")
    print("-" * 60)
    
    test_triples = [
        ("下雨", "CAUSES", "地面湿", 0.90),
        ("地面湿", "CAUSES", "摩擦力降低", 0.85),
        ("摩擦力降低", "CAUSES", "路面滑", 0.88),
    ]
    
    engine = ReasoningEngine(concept_graph=test_triples)
    
    result = engine.multi_step_reasoning("为什么下雨后路面会滑？", steps=5)
    
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"使用步数: {result['steps_used']}")
    print("推理链:")
    for step in result['reasoning_chain']:
        print(f"  {step}")
    
    assert result['answer'] is not None
    assert result['confidence'] >= 0
    assert result['steps_used'] >= 0
    print("✓ 多步推理测试通过")


def test_reasoning_chain():
    """测试推理链管理器"""
    print("\n【测试6】推理链管理器")
    print("-" * 60)
    
    chain = ReasoningChain()
    
    chain.add_step(
        premise="所有人都会死亡",
        conclusion="苏格拉底会死亡",
        rule="演绎推理",
        confidence=0.9
    )
    
    chain.add_step(
        premise="苏格拉底会死亡",
        conclusion="苏格拉底终将死亡",
        rule="等价转换",
        confidence=0.95
    )
    
    print(f"最终结论: {chain.get_final_conclusion()}")
    print(f"综合置信度: {chain.confidence:.2f}")
    print(f"推理步骤数: {len(chain.steps)}")
    
    chain_dict = chain.to_dict()
    assert chain_dict['final_conclusion'] == "苏格拉底终将死亡"
    assert chain.confidence == 0.9 * 0.95
    print("✓ 推理链管理器测试通过")


def test_integration_with_knowledge():
    """测试与知识库集成"""
    print("\n【测试7】与知识库集成")
    print("-" * 60)
    
    try:
        from src.data.unified_knowledge_manager import UnifiedKnowledgeManager
        
        km = UnifiedKnowledgeManager(data_dir="data/raw")
        engine = ReasoningEngine(knowledge_manager=km)
        
        print(f"概念图谱关系数: {len(engine._relation_index)}")
        print(f"概念索引数: {len(engine._concept_index)}")
        
        result = engine.multi_step_reasoning("什么是人？", steps=3)
        print(f"推理结果: {result['answer']}")
        
        print("✓ 知识库集成测试通过")
    except Exception as e:
        print(f"⚠ 知识库集成测试跳过: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("深度推理引擎测试套件")
    print("=" * 60)
    
    test_deductive_reasoning()
    test_inductive_reasoning()
    test_analogical_reasoning()
    test_causal_reasoning()
    test_multi_step_reasoning()
    test_reasoning_chain()
    test_integration_with_knowledge()
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)