"""
测试语言生成器优化效果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.language_generator import LanguageGenerator
from src.core.semantic_atoms import SemanticAtomManager
from src.data.unified_knowledge_manager import UnifiedKnowledgeManager
import logging

logging.basicConfig(level=logging.INFO)


def test_optimization():
    print("=" * 60)
    print("语言生成器优化测试")
    print("=" * 60)
    
    print("\n初始化组件...")
    knowledge_manager = UnifiedKnowledgeManager(
        data_dir="data/raw",
        concept_graph_path="data/raw/concept_graph.db",
        wiki_db_path="data/raw/zhwiki.db"
    )
    
    semantic_atoms = SemanticAtomManager(
        field_dim=512,
        atom_dim=128,
        initial_atoms=1000,
        knowledge_loader=knowledge_manager
    )
    
    generator = LanguageGenerator(
        semantic_atoms=semantic_atoms,
        knowledge_manager=knowledge_manager,
        field_dim=512,
        atom_dim=128
    )
    
    stats = generator.get_generation_stats()
    print("\n生成器统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("测试1: 续写模式 - '春风'")
    print("=" * 60)
    
    for i in range(3):
        result = generator.generate(
            prompt="春风",
            max_length=20,
            mode='continue',
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        coherence = generator.evaluate_coherence(result)
        idioms = generator.detect_idiom_usage(result)
        print(f"\n生成结果 {i+1}: {result}")
        print(f"连贯性分数: {coherence:.2f}")
        if idioms:
            print(f"检测到成语: {idioms}")
    
    print("\n" + "=" * 60)
    print("测试2: 问答模式 - '什么是量子？'")
    print("=" * 60)
    
    result = generator.generate(
        prompt="什么是量子？",
        max_length=50,
        mode='qa',
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    coherence = generator.evaluate_coherence(result)
    print(f"\n生成结果: {result}")
    print(f"连贯性分数: {coherence:.2f}")
    
    print("\n" + "=" * 60)
    print("测试3: 创作模式 - '月光'")
    print("=" * 60)
    
    for i in range(2):
        result = generator.generate(
            prompt="月光",
            max_length=30,
            mode='creative',
            temperature=1.0,
            top_k=50,
            top_p=0.9
        )
        coherence = generator.evaluate_coherence(result)
        print(f"\n生成结果 {i+1}: {result}")
        print(f"连贯性分数: {coherence:.2f}")
    
    print("\n" + "=" * 60)
    print("测试4: 连贯性评估")
    print("=" * 60)
    
    test_texts = [
        "春风得意马蹄疾",
        "春风中也。是《我她！你中...",
        "月光如水洒满大地",
        "量子力学是研究微观粒子运动规律的物理学分支"
    ]
    
    for text in test_texts:
        coherence = generator.evaluate_coherence(text)
        idioms = generator.detect_idiom_usage(text)
        print(f"\n文本: {text}")
        print(f"连贯性分数: {coherence:.2f}")
        if idioms:
            print(f"检测到成语: {idioms}")
    
    print("\n" + "=" * 60)
    print("测试5: 束搜索生成")
    print("=" * 60)
    
    beam_results = generator.beam_search(
        prompt="春风",
        beam_width=5,
        max_length=15,
        length_penalty=1.0
    )
    
    print("\n束搜索结果:")
    for i, (text, score) in enumerate(beam_results):
        coherence = generator.evaluate_coherence(text)
        print(f"{i+1}. {text} (分数: {score:.2f}, 连贯性: {coherence:.2f})")


if __name__ == "__main__":
    test_optimization()