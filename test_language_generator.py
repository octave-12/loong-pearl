"""
语言生成模块测试脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.language_generator import LanguageGenerator
from src.core.semantic_atoms import SemanticAtomManager
from src.data.unified_knowledge_manager import UnifiedKnowledgeManager
from src.data.knowledge_loader import KnowledgeLoader


def setup_test_environment():
    """设置测试环境"""
    print("=" * 60)
    print("初始化测试环境...")
    print("=" * 60)
    
    knowledge_loader = KnowledgeLoader()
    semantic_atoms = SemanticAtomManager(
        field_dim=512,
        atom_dim=128,
        initial_atoms=1000,
        knowledge_loader=knowledge_loader
    )
    
    print("\n构建测试语料...")
    test_corpus = [
        "春风得意马蹄疾，一日看尽长安花。",
        "春风吹绿了江南岸，明月何时照我还。",
        "春风不度玉门关，羌笛何须怨杨柳。",
        "量子力学是研究微观粒子运动规律的物理学分支。",
        "量子纠缠是量子力学中最神奇的现象之一。",
        "什么是量子？量子是能量的最小单位。",
        "人工智能正在改变我们的生活方式。",
        "深度学习是人工智能的重要分支。",
        "自然语言处理让机器理解人类语言。",
        "机器学习从数据中学习模式和规律。",
    ]
    
    print("计算PMI字对...")
    pmi_pairs = semantic_atoms.compute_pmi(
        test_corpus,
        window_size=5,
        min_count=1,
        pmi_threshold=0.5
    )
    print(f"  找到 {len(pmi_pairs)} 个高PMI字对")
    
    print("\n字符聚类...")
    clusters = semantic_atoms.cluster_characters(pmi_pairs, use_knowledge=False)
    print(f"  得到 {len(clusters)} 个聚类")
    
    print("\n初始化语义原子...")
    semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=512)
    print(f"  创建了 {semantic_atoms.get_num_atoms()} 个语义原子")
    
    knowledge_manager = UnifiedKnowledgeManager(
        data_dir="data/raw",
        concept_graph_path="data/raw/concept_graph.db",
        wiki_db_path="data/raw/zhwiki.db"
    )
    
    print("\n创建语言生成器...")
    generator = LanguageGenerator(
        semantic_atoms=semantic_atoms,
        knowledge_manager=knowledge_manager,
        field_dim=512,
        atom_dim=128
    )
    
    stats = generator.get_generation_stats()
    print("\n生成器配置:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return generator


def test_continue_generation(generator):
    """测试续写功能"""
    print("\n" + "=" * 60)
    print("测试1: 续写功能")
    print("=" * 60)
    
    test_cases = [
        ("春风得意", 20),
        ("人工智能", 20),
        ("深度学习", 20),
    ]
    
    for prompt, max_len in test_cases:
        print(f"\n提示: '{prompt}'")
        result = generator.generate(prompt, max_length=max_len, mode='continue', temperature=0.8)
        print(f"续写: '{result}'")
        print(f"新增: '{result[len(prompt):]}'")


def test_qa_generation(generator):
    """测试问答功能"""
    print("\n" + "=" * 60)
    print("测试2: 问答功能")
    print("=" * 60)
    
    questions = [
        "什么是量子？",
        "什么是人工智能？",
        "深度学习是什么？",
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        answer = generator.generate(question, max_length=30, mode='qa', temperature=0.7)
        print(f"回答: '{answer}'")


def test_creative_generation(generator):
    """测试创作功能"""
    print("\n" + "=" * 60)
    print("测试3: 创作功能")
    print("=" * 60)
    
    topics = [
        "春天",
        "科技",
        "未来",
    ]
    
    for topic in topics:
        print(f"\n主题: {topic}")
        creation = generator.generate(topic, max_length=30, mode='creative', temperature=1.0)
        print(f"创作: '{creation}'")


def test_knowledge_enhanced_generation(generator):
    """测试知识增强生成"""
    print("\n" + "=" * 60)
    print("测试4: 知识增强生成")
    print("=" * 60)
    
    test_cases = [
        ("量子力学研究", "量子"),
        ("人工智能发展", "人工智能"),
    ]
    
    for prompt, query in test_cases:
        print(f"\n提示: '{prompt}'")
        print(f"知识查询: '{query}'")
        result = generator.generate_with_knowledge(prompt, query, max_length=30)
        print(f"生成: '{result}'")


def test_beam_search(generator):
    """测试束搜索"""
    print("\n" + "=" * 60)
    print("测试5: 束搜索生成")
    print("=" * 60)
    
    prompt = "春风"
    print(f"\n提示: '{prompt}'")
    
    results = generator.beam_search(prompt, beam_width=3, max_length=15)
    
    print("\n束搜索结果:")
    for i, (text, score) in enumerate(results, 1):
        print(f"  {i}. '{text}' (分数: {score:.3f})")


def test_sampling_strategies(generator):
    """测试不同采样策略"""
    print("\n" + "=" * 60)
    print("测试6: 不同采样策略")
    print("=" * 60)
    
    prompt = "人工智能"
    
    print(f"\n提示: '{prompt}'")
    
    print("\n低温度 (temperature=0.5):")
    result = generator.generate(prompt, max_length=20, mode='continue', temperature=0.5)
    print(f"  '{result}'")
    
    print("\n中温度 (temperature=1.0):")
    result = generator.generate(prompt, max_length=20, mode='continue', temperature=1.0)
    print(f"  '{result}'")
    
    print("\n高温度 (temperature=1.5):")
    result = generator.generate(prompt, max_length=20, mode='continue', temperature=1.5)
    print(f"  '{result}'")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("语言生成模块测试")
    print("=" * 60)
    
    try:
        generator = setup_test_environment()
        
        test_continue_generation(generator)
        test_qa_generation(generator)
        test_creative_generation(generator)
        test_knowledge_enhanced_generation(generator)
        test_beam_search(generator)
        test_sampling_strategies(generator)
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)