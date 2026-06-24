"""
简化系统测试 - 逐步验证
"""
import sys
sys.path.insert(0, '/mnt/d/soso/projects/Loong-pearl')

print("=" * 70)
print("四代龙珠 - 简化系统测试")
print("=" * 70)

print("\n[测试1] 知识库...")
try:
    from src.data.unified_knowledge_manager import UnifiedKnowledgeManager
    km = UnifiedKnowledgeManager()
    stats = km.get_knowledge_stats()
    print(f"  ✓ 知识量: {sum(stats.values()):,}")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n[测试2] 语义原子...")
try:
    from src.core.semantic_atoms import SemanticAtomManager
    sa = SemanticAtomManager(field_dim=512, atom_dim=32, initial_atoms=100, device='cpu')
    print(f"  ✓ 原子管理器初始化成功")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n[测试3] 语言生成...")
try:
    from src.core.language_generator import LanguageGenerator
    gen = LanguageGenerator(sa, km, field_dim=512)
    result = gen.generate("春风", max_length=10, mode='continue')
    print(f"  ✓ 生成: {result[:30]}...")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n[测试4] 推理引擎...")
try:
    from src.core.reasoning_engine import ReasoningEngine
    re = ReasoningEngine(km, km.load_concept_graph(limit=1000))
    result = re.deductive_reasoning(["所有人都会死", "苏格拉底是人"], "苏格拉底会死")
    print(f"  ✓ 推理: valid={result['valid']}, conf={result['confidence']:.2f}")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n[测试5] 对话管理...")
try:
    from src.core.dialog_manager import DialogManager
    from src.core.context_memory import ContextMemory
    mem = ContextMemory()
    dlg = DialogManager(mem, gen)
    res = dlg.process_input("测试", session_id="test")
    print(f"  ✓ 对话: {res.get('response', 'N/A')[:30]}...")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)