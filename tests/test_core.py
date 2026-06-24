import sys
import os

if sys.platform == 'win32':
    torch_lib_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(torch_lib_path):
        os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
        try:
            os.add_dll_directory(torch_lib_path)
        except (AttributeError, OSError):
            pass

import torch
import numpy as np

torch.set_default_device('cpu')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.semantic_atoms import SemanticAtomManager
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian


def test_liquid_network():
    print("测试液态时间常数网络...")

    field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu')

    h = field.get_state()
    print(f"初始状态维度: {h.shape}")
    print(f"初始状态范数: {h.norm().item():.4f}")

    for i in range(10):
        h, tau = field.evolve(return_tau=True)
        print(f"步数 {i+1}: 范数={h.norm().item():.4f}, τ均值={tau.mean().item():.4f}")

    assert h.norm().item() < 100, "场状态应该有界（state_clip生效）"
    assert h.abs().max().item() <= 5.0 + 0.1, "场状态应在state_clip范围内"

    print("[OK] 液态网络测试通过\n")


def test_hebbian_learning():
    print("测试Hebbian学习...")

    updater = HebbianUpdater(field_dim=512, learning_rate=1e-4, device='cpu')

    h = torch.randn(512)
    h = h / h.norm() * 2.0

    initial_nnz = updater.get_activation_stats()["num_nonzero_weights"]

    for i in range(10):
        weight_matrix = updater.update(h)
        stats = updater.get_activation_stats()
        print(f"步数 {i+1}: 非零权重={stats['num_nonzero_weights']}")

    print("[OK] Hebbian学习测试通过\n")


def test_curiosity_drive():
    print("测试好奇心驱动...")

    drive = CuriosityDrive(field_dim=512, device='cpu')

    for i in range(5):
        h = torch.randn(512) * (i + 1) * 0.5
        entropy, exploration_atoms, noise, entropy_state = drive.detect_entropy_anomaly(h)
        print(f"测试 {i+1}: 熵={entropy:.3f}, 状态={entropy_state}, "
              f"探索原子={len(exploration_atoms)}, 噪声={'有' if noise is not None else '无'}")

    lr_normal = drive.adapt_learning_rate(1e-5, 0.3, "normal")
    lr_high = drive.adapt_learning_rate(1e-5, 0.3, "high_entropy")
    lr_low = drive.adapt_learning_rate(1e-5, 0.3, "low_entropy")
    print(f"学习率适应: 正常={lr_normal:.2e}, 高熵={lr_high:.2e}, 低熵={lr_low:.2e}")

    assert lr_high < lr_normal, "高熵时学习率应降低"
    assert lr_low > lr_high, "低熵时学习率应高于高熵"

    print("[OK] 好奇心驱动测试通过\n")


def test_semantic_atoms():
    print("测试语义原子管理...")

    manager = SemanticAtomManager(field_dim=512, atom_dim=64, initial_atoms=100, device='cpu')

    corpus = [
        "龙飞凤舞",
        "龙腾虎跃",
        "凤毛麟角",
        "虎啸龙吟",
        "春暖花开",
        "秋高气爽",
        "夏日炎炎",
        "冬雪皑皑"
    ]

    pmi_pairs = manager.compute_pmi(corpus, window_size=3, min_count=1, pmi_threshold=0.5)
    print(f"PMI字对数: {len(pmi_pairs)}")

    clusters = manager.cluster_characters(pmi_pairs)
    print(f"聚类数: {len(clusters)}")

    manager.initialize_atoms_from_clusters(clusters, field_dim=512)
    print(f"语义原子数: {manager.get_num_atoms()}")

    test_char = "龙"
    atom_id = manager.find_atom_for_char(test_char)
    print(f"字符 '{test_char}' 的原子ID: {atom_id}")

    state = manager.get_atoms_state()
    print(f"序列化原子数: {len(state['atoms'])}")

    manager2 = SemanticAtomManager(field_dim=512, atom_dim=64, initial_atoms=100, device='cpu')
    manager2.load_atoms_state(state)
    print(f"反序列化原子数: {manager2.get_num_atoms()}")

    assert manager2.get_num_atoms() == manager.get_num_atoms(), "序列化/反序列化应保持一致"

    print("[OK] 语义原子测试通过\n")


def test_field_interface():
    print("测试场接口...")

    semantic_atoms = SemanticAtomManager(field_dim=512, atom_dim=64, initial_atoms=50, device='cpu')

    corpus = ["测试文本", "龙飞凤舞", "春天来了"]
    pmi_pairs = semantic_atoms.compute_pmi(corpus, window_size=2, min_count=1, pmi_threshold=0.3)
    clusters = semantic_atoms.cluster_characters(pmi_pairs)
    semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=512)

    interface = FieldInterface(field_dim=512, atom_dim=64, device='cpu')

    test_text = "龙飞凤舞"
    perturbations = interface.encode_text_to_perturbation(test_text, semantic_atoms)
    print(f"文本 '{test_text}' 生成的扰动数: {len(perturbations)}")

    h = torch.randn(512)
    output = interface.decode_activation_to_text(h, semantic_atoms, top_k=5)
    print(f"解码输出: '{output}'")

    print("[OK] 场接口测试通过\n")


def test_guardian():
    print("测试守护进程...")

    field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu')
    hebbian = HebbianUpdater(field_dim=512, device='cpu')
    curiosity = CuriosityDrive(field_dim=512, device='cpu')
    semantic_atoms = SemanticAtomManager(field_dim=512, atom_dim=64, initial_atoms=50, device='cpu')

    corpus = ["测试", "龙", "春天"]
    pmi_pairs = semantic_atoms.compute_pmi(corpus, window_size=2, min_count=1, pmi_threshold=0.3)
    clusters = semantic_atoms.cluster_characters(pmi_pairs)
    semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=512)

    interface = FieldInterface(field_dim=512, atom_dim=64, device='cpu')

    guardian = FieldGuardian(
        field=field,
        hebbian_updater=hebbian,
        curiosity_drive=curiosity,
        semantic_atoms=semantic_atoms,
        field_interface=interface,
        checkpoint_dir="checkpoints",
        checkpoint_interval=100
    )

    for i in range(10):
        stats = guardian.evolve_step()
        if i % 3 == 0:
            print(f"步数: {stats['step']}, 熵: {stats['entropy']:.3f}, "
                  f"状态: {stats['entropy_state']}, 范数: {stats['field_norm']:.3f}")

    assert "entropy_state" in stats, "统计信息应包含entropy_state"
    assert "learning_rate" in stats, "统计信息应包含learning_rate"

    print("[OK] 守护进程测试通过\n")


def test_knowledge_loader():
    print("测试知识加载器...")
    
    from src.data.knowledge_loader import KnowledgeLoader
    loader = KnowledgeLoader()
    
    sources = loader.get_available_sources()
    print(f"可用知识源: {sum(sources.values())}/{len(sources)}")
    
    idioms = loader.load_idioms()
    print(f"成语数量: {len(idioms)}")
    
    idiom_atoms = loader.get_idiom_atoms()
    print(f"成语原子簇数: {len(idiom_atoms)}")
    if len(idiom_atoms) > 0:
        print(f"示例成语原子: {idiom_atoms[0]}")
    
    common_chars = loader.get_common_chars()
    print(f"常用字符数: {len(common_chars)}")
    
    cedict = loader.load_cedict()
    print(f"CEDICT词条数: {len(cedict)}")
    
    unihan = loader.load_unihan()
    print(f"Unihan字符数: {len(unihan)}")
    
    directed_pairs = loader.load_directed_pairs()
    print(f"预计算PMI字对数: {len(directed_pairs)}")
    
    print("[OK] 知识加载器测试通过\n")


def test_knowledge_injection():
    print("测试知识注入流程...")
    
    from src.data.knowledge_loader import KnowledgeLoader
    
    loader = KnowledgeLoader()
    manager = SemanticAtomManager(
        field_dim=512,
        atom_dim=64,
        initial_atoms=100,
        knowledge_loader=loader,
        device='cpu'
    )
    
    corpus = ["龙飞凤舞", "龙腾虎跃", "春暖花开"]
    pmi_pairs = manager.compute_pmi(corpus, window_size=2, min_count=1, pmi_threshold=0.3)
    print(f"语料PMI字对数: {len(pmi_pairs)}")
    
    enhanced_pairs = loader.get_enhanced_pmi_pairs(pmi_pairs, min_score=0.3)
    print(f"增强后PMI字对数: {len(enhanced_pairs)}")
    
    clusters = manager.cluster_characters(pmi_pairs, use_knowledge=True)
    print(f"知识增强聚类数: {len(clusters)}")
    
    manager.initialize_atoms_from_clusters(clusters, field_dim=512, inject_idiom_atoms=True)
    print(f"注入成语后的原子数: {manager.get_num_atoms()}")
    
    print("[OK] 知识注入流程测试通过\n")


def run_all_tests():
    print("=" * 60)
    print("四代龙珠 - 单元测试")
    print("=" * 60 + "\n")

    test_liquid_network()
    test_hebbian_learning()
    test_curiosity_drive()
    test_semantic_atoms()
    test_field_interface()
    test_guardian()
    test_knowledge_loader()
    test_knowledge_injection()

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
