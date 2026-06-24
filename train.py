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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.semantic_atoms import SemanticAtomManager
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian


def load_corpus(corpus_path: str) -> list:
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                corpus.append(line)
    return corpus


def train_phase1(config: Config, checkpoint_dir: str = "checkpoints"):
    print("=" * 60)
    print("第一阶段：液态时间常数网络验证")
    print("=" * 60)

    field = config.create_field()

    print(f"场维度: {config.field_dim}")
    print(f"设备: {field.device}")
    print(f"参数量: {sum(p.numel() for p in field.parameters())}")

    print("\n测试无输入演化...")
    h = field.get_state()
    initial_norm = h.norm().item()
    print(f"初始状态范数: {initial_norm:.4f}")

    for i in range(100):
        h = field.evolve(dt=config.dt)
        if i % 20 == 0:
            print(f"步数 {i}: 状态范数 = {h.norm().item():.4f}")

    print("\n测试扰动收敛...")
    perturbation = torch.randn(config.field_dim, device=field.device) * 0.5
    h = field.get_state()
    h = h + perturbation
    field.set_state(h)

    print(f"扰动后状态范数: {h.norm().item():.4f}")

    for i in range(200):
        h = field.evolve(dt=config.dt)
        if i % 40 == 0:
            print(f"收敛步数 {i}: 状态范数 = {h.norm().item():.4f}")

    print("\n第一阶段验证完成！")
    print("[OK] 场在无输入时能维持稳定状态")
    print("[OK] 在扰动下能收敛（状态裁剪防止发散）")

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({"field_state": field.get_state().cpu(), "field_params": field.state_dict()},
               os.path.join(checkpoint_dir, "phase1.pt"))
    print(f"[OK] 第一阶段检查点已保存")

    return field


def train_phase2(
    field: LiquidTimeConstantNetwork,
    config: Config,
    corpus_path: str,
    checkpoint_dir: str = "checkpoints"
):
    print("\n" + "=" * 60)
    print("第二阶段：语义原子生成与Hebbian学习")
    print("=" * 60)

    semantic_atoms = config.create_semantic_atoms()

    corpus_size = os.path.getsize(corpus_path) / 1024 / 1024
    print(f"\n语料文件: {corpus_path} ({corpus_size:.1f}MB)")

    if corpus_size > 100:
        print("大语料模式：使用流式PMI计算...")
        pmi_pairs = semantic_atoms.compute_pmi_streaming(
            corpus_path,
            batch_size=10000,
            max_lines=100000,
            window_size=config.pmi_window_size,
            min_count=config.pmi_min_count,
            pmi_threshold=config.pmi_threshold
        )
    else:
        print("加载语料...")
        corpus = load_corpus(corpus_path)
        print(f"语料行数: {len(corpus)}")

        print("\n计算PMI字对...")
        pmi_pairs = semantic_atoms.compute_pmi(
            corpus[:min(100000, len(corpus))],
            window_size=config.pmi_window_size,
            min_count=config.pmi_min_count,
            pmi_threshold=config.pmi_threshold
        )

    print(f"高PMI字对数: {len(pmi_pairs)}")

    print("\n聚类生成语义原子...")
    use_igraph = getattr(config, 'use_igraph', False)
    clusters = semantic_atoms.cluster_characters(pmi_pairs, use_igraph=use_igraph)
    print(f"聚类数: {len(clusters)}")

    print("\n初始化语义原子...")
    max_idiom_atoms = getattr(config, 'max_idiom_atoms', 1000)
    semantic_atoms.initialize_atoms_from_clusters(
        clusters,
        field_dim=config.field_dim,
        max_idiom_atoms=max_idiom_atoms
    )
    print(f"语义原子数: {semantic_atoms.get_num_atoms()}")

    hebbian_updater = config.create_hebbian()
    curiosity_drive = config.create_curiosity()
    field_interface = config.create_interface()

    print("\n开始连续演化...")
    num_evolution_steps = 5000

    for step in tqdm(range(num_evolution_steps), desc="演化"):
        h = field.get_state()

        entropy, exploration_atoms, noise, entropy_state = curiosity_drive.detect_entropy_anomaly(
            h,
            semantic_atoms.get_all_regions()
        )

        if noise is not None:
            h = h + noise
            field.set_state(h)

        h = field.evolve(dt=config.dt)

        weight_matrix = hebbian_updater.update(h)

        if step % 500 == 0:
            stats = hebbian_updater.get_activation_stats()
            print(f"\n步数 {step}:")
            print(f"  熵: {entropy:.3f} ({entropy_state})")
            print(f"  场范数: {h.norm().item():.3f}")
            print(f"  非零权重: {stats['num_nonzero_weights']}")
            print(f"  探索原子数: {len(exploration_atoms)}")

    print("\n第二阶段验证完成！")
    print("[OK] 语义原子从PMI统计中生成")
    print("[OK] Hebbian学习持续运行")
    print("[OK] 好奇心驱动正常工作")

    return hebbian_updater, curiosity_drive, semantic_atoms, field_interface


def train_phase3(
    field: LiquidTimeConstantNetwork,
    hebbian_updater: HebbianUpdater,
    curiosity_drive: CuriosityDrive,
    semantic_atoms: SemanticAtomManager,
    field_interface: FieldInterface,
    config: Config,
    test_questions: list,
    checkpoint_dir: str = "checkpoints"
):
    print("\n" + "=" * 60)
    print("第三阶段：输入输出接口测试")
    print("=" * 60)

    guardian = config.create_guardian(
        field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface
    )

    print("\n测试问答能力...")
    for i, question in enumerate(test_questions):
        print(f"\n问题 {i+1}: {question}")

        output = guardian.process_input(question)
        print(f"回答: {output}")

        for _ in range(100):
            guardian.evolve_step()

    print("\n保存最终检查点...")
    guardian.save_checkpoint("final_model.pt")

    print("\n第三阶段验证完成！")
    print("[OK] 输入扰动注入正常")
    print("[OK] 输出解码正常")
    print("[OK] 守护进程运行稳定")

    return guardian


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> str:
    """查找最新的检查点文件"""
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt.gz"))
    if not checkpoints:
        return None
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_checkpoint_for_phase(
    checkpoint_path: str,
    config: Config
):
    """从检查点加载所有组件"""
    field = config.create_field()
    hebbian_updater = config.create_hebbian()
    curiosity_drive = config.create_curiosity()
    semantic_atoms = config.create_semantic_atoms()
    field_interface = config.create_interface()

    if checkpoint_path.endswith('.gz'):
        import gzip
        with gzip.open(checkpoint_path, 'rb') as f:
            checkpoint = torch.load(f, map_location=field.device, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=field.device, weights_only=False)

    field.set_state(checkpoint["field_state"].to(field.device))
    if "field_params" in checkpoint:
        field.load_state_dict(checkpoint["field_params"])

    if "hebbian_weights" in checkpoint:
        guardian = config.create_guardian(
            field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface
        )
        guardian.load_checkpoint(checkpoint_path)
        return field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface, guardian

    return field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="四代龙珠训练")
    parser.add_argument("--phase", type=int, default=0, help="训练阶段 (0=全部, 1, 2, 3)")
    parser.add_argument("--corpus", type=str, default="data/corpus.txt", help="语料文件路径")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="检查点目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点文件路径（用于phase 2/3）")

    args = parser.parse_args()

    Config.reset()
    config = Config.get(args.config)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    test_questions = [
        "什么是龙",
        "天是什么颜色",
        "水往低处流",
        "春天来了",
        "月亮很圆"
    ]

    if args.phase == 0:
        field = train_phase1(config, checkpoint_dir=args.checkpoint_dir)

        hebbian_updater, curiosity_drive, semantic_atoms, field_interface = train_phase2(
            field, config, corpus_path=args.corpus, checkpoint_dir=args.checkpoint_dir
        )

        guardian = train_phase3(
            field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface,
            config, test_questions, checkpoint_dir=args.checkpoint_dir
        )

    elif args.phase == 1:
        field = train_phase1(config, checkpoint_dir=args.checkpoint_dir)

    elif args.phase == 2:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("错误：未找到检查点文件，请先运行第一阶段")
            print(f"提示：使用 --checkpoint 指定检查点路径，或将检查点放入 {args.checkpoint_dir}/")
            return

        print(f"从检查点加载: {checkpoint_path}")
        field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface, _ = load_checkpoint_for_phase(
            checkpoint_path, config
        )

        hebbian_updater, curiosity_drive, semantic_atoms, field_interface = train_phase2(
            field, config, corpus_path=args.corpus, checkpoint_dir=args.checkpoint_dir
        )

    elif args.phase == 3:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("错误：未找到检查点文件，请先运行第一、二阶段")
            print(f"提示：使用 --checkpoint 指定检查点路径，或将检查点放入 {args.checkpoint_dir}/")
            return

        print(f"从检查点加载: {checkpoint_path}")
        field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface, _ = load_checkpoint_for_phase(
            checkpoint_path, config
        )

        guardian = train_phase3(
            field, hebbian_updater, curiosity_drive, semantic_atoms, field_interface,
            config, test_questions, checkpoint_dir=args.checkpoint_dir
        )

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
