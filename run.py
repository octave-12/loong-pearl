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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.semantic_atoms import SemanticAtomManager
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian


def interactive_mode(config_path: str = "config.yaml"):
    print("=" * 60)
    print("四代龙珠 - 交互模式")
    print("=" * 60)

    print("\n初始化系统...")

    Config.reset()
    config = Config.get(config_path)

    field = config.create_field()
    hebbian = config.create_hebbian()
    curiosity = config.create_curiosity()
    semantic_atoms = config.create_semantic_atoms()
    interface = config.create_interface()

    corpus_path = "data/corpus.txt"
    if os.path.exists(corpus_path):
        print(f"从 {corpus_path} 加载语料...")
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus.append(line)

        print("生成语义原子...")
        pmi_pairs = semantic_atoms.compute_pmi(
            corpus[:min(500, len(corpus))],
            window_size=3,
            min_count=2,
            pmi_threshold=1.0
        )
        clusters = semantic_atoms.cluster_characters(pmi_pairs, use_igraph=config.use_igraph)
        semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=config.field_dim)
        print(f"语义原子数: {semantic_atoms.get_num_atoms()}")
    else:
        print("警告: 未找到语料文件，使用默认配置")
        test_corpus = ["龙", "凤", "天", "地", "水", "火"]
        pmi_pairs = semantic_atoms.compute_pmi(test_corpus, window_size=2, min_count=1, pmi_threshold=0.5)
        clusters = semantic_atoms.cluster_characters(pmi_pairs, use_igraph=config.use_igraph)
        semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=config.field_dim)

    guardian = config.create_guardian(field, hebbian, curiosity, semantic_atoms, interface)

    print("\n系统就绪！输入文本进行交互，输入 'quit' 退出")
    print("输入 'run' 启动持续演化模式")
    print("输入 'save' 保存检查点")
    print("输入 'load <file>' 加载检查点")
    print("输入 'stats' 查看系统状态")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n用户: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("保存状态并退出...")
                guardian.save_checkpoint("session_end.pt")
                break

            elif user_input.lower() == 'run':
                print("启动持续演化模式（按Ctrl+C停止）...")
                try:
                    guardian.run(max_steps=10000, auto_checkpoint=True)
                except KeyboardInterrupt:
                    print("\n演化模式已停止")

            elif user_input.lower() == 'save':
                filename = guardian.save_checkpoint()
                print(f"检查点已保存: {filename}")

            elif user_input.lower() == 'stats':
                h = field.get_state()
                entropy = curiosity.compute_entropy(h)
                print(f"步数: {guardian.step_count}")
                print(f"场范数: {h.norm().item():.4f}")
                print(f"信息熵: {entropy:.3f}")
                print(f"熵状态: {guardian.entropy_state}")
                print(f"输入频率: {guardian.input_frequency:.4f}")
                print(f"语义原子数: {semantic_atoms.get_num_atoms()}")

            elif user_input.lower().startswith('load '):
                filepath = user_input[5:].strip()
                if os.path.exists(filepath):
                    guardian.load_checkpoint(filepath)
                    print(f"检查点已加载: {filepath}")
                else:
                    print(f"文件不存在: {filepath}")

            else:
                print("处理中...")
                response = guardian.process_input(user_input)
                print(f"系统: {response}")

        except KeyboardInterrupt:
            print("\n检测到中断，保存并退出...")
            guardian.save_checkpoint("interrupt.pt")
            break

        except Exception as e:
            print(f"错误: {e}")


def demo_mode(config_path: str = "config.yaml"):
    print("=" * 60)
    print("四代龙珠 - 演示模式")
    print("=" * 60)

    Config.reset()
    config = Config.get(config_path)

    config._data["field_dim"] = 1024
    config._data["hidden_dim"] = 1024
    config._data["atom_dim"] = 64
    config._data["initial_atoms"] = 100

    print(f"\n使用简化配置 (场维度={config.field_dim}) 进行演示...")

    field = config.create_field()
    hebbian = config.create_hebbian()
    curiosity = config.create_curiosity()
    semantic_atoms = config.create_semantic_atoms()
    interface = config.create_interface()

    test_corpus = [
        "龙飞凤舞", "龙腾虎跃", "春暖花开", "秋高气爽",
        "天高云淡", "水清石见", "山高水长", "云淡风轻"
    ]

    print("\n生成语义原子...")
    pmi_pairs = semantic_atoms.compute_pmi(test_corpus, window_size=2, min_count=1, pmi_threshold=0.5)
    clusters = semantic_atoms.cluster_characters(pmi_pairs, use_igraph=config.use_igraph)
    semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=config.field_dim)
    print(f"语义原子数: {semantic_atoms.get_num_atoms()}")

    guardian = config.create_guardian(field, hebbian, curiosity, semantic_atoms, interface)

    test_questions = [
        "龙",
        "春天",
        "天",
        "水"
    ]

    print("\n开始演示问答...")
    print("-" * 60)

    for i, question in enumerate(test_questions):
        print(f"\n问题 {i+1}: {question}")
        response = guardian.process_input(question)
        print(f"回答: {response}")

        print("\n继续演化...")
        for _ in range(50):
            stats = guardian.evolve_step()
            if _ % 10 == 0:
                print(f"  步数: {stats['step']}, 熵: {stats['entropy']:.3f}, "
                      f"状态: {stats['entropy_state']}, 范数: {stats['field_norm']:.3f}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="四代龙珠运行脚本")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "interactive"], help="运行模式")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    args = parser.parse_args()

    if args.mode == "demo":
        demo_mode(args.config)
    elif args.mode == "interactive":
        interactive_mode(args.config)
