import sys
import os
import time
import json
from datetime import datetime

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.semantic_atoms import SemanticAtomManager
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian
from src.data.knowledge_loader import KnowledgeLoader


def benchmark_pmi(manager, corpus, num_runs=3):
    """PMI计算性能测试"""
    print("\n" + "=" * 60)
    print("PMI计算性能测试")
    print("=" * 60)
    
    corpus_sizes = [100, 1000, 5000]
    results = {}
    
    for size in corpus_sizes:
        if size > len(corpus):
            continue
        
        subset = corpus[:size]
        
        times_single = []
        for _ in range(num_runs):
            start = time.time()
            manager.compute_pmi(subset, num_workers=1)
            times_single.append(time.time() - start)
        
        times_parallel = []
        for _ in range(num_runs):
            start = time.time()
            manager.compute_pmi(subset, num_workers=4)
            times_parallel.append(time.time() - start)
        
        avg_single = np.mean(times_single)
        avg_parallel = np.mean(times_parallel)
        speedup = avg_single / avg_parallel if avg_parallel > 0 else 1.0
        
        results[size] = {
            "single_thread": avg_single,
            "multi_thread": avg_parallel,
            "speedup": speedup
        }
        
        print(f"\n语料大小: {size}")
        print(f"  单线程: {avg_single:.3f}s")
        print(f"  多线程(4核): {avg_parallel:.3f}s")
        print(f"  加速比: {speedup:.2f}x")
    
    return results


def benchmark_evolution(field, hebbian, curiosity, semantic_atoms, interface, num_steps=100):
    """演化性能测试"""
    print("\n" + "=" * 60)
    print("场演化性能测试")
    print("=" * 60)
    
    guardian = FieldGuardian(
        field=field,
        hebbian_updater=hebbian,
        curiosity_drive=curiosity,
        semantic_atoms=semantic_atoms,
        field_interface=interface,
        checkpoint_dir="checkpoints"
    )
    
    start = time.time()
    for _ in range(num_steps):
        guardian.evolve_step()
    elapsed = time.time() - start
    
    steps_per_sec = num_steps / elapsed
    ms_per_step = elapsed / num_steps * 1000
    
    print(f"\n演化步数: {num_steps}")
    print(f"总耗时: {elapsed:.3f}s")
    print(f"每秒步数: {steps_per_sec:.1f}")
    print(f"每步耗时: {ms_per_step:.3f}ms")
    
    return {
        "total_steps": num_steps,
        "total_time": elapsed,
        "steps_per_second": steps_per_sec,
        "ms_per_step": ms_per_step
    }


def benchmark_checkpoint(guardian, num_runs=5):
    """检查点保存/加载性能测试"""
    print("\n" + "=" * 60)
    print("检查点性能测试")
    print("=" * 60)
    
    save_times_compressed = []
    save_times_uncompressed = []
    file_sizes = []
    
    for i in range(num_runs):
        filename = f"benchmark_{i}.pt"
        
        start = time.time()
        filepath_compressed = guardian.save_checkpoint(filename, compress=True)
        save_times_compressed.append(time.time() - start)
        file_sizes.append(os.path.getsize(filepath_compressed) / 1024 / 1024)
        
        start = time.time()
        filepath_uncompressed = guardian.save_checkpoint(filename, compress=False)
        save_times_uncompressed.append(time.time() - start)
        
        os.remove(filepath_compressed) if os.path.exists(filepath_compressed) else None
        os.remove(filepath_uncompressed) if os.path.exists(filepath_uncompressed) else None
    
    avg_save_compressed = np.mean(save_times_compressed)
    avg_save_uncompressed = np.mean(save_times_uncompressed)
    avg_size = np.mean(file_sizes)
    compression_ratio = avg_save_uncompressed / avg_save_compressed
    
    print(f"\n保存次数: {num_runs}")
    print(f"压缩保存: {avg_save_compressed:.3f}s, 文件大小: {avg_size:.2f}MB")
    print(f"普通保存: {avg_save_uncompressed:.3f}s")
    print(f"压缩比: {compression_ratio:.2f}x")
    
    return {
        "save_compressed": avg_save_compressed,
        "save_uncompressed": avg_save_uncompressed,
        "file_size_mb": avg_size,
        "compression_ratio": compression_ratio
    }


def benchmark_batch_processing(guardian, num_texts=10):
    """批处理性能测试"""
    print("\n" + "=" * 60)
    print("批处理性能测试")
    print("=" * 60)
    
    test_texts = [
        "什么是龙",
        "天是什么颜色",
        "水往低处流",
        "春天来了",
        "月亮很圆"
    ] * (num_texts // 5 + 1)
    test_texts = test_texts[:num_texts]
    
    start = time.time()
    outputs = guardian.process_batch(test_texts)
    elapsed_sequential = time.time() - start
    
    print(f"\n处理文本数: {num_texts}")
    print(f"顺序处理: {elapsed_sequential:.3f}s")
    print(f"平均每文本: {elapsed_sequential/num_texts:.3f}s")
    
    return {
        "num_texts": num_texts,
        "total_time": elapsed_sequential,
        "time_per_text": elapsed_sequential / num_texts
    }


def run_benchmark():
    print("=" * 60)
    print("四代龙珠 - 性能基准测试")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    device = 'cpu'
    print(f"\n设备: {device}")
    
    print("\n初始化系统...")
    field = LiquidTimeConstantNetwork(field_dim=4096, device=device)
    hebbian = HebbianUpdater(field_dim=4096, device=device)
    curiosity = CuriosityDrive(field_dim=4096, device=device)
    knowledge = KnowledgeLoader()
    semantic_atoms = SemanticAtomManager(field_dim=4096, knowledge_loader=knowledge, device=device)
    interface = FieldInterface(field_dim=4096, device=device)
    
    print("加载测试语料...")
    corpus = []
    corpus_path = "data/corpus.txt"
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                line = line.strip()
                if line:
                    corpus.append(line)
    else:
        corpus = ["龙飞凤舞", "龙腾虎跃", "春暖花开"] * 1000
    
    print(f"语料行数: {len(corpus)}")
    
    results = {}
    
    results["pmi"] = benchmark_pmi(semantic_atoms, corpus)
    
    print("\n初始化语义原子...")
    pmi_pairs = semantic_atoms.compute_pmi(corpus[:1000], num_workers=4)
    clusters = semantic_atoms.cluster_characters(pmi_pairs)
    semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=4096)
    
    results["evolution"] = benchmark_evolution(field, hebbian, curiosity, semantic_atoms, interface)
    
    guardian = FieldGuardian(
        field=field,
        hebbian_updater=hebbian,
        curiosity_drive=curiosity,
        semantic_atoms=semantic_atoms,
        field_interface=interface,
        checkpoint_dir="checkpoints"
    )
    
    results["checkpoint"] = benchmark_checkpoint(guardian)
    results["batch"] = benchmark_batch_processing(guardian, num_texts=5)
    
    report_path = "benchmark_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("性能测试完成！")
    print(f"报告已保存: {report_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_benchmark()