"""
内存优化的完整训练方案
适用于15GB内存环境
"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def compute_pmi_batch(corpus_chunk, window_size=5, min_count=2):
    """计算单个批次的PMI（内存友好）"""
    char_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    total_chars = 0
    total_pairs = 0
    
    for text in corpus_chunk:
        chars = list(text)
        total_chars += len(chars)
        for char in chars:
            char_counts[char] += 1
        for i in range(len(chars)):
            for j in range(i + 1, min(i + window_size, len(chars))):
                pair = (chars[i], chars[j])
                pair_counts[pair] += 1
                total_pairs += 1
    
    return char_counts, pair_counts, total_chars, total_pairs

def compute_pmi_incremental(corpus_path, batch_size=2000, max_lines=50000, 
                           window_size=5, min_count=3, pmi_threshold=1.0,
                           max_pairs=100000):
    """增量PMI计算 - 分批读取语料，逐步累积统计，限制pair数量"""
    print(f'增量PMI计算 (batch_size={batch_size}, max_lines={max_lines}, max_pairs={max_pairs})')
    
    global_char_counts = defaultdict(int)
    global_pair_counts = defaultdict(int)
    global_total_chars = 0
    global_total_pairs = 0
    
    batch = []
    total_read = 0
    batch_num = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            batch.append(line)
            total_read += 1
            
            if len(batch) >= batch_size:
                batch_num += 1
                t0 = time.time()
                cc, pc, tc, tp = compute_pmi_batch(batch, window_size, min_count)
                
                # 合并到全局统计
                for char, count in cc.items():
                    global_char_counts[char] += count
                for pair, count in pc.items():
                    global_pair_counts[pair] += count
                global_total_chars += tc
                global_total_pairs += tp
                
                del cc, pc, batch
                gc.collect()
                
                # 定期修剪：只保留高频pair
                if len(global_pair_counts) > max_pairs:
                    sorted_pairs = sorted(global_pair_counts.items(), key=lambda x: x[1], reverse=True)
                    global_pair_counts = defaultdict(int, sorted_pairs[:max_pairs])
                    del sorted_pairs
                    gc.collect()
                    print(f'  Batch {batch_num}: {total_read} lines, '
                          f'{len(global_char_counts)} chars, {len(global_pair_counts)} pairs (trimmed) '
                          f'({time.time()-t0:.1f}s)')
                else:
                    print(f'  Batch {batch_num}: {total_read} lines, '
                          f'{len(global_char_counts)} chars, {len(global_pair_counts)} pairs '
                          f'({time.time()-t0:.1f}s)')
                batch = []
            
            if total_read >= max_lines:
                break
    
    # 处理最后一批
    if batch:
        cc, pc, tc, tp = compute_pmi_batch(batch, window_size, min_count)
        for char, count in cc.items():
            global_char_counts[char] += count
        for pair, count in pc.items():
            global_pair_counts[pair] += count
        global_total_chars += tc
        global_total_pairs += tp
        del cc, pc, batch
        gc.collect()
    
    print(f'计算PMI值...')
    pmi_pairs = []
    for (char_a, char_b), count in global_pair_counts.items():
        if count < min_count:
            continue
        p_a = global_char_counts[char_a] / global_total_chars
        p_b = global_char_counts[char_b] / global_total_chars
        p_ab = count / global_total_pairs
        pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)
        if pmi > pmi_threshold:
            pmi_pairs.append((char_a, char_b, float(pmi)))
    
    del global_char_counts, global_pair_counts
    gc.collect()
    
    print(f'PMI pairs: {len(pmi_pairs)}')
    return pmi_pairs

# ========== 主训练流程 ==========
t0 = time.time()

print('=== Phase 1: Liquid Network ===')
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
field = LiquidTimeConstantNetwork(field_dim=1024, hidden_dim=1024, device='cpu', use_amp=False)
h = field.get_state()
print(f'Device: {field.device}, Dim: {field.field_dim}, Initial norm: {h.norm().item():.4f}')
for i in range(50):
    h = field.evolve(dt=0.1)
print(f'Phase 1 OK ({time.time()-t0:.1f}s)')

print('\n=== Phase 2a: PMI + Clustering ===')
t1 = time.time()

# 增量PMI计算
pmi_pairs = compute_pmi_incremental(
    'data/corpus.txt',
    batch_size=2000,      # 每批2000行
    max_lines=10000,      # 总共1万行
    window_size=5,
    min_count=3,
    pmi_threshold=1.0,
    max_pairs=20000       # 最多保留2万个pair
)

print('Clustering...')
from src.core.semantic_atoms import SemanticAtomManager
from src.data.knowledge_loader import KnowledgeLoader
kl = KnowledgeLoader()
sa = SemanticAtomManager(field_dim=1024, atom_dim=64, initial_atoms=500, device='cpu', knowledge_loader=kl)
clusters = sa.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

sa.initialize_atoms_from_clusters(clusters, field_dim=1024, max_idiom_atoms=500)
print(f'Atoms: {sa.get_num_atoms()}')

# 清理PMI数据
del pmi_pairs, clusters
gc.collect()

print(f'Phase 2a OK ({time.time()-t1:.1f}s)')

print('\n=== Phase 2b: Evolution (1000 steps) ===')
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian

hebbian = HebbianUpdater(field_dim=1024, device='cpu')
curiosity = CuriosityDrive(field_dim=1024, device='cpu')
interface = FieldInterface(field_dim=1024, atom_dim=64, device='cpu')

t2 = time.time()
for step in range(1000):
    h = field.get_state()
    entropy, exploration_atoms, noise, entropy_state = curiosity.detect_entropy_anomaly(
        h, sa.get_all_regions()
    )
    if noise is not None:
        h = h + noise
        field.set_state(h)
    h = field.evolve(dt=0.1)
    hebbian.update(h)
    
    if step % 1000 == 0:
        nnz = hebbian._nnz()
        print(f'  Step {step}: norm={h.norm().item():.2f}, entropy={entropy:.3f}, '
              f'state={entropy_state}, nnz={nnz}')

print(f'Evolution done ({time.time()-t2:.1f}s)')

print('\n=== Phase 3: I/O Test ===')
guardian = FieldGuardian(field=field, hebbian_updater=hebbian, curiosity_drive=curiosity,
                         semantic_atoms=sa, field_interface=interface, checkpoint_dir='checkpoints')

questions = ["什么是龙", "天是什么颜色", "春天来了", "水往低处流", "月亮很圆"]
for q in questions:
    a = guardian.process_input(q)
    print(f'Q: {q}')
    print(f'A: {a}')

print('\n=== Saving Checkpoint ===')
guardian.save_checkpoint('trained_50k_final.pt', compress=True)

total = time.time() - t0
print(f'\n=== Complete! {total:.0f}s ({total/60:.1f}min) ===')