"""终身学习模式 - 带语料初始化"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("四代龙珠 - 终身学习模式")
print("加载语料 → 初始化语义原子 → 持续学习")
print("=" * 60)

t0 = time.time()

def compute_pmi_incremental(corpus_path, batch_size=1000, max_lines=10000, 
                           window_size=5, min_count=3, pmi_threshold=1.0,
                           max_pairs=20000):
    global_char_counts = defaultdict(int)
    global_pair_counts = defaultdict(int)
    global_total_chars = 0
    global_total_pairs = 0
    
    batch = []
    total_read = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            batch.append(line)
            total_read += 1
            
            if len(batch) >= batch_size:
                char_counts = defaultdict(int)
                pair_counts = defaultdict(int)
                tc = 0
                tp = 0
                
                for text in batch:
                    chars = list(text)
                    tc += len(chars)
                    for char in chars:
                        char_counts[char] += 1
                    for i in range(len(chars)):
                        for j in range(i + 1, min(i + window_size, len(chars))):
                            pair = (chars[i], chars[j])
                            pair_counts[pair] += 1
                            tp += 1
                
                for char, count in char_counts.items():
                    global_char_counts[char] += count
                for pair, count in pair_counts.items():
                    global_pair_counts[pair] += count
                global_total_chars += tc
                global_total_pairs += tp
                
                del char_counts, pair_counts, batch
                gc.collect()
                
                if len(global_pair_counts) > max_pairs:
                    sorted_pairs = sorted(global_pair_counts.items(), key=lambda x: x[1], reverse=True)
                    global_pair_counts = defaultdict(int, sorted_pairs[:max_pairs])
                    del sorted_pairs
                    gc.collect()
                
                batch = []
            
            if total_read >= max_lines:
                break
    
    if batch:
        char_counts = defaultdict(int)
        pair_counts = defaultdict(int)
        tc = 0
        tp = 0
        
        for text in batch:
            chars = list(text)
            tc += len(chars)
            for char in chars:
                char_counts[char] += 1
            for i in range(len(chars)):
                for j in range(i + 1, min(i + window_size, len(chars))):
                    pair = (chars[i], chars[j])
                    pair_counts[pair] += 1
                    tp += 1
        
        for char, count in char_counts.items():
            global_char_counts[char] += count
        for pair, count in pair_counts.items():
            global_pair_counts[pair] += count
        global_total_chars += tc
        global_total_pairs += tp
        del char_counts, pair_counts, batch
        gc.collect()
    
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
    
    return pmi_pairs

print('\n[Phase 1] 初始化连续神经场...')
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian
from src.core.semantic_atoms import SemanticAtomManager
from src.data.knowledge_loader import KnowledgeLoader

field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu', use_amp=False)
hebbian = HebbianUpdater(field_dim=512, device='cpu')
curiosity = CuriosityDrive(field_dim=512, device='cpu')
interface = FieldInterface(field_dim=512, atom_dim=32, device='cpu')
kl = KnowledgeLoader()
sa = SemanticAtomManager(field_dim=512, atom_dim=32, initial_atoms=200, device='cpu', knowledge_loader=kl)
print(f'初始化完成 ({time.time()-t0:.1f}s)')

print('\n[Phase 2] 加载语料并计算PMI...')
t1 = time.time()
pmi_pairs = compute_pmi_incremental('data/corpus.txt', batch_size=1000, max_lines=10000, 
                                   window_size=5, min_count=3, pmi_threshold=1.0, max_pairs=20000)
print(f'PMI pairs: {len(pmi_pairs)} ({time.time()-t1:.1f}s)')

print('\n[Phase 3] 聚类并初始化语义原子...')
t2 = time.time()
clusters = sa.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

sa.initialize_atoms_from_clusters(clusters, field_dim=512, max_idiom_atoms=200)
print(f'Semantic atoms: {sa.get_num_atoms()} ({time.time()-t2:.1f}s)')

del pmi_pairs, clusters
gc.collect()

print('\n[Phase 4] 创建守护进程并启动终身学习模式...')
guardian = FieldGuardian(
    field=field,
    hebbian_updater=hebbian,
    curiosity_drive=curiosity,
    semantic_atoms=sa,
    field_interface=interface,
    checkpoint_dir='checkpoints'
)

guardian.start_background_evolution(evolve_interval=0.05, auto_checkpoint=False)
print('后台演化线程已启动')

print('\n[Phase 5] 交互测试 - 系统在学习中回答...')
time.sleep(1)

questions = [
    "什么是龙",
    "天是什么颜色",
    "春天来了",
    "水往低处流",
    "月亮很圆"
]

for i, q in enumerate(questions):
    print(f'\n--- 第{i+1}轮 (演化{guardian.step_count}步, 权重{hebbian._nnz()}) ---')
    a = guardian.process_input_with_learning(q, evolve_steps=50)
    print(f'Q: {q}')
    print(f'A: {a}')
    time.sleep(1)

print(f'\n[状态] 总步数: {guardian.step_count}, 权重: {hebbian._nnz()}')

print('\n[Phase 6] 自主演化10秒...')
time.sleep(10)
print(f'[状态] 自主演化后: {guardian.step_count}步, 权重: {hebbian._nnz()}')

print('\n[Phase 7] 再次交互测试...')
for i, q in enumerate(["太阳从东方升起", "花开了"]):
    print(f'\n--- 第{i+1}轮 (演化{guardian.step_count}步) ---')
    a = guardian.process_input_with_learning(q, evolve_steps=50)
    print(f'Q: {q}')
    print(f'A: {a}')

guardian.stop_background_evolution()
guardian.save_checkpoint('lifelong_with_corpus.pt', compress=True)

total = time.time() - t0
print(f'\n=== 完成! {total:.0f}s ({total/60:.1f}min) ===')
print(f'总演化步数: {guardian.step_count}')
print(f'演化速度: {guardian.step_count/total:.1f} 步/秒')