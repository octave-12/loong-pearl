"""四代龙珠 - 永久终身学习模式
全量语料(456万行) + 持续演化 + 无限运行
按Ctrl+C优雅退出并保存检查点
"""
import sys, os, time, gc, signal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("四代龙珠 - 永久终身学习模式")
print("全量语料(456万行) + 持续演化 + 无限运行")
print("按Ctrl+C优雅退出并保存检查点")
print("=" * 70)

t0 = time.time()

def compute_pmi_incremental(corpus_path, batch_size=5000, max_lines=None, 
                           window_size=5, min_count=5, pmi_threshold=1.5,
                           max_pairs=50000):
    """增量PMI计算 - 支持全量语料"""
    print(f'增量PMI计算 (batch_size={batch_size}, max_pairs={max_pairs})')
    
    global_char_counts = defaultdict(int)
    global_pair_counts = defaultdict(int)
    global_total_chars = 0
    global_total_pairs = 0
    
    batch = []
    total_read = 0
    batch_num = 0
    t_start = time.time()
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            batch.append(line)
            total_read += 1
            
            if len(batch) >= batch_size:
                batch_num += 1
                t_batch = time.time()
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
                    trimmed = ' (trimmed)'
                else:
                    trimmed = ''
                
                elapsed = time.time() - t_start
                speed = total_read / elapsed if elapsed > 0 else 0
                print(f'  Batch {batch_num}: {total_read:,} lines, '
                      f'{len(global_char_counts)} chars, {len(global_pair_counts)} pairs{trimmed}, '
                      f'{elapsed:.1f}s ({speed:.0f} lines/s)')
                
                batch = []
            
            if max_lines is not None and total_read >= max_lines:
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
sa = SemanticAtomManager(field_dim=512, atom_dim=32, initial_atoms=500, device='cpu', knowledge_loader=kl)
print(f'初始化完成 ({time.time()-t0:.1f}s)')

print('\n[Phase 2] 加载全量语料并计算PMI (456万行)...')
t1 = time.time()
pmi_pairs = compute_pmi_incremental(
    'data/corpus.txt',
    batch_size=5000,
    max_lines=None,
    window_size=5,
    min_count=5,
    pmi_threshold=1.5,
    max_pairs=50000
)
print(f'PMI计算完成 ({time.time()-t1:.1f}s)')

print('\n[Phase 3] 聚类并初始化语义原子...')
t2 = time.time()
clusters = sa.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

sa.initialize_atoms_from_clusters(clusters, field_dim=512, max_idiom_atoms=500)
print(f'Semantic atoms: {sa.get_num_atoms()} ({time.time()-t2:.1f}s)')

del pmi_pairs, clusters
gc.collect()

print('\n[Phase 4] 创建守护进程...')
guardian = FieldGuardian(
    field=field,
    hebbian_updater=hebbian,
    curiosity_drive=curiosity,
    semantic_atoms=sa,
    field_interface=interface,
    checkpoint_dir='checkpoints',
    checkpoint_interval=10000
)

def signal_handler(signum, frame):
    print('\n\n[信号] 收到中断信号，优雅退出...')
    guardian.stop_background_evolution()
    print('[保存] 保存检查点...')
    checkpoint_path = guardian.save_checkpoint('lifelong_full_corpus.pt', compress=True)
    print(f'[完成] 检查点已保存: {checkpoint_path}')
    print(f'[统计] 总演化步数: {guardian.step_count}')
    print(f'[统计] 非零权重: {hebbian._nnz()}')
    print(f'[统计] 总用时: {(time.time()-t0)/60:.1f} 分钟')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print('\n[Phase 5] 启动永久终身学习模式...')
guardian.start_background_evolution(evolve_interval=0.05, auto_checkpoint=True)
print('后台演化线程已启动，按Ctrl+C退出')

print('\n[Phase 6] 进入永久学习循环 (每10分钟报告一次状态)...')
last_report = time.time()
report_count = 0

try:
    while True:
        time.sleep(60)
        
        elapsed_total = time.time() - t0
        elapsed_since_report = time.time() - last_report
        
        if elapsed_since_report >= 600:
            report_count += 1
            steps = guardian.step_count
            nnz = hebbian._nnz()
            speed = steps / elapsed_total if elapsed_total > 0 else 0
            
            hours = int(elapsed_total // 3600)
            minutes = int((elapsed_total % 3600) // 60)
            
            print(f'\n[运行{hours}小时{minutes}分钟] 步数: {steps:,}, 权重: {nnz:,}, 速度: {speed:.1f}步/秒')
            
            checkpoint_path = guardian.save_checkpoint(f'lifelong_{report_count}.pt', compress=True)
            print(f'  检查点: {checkpoint_path}')
            
            last_report = time.time()
            
except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)