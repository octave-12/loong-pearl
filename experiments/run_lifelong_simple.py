"""简化版永久终身学习 - 确保持续运行"""
import sys, os, time, gc
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("四代龙珠 - 终身学习模式 (简化版)")
print("=" * 70)

t0 = time.time()

def compute_pmi_incremental(corpus_path, batch_size=5000, max_lines=None, 
                           window_size=5, min_count=5, pmi_threshold=1.5,
                           max_pairs=50000):
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

print('\n[Phase 1] 初始化...')
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.field_interface import FieldInterface
from src.core.semantic_atoms import SemanticAtomManager
from src.data.knowledge_loader import KnowledgeLoader

field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu', use_amp=False)
hebbian = HebbianUpdater(field_dim=512, device='cpu')
curiosity = CuriosityDrive(field_dim=512, device='cpu')
interface = FieldInterface(field_dim=512, atom_dim=32, device='cpu')
kl = KnowledgeLoader()
sa = SemanticAtomManager(field_dim=512, atom_dim=32, initial_atoms=500, device='cpu', knowledge_loader=kl)

print('\n[Phase 2] PMI计算...')
pmi_pairs = compute_pmi_incremental(os.path.join(project_root, 'data/corpus.txt'), batch_size=5000, max_lines=None, 
                                   window_size=5, min_count=5, pmi_threshold=1.5, max_pairs=50000)
print(f'PMI pairs: {len(pmi_pairs)}')

print('\n[Phase 3] 聚类...')
clusters = sa.cluster_characters(pmi_pairs, use_igraph=False)
sa.initialize_atoms_from_clusters(clusters, field_dim=512, max_idiom_atoms=500)
print(f'Semantic atoms: {sa.get_num_atoms()}')

del pmi_pairs, clusters
gc.collect()

print('\n[Phase 4] 永久演化循环...')
step = 0
try:
    while True:
        h = field.get_state()
        entropy, exploration_atoms, noise, entropy_state = curiosity.detect_entropy_anomaly(
            h, sa.get_all_regions()
        )
        if noise is not None:
            h = h + noise
            field.set_state(h)
        h = field.evolve(dt=0.1)
        hebbian.update(h)
        step += 1
        
        if step % 100 == 0:
            elapsed = time.time() - t0
            speed = step / elapsed if elapsed > 0 else 0
            print(f'步数: {step:,}, 熵: {entropy:.3f}, 范数: {h.norm().item():.1f}, '
                  f'权重: {hebbian._nnz():,}, 速度: {speed:.1f}步/秒', flush=True)
        
        if step % 1000 == 0:
            gc.collect()
        
        if step % 10000 == 0:
            checkpoint_path = os.path.join(project_root, 'checkpoints', f'lifelong_{step}.pt')
            os.makedirs(os.path.join(project_root, 'checkpoints'), exist_ok=True)
            torch.save({
                'step': step,
                'field_state': field.get_state().cpu(),
                'hebbian_weights': hebbian.get_weight_matrix().cpu()
            }, checkpoint_path)
            print(f'  检查点: {checkpoint_path}', flush=True)
            
except KeyboardInterrupt:
    print(f'\n[停止] 总步数: {step:,}', flush=True)
    print(f'[统计] 总用时: {(time.time()-t0)/60:.1f} 分钟', flush=True)
except Exception as e:
    print(f'\n[异常] {type(e).__name__}: {e}', flush=True)
    import traceback
    traceback.print_exc()
    print(f'[统计] 已完成步数: {step:,}', flush=True)