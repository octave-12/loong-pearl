"""快速验证完整训练流程"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def compute_pmi_incremental(corpus_path, batch_size=1000, max_lines=5000, 
                           window_size=5, min_count=3, pmi_threshold=1.0,
                           max_pairs=10000):
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

t0 = time.time()

print('=== Phase 1: Liquid Network ===')
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu', use_amp=False)
h = field.get_state()
print(f'Device: {field.device}, Dim: {field.field_dim}')
for i in range(50):
    h = field.evolve(dt=0.1)
print(f'Phase 1 OK ({time.time()-t0:.1f}s)')

print('\n=== Phase 2a: PMI + Clustering ===')
t1 = time.time()

pmi_pairs = compute_pmi_incremental('data/corpus.txt', batch_size=1000, max_lines=5000, 
                                   window_size=5, min_count=3, pmi_threshold=1.0, max_pairs=10000)
print(f'PMI pairs: {len(pmi_pairs)}')

print('Clustering...')
from src.core.semantic_atoms import SemanticAtomManager
from src.data.knowledge_loader import KnowledgeLoader
kl = KnowledgeLoader()
sa = SemanticAtomManager(field_dim=512, atom_dim=32, initial_atoms=200, device='cpu', knowledge_loader=kl)
clusters = sa.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

sa.initialize_atoms_from_clusters(clusters, field_dim=512, max_idiom_atoms=200)
print(f'Atoms: {sa.get_num_atoms()}')

del pmi_pairs, clusters
gc.collect()
print(f'Phase 2a OK ({time.time()-t1:.1f}s)')

print('\n=== Phase 2b: Evolution (500 steps) ===')
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian

hebbian = HebbianUpdater(field_dim=512, device='cpu')
curiosity = CuriosityDrive(field_dim=512, device='cpu')
interface = FieldInterface(field_dim=512, atom_dim=32, device='cpu')

t2 = time.time()
for step in range(500):
    h = field.get_state()
    entropy, exploration_atoms, noise, entropy_state = curiosity.detect_entropy_anomaly(
        h, sa.get_all_regions()
    )
    if noise is not None:
        h = h + noise
        field.set_state(h)
    h = field.evolve(dt=0.1)
    hebbian.update(h)
    
    if step % 100 == 0:
        nnz = hebbian._nnz()
        print(f'  Step {step}: norm={h.norm().item():.2f}, entropy={entropy:.3f}, nnz={nnz}')

print(f'Evolution done ({time.time()-t2:.1f}s)')

print('\n=== Phase 3: I/O Test ===')
guardian = FieldGuardian(field=field, hebbian_updater=hebbian, curiosity_drive=curiosity,
                        semantic_atoms=sa, field_interface=interface, checkpoint_dir='checkpoints')

questions = ["什么是龙", "天是什么颜色", "春天来了"]
for q in questions:
    a = guardian.process_input(q)
    print(f'Q: {q}')
    print(f'A: {a}')

total = time.time() - t0
print(f'\n=== Complete! {total:.0f}s ({total/60:.1f}min) ===')