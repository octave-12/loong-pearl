import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import warnings
warnings.filterwarnings("ignore")

from src.utils.config import Config
Config.reset()
config = Config.get('config.yaml')


t0 = time.time()

print('=== Phase 1: Liquid Network ===')
field = config.create_field()
h = field.get_state()
print(f'Device: {field.device}, Initial norm: {h.norm().item():.4f}')
for i in range(50):
    h = field.evolve(dt=0.1)
print(f'Phase 1 OK ({time.time()-t0:.1f}s)')

print('\n=== Phase 2a: PMI + Clustering ===')
t1 = time.time()
semantic_atoms = config.create_semantic_atoms()

print('Loading 1000 lines...')
corpus = []
with open('data/corpus.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        line = line.strip()
        if line:
            corpus.append(line)
print(f'Corpus: {len(corpus)}')

print('Computing PMI (simple mode)...')
from collections import defaultdict
import numpy as np

char_counts = defaultdict(int)
pair_counts = defaultdict(int)
total_chars = 0
total_pairs = 0

for text in corpus:
    chars = list(text)
    total_chars += len(chars)
    for char in chars:
        char_counts[char] += 1
    for i in range(len(chars)):
        for j in range(i + 1, min(i + 5, len(chars))):
            pair = (chars[i], chars[j])
            pair_counts[pair] += 1
            total_pairs += 1

pmi_pairs = []
for (char_a, char_b), count in pair_counts.items():
    if count < 2:
        continue
    p_a = char_counts[char_a] / total_chars
    p_b = char_counts[char_b] / total_chars
    p_ab = count / total_pairs
    pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)
    if pmi > 0.5:
        pmi_pairs.append((char_a, char_b, pmi))
print(f'PMI pairs: {len(pmi_pairs)}')

print('Clustering...')
clusters = semantic_atoms.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=config.field_dim, max_idiom_atoms=1000)
print(f'Atoms: {semantic_atoms.get_num_atoms()} ({time.time()-t1:.1f}s)')

print('\n=== Phase 2b: Evolution (2000 steps) ===')
hebbian = config.create_hebbian()
curiosity = config.create_curiosity()
interface = config.create_interface()

t2 = time.time()
for step in range(2000):
    h = field.get_state()
    entropy, exploration_atoms, noise, entropy_state = curiosity.detect_entropy_anomaly(
        h, semantic_atoms.get_all_regions()
    )
    if noise is not None:
        h = h + noise
        field.set_state(h)
    h = field.evolve(dt=0.1)
    hebbian.update(h)
    
    if step % 500 == 0:
        nnz = hebbian._nnz()
        elapsed = time.time() - t2
        sps = (step + 1) / elapsed if elapsed > 0 else 0
        print(f'  Step {step:5d}: norm={h.norm().item():8.2f}, entropy={entropy:.3f}, '
              f'state={entropy_state}, nnz={nnz}, {sps:.0f} steps/s')

print(f'Evolution done ({time.time()-t2:.1f}s)')

print('\n=== Phase 3: I/O Test ===')
guardian = config.create_guardian(field, hebbian, curiosity, semantic_atoms, interface)

questions = ["什么是龙", "天是什么颜色", "春天来了", "水往低处流", "月亮很圆"]
for q in questions:
    a = guardian.process_input(q)
    print(f'Q: {q} -> A: {a}')

print('\n=== Saving ===')
guardian.save_checkpoint('trained_50k.pt', compress=True)

total = time.time() - t0
print(f'\n=== Complete! {total:.0f}s ({total/60:.1f}min) ===')
