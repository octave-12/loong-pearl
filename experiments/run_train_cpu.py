import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict

t0 = time.time()

print('=== Phase 1: Liquid Network (CPU, dim=512) ===')
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu')
h = field.get_state()
print(f'Initial norm: {h.norm().item():.4f}')
for i in range(50):
    h = field.evolve(dt=0.1)
print(f'Phase 1 OK ({time.time()-t0:.1f}s)')

print('\n=== Phase 2a: PMI + Clustering ===')
t1 = time.time()

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

print('Computing PMI...')
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
print(f'PMI pairs: {len(pmi_pairs)} ({time.time()-t1:.1f}s)')

print('Clustering...')
from src.core.semantic_atoms import SemanticAtomManager
from src.data.knowledge_loader import KnowledgeLoader
kl = KnowledgeLoader()
sa = SemanticAtomManager(field_dim=512, atom_dim=64, initial_atoms=200, device='cpu', knowledge_loader=kl)
clusters = sa.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

sa.initialize_atoms_from_clusters(clusters, field_dim=512, max_idiom_atoms=200)
print(f'Atoms: {sa.get_num_atoms()}')

print('\n=== Phase 2b: Evolution (500 steps) ===')
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian

hebbian = HebbianUpdater(field_dim=512, device='cpu')
curiosity = CuriosityDrive(field_dim=512, device='cpu')
interface = FieldInterface(field_dim=512, atom_dim=64, device='cpu')

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
        print(f'  Step {step}: norm={h.norm().item():.2f}, entropy={entropy:.3f}, state={entropy_state}')

print(f'Evolution done ({time.time()-t2:.1f}s)')

print('\n=== Phase 3: I/O Test ===')
guardian = FieldGuardian(field=field, hebbian_updater=hebbian, curiosity_drive=curiosity,
                         semantic_atoms=sa, field_interface=interface, checkpoint_dir='checkpoints')

for q in ["什么是龙", "春天来了", "水往低处流"]:
    a = guardian.process_input(q)
    print(f'Q: {q} -> A: {a}')

total = time.time() - t0
print(f'\n=== Complete! {total:.0f}s ({total/60:.1f}min) ===')