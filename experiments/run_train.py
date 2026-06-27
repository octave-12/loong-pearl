import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from src.utils.config import Config
Config.reset()
config = Config.get('config.yaml')

print('=== Phase 1: Liquid Network ===')
field = config.create_field()
print(f'Field device: {field.device}')

h = field.get_state()
print(f'Initial norm: {h.norm().item():.4f}')

for i in range(50):
    h = field.evolve(dt=0.1)
print(f'After 50 steps norm: {h.norm().item():.4f}')

print('\n=== Phase 2: Semantic Atoms ===')
semantic_atoms = config.create_semantic_atoms()

print('Loading corpus (5000 lines)...')
corpus = []
with open('data/corpus.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5000:
            break
        line = line.strip()
        if line:
            corpus.append(line)
print(f'Corpus size: {len(corpus)}')

print('Computing PMI...')
pmi_pairs = semantic_atoms.compute_pmi(corpus, window_size=5, min_count=3, pmi_threshold=1.0, num_workers=1)
print(f'PMI pairs: {len(pmi_pairs)}')

print('Clustering...')
clusters = semantic_atoms.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=config.field_dim, max_idiom_atoms=500)
print(f'Atoms: {semantic_atoms.get_num_atoms()}')

print('\n=== Phase 2b: Evolution ===')
hebbian = config.create_hebbian()
curiosity = config.create_curiosity()
interface = config.create_interface()

for step in range(200):
    h = field.get_state()
    entropy, exploration_atoms, noise, entropy_state = curiosity.detect_entropy_anomaly(
        h, semantic_atoms.get_all_regions()
    )
    if noise is not None:
        h = h + noise
        field.set_state(h)
    h = field.evolve(dt=0.1)
    hebbian.update(h)
    if step % 50 == 0:
        print(f'  Step {step}: norm={h.norm().item():.2f}, entropy={entropy:.3f}, state={entropy_state}')

print('\n=== Phase 3: I/O Test ===')
guardian = config.create_guardian(field, hebbian, curiosity, semantic_atoms, interface)

for q in ["什么是龙", "春天来了"]:
    print(f'\nQ: {q}')
    a = guardian.process_input(q)
    print(f'A: {a}')

print('\n=== Training Complete! ===')