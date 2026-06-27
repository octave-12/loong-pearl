import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.semantic_atoms import SemanticAtomManager
from src.data.knowledge_loader import KnowledgeLoader

kl = KnowledgeLoader()
sa = SemanticAtomManager(field_dim=4096, knowledge_loader=kl, device='cpu')

print('Loading 50000 lines...')
t0 = time.time()
corpus = []
with open('data/corpus.txt', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 50000:
            break
        line = line.strip()
        if line:
            corpus.append(line)
print(f'Corpus: {len(corpus)} ({time.time()-t0:.1f}s)')

print('Computing PMI (4 workers)...')
t1 = time.time()
pmi = sa.compute_pmi(corpus, window_size=5, min_count=3, pmi_threshold=1.0, num_workers=4)
print(f'PMI pairs: {len(pmi)} ({time.time()-t1:.1f}s)')

print('Clustering...')
t2 = time.time()
cl = sa.cluster_characters(pmi, use_igraph=False)
print(f'Clusters: {len(cl)} ({time.time()-t2:.1f}s)')

sa.initialize_atoms_from_clusters(cl, field_dim=4096, max_idiom_atoms=1000)
print(f'Atoms: {sa.get_num_atoms()}')

import json
with open('pmi_results.json', 'w') as f:
    json.dump({'pmi_count': len(pmi), 'cluster_count': len(cl), 'atom_count': sa.get_num_atoms(),
               'elapsed_pmi': time.time()-t1, 'elapsed_cluster': time.time()-t2}, f)
print('Results saved to pmi_results.json')