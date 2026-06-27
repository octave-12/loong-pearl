import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config
Config.reset()
config = Config.get('config.yaml')

from src.core.liquid_time_constant import LiquidTimeConstantNetwork
field = config.create_field()
print(f'Field created on {field.device}')

from src.core.semantic_atoms import SemanticAtomManager
semantic_atoms = config.create_semantic_atoms()

print('Computing PMI streaming (5000 lines)...')
pmi_pairs = semantic_atoms.compute_pmi_streaming('data/corpus.txt', batch_size=5000, max_lines=5000)
print(f'PMI pairs: {len(pmi_pairs)}')

print('Clustering...')
clusters = semantic_atoms.cluster_characters(pmi_pairs, use_igraph=False)
print(f'Clusters: {len(clusters)}')

semantic_atoms.initialize_atoms_from_clusters(clusters, field_dim=config.field_dim, max_idiom_atoms=500)
print(f'Atoms: {semantic_atoms.get_num_atoms()}')

from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian

hebbian = config.create_hebbian()
curiosity = config.create_curiosity()
interface = config.create_interface()

print('Running 500 evolution steps...')
for step in range(500):
    stats = {
        'step': step,
        'entropy': 0,
        'entropy_state': 'normal',
        'field_norm': 0,
        'learning_rate': 0
    }
    h = field.get_state()
    entropy, exploration_atoms, noise, entropy_state = curiosity.detect_entropy_anomaly(
        h, semantic_atoms.get_all_regions()
    )
    if noise is not None:
        h = h + noise
        field.set_state(h)
    h = field.evolve(dt=config.dt)
    hebbian.update(h)
    if step % 100 == 0:
        print(f'  Step {step}: norm={h.norm().item():.2f}, entropy={entropy:.3f}, state={entropy_state}')

print('Phase 2 DONE!')