import yaml
import os
from typing import Any, Dict


_config_defaults = {
    "field_dim": 4096,
    "hidden_dim": 4096,
    "atom_dim": 128,
    "initial_atoms": 5000,
    "tau_max": 1.0,
    "noise_std": 0.01,
    "state_clip": 5.0,
    "dt": 0.1,
    "learning_rate": 1.0e-5,
    "decay_rate": 1.0e-6,
    "activation_threshold": 0.7,
    "dormancy_threshold": 1000,
    "weight_prune_threshold": 1.0e-4,
    "max_density": 0.05,
    "num_bins": 64,
    "entropy_low_threshold": 2.0,
    "entropy_high_threshold": 5.0,
    "exploration_noise_std": 0.05,
    "max_entropy_history": 10000,
    "checkpoint_interval": 1000,
    "atom_evolve_interval": 2000,
    "max_steps": 100000,
    "projection_update_interval": 10,
    "projection_top_k": 5,
    "corpus_path": "data/corpus.txt",
    "pmi_window_size": 5,
    "pmi_min_count": 10,
    "pmi_threshold": 2.0,
    "max_idiom_atoms": 1000,
    "knowledge_data_dir": "data/raw",
    "use_fp16": True,
    "use_amp": True,
    "use_cuda_graph": False,
    "use_igraph": False,
    "device": "auto",
}


class Config:
    _instance = None
    _cached_device = None

    def __init__(self, config_path: str = None):
        self._data = dict(_config_defaults)
        if config_path and os.path.exists(config_path):
            self.load(config_path)

    @classmethod
    def get(cls, config_path: str = None) -> "Config":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
        cls._cached_device = None

    def load(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            self._data.update(user_config)
        Config._cached_device = None

    def get_value(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._data.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def get_device(self) -> str:
        if Config._cached_device is not None:
            return Config._cached_device
        import torch
        device = self._data.get("device", "auto")
        if device == "auto":
            result = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            result = device
        Config._cached_device = result
        return result

    def create_field(self):
        from src.core.liquid_time_constant import LiquidTimeConstantNetwork
        return LiquidTimeConstantNetwork(
            field_dim=self.field_dim,
            hidden_dim=self.hidden_dim,
            tau_max=self.tau_max,
            noise_std=self.noise_std,
            state_clip=self.state_clip,
            use_fp16=self.use_fp16,
            use_amp=self.use_amp,
            use_cuda_graph=self.use_cuda_graph,
            dt=self.dt,
            device=self.get_device()
        )

    def create_hebbian(self):
        from src.core.hebbian_learning import HebbianUpdater
        return HebbianUpdater(
            field_dim=self.field_dim,
            learning_rate=self.learning_rate,
            decay_rate=self.decay_rate,
            activation_threshold=self.activation_threshold,
            dormancy_threshold=self.dormancy_threshold,
            weight_prune_threshold=self.weight_prune_threshold,
            max_density=self.max_density,
            device=self.get_device()
        )

    def create_curiosity(self):
        from src.core.curiosity_drive import CuriosityDrive
        return CuriosityDrive(
            field_dim=self.field_dim,
            num_bins=self.num_bins,
            entropy_low_threshold=self.entropy_low_threshold,
            entropy_high_threshold=self.entropy_high_threshold,
            exploration_noise_std=self.exploration_noise_std,
            max_history=self.max_entropy_history,
            device=self.get_device()
        )

    def create_semantic_atoms(self):
        from src.core.semantic_atoms import SemanticAtomManager
        from src.data.knowledge_loader import KnowledgeLoader
        knowledge = KnowledgeLoader(data_dir=self.knowledge_data_dir)
        return SemanticAtomManager(
            field_dim=self.field_dim,
            atom_dim=self.atom_dim,
            initial_atoms=self.initial_atoms,
            device=self.get_device(),
            knowledge_loader=knowledge
        )

    def create_interface(self):
        from src.core.field_interface import FieldInterface
        return FieldInterface(
            field_dim=self.field_dim,
            atom_dim=self.atom_dim,
            device=self.get_device()
        )

    def create_guardian(self, field, hebbian, curiosity, semantic_atoms, interface):
        from src.core.field_guardian import FieldGuardian
        return FieldGuardian(
            field=field,
            hebbian_updater=hebbian,
            curiosity_drive=curiosity,
            semantic_atoms=semantic_atoms,
            field_interface=interface,
            checkpoint_dir="checkpoints",
            checkpoint_interval=self.checkpoint_interval,
            atom_evolve_interval=self.atom_evolve_interval,
            dt=self.dt,
            projection_update_interval=self.projection_update_interval,
            projection_top_k=self.projection_top_k
        )
