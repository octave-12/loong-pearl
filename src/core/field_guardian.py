import torch
import time
import os
import logging
import queue
import gzip
import threading
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime


class SearchDriver:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._search_callback = None

    def set_search_callback(self, callback: Callable[[List[str]], List[str]]):
        self._search_callback = callback

    def search(self, keywords: List[str]) -> List[str]:
        if self._search_callback is not None:
            try:
                results = self._search_callback(keywords)
                self.logger.info(f"搜索驱动: 关键词={keywords}, 结果数={len(results)}")
                return results
            except Exception as e:
                self.logger.warning(f"搜索驱动失败: {e}")
                return []
        return []


class FieldGuardian:
    def __init__(
        self,
        field,
        hebbian_updater,
        curiosity_drive,
        semantic_atoms,
        field_interface,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 1000,
        atom_evolve_interval: int = 2000,
        dt: float = 0.1,
        projection_update_interval: int = 10,
        projection_top_k: int = 5
    ):
        self.field = field
        self.hebbian_updater = hebbian_updater
        self.curiosity_drive = curiosity_drive
        self.semantic_atoms = semantic_atoms
        self.field_interface = field_interface

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.atom_evolve_interval = atom_evolve_interval
        self.dt = dt
        self.projection_update_interval = projection_update_interval
        self.projection_top_k = projection_top_k

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.running = False
        self.step_count = 0
        self.last_checkpoint_step = 0
        self.last_atom_evolve_step = 0
        self.input_frequency = 0.0
        self.input_count = 0
        self.total_steps = 0
        self.entropy_state = "normal"

        self._logger_instance = None

        self.search_driver = SearchDriver(self._get_logger())

        self._input_queue = queue.Queue()

        self._cached_nnz = 0
        self._last_checkpoint_state = None
        self._last_checkpoint_path = None
        self._nnz_cache_step = 0
        self._nnz_cache_interval = 50

        self._background_thread = None
        self._realtime_mode = False
        self._lock = threading.Lock()

    def _get_logger(self) -> logging.Logger:
        if self._logger_instance is None:
            self._logger_instance = logging.getLogger("FieldGuardian")
            self._logger_instance.setLevel(logging.INFO)
            if not self._logger_instance.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s'))
                self._logger_instance.addHandler(handler)

                log_dir = "logs"
                os.makedirs(log_dir, exist_ok=True)
                log_filename = os.path.join(log_dir, "field_{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
                file_handler = logging.FileHandler(log_filename)
                file_handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s'))
                self._logger_instance.addHandler(file_handler)
        return self._logger_instance

    @property
    def logger(self) -> logging.Logger:
        return self._get_logger()

    def set_search_callback(self, callback: Callable[[List[str]], List[str]]):
        self.search_driver.set_search_callback(callback)

    def submit_input(self, text: str):
        self._input_queue.put(text)

    def evolve_step(
        self,
        external_input: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        with self._lock:
            h = self.field.get_state()

            entropy, exploration_atoms, noise, entropy_state = self.curiosity_drive.detect_entropy_anomaly(
                h,
                self.semantic_atoms.get_all_regions()
            )
            self.entropy_state = entropy_state

            if noise is not None:
                h = h + noise
                self.field.set_state(h)

            h = self.field.evolve(h, u=external_input, dt=self.dt)

            weight_matrix = self.hebbian_updater.update(h)

            if self.step_count % self.projection_update_interval == 0:
                self.field_interface.evolve_projections(
                    h, self.semantic_atoms, learning_rate=1e-6,
                    top_k=self.projection_top_k
                )

            adapted_lr = self.curiosity_drive.adapt_learning_rate(
                self.hebbian_updater.base_learning_rate,
                self.input_frequency,
                entropy_state
            )
            self.hebbian_updater.learning_rate = adapted_lr

            self.step_count += 1
            self.total_steps += 1

            if external_input is not None:
                self.input_count += 1

            if self.total_steps > 0:
                self.input_frequency = self.input_count / self.total_steps

            if self.step_count - self.last_atom_evolve_step >= self.atom_evolve_interval:
                self.semantic_atoms.evolve_atoms(self.step_count)
                self.field_interface.invalidate_cache()
                self.last_atom_evolve_step = self.step_count

            if len(exploration_atoms) > 0:
                self._handle_exploration(exploration_atoms)

            for atom_id in exploration_atoms:
                self.semantic_atoms.update_atom_activation(atom_id, self.step_count)

            if self.step_count - self._nnz_cache_step >= self._nnz_cache_interval:
                self._cached_nnz = self.hebbian_updater._nnz()
                self._nnz_cache_step = self.step_count

        stats = {
            "step": self.step_count,
            "entropy": entropy,
            "entropy_state": entropy_state,
            "exploration_atoms": len(exploration_atoms),
            "field_norm": h.norm().item(),
            "num_nonzero_weights": self._cached_nnz,
            "input_frequency": self.input_frequency,
            "learning_rate": adapted_lr
        }

        return stats

    def _handle_exploration(self, exploration_atom_ids: List[int]):
        keywords = self.semantic_atoms.get_exploration_keywords(exploration_atom_ids)
        if not keywords:
            return

        search_results = self.search_driver.search(keywords)
        if search_results:
            for result_text in search_results[:3]:
                self.semantic_atoms.inject_search_result(
                    result_text, self.field, self.field_interface, strength=0.3
                )
            self.logger.info(f"搜索驱动注入: 关键词={keywords}, 结果数={len(search_results)}")

    def process_input(self, text: str) -> str:
        self.logger.info(f"输入: {text}")

        perturbations = self.field_interface.encode_text_to_perturbation(
            text,
            self.semantic_atoms
        )

        for perturbation, delay in perturbations:
            time.sleep(delay)
            h = self.field.get_state()
            h = self.field_interface.inject_perturbation(h, perturbation)
            self.field.set_state(h)

            for _ in range(10):
                stats = self.evolve_step()

        for _ in range(50):
            stats = self.evolve_step()
            if stats["entropy_state"] != "high_entropy" and stats["entropy"] < 3.0:
                break

        h = self.field.get_state()
        output = self.field_interface.decode_activation_to_text(
            h,
            self.semantic_atoms,
            top_k=20
        )

        self.logger.info(f"输出: {output}")
        return output

    def process_input_with_learning(self, text: str, evolve_steps: int = 100) -> str:
        """处理输入并在过程中持续学习（终身学习模式）"""
        self.logger.info(f"[学习模式] 输入: {text}")
        
        perturbations = self.field_interface.encode_text_to_perturbation(
            text,
            self.semantic_atoms
        )

        for perturbation, delay in perturbations:
            time.sleep(delay)
            h = self.field.get_state()
            h = self.field_interface.inject_perturbation(h, perturbation)
            self.field.set_state(h)

            for _ in range(10):
                stats = self.evolve_step()

        for _ in range(evolve_steps):
            stats = self.evolve_step()

        h = self.field.get_state()
        output = self.field_interface.decode_activation_to_text(
            h,
            self.semantic_atoms,
            top_k=20
        )

        self.logger.info(f"[学习模式] 输出: {output}, 演化步数: {evolve_steps}")
        return output

    def process_batch(self, texts: List[str]) -> List[str]:
        """批量处理多个输入文本
        
        注意: 由于共享场状态，文本必须顺序处理。
        如需真正并行，需为每个worker创建独立的场实例。
        
        Args:
            texts: 输入文本列表
        
        Returns:
            输出文本列表
        """
        outputs = []
        for i, text in enumerate(texts):
            self.logger.info(f"批量处理 [{i+1}/{len(texts)}]: {text}")
            output = self.process_input(text)
            outputs.append(output)
        return outputs

    def save_checkpoint(self, filename: Optional[str] = None, compress: bool = True):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_step{self.step_count}_{timestamp}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)

        wm = self.hebbian_updater.get_weight_matrix()
        if wm.is_sparse_csr:
            wm = wm.to_sparse_coo().coalesce()

        checkpoint = {
            "step_count": self.step_count,
            "total_steps": self.total_steps,
            "input_count": self.input_count,
            "input_frequency": self.input_frequency,
            "entropy_state": self.entropy_state,
            "last_atom_evolve_step": self.last_atom_evolve_step,
            "field_state": self.field.get_state().cpu(),
            "field_params": self.field.state_dict(),
            "hebbian_weights": wm.cpu(),
            "hebbian_stats": self.hebbian_updater.get_activation_stats(),
            "entropy_history": self.curiosity_drive.get_entropy_history(),
            "input_projection": self.field_interface.input_projection.cpu(),
            "output_projection": self.field_interface.output_projection.cpu(),
            "char_embeddings": {
                k: v.cpu() for k, v in self.field_interface.char_embeddings.items()
            },
            "semantic_atoms": self.semantic_atoms.get_atoms_state()
        }

        if compress:
            filepath_gz = filepath + '.gz'
            with gzip.open(filepath_gz, 'wb', compresslevel=6) as f:
                torch.save(checkpoint, f, _use_new_zipfile_serialization=True)
            filepath = filepath_gz
            self.logger.info(f"检查点已保存(压缩): {filepath}")
        else:
            torch.save(checkpoint, filepath)
            self.logger.info(f"检查点已保存: {filepath}")

        return filepath

    def load_checkpoint(self, filepath: str):
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.field.device, weights_only=False)
        else:
            checkpoint = torch.load(filepath, map_location=self.field.device, weights_only=False)

        self.step_count = checkpoint["step_count"]
        self.total_steps = checkpoint.get("total_steps", self.step_count)
        self.input_count = checkpoint.get("input_count", 0)
        self.input_frequency = checkpoint.get("input_frequency", 0.0)
        self.entropy_state = checkpoint.get("entropy_state", "normal")
        self.last_atom_evolve_step = checkpoint.get("last_atom_evolve_step", 0)

        self.field.set_state(checkpoint["field_state"].to(self.field.device))
        self.field.load_state_dict(checkpoint["field_params"])

        self.hebbian_updater.weight_matrix = checkpoint["hebbian_weights"].to(self.field.device)
        hebbian_stats = checkpoint["hebbian_stats"]
        self.hebbian_updater.activation_counts = torch.tensor(
            hebbian_stats["activation_counts"],
            device=self.field.device
        )
        self.hebbian_updater.last_activation_time = torch.tensor(
            hebbian_stats["last_activation_time"],
            device=self.field.device
        )
        self.hebbian_updater.current_step = hebbian_stats["current_step"]

        self.curiosity_drive.entropy_history = checkpoint["entropy_history"]

        self.field_interface.input_projection = checkpoint["input_projection"].to(self.field.device)
        self.field_interface.output_projection = checkpoint["output_projection"].to(self.field.device)

        if "char_embeddings" in checkpoint:
            self.field_interface.char_embeddings = {
                k: v.to(self.field.device) for k, v in checkpoint["char_embeddings"].items()
            }

        if "semantic_atoms" in checkpoint:
            self.semantic_atoms.load_atoms_state(checkpoint["semantic_atoms"])
            self.field_interface.invalidate_cache()

        self.logger.info(f"检查点已加载: {filepath}")

    def save_checkpoint_incremental(self, filename: Optional[str] = None):
        """增量保存检查点（只保存变化的部分）"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_incremental_step{self.step_count}_{timestamp}.pt.gz"

        filepath = os.path.join(self.checkpoint_dir, filename)

        current_state = {
            "step_count": self.step_count,
            "total_steps": self.total_steps,
            "input_count": self.input_count,
            "input_frequency": self.input_frequency,
            "entropy_state": self.entropy_state,
            "last_atom_evolve_step": self.last_atom_evolve_step,
            "field_state": self.field.get_state().cpu(),
            "hebbian_nnz": self.hebbian_updater._nnz(),
        }

        if self._last_checkpoint_state is not None:
            delta = {}
            
            field_diff = (current_state["field_state"] - self._last_checkpoint_state["field_state"])
            field_diff_norm = field_diff.norm().item()
            if field_diff_norm > 1e-6:
                delta["field_state_diff"] = field_diff
                delta["field_diff_norm"] = field_diff_norm
            
            delta["step_delta"] = current_state["step_count"] - self._last_checkpoint_state["step_count"]
            delta["input_delta"] = current_state["input_count"] - self._last_checkpoint_state["input_count"]
            
            if delta:
                delta["base_checkpoint"] = self._last_checkpoint_path
                delta["current_state"] = current_state
                
                with gzip.open(filepath, 'wb', compresslevel=6) as f:
                    torch.save(delta, f, _use_new_zipfile_serialization=True)
                self.logger.info(f"增量检查点已保存: {filepath} (变化: {len(delta)}项)")
            else:
                self.logger.info("无显著变化，跳过增量保存")
                # 即使跳过保存，也要更新state以保持基线正确
                self._last_checkpoint_state = current_state
                return None
        else:
            with gzip.open(filepath, 'wb', compresslevel=6) as f:
                torch.save({"full_state": current_state}, f, _use_new_zipfile_serialization=True)
            self.logger.info(f"首次增量检查点已保存: {filepath}")

        self._last_checkpoint_state = current_state
        self._last_checkpoint_path = filepath
        
        return filepath

    def run(
        self,
        max_steps: Optional[int] = None,
        auto_checkpoint: bool = True,
        realtime: bool = True
    ):
        """启动持续演化循环
        
        Args:
            max_steps: 最大步数（None表示无限）
            auto_checkpoint: 是否自动保存检查点
            realtime: 是否按dt时间间隔实时演化（False时全速运行）
        """
        self.running = True
        self.logger.info(f"守护进程启动，时间步长: {self.dt}s, 实时模式: {realtime}")

        try:
            while self.running:
                if max_steps is not None and self.step_count >= max_steps:
                    self.logger.info(f"达到最大步数 {max_steps}，停止运行")
                    break

                while not self._input_queue.empty():
                    try:
                        text = self._input_queue.get_nowait()
                        self.process_input(text)
                    except queue.Empty:
                        break

                stats = self.evolve_step()

                if self.step_count % 100 == 0:
                    self.logger.info(
                        f"步数: {stats['step']}, 熵: {stats['entropy']:.3f}, "
                        f"状态: {stats['entropy_state']}, "
                        f"场范数: {stats['field_norm']:.3f}, "
                        f"非零权重: {stats['num_nonzero_weights']}, "
                        f"学习率: {stats['learning_rate']:.2e}"
                    )

                if auto_checkpoint and self.step_count - self.last_checkpoint_step >= self.checkpoint_interval:
                    self.save_checkpoint()
                    self.last_checkpoint_step = self.step_count

                if realtime:
                    time.sleep(self.dt)

        except KeyboardInterrupt:
            self.logger.info("守护进程被中断")
            self.running = False

        if auto_checkpoint:
            self.save_checkpoint()

    def stop(self):
        self.running = False
        if self._background_thread is not None:
            self._background_thread.join(timeout=5.0)
            self._background_thread = None
        self.logger.info("守护进程已停止")

    def start_background_evolution(self, evolve_interval: float = 0.1, auto_checkpoint: bool = False):
        """启动后台持续演化线程（终身学习模式）"""
        if self._background_thread is not None and self._background_thread.is_alive():
            self.logger.warning("后台演化线程已在运行")
            return

        self.running = True
        self._realtime_mode = True

        def _background_loop():
            try:
                while self.running:
                    if not self._input_queue.empty():
                        try:
                            text = self._input_queue.get_nowait()
                            self.process_input_with_learning(text, evolve_steps=50)
                        except queue.Empty:
                            pass

                    stats = self.evolve_step()

                    if self.step_count % 100 == 0:
                        self.logger.info(
                            f"[后台演化] 步数: {stats['step']}, 熵: {stats['entropy']:.3f}, "
                            f"状态: {stats['entropy_state']}, "
                            f"场范数: {stats['field_norm']:.3f}, "
                            f"非零权重: {stats['num_nonzero_weights']}"
                        )

                    if auto_checkpoint and self.step_count - self.last_checkpoint_step >= self.checkpoint_interval:
                        self.save_checkpoint()
                        self.last_checkpoint_step = self.step_count

                    time.sleep(evolve_interval)

            except Exception as e:
                self.logger.error(f"后台演化线程异常: {e}")
            finally:
                self.running = False
                self.logger.info("后台演化线程结束")

        self._background_thread = threading.Thread(target=_background_loop, daemon=True)
        self._background_thread.start()
        self.logger.info(f"后台演化线程已启动 (间隔={evolve_interval}s)")

    def stop_background_evolution(self):
        """停止后台演化线程"""
        self.running = False
        if self._background_thread is not None:
            self._background_thread.join(timeout=5.0)
        self._background_thread = None
        self._realtime_mode = False
        self._lock = threading.Lock()
        self.logger.info("后台演化线程已停止")

    def interact(self, text: str) -> str:
        """交互式输入（终身学习模式）：如果后台线程运行则提交到队列，否则直接处理"""
        if self._realtime_mode and self._background_thread is not None and self._background_thread.is_alive():
            self._input_queue.put(text)
            self.logger.info(f"输入已提交到队列: {text}")
            return "[已提交到学习队列]"
        else:
            return self.process_input_with_learning(text)
