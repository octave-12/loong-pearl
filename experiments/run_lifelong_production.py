"""四代龙珠 - 产品级终身学习系统
特性：
1. 完善的异常处理和日志系统
2. 信号处理（优雅退出）
3. 自动重启机制
4. 健康检查和监控
5. 配置文件支持
6. 进程守护（可配合screen/supervisor）
"""
import sys, os, time, gc, signal, json, logging
from datetime import datetime
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# ============ 配置管理 ============
class Config:
    def __init__(self, config_file=None):
        self.field_dim = 512
        self.hidden_dim = 512
        self.atom_dim = 32
        self.initial_atoms = 500
        self.device = 'cpu'
        self.use_amp = False
        self.dt = 0.1
        
        self.pmi_batch_size = 5000
        self.pmi_max_pairs = 50000
        self.pmi_window_size = 5
        self.pmi_min_count = 5
        self.pmi_threshold = 1.5
        
        self.checkpoint_interval = 10000
        self.gc_interval = 1000
        self.log_interval = 100
        self.health_check_interval = 500
        self.progress_interval = 100
        
        self.max_restart_attempts = 5
        self.restart_delay = 10
        
        self.webhook_url = None
        self.enable_alerts = False
        
        if config_file and os.path.exists(config_file):
            self.load(config_file)
    
    def load(self, config_file):
        with open(config_file, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)

# ============ 日志系统 ============
class Logger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.log_dir / f"lifelong_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.logger = logging.getLogger('LifelongLearning')
        self.logger.setLevel(logging.INFO)
        
        # 文件handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)
        
        # 控制台handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(ch)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def error(self, msg):
        self.logger.error(msg, exc_info=True)
    
    def warning(self, msg):
        self.logger.warning(msg)

# ============ 健康检查 ============
class HealthMonitor:
    def __init__(self):
        self.metrics = {
            'step': 0,
            'entropy': 0,
            'field_norm': 0,
            'weights': 0,
            'speed': 0,
            'memory_mb': 0,
            'errors': 0,
            'restarts': 0
        }
        self.start_time = time.time()
        self.last_check = time.time()
    
    def update(self, key, value):
        self.metrics[key] = value
    
    def check(self):
        """健康检查：检测异常状态"""
        issues = []
        
        # 检查场范数爆炸
        if self.metrics['field_norm'] > 1000:
            issues.append(f"场范数爆炸: {self.metrics['field_norm']:.1f}")
        
        # 检查熵值异常
        if self.metrics['entropy'] < 0 or self.metrics['entropy'] > 10:
            issues.append(f"熵值异常: {self.metrics['entropy']:.3f}")
        
        # 检查内存使用
        import psutil
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        self.metrics['memory_mb'] = mem
        if mem > 14000:  # 14GB
            issues.append(f"内存过高: {mem:.0f}MB")
        
        # 检查演化停滞
        if time.time() - self.last_check > 60 and self.metrics['speed'] < 0.1:
            issues.append(f"演化停滞: {self.metrics['speed']:.2f}步/秒")
        
        self.last_check = time.time()
        return issues
    
    def save_progress(self, progress_file):
        """保存进度到JSON（快速恢复）"""
        progress = {
            'step': self.metrics['step'],
            'entropy': self.metrics['entropy'],
            'field_norm': self.metrics['field_norm'],
            'weights': self.metrics['weights'],
            'speed': self.metrics['speed'],
            'memory_mb': self.metrics['memory_mb'],
            'elapsed_hours': (time.time() - self.start_time) / 3600,
            'timestamp': datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def report(self):
        elapsed = time.time() - self.start_time
        return {
            **self.metrics,
            'elapsed_hours': elapsed / 3600,
            'steps_per_hour': self.metrics['step'] / (elapsed / 3600) if elapsed > 0 else 0
        }

# ============ 检查点管理 ============
class CheckpointManager:
    def __init__(self, checkpoint_dir, max_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
    
    def save(self, step, field, hebbian, config, metrics):
        checkpoint = {
            'step': step,
            'config': config.__dict__,
            'metrics': metrics,
            'field_state': field.get_state().cpu(),
            'hebbian_weights': hebbian.get_weight_matrix().cpu(),
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.checkpoint_dir / f'lifelong_{step}.pt'
        torch.save(checkpoint, path)
        
        # 清理旧检查点
        self._cleanup()
        return path
    
    def load_latest(self):
        checkpoints = sorted(self.checkpoint_dir.glob('lifelong_*.pt'))
        if checkpoints:
            return torch.load(checkpoints[-1], map_location='cpu')
        return None
    
    def _cleanup(self):
        checkpoints = sorted(self.checkpoint_dir.glob('lifelong_*.pt'))
        while len(checkpoints) > self.max_checkpoints:
            checkpoints[0].unlink()
            checkpoints = checkpoints[1:]

# ============ PMI计算 ============
def compute_pmi_incremental(corpus_path, config, logger):
    logger.info(f'开始PMI计算: {corpus_path}')
    
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
            
            if len(batch) >= config.pmi_batch_size:
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
                        for j in range(i + 1, min(i + config.pmi_window_size, len(chars))):
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
                
                if len(global_pair_counts) > config.pmi_max_pairs:
                    sorted_pairs = sorted(global_pair_counts.items(), key=lambda x: x[1], reverse=True)
                    global_pair_counts = defaultdict(int, sorted_pairs[:config.pmi_max_pairs])
                    del sorted_pairs
                    gc.collect()
                
                batch = []
                
                if batch_num % 100 == 0:
                    elapsed = time.time() - t_start
                    speed = total_read / elapsed if elapsed > 0 else 0
                    logger.info(f'  Batch {batch_num}: {total_read:,} lines, {len(global_pair_counts)} pairs, {elapsed:.1f}s ({speed:.0f} lines/s)')
    
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
                for j in range(i + 1, min(i + config.pmi_window_size, len(chars))):
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
    
    logger.info('计算PMI值...')
    pmi_pairs = []
    for (char_a, char_b), count in global_pair_counts.items():
        if count < config.pmi_min_count:
            continue
        p_a = global_char_counts[char_a] / global_total_chars
        p_b = global_char_counts[char_b] / global_total_chars
        p_ab = count / global_total_pairs
        pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)
        if pmi > config.pmi_threshold:
            pmi_pairs.append((char_a, char_b, float(pmi)))
    
    del global_char_counts, global_pair_counts
    gc.collect()
    
    logger.info(f'PMI计算完成: {len(pmi_pairs)} pairs')
    return pmi_pairs

# ============ 主系统 ============
class LifelongLearningSystem:
    def __init__(self, config_file=None):
        self.config = Config(config_file)
        self.logger = Logger(PROJECT_ROOT / 'outputs' / 'logs')
        self.monitor = HealthMonitor()
        self.checkpoint_mgr = CheckpointManager(PROJECT_ROOT / 'checkpoints')
        
        self.running = True
        self.step = 0
        self.restart_count = 0
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.logger.info(f'收到信号 {signum}，准备优雅退出...')
        self.running = False
    
    def initialize(self):
        """初始化系统"""
        self.logger.info('=' * 70)
        self.logger.info('四代龙珠 - 产品级终身学习系统')
        self.logger.info('=' * 70)
        
        # 检查是否恢复检查点
        checkpoint = self.checkpoint_mgr.load_latest()
        if checkpoint:
            self.logger.info(f'从检查点恢复: step={checkpoint["step"]}')
            self._restore_checkpoint(checkpoint)
        else:
            self._initialize_fresh()
    
    def _initialize_fresh(self):
        """全新初始化"""
        self.logger.info('[Phase 1] 初始化连续神经场...')
        
        from src.core.liquid_time_constant import LiquidTimeConstantNetwork
        from src.core.hebbian_learning import HebbianUpdater
        from src.core.curiosity_drive import CuriosityDrive
        from src.core.field_interface import FieldInterface
        from src.core.semantic_atoms import SemanticAtomManager
        from src.data.knowledge_loader import KnowledgeLoader
        
        self.field = LiquidTimeConstantNetwork(
            field_dim=self.config.field_dim,
            hidden_dim=self.config.hidden_dim,
            device=self.config.device,
            use_amp=self.config.use_amp
        )
        self.hebbian = HebbianUpdater(field_dim=self.config.field_dim, device=self.config.device)
        self.curiosity = CuriosityDrive(field_dim=self.config.field_dim, device=self.config.device)
        self.interface = FieldInterface(
            field_dim=self.config.field_dim,
            atom_dim=self.config.atom_dim,
            device=self.config.device
        )
        kl = KnowledgeLoader()
        self.sa = SemanticAtomManager(
            field_dim=self.config.field_dim,
            atom_dim=self.config.atom_dim,
            initial_atoms=self.config.initial_atoms,
            device=self.config.device,
            knowledge_loader=kl
        )
        
        self.logger.info('[Phase 2] PMI计算...')
        pmi_pairs = compute_pmi_incremental(
            PROJECT_ROOT / 'data' / 'corpus.txt',
            self.config,
            self.logger
        )
        
        self.logger.info('[Phase 3] 聚类...')
        clusters = self.sa.cluster_characters(pmi_pairs, use_igraph=False)
        self.sa.initialize_atoms_from_clusters(
            clusters,
            field_dim=self.config.field_dim,
            max_idiom_atoms=self.config.initial_atoms
        )
        self.logger.info(f'Semantic atoms: {self.sa.get_num_atoms()}')
        
        del pmi_pairs, clusters
        gc.collect()
    
    def _restore_checkpoint(self, checkpoint):
        """从检查点恢复"""
        from src.core.liquid_time_constant import LiquidTimeConstantNetwork
        from src.core.hebbian_learning import HebbianUpdater
        
        self.step = checkpoint['step']
        self.field = LiquidTimeConstantNetwork(
            field_dim=self.config.field_dim,
            hidden_dim=self.config.hidden_dim,
            device=self.config.device,
            use_amp=self.config.use_amp
        )
        self.field.set_state(checkpoint['field_state'].to(self.config.device))
        
        self.hebbian = HebbianUpdater(field_dim=self.config.field_dim, device=self.config.device)
        # 恢复Hebbian权重需要特殊处理
        # ...
    
    def run(self):
        """主演化循环（带自动重启）"""
        while self.restart_count < self.config.max_restart_attempts:
            try:
                self._evolution_loop()
                break  # 正常退出
            except Exception as e:
                self.restart_count += 1
                self.monitor.metrics['restarts'] = self.restart_count
                self.logger.error(f'演化异常 (重启 {self.restart_count}/{self.config.max_restart_attempts}): {e}')
                
                if self.restart_count < self.config.max_restart_attempts:
                    # 保存检查点
                    self._save_checkpoint()
                    
                    # 等待后重启
                    self.logger.info(f'等待 {self.config.restart_delay}s 后重启...')
                    time.sleep(self.config.restart_delay)
                else:
                    self.logger.error('达到最大重启次数，退出')
                    raise
    
    def _evolution_loop(self):
        """演化循环"""
        self.logger.info('[Phase 4] 开始永久演化循环...')
        t0 = time.time()
        
        while self.running:
            try:
                # 演化一步
                h = self.field.get_state()
                entropy, exploration_atoms, noise, entropy_state = self.curiosity.detect_entropy_anomaly(
                    h, self.sa.get_all_regions()
                )
                
                if noise is not None:
                    h = h + noise
                    self.field.set_state(h)
                
                h = self.field.evolve(dt=self.config.dt)
                self.hebbian.update(h)
                self.step += 1
                
                # 更新监控指标
                self.monitor.update('step', self.step)
                self.monitor.update('entropy', entropy)
                self.monitor.update('field_norm', h.norm().item())
                self.monitor.update('weights', self.hebbian._nnz())
                
                elapsed = time.time() - t0
                speed = self.step / elapsed if elapsed > 0 else 0
                self.monitor.update('speed', speed)
                
                # 日志输出
                if self.step % self.config.log_interval == 0:
                    self.logger.info(
                        f'步数: {self.step:,}, 熵: {entropy:.3f}, 范数: {h.norm().item():.1f}, '
                        f'权重: {self.hebbian._nnz():,}, 速度: {speed:.1f}步/秒'
                    )
                
                # 内存回收
                if self.step % self.config.gc_interval == 0:
                    gc.collect()
                
                # 健康检查
                if self.step % self.config.health_check_interval == 0:
                    issues = self.monitor.check()
                    if issues:
                        self.logger.warning(f'健康检查发现问题: {issues}')
                        # 发送告警
                        if self.config.enable_alerts:
                            self.monitor.alert(f'四代龙珠健康检查告警: {issues}', self.logger)
                
                # 保存进度（快速恢复）
                if self.step % self.config.progress_interval == 0:
                    progress_file = Path(self.config.checkpoint_dir if hasattr(self.config, 'checkpoint_dir') else 'outputs') / 'progress.json'
                    self.monitor.save_progress(progress_file)
                
                # 保存检查点
                if self.step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                    
            except RuntimeError as e:
                # PyTorch运行时错误（如CUDA OOM）
                self.logger.error(f'运行时错误: {e}')
                self.monitor.metrics['errors'] += 1
                gc.collect()
                if 'out of memory' in str(e).lower():
                    # 内存不足，降低batch size或清理缓存
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                else:
                    raise
        
        # 最终统计
        report = self.monitor.report()
        self.logger.info('=' * 70)
        self.logger.info('演化结束')
        self.logger.info(f'总步数: {report["step"]:,}')
        self.logger.info(f'总时长: {report["elapsed_hours"]:.2f} 小时')
        self.logger.info(f'平均速度: {report["steps_per_hour"]:.0f} 步/小时')
        self.logger.info(f'重启次数: {report["restarts"]}')
        self.logger.info(f'错误次数: {report["errors"]}')
        self.logger.info('=' * 70)
    
    def _save_checkpoint(self):
        """保存检查点"""
        try:
            path = self.checkpoint_mgr.save(
                self.step,
                self.field,
                self.hebbian,
                self.config,
                self.monitor.report()
            )
            self.logger.info(f'检查点已保存: {path}')
        except Exception as e:
            self.logger.error(f'保存检查点失败: {e}')

# ============ 入口 ============
if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    system = LifelongLearningSystem(config_file)
    system.initialize()
    system.run()