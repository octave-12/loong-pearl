#!/usr/bin/env python3
"""产品级终身学习启动脚本"""
import sys, os, time, gc, signal, json, logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# 简化版Logger（避免bug）
class SimpleLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 配置
config = {
    'field_dim': 512,
    'hidden_dim': 512,
    'atom_dim': 32,
    'initial_atoms': 500,
    'device': 'cpu',
    'dt': 0.1,
    'checkpoint_interval': 10000,
    'gc_interval': 1000,
    'log_interval': 100,
    'progress_interval': 100,
    'max_restart_attempts': 5
}

# 日志
log_dir = PROJECT_ROOT / 'outputs' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = SimpleLogger(log_file)

print('=' * 70, flush=True)
print('四代龙珠 - 产品级终身学习系统', flush=True)
print(f'日志文件: {log_file}', flush=True)
print('=' * 70, flush=True)

# 信号处理
running = True
def signal_handler(signum, frame):
    global running
    print(f'\n收到信号 {signum}，准备退出...', flush=True)
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# PMI计算
def compute_pmi(corpus_path):
    print(f'\n[PMI] 计算中...', flush=True)
    from collections import defaultdict
    
    char_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    total_chars = 0
    total_pairs = 0
    
    batch = []
    total_read = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            batch.append(line)
            total_read += 1
            
            if len(batch) >= 5000:
                for text in batch:
                    chars = list(text)
                    total_chars += len(chars)
                    for char in chars:
                        char_counts[char] += 1
                    for i in range(len(chars)):
                        for j in range(i + 1, min(i + 5, len(chars))):
                            pair = (chars[i], chars[j])
                            pair_counts[pair] += 1
                            total_pairs += 1
                
                if len(pair_counts) > 50000:
                    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:50000]
                    pair_counts = defaultdict(int, sorted_pairs)
                
                batch = []
                
                if total_read % 100000 == 0:
                    print(f'  已处理 {total_read:,} 行', flush=True)
    
    print(f'  计算PMI值...', flush=True)
    pmi_pairs = []
    for (char_a, char_b), count in pair_counts.items():
        if count < 5:
            continue
        p_a = char_counts[char_a] / total_chars
        p_b = char_counts[char_b] / total_chars
        p_ab = count / total_pairs
        pmi = np.log(p_ab / (p_a * p_b + 1e-10) + 1e-10)
        if pmi > 1.5:
            pmi_pairs.append((char_a, char_b, float(pmi)))
    
    print(f'  PMI pairs: {len(pmi_pairs)}', flush=True)
    return pmi_pairs

# 主流程
try:
    t0 = time.time()
    
    print('\n[Phase 1] 初始化...', flush=True)
    from src.core.liquid_time_constant import LiquidTimeConstantNetwork
    from src.core.hebbian_learning import HebbianUpdater
    from src.core.curiosity_drive import CuriosityDrive
    from src.core.field_interface import FieldInterface
    from src.core.semantic_atoms import SemanticAtomManager
    from src.data.unified_knowledge_manager import UnifiedKnowledgeManager
    
    field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu', use_amp=False)
    hebbian = HebbianUpdater(field_dim=512, device='cpu')
    curiosity = CuriosityDrive(field_dim=512, device='cpu')
    interface = FieldInterface(field_dim=512, atom_dim=32, device='cpu')
    km = UnifiedKnowledgeManager()
    
    print('\n[知识库统计]', flush=True)
    stats = km.get_knowledge_stats()
    for name, count in stats.items():
        print(f'  {name}: {count:,}', flush=True)
    
    print('\n[Phase 2] PMI计算...', flush=True)
    pmi_pairs = compute_pmi(PROJECT_ROOT / 'data' / 'corpus.txt')
    
    print('\n[Phase 3] 注入知识增强PMI...', flush=True)
    enhanced_pmi = km.get_enhanced_pmi_pairs(
        pmi_pairs, 
        inject_concepts=True, 
        inject_wiki=True,
        min_score=0.3,
        concept_limit=500000,
        wiki_limit=10000
    )
    print(f'  原始PMI: {len(pmi_pairs)}, 增强后: {len(enhanced_pmi)}', flush=True)
    
    print('\n[Phase 4] 聚类...', flush=True)
    sa = SemanticAtomManager(field_dim=512, atom_dim=32, initial_atoms=500, device='cpu')
    clusters = sa.cluster_characters(enhanced_pmi, use_igraph=False)
    sa.initialize_atoms_from_clusters(clusters, field_dim=512, max_idiom_atoms=2000)
    print(f'Semantic atoms: {sa.get_num_atoms()}', flush=True)
    
    del pmi_pairs, enhanced_pmi, clusters
    gc.collect()
    
    print('\n[Phase 5] 永久演化循环...', flush=True)
    step = 0
    errors = 0
    restarts = 0
    
    while running:
        try:
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
                mem_mb = 0
                try:
                    import psutil
                    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                except:
                    pass
                
                print(f'步数: {step:,}, 熵: {entropy:.3f}, 范数: {h.norm().item():.1f}, '
                      f'权重: {hebbian._nnz():,}, 速度: {speed:.1f}步/秒, 内存: {mem_mb:.0f}MB', flush=True)
            
            if step % 1000 == 0:
                gc.collect()
            
            if step % 100 == 0:
                progress_file = PROJECT_ROOT / 'outputs' / 'progress.json'
                with open(progress_file, 'w') as f:
                    json.dump({
                        'step': step,
                        'entropy': entropy,
                        'field_norm': h.norm().item(),
                        'weights': hebbian._nnz(),
                        'speed': speed,
                        'memory_mb': mem_mb,
                        'errors': errors,
                        'restarts': restarts,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
            
            if step % 10000 == 0:
                checkpoint_dir = PROJECT_ROOT / 'checkpoints'
                checkpoint_dir.mkdir(exist_ok=True)
                checkpoint_file = checkpoint_dir / f'lifelong_{step}.pt'
                torch.save({
                    'step': step,
                    'field_state': field.get_state().cpu(),
                    'hebbian_weights': hebbian.get_weight_matrix().cpu()
                }, checkpoint_file)
                print(f'  检查点: {checkpoint_file}', flush=True)
                
        except RuntimeError as e:
            errors += 1
            print(f'[错误] {e}', flush=True)
            gc.collect()
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                raise
        except Exception as e:
            errors += 1
            print(f'[异常] {type(e).__name__}: {e}', flush=True)
            if errors > 10:
                print('错误次数过多，退出', flush=True)
                raise
    
    # 最终统计
    elapsed = time.time() - t0
    print('\n' + '=' * 70, flush=True)
    print('演化结束', flush=True)
    print(f'总步数: {step:,}', flush=True)
    print(f'总时长: {elapsed/3600:.2f} 小时', flush=True)
    print(f'平均速度: {step/(elapsed/3600):.0f} 步/小时', flush=True)
    print(f'错误次数: {errors}', flush=True)
    print('=' * 70, flush=True)
    
except Exception as e:
    print(f'\n[致命错误] {type(e).__name__}: {e}', flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)