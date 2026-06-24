"""终身学习模式演示 - 时时刻刻都在进化"""
import sys, os, time, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("四代龙珠 - 终身学习模式")
print("时时刻刻都在学习、训练、进化")
print("=" * 60)

t0 = time.time()

print('\n[初始化] 创建连续神经场...')
from src.core.liquid_time_constant import LiquidTimeConstantNetwork
from src.core.hebbian_learning import HebbianUpdater
from src.core.curiosity_drive import CuriosityDrive
from src.core.field_interface import FieldInterface
from src.core.field_guardian import FieldGuardian
from src.core.semantic_atoms import SemanticAtomManager
from src.data.knowledge_loader import KnowledgeLoader

field = LiquidTimeConstantNetwork(field_dim=512, hidden_dim=512, device='cpu', use_amp=False)
hebbian = HebbianUpdater(field_dim=512, device='cpu')
curiosity = CuriosityDrive(field_dim=512, device='cpu')
interface = FieldInterface(field_dim=512, atom_dim=32, device='cpu')
kl = KnowledgeLoader()
sa = SemanticAtomManager(field_dim=512, atom_dim=32, initial_atoms=100, device='cpu', knowledge_loader=kl)
print(f'语义原子数: {sa.get_num_atoms()}')

guardian = FieldGuardian(
    field=field,
    hebbian_updater=hebbian,
    curiosity_drive=curiosity,
    semantic_atoms=sa,
    field_interface=interface,
    checkpoint_dir='checkpoints'
)

print(f'初始化完成 ({time.time()-t0:.1f}s)')

print('\n[启动后台演化] 开始持续学习...')
guardian.start_background_evolution(evolve_interval=0.05, auto_checkpoint=False)
time.sleep(1)

print('\n[交互测试] 输入问题，系统在学习中回答...')
questions = [
    "什么是龙",
    "天是什么颜色",
    "春天来了",
    "水往低处流",
    "月亮很圆",
    "太阳从东方升起",
    "花开了",
    "鸟儿在歌唱"
]

for i, q in enumerate(questions):
    print(f'\n--- 第{i+1}轮交互 (后台已演化{guardian.step_count}步) ---')
    a = guardian.process_input_with_learning(q, evolve_steps=50)
    print(f'Q: {q}')
    print(f'A: {a}')
    
    time.sleep(1)

print(f'\n[状态] 总演化步数: {guardian.step_count}')
print(f'[状态] 非零权重数: {hebbian._nnz()}')
print(f'[状态] 输入频率: {guardian.input_frequency:.3f}')

print('\n[持续学习] 让系统自主演化10秒...')
time.sleep(10)
print(f'[状态] 自主演化后步数: {guardian.step_count}')

print('\n[停止后台演化]')
guardian.stop_background_evolution()

print('\n[保存检查点]')
guardian.save_checkpoint('lifelong_learning_demo.pt', compress=True)

total = time.time() - t0
print(f'\n完成! 总用时: {total:.0f}s')
print(f'总演化步数: {guardian.step_count}')
print(f'平均演化速度: {guardian.step_count/total:.1f} 步/秒')