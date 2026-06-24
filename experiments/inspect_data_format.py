"""检查训练数据保存格式"""
import torch
import os
import gzip
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

print("=" * 70)
print("训练数据保存格式检查")
print("=" * 70)

# 1. 检查检查点文件
print("\n[1] 检查点文件 (.pt / .pt.gz)")
print("-" * 70)

checkpoint_files = [
    'checkpoints/lifelong_with_corpus.pt.gz',
    'checkpoints/lifelong_learning_demo.pt.gz',
    'checkpoints/phase1.pt'
]

for f in checkpoint_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f'\n文件: {f}')
        print(f'大小: {size_mb:.2f} MB')
        
        try:
            if f.endswith('.gz'):
                with gzip.open(f, 'rb') as file:
                    checkpoint = torch.load(file, map_location='cpu', weights_only=False)
            else:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            
            print(f'包含键: {list(checkpoint.keys())}')
            
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    print(f'  {key}: Tensor {value.shape}, dtype={value.dtype}, size={value.numel() * value.element_size() / 1024 / 1024:.2f}MB')
                elif isinstance(value, dict):
                    print(f'  {key}: dict with {len(value)} keys')
                elif isinstance(value, (int, float)):
                    print(f'  {key}: {type(value).__name__} = {value}')
                elif isinstance(value, str):
                    print(f'  {key}: str = "{value[:50]}..."' if len(value) > 50 else f'  {key}: str = "{value}"')
                else:
                    print(f'  {key}: {type(value).__name__}')
        except Exception as e:
            print(f'读取失败: {e}')

# 2. 检查进度文件
print("\n\n[2] 进度文件 (progress.json)")
print("-" * 70)

progress_file = 'outputs/progress.json'
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    print(f'文件: {progress_file}')
    print(f'内容:')
    for key, value in progress.items():
        print(f'  {key}: {value}')
else:
    print(f'{progress_file} 不存在')

# 3. 检查PMI结果
print("\n\n[3] PMI结果 (pmi_results.json)")
print("-" * 70)

pmi_file = 'outputs/pmi_results.json'
if os.path.exists(pmi_file):
    with open(pmi_file, 'r') as f:
        pmi = json.load(f)
    print(f'文件: {pmi_file}')
    print(f'内容: {pmi}')
else:
    print(f'{pmi_file} 不存在')

# 4. 检查日志文件
print("\n\n[4] 日志文件 (.log)")
print("-" * 70)

log_dirs = ['outputs', 'outputs/logs', 'logs']
for log_dir in log_dirs:
    if os.path.exists(log_dir):
        logs = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if logs:
            print(f'\n目录: {log_dir}/')
            for log in sorted(logs, reverse=True)[:5]:
                path = os.path.join(log_dir, log)
                size_kb = os.path.getsize(path) / 1024
                print(f'  {log}: {size_kb:.2f} KB')

# 5. 总结
print("\n\n" + "=" * 70)
print("数据保存格式总结")
print("=" * 70)

print("""
训练数据以以下格式保存：

1. 检查点文件 (.pt / .pt.gz)
   - 格式: PyTorch序列化格式
   - 压缩: gzip压缩（可选）
   - 内容:
     * step: 当前步数
     * field_state: 连续神经场状态 (Tensor)
     * hebbian_weights: Hebbian学习权重矩阵 (Sparse Tensor)
     * field_params: 场参数
     * entropy_history: 熵历史
     * semantic_atoms: 语义原子状态
     * timestamp: 时间戳

2. 进度文件 (progress.json)
   - 格式: JSON
   - 更新频率: 每100步
   - 内容:
     * step: 当前步数
     * entropy: 当前熵值
     * field_norm: 场范数
     * weights: 非零权重数
     * speed: 演化速度
     * memory_mb: 内存使用
     * timestamp: 时间戳

3. PMI结果 (pmi_results.json)
   - 格式: JSON
   - 内容: PMI统计结果

4. 日志文件 (.log)
   - 格式: 纯文本
   - 内容: 结构化日志（时间戳 + 级别 + 消息）

文件大小参考:
- 检查点: 3-140 MB（取决于是否压缩和演化步数）
- 进度: <1 KB
- PMI结果: <1 KB
- 日志: 1-10 MB
""")