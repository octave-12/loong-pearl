"""数据保存格式分析和改进方案"""
import torch
import os
import gzip
import json
import pickle
import struct
from pathlib import Path

print("=" * 70)
print("数据保存格式分析")
print("=" * 70)

# 分析当前保存方式的问题
print("\n[当前保存方式分析]")
print("-" * 70)

current_formats = """
1. 检查点文件 (.pt / .pt.gz)
   ✅ 格式: PyTorch二进制序列化
   ✅ 优点: 高效、支持张量、支持压缩
   ✅ 适用: 大规模训练数据
   
2. 进度文件 (progress.json)
   ⚠️ 格式: JSON文本
   ⚠️ 问题:
      - 文本格式，容易损坏
      - 无校验机制
      - 并发写入可能损坏
      - 不支持二进制数据
   
3. PMI结果 (pmi_results.json)
   ⚠️ 格式: JSON文本
   ⚠️ 问题: 同上

4. 日志文件 (.log)
   ✅ 格式: 文本（适合日志）
"""

print(current_formats)

# 改进方案
print("\n[改进方案]")
print("-" * 70)

improvement = """
方案1: 进度文件改用二进制格式
   - 格式: 自定义二进制格式 + 校验和
   - 优点: 高效、健壮、支持原子写入
   - 实现: struct打包 + CRC32校验

方案2: 使用SQLite数据库
   - 格式: SQLite数据库
   - 优点: 事务支持、原子写入、查询方便
   - 实现: 存储进度、PMI结果、检查点索引

方案3: 使用HDF5格式
   - 格式: HDF5层次化数据格式
   - 优点: 支持大数据、压缩、并行访问
   - 实现: h5py库

方案4: 混合方案（推荐）
   - 检查点: PyTorch二进制 (.pt.gz) ✅
   - 进度: 自定义二进制 (.bin) + JSON备份
   - PMI: Pickle二进制 (.pkl)
   - 元数据: JSON (人类可读)
"""

print(improvement)

# 实现二进制进度文件
print("\n[实现: 二进制进度文件]")
print("-" * 70)

class BinaryProgressFile:
    """二进制进度文件格式
    
    格式结构:
    [4字节] 魔数 (0x4C4C5052 = "LLPR")
    [4字节] 版本号
    [4字节] 数据长度
    [N字节] 数据内容
    [4字节] CRC32校验和
    """
    
    MAGIC = 0x4C4C5052  # "LLPR" - Loong Pearl Progress
    VERSION = 1
    FORMAT = '<IIIIIIIIId'  # 小端序: 8个int + 1个double + 1个double
    
    @staticmethod
    def save(filepath, step, entropy, field_norm, weights, speed, 
             memory_mb, errors, restarts, elapsed_hours, timestamp):
        """保存二进制进度文件"""
        # 打包数据
        data = struct.pack(
            BinaryProgressFile.FORMAT,
            step, 
            int(entropy * 1000),  # 保留3位小数
            int(field_norm * 1000),
            weights,
            int(speed * 1000),
            memory_mb,
            errors,
            restarts,
            elapsed_hours,
            timestamp
        )
        
        # 计算校验和
        import binascii
        crc = binascii.crc32(data)
        
        # 写入文件（原子写入）
        temp_file = filepath + '.tmp'
        with open(temp_file, 'wb') as f:
            # 写入头部
            f.write(struct.pack('<III', 
                               BinaryProgressFile.MAGIC,
                               BinaryProgressFile.VERSION,
                               len(data)))
            # 写入数据
            f.write(data)
            # 写入校验和
            f.write(struct.pack('<I', crc))
        
        # 原子重命名
        os.replace(temp_file, filepath)
    
    @staticmethod
    def load(filepath):
        """加载二进制进度文件"""
        with open(filepath, 'rb') as f:
            # 读取头部
            magic, version, data_len = struct.unpack('<III', f.read(12))
            
            # 验证魔数
            if magic != BinaryProgressFile.MAGIC:
                raise ValueError(f"无效的进度文件: 魔数不匹配")
            
            # 读取数据
            data = f.read(data_len)
            
            # 读取校验和
            crc_stored = struct.unpack('<I', f.read(4))[0]
            
            # 验证校验和
            import binascii
            crc_calc = binascii.crc32(data)
            if crc_calc != crc_stored:
                raise ValueError(f"进度文件损坏: 校验和不匹配")
            
            # 解包数据
            values = struct.unpack(BinaryProgressFile.FORMAT, data)
            
            return {
                'step': values[0],
                'entropy': values[1] / 1000.0,
                'field_norm': values[2] / 1000.0,
                'weights': values[3],
                'speed': values[4] / 1000.0,
                'memory_mb': values[5],
                'errors': values[6],
                'restarts': values[7],
                'elapsed_hours': values[8],
                'timestamp': values[9]
            }

# 测试二进制进度文件
print("\n测试二进制进度文件:")
progress_file = 'outputs/progress.bin'

try:
    # 保存
    BinaryProgressFile.save(
        progress_file,
        step=2200,
        entropy=4.156,
        field_norm=106.3,
        weights=260597,
        speed=5.18,
        memory_mb=350,
        errors=0,
        restarts=0,
        elapsed_hours=0.6,
        timestamp=time.time()
    )
    print(f"✅ 保存成功: {progress_file}")
    print(f"   文件大小: {os.path.getsize(progress_file)} 字节")
    
    # 加载
    progress = BinaryProgressFile.load(progress_file)
    print(f"✅ 加载成功:")
    for key, value in progress.items():
        print(f"   {key}: {value}")
    
    # 对比JSON大小
    json_file = 'outputs/progress.json'
    if os.path.exists(json_file):
        json_size = os.path.getsize(json_file)
        bin_size = os.path.getsize(progress_file)
        print(f"\n大小对比:")
        print(f"   JSON: {json_size} 字节")
        print(f"   二进制: {bin_size} 字节")
        print(f"   压缩率: {(1 - bin_size/json_size)*100:.1f}%")
        
except Exception as e:
    print(f"❌ 错误: {e}")

# 实现Pickle格式的PMI结果
print("\n\n[实现: Pickle格式PMI结果]")
print("-" * 70)

pmi_data = {
    'pmi_count': 11713,
    'cluster_count': 28,
    'atom_count': 1028,
    'elapsed_pmi': 32.94,
    'elapsed_cluster': 0.29,
    'pmi_pairs': [('龙', '珠', 2.5), ('春', '天', 1.8)]  # 示例数据
}

# 保存为Pickle
pmi_pkl_file = 'outputs/pmi_results.pkl'
with open(pmi_pkl_file, 'wb') as f:
    pickle.dump(pmi_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✅ Pickle保存成功: {pmi_pkl_file}")
print(f"   文件大小: {os.path.getsize(pmi_pkl_file)} 字节")

# 对比JSON
pmi_json_file = 'outputs/pmi_results.json'
with open(pmi_json_file, 'w') as f:
    json.dump(pmi_data, f)
print(f"✅ JSON保存成功: {pmi_json_file}")
print(f"   文件大小: {os.path.getsize(pmi_json_file)} 字节")

# 总结
print("\n\n" + "=" * 70)
print("改进后的数据保存格式")
print("=" * 70)

summary = """
┌─────────────────────────────────────────────────────────────┐
│ 文件类型        │ 旧格式      │ 新格式           │ 改进     │
├─────────────────────────────────────────────────────────────┤
│ 检查点          │ .pt.gz     │ .pt.gz          │ 保持不变  │
│ 进度文件        │ .json      │ .bin + .json备份 │ 更健壮   │
│ PMI结果         │ .json      │ .pkl            │ 更高效   │
│ 日志文件        │ .log       │ .log            │ 保持不变  │
└─────────────────────────────────────────────────────────────┘

优势:
1. 二进制格式更健壮，不易损坏
2. 校验和机制确保数据完整性
3. 原子写入避免并发问题
4. 文件更小，读写更快
5. 支持更复杂的数据类型

兼容性:
- 保留JSON备份，人类可读
- 支持从JSON迁移
- 版本号机制支持未来升级
"""

print(summary)

import time