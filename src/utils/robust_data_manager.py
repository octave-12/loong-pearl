"""健壮的二进制数据管理模块
解决JSON格式易损坏、无校验的问题
"""
import os
import struct
import pickle
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch

class RobustDataManager:
    """健壮的数据管理器
    
    特性:
    1. 二进制格式存储
    2. CRC32校验和验证
    3. 原子写入（避免损坏）
    4. 自动备份
    5. 版本控制
    """
    
    # 文件格式常量
    PROGRESS_MAGIC = 0x4C4C5052  # "LLPR" - Loong Pearl Progress
    PMI_MAGIC = 0x4C4C504D      # "LLPM" - Loong Pearl PMI
    VERSION = 1
    
    def __init__(self, data_dir: str = 'outputs'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份目录
        self.backup_dir = self.data_dir / 'backups'
        self.backup_dir.mkdir(exist_ok=True)
    
    # ==================== 进度文件 ====================
    
    def save_progress(self, 
                     step: int,
                     entropy: float,
                     field_norm: float,
                     weights: int,
                     speed: float,
                     memory_mb: float = 0,
                     errors: int = 0,
                     restarts: int = 0,
                     elapsed_hours: float = 0,
                     timestamp: float = None) -> Path:
        """保存进度（二进制格式 + JSON备份）
        
        二进制格式结构:
        [4字节] 魔数 (0x4C4C5052)
        [4字节] 版本号
        [4字节] 数据长度
        [N字节] 数据内容
        [4字节] CRC32校验和
        """
        if timestamp is None:
            import time
            timestamp = time.time()
        
        # 打包数据（使用定精度避免浮点误差）
        data = struct.pack(
            '<QIIIIIIId',  # 小端序
            step,                    # 8字节 unsigned long long
            int(entropy * 10000),    # 4字节 (保留4位小数)
            int(field_norm * 10000), # 4字节
            weights,                 # 4字节 unsigned int
            int(speed * 10000),      # 4字节
            int(memory_mb),          # 4字节
            errors,                  # 4字节
            restarts,                # 4字节
            elapsed_hours            # 8字节 double
        )
        
        # 计算CRC32校验和
        import binascii
        crc = binascii.crc32(data)
        
        # 构建完整数据
        header = struct.pack('<III', 
                            self.PROGRESS_MAGIC,
                            self.VERSION,
                            len(data))
        footer = struct.pack('<I', crc)
        full_data = header + data + footer
        
        # 原子写入
        progress_file = self.data_dir / 'progress.bin'
        temp_file = progress_file.with_suffix('.tmp')
        
        with open(temp_file, 'wb') as f:
            f.write(full_data)
        
        os.replace(temp_file, progress_file)
        
        # JSON备份（人类可读）
        json_backup = self.data_dir / 'progress.json'
        progress_dict = {
            'step': step,
            'entropy': entropy,
            'field_norm': field_norm,
            'weights': weights,
            'speed': speed,
            'memory_mb': memory_mb,
            'errors': errors,
            'restarts': restarts,
            'elapsed_hours': elapsed_hours,
            'timestamp': timestamp
        }
        with open(json_backup, 'w') as f:
            json.dump(progress_dict, f, indent=2)
        
        return progress_file
    
    def load_progress(self) -> Optional[Dict[str, Any]]:
        """加载进度（自动检测格式）"""
        progress_file = self.data_dir / 'progress.bin'
        
        if not progress_file.exists():
            # 尝试从JSON加载
            json_file = self.data_dir / 'progress.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    return json.load(f)
            return None
        
        try:
            with open(progress_file, 'rb') as f:
                # 读取头部
                header = f.read(12)
                if len(header) < 12:
                    raise ValueError("文件损坏: 头部不完整")
                
                magic, version, data_len = struct.unpack('<III', header)
                
                # 验证魔数
                if magic != self.PROGRESS_MAGIC:
                    raise ValueError(f"无效文件: 魔数不匹配 (0x{magic:08X})")
                
                # 验证版本
                if version > self.VERSION:
                    raise ValueError(f"版本不兼容: 文件版本{version} > 当前版本{self.VERSION}")
                
                # 读取数据
                data = f.read(data_len)
                if len(data) < data_len:
                    raise ValueError("文件损坏: 数据不完整")
                
                # 读取校验和
                footer = f.read(4)
                if len(footer) < 4:
                    raise ValueError("文件损坏: 校验和不完整")
                
                crc_stored = struct.unpack('<I', footer)[0]
                
                # 验证校验和
                import binascii
                crc_calc = binascii.crc32(data)
                if crc_calc != crc_stored:
                    raise ValueError(f"文件损坏: 校验和不匹配 (存储=0x{crc_stored:08X}, 计算=0x{crc_calc:08X})")
                
                # 解包数据
                values = struct.unpack('<QIIIIIIId', data)
                
                return {
                    'step': values[0],
                    'entropy': values[1] / 10000.0,
                    'field_norm': values[2] / 10000.0,
                    'weights': values[3],
                    'speed': values[4] / 10000.0,
                    'memory_mb': values[5],
                    'errors': values[6],
                    'restarts': values[7],
                    'elapsed_hours': values[8]
                }
                
        except Exception as e:
            print(f"二进制进度文件损坏: {e}")
            print("尝试从JSON备份恢复...")
            
            json_file = self.data_dir / 'progress.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    return json.load(f)
            
            return None
    
    # ==================== PMI结果 ====================
    
    def save_pmi(self, pmi_data: Dict[str, Any]) -> Path:
        """保存PMI结果（Pickle二进制格式）"""
        pmi_file = self.data_dir / 'pmi_results.pkl'
        temp_file = pmi_file.with_suffix('.tmp')
        
        # 添加校验信息
        pmi_data['_version'] = self.VERSION
        pmi_data['_magic'] = self.PMI_MAGIC
        
        # Pickle序列化
        with open(temp_file, 'wb') as f:
            pickle.dump(pmi_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.replace(temp_file, pmi_file)
        
        # JSON备份
        json_backup = self.data_dir / 'pmi_results.json'
        json_data = {k: v for k, v in pmi_data.items() if not k.startswith('_')}
        with open(json_backup, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return pmi_file
    
    def load_pmi(self) -> Optional[Dict[str, Any]]:
        """加载PMI结果"""
        pmi_file = self.data_dir / 'pmi_results.pkl'
        
        if not pmi_file.exists():
            # 尝试从JSON加载
            json_file = self.data_dir / 'pmi_results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    return json.load(f)
            return None
        
        try:
            with open(pmi_file, 'rb') as f:
                data = pickle.load(f)
            
            # 验证魔数
            if data.get('_magic') != self.PMI_MAGIC:
                raise ValueError("无效PMI文件: 魔数不匹配")
            
            # 移除元数据
            return {k: v for k, v in data.items() if not k.startswith('_')}
            
        except Exception as e:
            print(f"PMI文件损坏: {e}")
            
            json_file = self.data_dir / 'pmi_results.json'
            if json_file.exists():
                with open(json_file, 'r') as f:
                    return json.load(f)
            
            return None
    
    # ==================== 检查点 ====================
    
    def save_checkpoint(self, 
                       step: int,
                       field_state: torch.Tensor,
                       hebbian_weights: torch.Tensor,
                       extra_data: Dict[str, Any] = None,
                       compress: bool = True) -> Path:
        """保存检查点（PyTorch二进制格式）"""
        checkpoint = {
            'step': step,
            'field_state': field_state.cpu(),
            'hebbian_weights': hebbian_weights.cpu(),
            '_version': self.VERSION,
            '_magic': 0x4C4C5043  # "LLPC" - Loong Pearl Checkpoint
        }
        
        if extra_data:
            checkpoint.update(extra_data)
        
        checkpoint_file = self.data_dir / 'checkpoints' / f'checkpoint_{step}.pt'
        checkpoint_file.parent.mkdir(exist_ok=True)
        
        temp_file = checkpoint_file.with_suffix('.tmp')
        
        if compress:
            with gzip.open(temp_file, 'wb', compresslevel=6) as f:
                torch.save(checkpoint, f, _use_new_zipfile_serialization=True)
            checkpoint_file = checkpoint_file.with_suffix('.pt.gz')
        else:
            torch.save(checkpoint, temp_file, _use_new_zipfile_serialization=True)
        
        os.replace(temp_file, checkpoint_file)
        
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_path: Path = None) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        if checkpoint_path is None:
            # 查找最新检查点
            checkpoint_dir = self.data_dir / 'checkpoints'
            if not checkpoint_dir.exists():
                return None
            
            checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt*'))
            if not checkpoints:
                return None
            
            checkpoint_path = checkpoints[-1]
        
        try:
            if checkpoint_path.suffix == '.gz':
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # 验证魔数
            if checkpoint.get('_magic') != 0x4C4C5043:
                raise ValueError("无效检查点: 魔数不匹配")
            
            return checkpoint
            
        except Exception as e:
            print(f"检查点加载失败: {e}")
            return None
    
    # ==================== 备份管理 ====================
    
    def create_backup(self, reason: str = 'manual') -> Path:
        """创建完整备份"""
        import time
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_name = f'backup_{timestamp}_{reason}'
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir()
        
        # 复制所有文件
        for file in self.data_dir.glob('*'):
            if file.is_file():
                import shutil
                shutil.copy2(file, backup_path / file.name)
        
        return backup_path
    
    def list_backups(self) -> list:
        """列出所有备份"""
        return sorted(self.backup_dir.glob('backup_*'))
    
    def restore_backup(self, backup_path: Path):
        """从备份恢复"""
        if not backup_path.exists():
            raise ValueError(f"备份不存在: {backup_path}")
        
        for file in backup_path.glob('*'):
            if file.is_file():
                import shutil
                shutil.copy2(file, self.data_dir / file.name)

# 使用示例
if __name__ == '__main__':
    import time
    
    print("=" * 70)
    print("健壮数据管理器测试")
    print("=" * 70)
    
    dm = RobustDataManager('outputs')
    
    # 测试进度保存
    print("\n[测试进度保存]")
    progress_file = dm.save_progress(
        step=5000,
        entropy=4.156,
        field_norm=120.5,
        weights=300000,
        speed=5.5,
        memory_mb=450,
        errors=0,
        restarts=0,
        elapsed_hours=1.5
    )
    print(f"✅ 进度已保存: {progress_file}")
    print(f"   文件大小: {progress_file.stat().st_size} 字节")
    
    # 测试进度加载
    print("\n[测试进度加载]")
    progress = dm.load_progress()
    if progress:
        print("✅ 进度加载成功:")
        for k, v in progress.items():
            print(f"   {k}: {v}")
    
    # 测试PMI保存
    print("\n[测试PMI保存]")
    pmi_data = {
        'pmi_count': 13729,
        'cluster_count': 82,
        'atom_count': 500,
        'elapsed_pmi': 486.1,
        'elapsed_cluster': 0.5
    }
    pmi_file = dm.save_pmi(pmi_data)
    print(f"✅ PMI已保存: {pmi_file}")
    print(f"   文件大小: {pmi_file.stat().st_size} 字节")
    
    # 测试PMI加载
    print("\n[测试PMI加载]")
    pmi = dm.load_pmi()
    if pmi:
        print("✅ PMI加载成功:")
        for k, v in pmi.items():
            print(f"   {k}: {v}")
    
    # 文件大小对比
    print("\n[文件大小对比]")
    files = [
        ('progress.bin', '二进制进度'),
        ('progress.json', 'JSON进度'),
        ('pmi_results.pkl', 'Pickle PMI'),
        ('pmi_results.json', 'JSON PMI')
    ]
    
    for filename, desc in files:
        filepath = dm.data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"   {desc:15s}: {size:5d} 字节")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)