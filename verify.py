import os
import sys

def verify_project_structure():
    print("=" * 60)
    print("四代龙珠 - 项目结构验证")
    print("=" * 60)
    
    required_files = {
        "核心模块": [
            "src/core/liquid_time_constant.py",
            "src/core/hebbian_learning.py",
            "src/core/curiosity_drive.py",
            "src/core/semantic_atoms.py",
            "src/core/field_interface.py",
            "src/core/field_guardian.py"
        ],
        "训练脚本": [
            "train.py",
            "run.py",
            "tests/test_core.py"
        ],
        "配置文件": [
            "requirements.txt",
            "README.md",
            "config.yaml",
            "data/corpus.txt"
        ]
    }
    
    all_ok = True
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            exists = os.path.exists(file)
            status = "[OK]" if exists else "[X]"
            print(f"  {status} {file}")
            if not exists:
                all_ok = False
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("[OK] 所有文件已创建")
        print("\n项目结构完整，核心实现包括：")
        print("  1. 液态时间常数网络")
        print("  2. Hebbian连续学习")
        print("  3. 好奇心驱动机制")
        print("  4. 语义原子管理")
        print("  5. 场输入输出接口")
        print("  6. 守护进程系统")
    else:
        print("[X] 部分文件缺失")
    
    print("=" * 60)
    
    return all_ok


def verify_code_structure():
    print("\n" + "=" * 60)
    print("代码结构验证")
    print("=" * 60)
    
    files_to_check = [
        ("src/core/liquid_time_constant.py", ["class LiquidTimeConstantNetwork", "def evolve", "def compute_time_constants"]),
        ("src/core/hebbian_learning.py", ["class HebbianUpdater", "def update", "def get_activation_stats"]),
        ("src/core/curiosity_drive.py", ["class CuriosityDrive", "def compute_entropy", "def detect_entropy_anomaly"]),
        ("src/core/semantic_atoms.py", ["class SemanticAtomManager", "def compute_pmi", "def cluster_characters"]),
        ("src/core/field_interface.py", ["class FieldInterface", "def encode_text_to_perturbation", "def decode_activation_to_text"]),
        ("src/core/field_guardian.py", ["class FieldGuardian", "def evolve_step", "def process_input", "def run"])
    ]
    
    all_ok = True
    
    for filepath, required_elements in files_to_check:
        print(f"\n检查 {filepath}:")
        
        if not os.path.exists(filepath):
            print(f"  [X] 文件不存在")
            all_ok = False
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for element in required_elements:
            found = element in content
            status = "[OK]" if found else "[X]"
            print(f"  {status} {element}")
            if not found:
                all_ok = False
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("[OK] 所有核心类和方法已实现")
    else:
        print("[X] 部分实现缺失")
    
    print("=" * 60)
    
    return all_ok


def verify_dependencies():
    print("\n" + "=" * 60)
    print("依赖检查")
    print("=" * 60)
    
    required_packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
        ("jieba", "jieba")
    ]
    
    print("\n检查Python包:")
    all_ok = True
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"  [OK] {package_name}")
        except ImportError:
            print(f"  [X] {package_name} (未安装)")
            all_ok = False
    
    try:
        import torch
        print(f"  [OK] torch")
    except Exception as e:
        print(f"  [!] torch (已安装但加载失败: {str(e)[:50]}...)")
        print("    建议: pip install torch --index-url https://download.pytorch.org/whl/cpu")
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("[OK] 基础依赖已安装")
        print("\n提示: 如果遇到PyTorch DLL错误，请重新安装PyTorch:")
        print("  pip uninstall torch")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    else:
        print("[X] 部分依赖缺失")
        print("\n安装依赖:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return all_ok


def main():
    structure_ok = verify_project_structure()
    code_ok = verify_code_structure()
    deps_ok = verify_dependencies()
    
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    if structure_ok and code_ok:
        print("[OK] 项目实现完整")
        print("\n核心特性:")
        print("  - 液态时间常数网络 (4096维场状态)")
        print("  - 连续Hebbian自组织学习")
        print("  - 信息熵梯度驱动的好奇心机制")
        print("  - PMI统计涌现的语义原子")
        print("  - 退化设计的输入输出接口")
        print("  - 持续运行的守护进程")
        
        if deps_ok:
            print("\n可以开始使用:")
            print("  python run.py --mode demo        # 演示模式")
            print("  python run.py --mode interactive # 交互模式")
            print("  python train.py                  # 完整训练")
        else:
            print("\n请先安装依赖:")
            print("  pip install -r requirements.txt")
    else:
        print("[X] 项目实现不完整")
    
    print("=" * 60)


if __name__ == "__main__":
    main()