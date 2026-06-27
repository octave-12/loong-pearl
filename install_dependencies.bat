@echo off
echo ========================================================================
echo 四代龙珠 - Python 3.12 依赖安装
echo ========================================================================
echo.

echo [1] 安装核心依赖...
py -3.12 -m pip install torch numpy scipy scikit-learn tqdm jieba python-louvain pyyaml networkx

echo.
echo [2] 安装可选依赖...
py -3.12 -m pip install python-igraph faiss-cpu

echo.
echo [3] 验证安装...
py -3.12 -c "import torch; print('PyTorch:', torch.__version__)"
py -3.12 -c "import numpy; print('NumPy:', numpy.__version__)"
py -3.12 -c "import jieba; print('Jieba: OK')"

echo.
echo ========================================================================
echo 安装完成！
echo ========================================================================
pause