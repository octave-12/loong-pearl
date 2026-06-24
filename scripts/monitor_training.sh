#!/bin/bash
# 训练监控脚本

cd /mnt/d/soso/projects/Loong-pearl

PID_FILE="outputs/training.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "训练进程运行中 (PID: $PID)"
        echo ""
        echo "最新日志:"
        tail -20 outputs/knowledge_training_*.log 2>/dev/null | tail -20
        echo ""
        echo "内存使用:"
        ps -p $PID -o pid,vsz,rss,comm 2>/dev/null
    else
        echo "训练进程已停止"
    fi
else
    echo "未找到训练进程"
fi

echo ""
echo "日志文件:"
ls -lh outputs/knowledge_training_*.log 2>/dev/null | tail -5