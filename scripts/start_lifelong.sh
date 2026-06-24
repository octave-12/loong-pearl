#!/bin/bash
# 四代龙珠终身学习 - 守护启动脚本
# 用法: ./start_lifelong.sh [start|stop|status|restart]

PROJECT_ROOT="/mnt/d/soso/projects/Loong-pearl"
SCRIPT="$PROJECT_ROOT/experiments/run_lifelong_production.py"
CONFIG="$PROJECT_ROOT/config/lifelong.json"
LOG_DIR="$PROJECT_ROOT/outputs"
PID_FILE="$LOG_DIR/lifelong.pid"
SCREEN_NAME="lifelong-learning"

cd $PROJECT_ROOT

case "$1" in
    start)
        if [ -f "$PID_FILE" ] && kill -0 $(cat $PID_FILE) 2>/dev/null; then
            echo "进程已在运行 (PID: $(cat $PID_FILE))"
            exit 1
        fi
        
        echo "启动终身学习系统..."
        
        # 方案A: 使用screen（推荐）
        screen -dmS $SCREEN_NAME bash -c "
            python3 -u $SCRIPT $CONFIG 2>&1 | tee -a $LOG_DIR/lifelong_console.log
        "
        
        # 方案B: 直接nohup（备选）
        # nohup python3 -u $SCRIPT $CONFIG > $LOG_DIR/lifelong_console.log 2>&1 &
        
        sleep 2
        screen -list | grep $SCREEN_NAME
        echo "启动完成！使用 'screen -r $SCREEN_NAME' 查看实时输出"
        ;;
    
    stop)
        echo "停止终身学习系统..."
        screen -S $SCREEN_NAME -X quit 2>/dev/null
        
        if [ -f "$PID_FILE" ]; then
            kill -TERM $(cat $PID_FILE) 2>/dev/null
            rm -f $PID_FILE
        fi
        
        echo "已停止"
        ;;
    
    status)
        if screen -list | grep -q $SCREEN_NAME; then
            echo "状态: 运行中 (screen: $SCREEN_NAME)"
            screen -list | grep $SCREEN_NAME
        elif [ -f "$PID_FILE" ] && kill -0 $(cat $PID_FILE) 2>/dev/null; then
            echo "状态: 运行中 (PID: $(cat $PID_FILE))"
        else
            echo "状态: 未运行"
        fi
        
        # 显示最新日志
        if [ -f "$LOG_DIR/lifelong_console.log" ]; then
            echo ""
            echo "最新日志:"
            tail -10 "$LOG_DIR/lifelong_console.log"
        fi
        ;;
    
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    
    logs)
        # 实时查看日志
        tail -f "$LOG_DIR/lifelong_console.log"
        ;;
    
    attach)
        # 连接到screen会话
        screen -r $SCREEN_NAME
        ;;
    
    *)
        echo "用法: $0 {start|stop|status|restart|logs|attach}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动终身学习系统（使用screen守护）"
        echo "  stop    - 停止系统"
        echo "  status  - 查看运行状态"
        echo "  restart - 重启系统"
        echo "  logs    - 实时查看日志"
        echo "  attach  - 连接到screen会话（Ctrl+A+D分离）"
        exit 1
        ;;
esac