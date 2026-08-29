#!/bin/bash

# 执行命令
CMD="/usr/local/bin/python"  # 编译器/解释器路径
EXE_FILE=$1  # 文件名称

FINAL_CMD="${CMD} ${EXE_FILE}"

# 后缀名
FILE_EXT="py"

# 检查输入参数
if [[ ! -e "$EXE_FILE" ]]; then
    echo "请提供要执行的脚本路径，例如：./start.sh /path/to/script.ext"
    exit 1
fi

# 日志目录（根据需要替换成合适的路径）
LOG_DIR="log"
mkdir -p "$LOG_DIR"  # 创建日志目录（若不存在）

# 获取脚本名称并去掉后缀
script_name=$(basename "$EXE_FILE" ".$FILE_EXT")
# 设置日志文件大小上限（5MB）
MAX_LOG_SIZE=5242880  # 5MB = 5 * 1024 * 1024 bytes

# 日志文件路径和日期
current_date=$(date '+%Y-%m-%d')
LOG_FILE="$LOG_DIR/${script_name}_${current_date}.log"
echo $LOG_FILE

# 检查日志文件是否需要轮替（按天或大小限制）
rotate_log() {
    # 如果日期变更，则更新日志文件名
    new_date=$(date '+%Y-%m-%d')
    if [[ "$new_date" != "$current_date" ]]; then
        current_date="$new_date"
        LOG_FILE="$LOG_DIR/${script_name}_${current_date}.log"
    fi
    
    # 如果文件大小超过 MAX_LOG_SIZE，则重命名旧日志并创建新文件
    if [[ -f "$LOG_FILE" && $(stat -c%s "$LOG_FILE") -ge $MAX_LOG_SIZE ]]; then
        mv "$LOG_FILE" "$LOG_FILE.$(date '+%H%M%S')"  # 添加时间作为后缀
    fi
    
    # 如果日志文件不存在，则创建新的日志文件
    touch "$LOG_FILE"
}

# 检查程序是否正在运行的函数
check_process() { 
    pgrep -f "$FINAL_CMD"
    return $? 
}

cleanup() {
    echo "终止进程中..."
    SUB_PID=$(check_process);
    if [ $SUB_PID ]; then
        kill -9 $SUB_PID
    fi
    pkill -P $$
    exit 1;
}

trap cleanup SIGINT

# 输出启动信息
echo "$(date '+%Y-%m-%d %H:%M:%S') - 启动监控脚本" | tee -a "$LOG_FILE"
echo "监控程序：$FINAL_CMD" | tee -a "$LOG_FILE"

# 主循环
while true; do
    rotate_log  # 检查并更新日志文件

    # 检查程序是否正在运行
    FINAL_CMD_PID=$(check_process)
    if [[ ! $FINAL_CMD_PID ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 程序正在启动..." | tee -a "$LOG_FILE"
        # 重启程序
        nohup $FINAL_CMD &>> "$LOG_FILE" &
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 程序已启动" | tee -a "$LOG_FILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 程序运行PID:$(check_process)"
    fi
    # 每隔指定的时间检查一次
    sleep 10  # 设置检查间隔时间（秒）
done
