#!/bin/bash

# 设置要清理的根目录
# 默认是当前目录，您可以修改为指定的路径，例如: BASE_DIR="./prompt/prompts_1"
BASE_DIR="./prompt/prompts_1/editing"

# 检查根目录是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 根目录 '$BASE_DIR' 不存在。"
    exit 1
fi

echo "将在 '$BASE_DIR' 目录下查找并清理..."

# 使用find命令查找所有名为 set_* 的目录
# -type d: 只查找目录
# -name "set_*": 匹配名称
# -maxdepth 2: 可以根据你的目录结构调整，如果set_*就在prompts_1下，-maxdepth 2是合适的
# -exec ...: 对找到的每个目录执行后续命令
find "$BASE_DIR" -type d -name "set_*" -print0 | while IFS= read -r -d '' set_dir; do
    
    # --- 清理 relighting 文件夹 ---
    RELIGHTING_DIR="$set_dir/relighting"
    if [ -d "$RELIGHTING_DIR" ]; then
        echo "正在删除: $RELIGHTING_DIR"
        # 使用 rm -rf 来强制删除目录及其所有内容
        rm -rf "$RELIGHTING_DIR"
    else
        # 可选：如果文件夹不存在，打印一条消息
        echo "未找到: $RELIGHTING_DIR (跳过)"
    fi

    # --- 清理 video 文件夹 ---
    VIDEO_DIR="$set_dir/video"
    if [ -d "$VIDEO_DIR" ]; then
        echo "正在删除: $VIDEO_DIR"
        rm -rf "$VIDEO_DIR"
    else
        echo "未找到: $VIDEO_DIR (跳过)"
    fi

done

echo "清理完成。"