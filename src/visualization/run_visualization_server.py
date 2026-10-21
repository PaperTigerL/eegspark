#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化服务器启动脚本

此脚本用于启动EEG数据可视化的Web服务器
确保eventlet的monkey_patch在所有导入之前执行
"""

# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet

eventlet.monkey_patch()

import os
import sys
import logging

# 导入Flask服务器模块
from visualization.flask_server import app, socketio, start_visualization_server
from visualization.websocket_server import EEGWebSocketServer

# 配置日志
# 使用日志目录初始化器确保日志目录和文件正确创建
from utils.log_directory_initializer import LogDirectoryInitializer

# 初始化日志目录
log_initializer = LogDirectoryInitializer()
log_dir = log_initializer.initialize()

if not log_dir:
    # 如果所有尝试都失败，使用默认目录
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"警告: 使用默认日志目录: {log_dir}")

# 确保可视化日志文件存在
vis_log_path = log_initializer.ensure_log_file_exists("visualization.log")
if not vis_log_path:
    vis_log_path = os.path.join(log_dir, "visualization.log")

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 移除所有现有处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 添加文件处理器
file_handler = logging.FileHandler(vis_log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

logger.info(f"已添加文件日志处理器，日志文件路径: {vis_log_path}")
logger.info(f"日志目录初始化完成: {log_dir}")


def main():
    """主函数，启动可视化服务器"""
    # 初始化WebSocket服务器
    # 注意：这里不需要创建visualizer实例，因为我们只是启动服务器
    # 实际的visualizer会在main.py中创建并传递给WebSocketServer

    # 启动Flask服务器
    logger.info("启动可视化服务器...")
    logger.info("正在初始化WebSocket服务器...")
    start_visualization_server(host="0.0.0.0", port=5000, debug=False)
    logger.info("WebSocket服务器已启动，监听端口: 5000")


if __name__ == "__main__":
    main()
