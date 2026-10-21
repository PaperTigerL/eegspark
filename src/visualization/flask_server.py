#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask Web服务器模块，提供可视化界面

此模块启动一个Flask Web服务器，用于展示EEG数据的可视化结果
确保eventlet的monkey_patch在所有导入之前执行
"""

# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet

eventlet.monkey_patch()

import os
import sys
import logging
import json
import threading

# 导入Flask相关模块 - 确保在创建应用上下文之前导入
from flask import Flask, render_template, jsonify, request, current_app
from flask_cors import CORS
from flask_socketio import SocketIO

# 修复导入路径，从utils.spark_fixes.ivy_config导入而不是fix_ivy_config
from utils.spark_fixes.ivy_config import fix_ivy_repository_path

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # 启用CORS支持

# 初始化SocketIO
socketio = SocketIO(app, 
    cors_allowed_origins="*", path='/ws')

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    logger.info(f"WebSocket客户端连接: {request.sid}")
    print(f"WebSocket客户端连接: {request.sid}")
    return True

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"WebSocket客户端断开连接: {request.sid}")
    print(f"WebSocket客户端断开连接: {request.sid}")

@socketio.on('message')
def handle_message(data):
    logger.info(f"收到来自 {request.sid} 的WebSocket消息: {data}")
    print(f"收到来自 {request.sid} 的WebSocket消息: {data}")
    socketio.emit('response', {'status': 'received'}, room=request.sid)

# 导入并初始化API服务器
from .api_server import EEGAPIServer
from .visualizer import EEGVisualizer

# 创建可视化器实例
visualizer = EEGVisualizer()

# 初始化API服务器
api_server = EEGAPIServer(app, visualizer)

# 全局变量
visualization_data = {}
visualization_lock = threading.Lock()


@app.route("/")
def index():
    """网站根路由"""
    return render_template("index.html")

@app.route("/eeg")
def eeg_index():
    """EEG可视化主页"""
    return render_template("index.html")

@app.route("/ws")
def websocket_endpoint():
    """WebSocket端点"""
    logger.info("WebSocket端点被访问")
    print("WebSocket端点被访问")
    return "WebSocket服务已启用"


@app.route("/api/data", methods=["GET"])
def get_data():
    """API端点，返回可视化数据"""
    with visualization_lock:
        return jsonify(visualization_data)


@app.route("/api/update", methods=["POST"])
def update_data():
    """API端点，更新可视化数据"""
    global visualization_data
    try:
        data = request.json
        with visualization_lock:
            visualization_data.update(data)
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"更新数据失败: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


def start_visualization_server(host="0.0.0.0", port=5000, debug=False):
    """启动Flask服务器

    Args:
        host: 服务器主机地址
        port: 服务器端口
        debug: 是否启用调试模式
    """
    # 修复Ivy仓库路径
    fix_ivy_repository_path()

    # 确保应用上下文正确设置
    with app.app_context():
        # 确保绑定到所有网络接口，解决Docker容器中的网络问题
        host = "0.0.0.0"  # 强制使用0.0.0.0以确保在Docker容器中可访问

        # 设置允许的CORS源
        socketio.cors_allowed_origins = "*"

        # 启动Flask服务器
        logger.info(f"启动可视化服务器: http://{host}:{port}")
        logger.info("正在启动Socket.IO服务器...")
        try:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False,
                allow_unsafe_werkzeug=True,
            )
            logger.info("Socket.IO服务器已成功启动")
        except Exception as e:
            logger.error(f"Socket.IO服务器启动失败: {str(e)}")
            raise


if __name__ == "__main__":
    # 直接运行此文件时启动服务器
    start_visualization_server(debug=True)
