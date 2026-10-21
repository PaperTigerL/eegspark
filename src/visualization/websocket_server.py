#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG WebSocket服务器模块 - 处理实时数据推送

此模块确保eventlet的monkey_patch在所有导入之前执行
"""

# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet
import sys

eventlet.monkey_patch()

import os
import logging
import threading
import time
from typing import Optional, Dict, Any, List
import queue

# 导入Flask相关模块 - 确保在创建应用上下文之前导入
from flask import request, current_app
from flask_socketio import SocketIO, emit

# 初始化SocketIO，设置path为/ws
socketio = SocketIO(path='/ws')

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

# 确保WebSocket服务器日志文件存在
websocket_log_path = log_initializer.ensure_log_file_exists("websocket.log")
if not websocket_log_path:
    websocket_log_path = os.path.join(log_dir, "websocket.log")

# 配置日志 - 仅保留WebSocket相关日志
logger = logging.getLogger("EEGWebSocketServer")
logger.setLevel(logging.DEBUG)

# 配置SocketIO和EngineIO日志
socketio_logger = logging.getLogger('socketio')
socketio_logger.setLevel(logging.DEBUG)
engineio_logger = logging.getLogger('engineio')
engineio_logger.setLevel(logging.DEBUG)

# 移除所有现有处理器
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 添加文件处理器
file_handler = logging.FileHandler(websocket_log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# 添加控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)


class EEGWebSocketServer:
    """EEG WebSocket服务器 - 处理实时数据推送"""

    def __init__(self, socketio, visualizer):
        """初始化WebSocket服务器

        Args:
            socketio: Flask-SocketIO实例
            visualizer: EEG可视化器实例
        """
        self.socketio = socketio
        self.visualizer = visualizer
        if not hasattr(self.visualizer, 'get_data_source_ids'):
            logger.error("Visualizer实例缺少必要方法，请检查visualizer参数")
            raise ValueError("Visualizer实例必须实现get_data_source_ids方法")
        
        self.data_update_queue = queue.Queue(maxsize=200)  # 增大队列容量以减少数据丢弃
        self.active_subscriptions = {}
        self.update_thread = None
        self.running = False

        # 注册事件处理程序
        self._register_event_handlers()

        logger.info("EEG WebSocket服务器初始化完成")

    def _register_event_handlers(self):
        """注册WebSocket事件处理程序"""

        @self.socketio.on("connect")
        def handle_connect():
            # 确保在应用上下文中运行
            with self.socketio.server.flask_app.app_context():
                try:
                    logger.info(f"WebSocket连接开始建立，客户端ID: {request.sid}")
                    logger.debug(f"当前应用上下文: {current_app.name}")
                    logger.debug(f"SocketIO实例类型: {type(self.socketio)}")
                    logger.debug(f"请求头信息: {dict(request.headers)}")
                    logger.debug(f"客户端环境: {request.environ}")
                    logger.debug(f"WebSocket路径: {request.path}")
                    logger.debug(f"WebSocket协议: {request.environ.get('wsgi.websocket')}")
                    logger.debug(f"请求方法: {request.method}")
                    logger.debug(f"请求URL: {request.url}")
                    
                    # 检查端口是否可用
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    port_in_use = sock.connect_ex(('localhost', 5000)) == 0
                    sock.close()
                    
                    logger.debug(f"端口5000检查结果: {'被占用' if port_in_use else '可用'}")
                    if port_in_use:
                        logger.error(f"端口5000已被占用，无法建立WebSocket连接")
                        emit('connection_error', {'message': "端口5000已被占用", 'client_id': request.sid, 'timestamp': time.time()})
                        return False
                    
                    logger.info(f"WebSocket连接建立: {request.sid}, IP: {request.remote_addr}, 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    emit('connection_status', {'status': 'connected', 'client_id': request.sid, 'timestamp': time.time()})
                    
                    # 检查socketio实例是否有效
                    if not hasattr(self.socketio, 'emit'):
                        logger.error(f"无效的SocketIO实例: {type(self.socketio)}")
                        return False
                    
                    return True
                except Exception as e:
                    logger.error(f"处理连接请求出错: {e}", exc_info=True)
                    logger.debug(f"错误发生时的请求信息: {request.data}")
                    logger.debug(f"当前活跃订阅: {self.active_subscriptions}")
                    logger.debug(f"当前队列状态: {self.data_update_queue.qsize()}/{self.data_update_queue.maxsize}")
                    emit('connection_error', {'message': f"连接出错: {str(e)}", 'client_id': request.sid, 'timestamp': time.time()})
                    return False

        @self.socketio.on("disconnect")
        def handle_disconnect():
            # 确保在应用上下文中运行
            with self.socketio.server.flask_app.app_context():
                client_id = request.sid
                disconnect_time = time.strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"WebSocket连接断开: {client_id}, 时间: {disconnect_time}, 原因: {request.disconnected}")
                logger.debug(f"断开时的WebSocket状态: {socketio.server.get_session(request.sid)}")
                logger.debug(f"当前活跃连接数: {len(socketio.server.manager.rooms['/'])}")
                
                # 清理该客户端的所有订阅
                for source_id in list(self.active_subscriptions.keys()):
                    if client_id in self.active_subscriptions.get(source_id, set()):
                        self.active_subscriptions[source_id].remove(client_id)
                        logger.info(f"客户端 {client_id} 取消订阅数据源 {source_id}, 当前订阅数: {len(self.active_subscriptions[source_id])}")

        @self.socketio.on("subscribe")
        def handle_subscribe(data):
            # 确保在应用上下文中运行
            with self.socketio.server.flask_app.app_context():
                try:
                    source_id = data.get("source_id")
                    if not source_id:
                        emit("subscription_error", {"message": "缺少source_id参数"})
                        return

                    client_id = request.sid

                    # 添加到订阅列表
                    if source_id not in self.active_subscriptions:
                        self.active_subscriptions[source_id] = set()
                    self.active_subscriptions[source_id].add(client_id)

                    logger.info(f"客户端 {client_id} 订阅数据源 {source_id}")
                    emit(
                        "subscription_status",
                        {"status": "subscribed", "source_id": source_id},
                    )
                except Exception as e:
                    logger.error(f"处理订阅请求出错: {e}, 客户端: {request.sid}, 当前连接数: {len(self.active_subscriptions)}, 队列状态: {self.data_update_queue.qsize()}/{self.data_update_queue.maxsize}", exc_info=True)
                    emit('subscription_error', {'message': f"订阅出错: {str(e)}", 'client_id': request.sid, 'timestamp': time.time()})

        @self.socketio.on("unsubscribe")
        def handle_unsubscribe(data):
            # 确保在应用上下文中运行
            with self.socketio.server.flask_app.app_context():
                try:
                    source_id = data.get("source_id")
                    if not source_id:
                        emit("subscription_error", {"message": "缺少source_id参数"})
                        return

                    client_id = request.sid

                    # 从订阅列表中移除
                    if (
                        source_id in self.active_subscriptions
                        and client_id in self.active_subscriptions[source_id]
                    ):
                        self.active_subscriptions[source_id].remove(client_id)
                        logger.info(f"客户端 {client_id} 取消订阅数据源 {source_id}")

                    emit(
                        "subscription_status",
                        {"status": "unsubscribed", "source_id": source_id},
                    )
                except Exception as e:
                    logger.error(f"处理取消订阅请求出错: {e}", exc_info=True)
                    emit("subscription_error", {"message": f"取消订阅出错: {str(e)}"})
                    
    def _process_data_updates(self):
        """处理数据更新队列并推送给订阅客户端"""
        while self.running:
            try:
                # 从队列获取数据更新
                update_data = self.data_update_queue.get(timeout=1)
                if not update_data:
                    continue
                    
                # 验证数据格式
                if not all(key in update_data for key in ['source_id', 'data', 'channel_names', 'sampling_rate']):
                    logger.warning(f"无效的数据格式: {update_data}")
                    continue
                    
                # 获取订阅该数据源的所有客户端
                subscribers = self.active_subscriptions.get(update_data['source_id'], set())
                if not subscribers:
                    continue
                    
                # 准备推送数据
                push_data = {
                    'source_id': update_data['source_id'],
                    'data': update_data['data'],
                    'channel_names': update_data['channel_names'],
                    'sampling_rate': update_data['sampling_rate'],
                    'timestamp': time.time()
                }
                
                # 推送给所有订阅客户端
                for client_id in subscribers:
                    try:
                        self.socketio.emit('data_update', push_data, room=client_id)
                    except Exception as e:
                        logger.error(f"推送数据给客户端 {client_id} 失败: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理数据更新出错: {e}")
                
    def start(self):
        """启动WebSocket服务器"""
        if self.running:
            return
            
        self.running = True
        self.update_thread = threading.Thread(target=self._process_data_updates, daemon=True)
        self.update_thread.start()
        logger.info("WebSocket数据推送线程已启动")
        
    def stop(self):
        """停止WebSocket服务器"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("WebSocket数据推送线程已停止")

    def add_data_update(
        self, source_id: str, data: Any, features: Optional[Dict] = None
    ):
        """添加数据更新到队列

        Args:
            source_id: 数据源ID
            data: 更新的数据
            features: 特征数据
        """
        try:
            # 检查是否有客户端订阅了该数据源
            if (
                source_id in self.active_subscriptions
                and self.active_subscriptions[source_id]
            ):
                # 将更新添加到队列
                update_info = {
                    "source_id": source_id,
                    "timestamp": time.time(),
                    "data_shape": data.shape if hasattr(data, "shape") else None,
                    "features": features,
                }

                # 使用非阻塞方式添加到队列，如果队列已满则丢弃
                try:
                    self.data_update_queue.put_nowait(update_info)
                    logger.debug(f"数据更新已添加到队列: {source_id}")
                except queue.Full:
                    logger.warning("更新队列已满，丢弃更新")
        except Exception as e:
            logger.error(f"添加数据更新出错: {e}", exc_info=True)

    def _process_data_updates(self):
        """处理数据更新队列并推送给订阅客户端"""
        while self.running:
            try:
                # 从队列获取数据更新
                update_data = self.data_update_queue.get(timeout=1)
                if not update_data:
                    continue
                    
                # 验证数据格式
                if not all(key in update_data for key in ['source_id', 'data', 'channel_names', 'sampling_rate']):
                    logger.warning(f"无效的数据格式: {update_data}")
                    continue
                    
                # 获取订阅该数据源的所有客户端
                subscribers = self.active_subscriptions.get(update_data['source_id'], set())
                if not subscribers:
                    continue
                    
                # 准备推送数据
                push_data = {
                    'source_id': update_data['source_id'],
                    'data': update_data['data'],
                    'channel_names': update_data['channel_names'],
                    'sampling_rate': update_data['sampling_rate'],
                    'timestamp': time.time()
                }
                
                # 推送给所有订阅客户端
                for client_id in subscribers:
                    try:
                        with self.socketio.server.flask_app.app_context():
                            self.socketio.emit('data_update', push_data, room=client_id)
                        logger.debug(f"成功推送数据给客户端 {client_id}")
                    except Exception as e:
                        logger.error(f"推送数据给客户端 {client_id} 失败: {e}", exc_info=True)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理数据更新出错: {e}", exc_info=True)
                time.sleep(0.1)
