# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet
eventlet.monkey_patch()

import os
import logging
import threading
import time
from typing import Optional, Dict, Any, List
from flask import request
from flask_socketio import SocketIO, emit
import queue

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EEGWebSocketServer')

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
        self.data_update_queue = queue.Queue(maxsize=100)
        self.active_subscriptions = {}
        self.update_thread = None
        self.running = False
        
        # 注册事件处理程序
        self._register_event_handlers()
        
        logger.info("EEG WebSocket服务器初始化完成")
    
    def _register_event_handlers(self):
        """注册WebSocket事件处理程序"""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"客户端已连接: {request.sid}")
            emit('connection_status', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            # 清理该客户端的所有订阅
            client_id = request.sid
            for source_id in list(self.active_subscriptions.keys()):
                if client_id in self.active_subscriptions.get(source_id, set()):
                    self.active_subscriptions[source_id].remove(client_id)
                    logger.info(f"客户端 {client_id} 断开连接，取消订阅数据源 {source_id}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            try:
                source_id = data.get('source_id')
                if not source_id:
                    emit('subscription_error', {'message': '缺少source_id参数'})
                    return
                
                client_id = request.sid
                
                # 添加到订阅列表
                if source_id not in self.active_subscriptions:
                    self.active_subscriptions[source_id] = set()
                self.active_subscriptions[source_id].add(client_id)
                
                logger.info(f"客户端 {client_id} 订阅数据源 {source_id}")
                emit('subscription_status', {
                    'status': 'subscribed',
                    'source_id': source_id
                })
            except Exception as e:
                logger.error(f"处理订阅请求出错: {e}", exc_info=True)
                emit('subscription_error', {'message': f'订阅出错: {str(e)}'})
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            try:
                source_id = data.get('source_id')
                if not source_id:
                    emit('subscription_error', {'message': '缺少source_id参数'})
                    return
                
                client_id = request.sid
                
                # 从订阅列表中移除
                if source_id in self.active_subscriptions and client_id in self.active_subscriptions[source_id]:
                    self.active_subscriptions[source_id].remove(client_id)
                    logger.info(f"客户端 {client_id} 取消订阅数据源 {source_id}")
                
                emit('subscription_status', {
                    'status': 'unsubscribed',
                    'source_id': source_id
                })
            except Exception as e:
                logger.error(f"处理取消订阅请求出错: {e}", exc_info=True)
                emit('subscription_error', {'message': f'取消订阅出错: {str(e)}'})
    
    def start(self):
        """启动WebSocket服务器"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.running = True
            self.update_thread = threading.Thread(target=self._process_updates)
            self.update_thread.daemon = True
            self.update_thread.start()
            logger.info("WebSocket更新处理线程已启动")
    
    def stop(self):
        """停止WebSocket服务器"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
            logger.info("WebSocket更新处理线程已停止")
    
    def add_data_update(self, source_id: str, data: Any, features: Optional[Dict] = None):
        """添加数据更新到队列
        
        Args:
            source_id: 数据源ID
            data: 更新的数据
            features: 特征数据
        """
        try:
            # 检查是否有客户端订阅了该数据源
            if source_id in self.active_subscriptions and self.active_subscriptions[source_id]:
                # 将更新添加到队列
                update_info = {
                    'source_id': source_id,
                    'timestamp': time.time(),
                    'data_shape': data.shape if hasattr(data, 'shape') else None,
                    'features': features
                }
                
                # 使用非阻塞方式添加到队列，如果队列已满则丢弃
                try:
                    self.data_update_queue.put_nowait(update_info)
                    logger.debug(f"数据更新已添加到队列: {source_id}")
                except queue.Full:
                    logger.warning("更新队列已满，丢弃更新")
        except Exception as e:
            logger.error(f"添加数据更新出错: {e}", exc_info=True)
    
    def _process_updates(self):
        """处理数据更新队列"""
        while self.running:
            try:
                # 从队列中获取更新，最多等待100ms
                try:
                    update_info = self.data_update_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                source_id = update_info['source_id']
                timestamp = update_info['timestamp']
                data_shape = update_info['data_shape']
                features = update_info['features']
                
                # 检查是否有客户端订阅了该数据源
                if source_id in self.active_subscriptions and self.active_subscriptions[source_id]:
                    # 准备要发送的数据
                    emit_data = {
                        'source_id': source_id,
                        'timestamp': timestamp,
                        'data_shape': data_shape,
                        'features': features
                    }
                    
                    # 向所有订阅的客户端发送更新
                    self.socketio.emit('data_update', emit_data, room=list(self.active_subscriptions[source_id]))
                    logger.debug(f"已向 {len(self.active_subscriptions[source_id])} 个客户端推送数据更新: {source_id}")
                
                # 标记任务完成
                self.data_update_queue.task_done()
            except Exception as e:
                logger.error(f"处理数据更新出错: {e}", exc_info=True)
                time.sleep(0.1)  # 避免在出错时过度消耗CPU