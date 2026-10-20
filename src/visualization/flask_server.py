import os
import logging
import threading
import time
import eventlet
eventlet.monkey_patch()
from typing import Optional, Dict, Any
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import numpy as np
import queue
import hashlib
import json

# 导入可视化器和服务器组件
from .visualizer import EEGVisualizer
from .api_server import EEGAPIServer
from .websocket_server import EEGWebSocketServer

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EEGVisualizationServer')

# 全局变量
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'eeg-visualization-secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
visualizer = None
api_server = None
websocket_server = None
server_thread = None
data_update_queue = queue.Queue(maxsize=100)  # 数据更新队列
active_subscriptions = {}  # 活动订阅

# Create templates directory
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Create static files directory
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

# Create HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Real-time Visualization System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
        }
        .visualization-panel {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 20px;
        }
        .viz-card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            flex: 1;
            min-width: 300px;
        }
        .viz-card h3 {
            margin-top: 0;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            color: #2c3e50;
        }
        .viz-image {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            margin-top: 10px;
            cursor: pointer;
        }
        .controls {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 15px;
            margin-bottom: 20px;
        }
        .control-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #2c3e50;
        }
        select, input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #2c3e50;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #1a252f;
        }
        .channel-selector {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .channel-checkbox {
            display: flex;
            align-items: center;
            margin-right: 10px;
        }
        .channel-checkbox input {
            width: auto;
            margin-right: 5px;
        }
        .status {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            background-color: #e3f2fd;
            color: #0d47a1;
        }
        .refresh-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
        }
        .auto-refresh {
            display: flex;
            align-items: center;
        }
        .auto-refresh input {
            width: auto;
            margin-right: 5px;
        }
        .data-source-info {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 4px;
            background-color: #f0f7ff;
        }
        .data-source-info.active {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .data-source-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            background-color: #2c3e50;
            color: white;
            font-size: 12px;
            margin-left: 10px;
        }
        .multi-viz-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .slider-container {
            display: flex;
            align-items: center;
            margin-top: 10px;
        }
        .slider-container input[type="range"] {
            flex-grow: 1;
            margin: 0 10px;
        }
        .slider-value {
            min-width: 40px;
            text-align: center;
        }
        .tabs {
            display: flex;
            margin-bottom: 15px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 15px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            margin-right: 5px;
        }
        .tab.active {
            background-color: #fff;
            border-color: #ddd;
            color: #2c3e50;
            font-weight: bold;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .signal-quality-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .signal-good {
            background-color: #4CAF50;
        }
        .signal-medium {
            background-color: #FFC107;
        }
        .signal-poor {
            background-color: #F44336;
        }
        .time-marker {
            position: absolute;
            width: 2px;
            background-color: red;
            height: 100%;
            pointer-events: none;
            z-index: 10;
        }
        .time-marker-label {
            position: absolute;
            background-color: rgba(255,0,0,0.7);
            color: white;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 10px;
            white-space: nowrap;
            pointer-events: none;
            z-index: 11;
        }
        .viz-container {
            position: relative;
        }
        .overlay-controls {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            display: flex;
            gap: 5px;
        }
        .overlay-btn {
            background-color: rgba(44, 62, 80, 0.7);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 12px;
            cursor: pointer;
        }
        .overlay-btn:hover {
            background-color: rgba(44, 62, 80, 0.9);
        }
        .advanced-options {
            margin-top: 10px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
        .advanced-options-toggle {
            cursor: pointer;
            color: #2c3e50;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        .advanced-options-toggle::after {
            content: '▼';
            margin-left: 5px;
            font-size: 10px;
        }
        .advanced-options-toggle.collapsed::after {
            content: '►';
        }
        .advanced-options-content {
            margin-top: 10px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>EEG Real-time Visualization System</h1>
        </div>
        <div id="app"></div>
    </div>
    <script src="/static/app.js"></script>
    <script src="/static/websocket_client.js"></script>
</body>
</html>
"""

# 写入HTML模板到文件
with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w', encoding='utf-8') as f:
    f.write(html_template)

# 主页路由
@app.route('/')
def index():
    return render_template('index.html')

# API路由 - 获取数据源列表
@app.route('/api/data_sources')
def get_data_sources():
    global visualizer
    
    if visualizer is None:
        return jsonify({
            'success': False,
            'error': 'Visualizer not initialized'
        })
    
    try:
        # 获取真实数据源列表
        source_ids = visualizer.get_data_source_ids()
        sources = []
        
        for source_id in source_ids:
            source_data = visualizer.get_data_source(source_id)
            if source_data:
                sources.append({
                    'id': source_id,
                    'name': source_id,  # 使用ID作为名称
                    'subject': source_data.get('metadata', {}).get('subject', 'Unknown'),
                    'channel_count': len(source_data['channel_names']),
                    'channel_names': source_data['channel_names'],
                    'sampling_rate': source_data['sampling_rate']
                })
        
        return jsonify({
            'success': True,
            'sources': sources,
            'active_source': visualizer.active_source
        })
    except Exception as e:
        logger.error(f"Error getting data sources: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# API路由 - 设置活动数据源
@app.route('/api/set_active_source')
def set_active_source():
    global visualizer
    
    if visualizer is None:
        return jsonify({
            'success': False,
            'error': 'Visualizer not initialized'
        })
    
    source_id = request.args.get('source_id')
    if not source_id:
        return jsonify({
            'success': False,
            'error': 'No source_id provided'
        })
    
    try:
        success = visualizer.set_active_source(source_id)
        return jsonify({
            'success': success,
            'message': f"Active source set to {source_id}" if success else f"Failed to set active source to {source_id}"
        })
    except Exception as e:
        logger.error(f"Error setting active source: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# API路由 - 获取通道名称
@app.route('/api/channel_names')
def get_channel_names():
    global visualizer
    
    if visualizer is None:
        return jsonify({
            'success': False,
            'error': 'Visualizer not initialized'
        })
    
    source_id = request.args.get('source_id')
    
    try:
        source_data = visualizer.get_data_source(source_id)
        if source_data:
            return jsonify({
                'success': True,
                'channel_names': source_data['channel_names']
            })
        else:
            return jsonify({
                'success': False,
                'error': f"Data source '{source_id}' not found"
            })
    except Exception as e:
        logger.error(f"Error getting channel names: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# API路由 - 可视化
@app.route('/api/visualize', methods=['POST'])
def visualize():
    global visualizer, api_cache
    
    if visualizer is None:
        return jsonify({
            'success': False,
            'error': 'Visualizer not initialized'
        })
    
    # 获取请求参数
    try:
        params = request.json
        
        # 验证必要参数
        if not params.get('source_id'):
            return jsonify({
                'success': False,
                'error': 'No source_id provided'
            })
            
        if not params.get('visualization_type'):
            return jsonify({
                'success': False,
                'error': 'No visualization_type provided'
            })
            
        if not params.get('channels') or len(params.get('channels')) == 0:
            return jsonify({
                'success': False,
                'error': 'No channels provided'
            })
        
        # 生成缓存键
        cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        
        # 检查缓存
        if cache_key in api_cache and time.time() - api_cache[cache_key]['timestamp'] < api_cache_ttl:
            return jsonify(api_cache[cache_key]['response'])
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取数据源
        source_id = params.get('source_id')
        source_data = visualizer.get_data_source(source_id)
        
        if not source_data:
            return jsonify({
                'success': False,
                'error': f"Data source '{source_id}' not found"
            })
        
        # 获取通道数据
        channels = params.get('channels')
        channel_indices = [source_data['channel_names'].index(ch) for ch in channels if ch in source_data['channel_names']]
        
        if len(channel_indices) == 0:
            return jsonify({
                'success': False,
                'error': 'No valid channels found'
            })
        
        # 获取数据
        data = source_data['data'][channel_indices]
        channel_names = [source_data['channel_names'][i] for i in channel_indices]
        
        # 根据可视化类型生成图像
        image_base64 = None
        viz_type = params.get('visualization_type')
        
        if viz_type == 'time':
            # 时域可视化
            time_window = float(params.get('time_window', 5.0))
            baseline_correction = bool(params.get('baseline_correction', True))
            
            image_base64 = visualizer.plot_time_domain(
                data=data,
                channel_names=channel_names,
                duration=time_window,
                baseline_correction=baseline_correction,
                source_id=source_id
            )
        elif viz_type == 'frequency':
            # 频域可视化
            freq_min = int(params.get('freq_min', 0))
            freq_max = int(params.get('freq_max', 50))
            
            image_base64 = visualizer.plot_frequency_domain(
                data=data,
                channel_names=channel_names,
                freq_range=(freq_min, freq_max),
                source_id=source_id
            )
        elif viz_type == 'topographic':
            # 脑地形图可视化
            freq_band = params.get('freq_band', 'alpha')
            
            image_base64 = visualizer.plot_topographic_map(
                data=data,
                channel_names=channel_names,
                freq_band=freq_band,
                source_id=source_id
            )
        elif viz_type == 'combined':
            # 组合可视化
            time_window = float(params.get('time_window', 5.0))
            freq_band = params.get('freq_band', 'alpha')
            
            image_base64 = visualizer.plot_combined_view(
                data=data,
                channel_names=channel_names,
                duration=time_window,
                freq_band=freq_band,
                source_id=source_id
            )
        else:
            return jsonify({
                'success': False,
                'error': f"Unknown visualization type: {viz_type}"
            })
        
        # 计算响应时间
        response_time = f"{(time.time() - start_time) * 1000:.2f}ms"
        
        # 准备响应
        response = {
            'success': True,
            'image': image_base64,
            'response_time': response_time
        }
        
        # 缓存响应
        api_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        
        # 清理过期缓存
        if len(api_cache) > api_cache_size_limit:
            # 删除最旧的缓存项
            oldest_key = min(api_cache.keys(), key=lambda k: api_cache[k]['timestamp'])
            del api_cache[oldest_key]
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in visualization: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        })

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    global active_subscriptions
    logger.info(f"WebSocket客户端已连接: {request.sid}")
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    global active_subscriptions
    logger.info(f"WebSocket客户端已断开: {request.sid}")
    # 清理订阅
    for source_id, subscribers in list(active_subscriptions.items()):
        if request.sid in subscribers:
            subscribers.remove(request.sid)
            logger.info(f"已移除客户端 {request.sid} 对数据源 {source_id} 的订阅")

@socketio.on('subscribe')
def handle_subscribe(data):
    global active_subscriptions
    source_id = data.get('source_id')
    if not source_id:
        emit('error', {'message': '未提供数据源ID'})
        return
    
    # 初始化订阅列表（如果不存在）
    if source_id not in active_subscriptions:
        active_subscriptions[source_id] = set()
    
    # 添加订阅
    active_subscriptions[source_id].add(request.sid)
    logger.info(f"客户端 {request.sid} 已订阅数据源 {source_id}")
    emit('subscription_status', {'status': 'subscribed', 'source_id': source_id})

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    global active_subscriptions
    source_id = data.get('source_id')
    if not source_id or source_id not in active_subscriptions:
        return
    
    # 移除订阅
    if request.sid in active_subscriptions[source_id]:
        active_subscriptions[source_id].remove(request.sid)
        logger.info(f"客户端 {request.sid} 已取消订阅数据源 {source_id}")
    
    emit('subscription_status', {'status': 'unsubscribed', 'source_id': source_id})

# 数据推送线程函数
def push_data_updates():
    global data_update_queue, active_subscriptions
    while True:
        try:
            # 从队列获取数据更新
            update = data_update_queue.get(timeout=1.0)
            source_id = update.get('source_id')
            
            # 检查是否有订阅者
            if source_id in active_subscriptions and active_subscriptions[source_id]:
                # 向所有订阅者推送更新
                socketio.emit('data_update', update, room=list(active_subscriptions[source_id]))
                logger.info(f"已向 {len(active_subscriptions[source_id])} 个客户端推送数据源 {source_id} 的更新")
            
            # 标记任务完成
            data_update_queue.task_done()
        except queue.Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            logger.error(f"推送数据更新时出错: {e}")

# 添加数据更新到队列
def add_data_update(source_id, data, features=None):
    global data_update_queue, active_subscriptions
    try:
        # 准备更新数据
        update = {
            'type': 'data_update',
            'source_id': source_id,
            'timestamp': time.time(),
            'data': {
                'source_id': source_id,
                'packet_count': len(data_update_queue.queue) + 1,
                'latency': 0.0  # 将在前端计算
            }
        }
        
        # 如果队列已满，移除最旧的项
        if data_update_queue.full():
            try:
                data_update_queue.get_nowait()
                data_update_queue.task_done()
            except queue.Empty:
                pass
        
        # 添加到队列
        data_update_queue.put(update, block=False)
        logger.info(f"已添加数据源 {source_id} 的更新到队列")
        return True
    except Exception as e:
        logger.error(f"添加数据更新到队列时出错: {e}")
        return False

# 启动可视化服务器
def start_visualization_server(eeg_visualizer, port=5000):
    global visualizer, server_thread, data_update_queue, active_subscriptions
    
    visualizer = eeg_visualizer
    
    # 确保数据更新队列和活动订阅已初始化
    if data_update_queue is None:
        data_update_queue = queue.Queue(maxsize=100)
    
    if active_subscriptions is None:
        active_subscriptions = {}
    
    # 启动数据推送线程
    push_thread = threading.Thread(target=push_data_updates)
    push_thread.daemon = True
    push_thread.start()
    logger.info("数据推送线程已启动")
    
    # 启动WebSocket服务器
    server_thread = threading.Thread(target=lambda: socketio.run(app, host='0.0.0.0', port=port, debug=False))
    server_thread.daemon = True
    server_thread.start()
    
    logger.info(f"可视化服务器已启动，端口 {port}")
    return push_thread

