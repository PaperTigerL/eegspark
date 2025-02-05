from flask import Flask
from flask_socketio import SocketIO
import threading
import time
import numpy as np

class EEGVisualizer:
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        # 关闭SocketIO日志
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins='*',
            async_mode='threading',
            logger=False,  # 关闭SocketIO日志
            engineio_logger=False,  # 关闭Engine.IO日志
            ping_timeout=5,
            ping_interval=1
        )
        self.channels_data = {}  # 存储所有通道数据
        self.max_data_points = 5000  # 每个通道保留的最大数据点数
        self.update_counter = 0
        self.last_update = time.time()
        
        @self.app.route('/')
        def index():
            return self.index_template
            
    def start(self):
        """启动Web服务器"""
        try:
            threading.Thread(
                target=self.socketio.run,
                args=(self.app,),
                kwargs={
                    'host': self.host,
                    'port': self.port,
                    'debug': False,
                    'use_reloader': False
                }
            ).start()
            print(f"[启动] 可视化服务器运行在 http://{self.host}:{self.port}")
        except Exception as e:
            print(f"[错误] 启动失败: {e}")

    def process(self, eegrdd):
        try:
            channel = eegrdd.channel
            current_time = time.time()
            
            if channel not in self.channels_data:
                self.channels_data[channel] = {
                    'time': eegrdd.time_points.tolist(),
                    'voltage': eegrdd.voltage_data.tolist(),
                    'visible': False,  # 新通道默认不可见
                    'color': f'hsl({(channel * 30) % 360}, 70%, 50%)'
                }
            else:
                self.channels_data[channel]['time'].extend(eegrdd.time_points.tolist())
                self.channels_data[channel]['voltage'].extend(eegrdd.voltage_data.tolist())
                
                if len(self.channels_data[channel]['time']) > self.max_data_points:
                    self.channels_data[channel]['time'] = self.channels_data[channel]['time'][-self.max_data_points:]
                    self.channels_data[channel]['voltage'] = self.channels_data[channel]['voltage'][-self.max_data_points:]

            if current_time - self.last_update >= 0.1:
                self.socketio.emit('eeg_update', {
                    'channels': self.channels_data
                })
                self.last_update = current_time
                
        except Exception as e:
            print(f"[错误] 数据处理失败: {e}")
            
    @property 
    def index_template(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EEG多通道分析</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    margin: 0; 
                    background: #1a1a1a; 
                    color: #fff;
                    display: flex;
                    height: 100vh;
                }
                #sidebar {
                    width: 160px;
                    background: rgba(0,0,0,0.7);
                    padding: 15px;
                    overflow-y: auto;
                }
                #plot-container {
                    flex: 1;
                    height: 100vh;
                    padding: 20px;
                    box-sizing: border-box;
                }
                #plot {
                    width: 100%;
                    height: calc(100% - 40px);
                    background: #262626;
                    border-radius: 5px;
                }
                #controls {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: rgba(0,0,0,0.7);
                    padding: 10px;
                    border-radius: 4px;
                    z-index: 1000;
                }
                .control-btn {
                    background: #333;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .control-btn.active {
                    background: #0066cc;
                }
                .channel-btn {
                    width: 100%;
                    padding: 8px;
                    margin: 5px 0;
                    background: #333;
                    border: none;
                    color: white;
                    cursor: pointer;
                    border-radius: 4px;
                    transition: all 0.3s;
                }
                .channel-btn.active {
                    background: #0066cc;
                }
            </style>
        </head>
        <body>
            <div id="sidebar">
                <div id="channel-buttons"></div>
            </div>
            <div id="plot-container">
                <div id="plot"></div>
                <div id="controls">
                    <button id="autoScrollBtn" class="control-btn active" onclick="toggleAutoScroll()">
                        自动滚动: 开
                    </button>
                </div>
            </div>
            
            <script>
                const socket = io();
                let channelsData = {};
                let selectedChannels = new Set();  // 使用Set存储选中的通道
                let isAutoScroll = true;
                let currentXRange = null;

                function updateChannelButtons() {
                    const container = document.getElementById('channel-buttons');
                    container.innerHTML = Object.keys(channelsData)
                        .sort((a, b) => Number(a) - Number(b))
                        .map(channel => `
                            <button 
                                class="channel-btn ${selectedChannels.has(Number(channel)) ? 'active' : ''}"
                                onclick="toggleChannel(${channel})"
                            >通道 ${channel}</button>
                        `).join('');
                }
                
                function toggleChannel(channel) {
                    if (selectedChannels.has(channel)) {
                        selectedChannels.delete(channel);
                    } else {
                        selectedChannels.add(channel);
                    }
                    updateChannelButtons();
                    updatePlot();
                }
                
                function toggleAutoScroll() {
                    isAutoScroll = !isAutoScroll;
                    document.getElementById('autoScrollBtn').textContent = 
                        `自动滚动: ${isAutoScroll ? '开' : '关'}`;
                    document.getElementById('autoScrollBtn').classList.toggle('active');
                }
                
                function updatePlot() {
                    const traces = Array.from(selectedChannels)
                        .filter(channel => channelsData[channel])
                        .map(channel => ({
                            x: channelsData[channel].time,
                            y: channelsData[channel].voltage,
                            name: `通道 ${channel}`,
                            type: 'scatter',
                            mode: 'lines',
                            line: { 
                                color: `hsl(${channel * 30 % 360}, 70%, 50%)`,
                                width: 1.5 
                            }
                        }));
                    
                    const layout = {
                        paper_bgcolor: '#1a1a1a',
                        plot_bgcolor: '#1a1a1a',
                        font: { color: '#fff' },
                        margin: { t: 30, l: 60, r: 30, b: 40 },
                        showlegend: true,
                        xaxis: {
                            title: '时间 (秒)',
                            gridcolor: '#333',
                            zerolinecolor: '#333',
                            ticksuffix: ' s',
                            rangeslider: {  // 添加时间轴滑块
                                visible: true,
                                bgcolor: '#262626',
                                bordercolor: '#404040',
                                thickness: 0.1
                            }
                        },
                        yaxis: {
                            title: '电压 (μV)',
                            gridcolor: '#333',
                            zerolinecolor: '#333',
                            ticksuffix: ' μV'
                        },
                        legend: {
                            x: 1,
                            xanchor: 'right',
                            y: 1,
                            bgcolor: 'rgba(0,0,0,0.3)'
                        }
                    };

                    // 如果不是自动滚动且有保存的视图范围，则使用保存的范围
                    if (!isAutoScroll && currentXRange) {
                        layout.xaxis.range = currentXRange;
                    }
                    
                    Plotly.newPlot('plot', traces, layout, {
                        scrollZoom: true,  // 启用滚轮缩放
                        displayModeBar: true,  // 显示工具栏
                        modeBarButtons: [[  // 自定义工具栏按钮
                            'zoom2d',
                            'pan2d',
                            'resetScale2d',
                            'autoScale2d'
                        ]],
                        displaylogo: false
                    });

                    // 保存当前视图范围
                    const plotDiv = document.getElementById('plot');
                    plotDiv.on('plotly_relayout', function(eventData) {
                        if (eventData['xaxis.range[0]'] !== undefined) {
                            currentXRange = [
                                eventData['xaxis.range[0]'],
                                eventData['xaxis.range[1]']
                            ];
                            isAutoScroll = false;
                            document.getElementById('autoScrollBtn').textContent = '自动滚动: 关';
                            document.getElementById('autoScrollBtn').classList.remove('active');
                        }
                    });
                }

                // 初始化空图表
                updatePlot();
                
                socket.on('eeg_update', (data) => {
                    channelsData = data.channels;
                    updateChannelButtons();
                    if (selectedChannels.size > 0) {
                        if (isAutoScroll) {
                            currentXRange = null;  // 重置保存的范围
                        }
                        updatePlot();
                    }
                });
            </script>
        </body>
        </html>
        """