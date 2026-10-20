// WebSocket客户端实现 - 用于EEG实时可视化系统

let socket = null;
let isConnected = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 5;
let reconnectInterval = 2000; // 重连间隔(毫秒)

// 初始化WebSocket连接
function initWebSocket() {
    // 确定WebSocket URL (使用相对路径，自动适应不同部署环境)
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    console.log(`正在连接WebSocket: ${wsUrl}`);
    
    // 创建WebSocket连接
    socket = new WebSocket(wsUrl);
    
    // 连接打开事件
    socket.onopen = function() {
        console.log('WebSocket连接已建立');
        isConnected = true;
        reconnectAttempts = 0;
        showConnectionStatus('已连接', 'success');
        
        // 订阅数据更新
        if (activeSource) {
            subscribeToDataSource(activeSource);
        }
    };
    
    // 接收消息事件
    socket.onmessage = function(event) {
        try {
            const message = JSON.parse(event.data);
            
            // 处理不同类型的消息
            switch (message.type) {
                case 'data_update':
                    handleDataUpdate(message.data);
                    break;
                    
                case 'new_data_source':
                    handleNewDataSource(message.data);
                    break;
                    
                case 'error':
                    showStatusMessage(message.error, 'error');
                    break;
                    
                default:
                    console.log('收到未知类型的消息:', message);
            }
        } catch (error) {
            console.error('处理WebSocket消息时出错:', error);
        }
    };
    
    // 连接关闭事件
    socket.onclose = function() {
        console.log('WebSocket连接已关闭');
        isConnected = false;
        showConnectionStatus('已断开', 'error');
        
        // 尝试重新连接
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            showConnectionStatus(`已断开，${reconnectInterval/1000}秒后重连(${reconnectAttempts}/${maxReconnectAttempts})`, 'warning');
            
            setTimeout(function() {
                initWebSocket();
            }, reconnectInterval);
        } else {
            showConnectionStatus('连接失败，请刷新页面重试', 'error');
        }
    };
    
    // 连接错误事件
    socket.onerror = function(error) {
        console.error('WebSocket连接错误:', error);
        showConnectionStatus('连接错误', 'error');
    };
}

// 订阅数据源更新
function subscribeToDataSource(sourceId) {
    if (!isConnected || !socket) {
        console.warn('WebSocket未连接，无法订阅数据源');
        return false;
    }
    
    try {
        socket.send(JSON.stringify({
            action: 'subscribe',
            source_id: sourceId
        }));
        console.log(`已订阅数据源: ${sourceId}`);
        return true;
    } catch (error) {
        console.error('订阅数据源时出错:', error);
        return false;
    }
}

// 取消订阅数据源
function unsubscribeFromDataSource(sourceId) {
    if (!isConnected || !socket) {
        return false;
    }
    
    try {
        socket.send(JSON.stringify({
            action: 'unsubscribe',
            source_id: sourceId
        }));
        console.log(`已取消订阅数据源: ${sourceId}`);
        return true;
    } catch (error) {
        console.error('取消订阅数据源时出错:', error);
        return false;
    }
}

// 处理数据更新
function handleDataUpdate(data) {
    // 检查是否是当前活动数据源的更新
    if (data.source_id === activeSource) {
        // 自动更新可视化
        if (document.getElementById('auto-refresh').checked) {
            refreshVisualization(data.visualization_data);
        }
        
        // 更新数据源信息
        updateDataSourceInfo(data);
    }
}

// 处理新数据源
function handleNewDataSource(data) {
    // 添加新数据源到列表
    dataSources.push(data);
    
    // 更新数据源选择器
    updateDataSourceSelector();
    
    // 显示通知
    showStatusMessage(`发现新数据源: ${data.name}`, 'info');
}

// 显示连接状态
function showConnectionStatus(message, type) {
    // 创建或获取连接状态元素
    let statusElement = document.getElementById('connection-status');
    
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = 'connection-status';
        statusElement.className = 'status';
        
        // 添加到控制面板顶部
        const controlsElement = document.querySelector('.controls');
        if (controlsElement) {
            controlsElement.insertBefore(statusElement, controlsElement.firstChild);
        }
    }
    
    // 设置消息和样式
    statusElement.textContent = `WebSocket: ${message}`;
    
    // 根据消息类型设置样式
    statusElement.className = 'status';
    if (type === 'error') {
        statusElement.style.backgroundColor = '#ffebee';
        statusElement.style.color = '#c62828';
    } else if (type === 'warning') {
        statusElement.style.backgroundColor = '#fff8e1';
        statusElement.style.color = '#ff8f00';
    } else if (type === 'success') {
        statusElement.style.backgroundColor = '#e8f5e9';
        statusElement.style.color = '#2e7d32';
    } else {
        statusElement.style.backgroundColor = '#e3f2fd';
        statusElement.style.color = '#0d47a1';
    }
}

// 更新数据源信息
function updateDataSourceInfo(data) {
    // 更新数据源面板
    const sourcePanel = document.getElementById('data-source-panel');
    if (sourcePanel) {
        // 添加实时指标
        const infoElement = document.createElement('div');
        infoElement.className = 'data-source-info-metrics';
        infoElement.innerHTML = `
            <div>最后更新: ${new Date().toLocaleTimeString()}</div>
            <div>数据包: ${data.packet_count || 'N/A'}</div>
            <div>延迟: ${data.latency ? data.latency.toFixed(2) + 'ms' : 'N/A'}</div>
        `;
        
        // 替换或添加指标元素
        const existingMetrics = sourcePanel.querySelector('.data-source-info-metrics');
        if (existingMetrics) {
            sourcePanel.replaceChild(infoElement, existingMetrics);
        } else {
            sourcePanel.appendChild(infoElement);
        }
    }
}

// 在页面加载完成后初始化WebSocket
document.addEventListener('DOMContentLoaded', function() {
    // 在应用初始化后启动WebSocket
    setTimeout(initWebSocket, 1000);
});

// 在窗口关闭前关闭WebSocket连接
window.addEventListener('beforeunload', function() {
    if (socket && isConnected) {
        socket.close();
    }
});