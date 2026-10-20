// EEG Visualization System - Frontend Application

// 全局变量
let dataSources = [];
let activeSource = null;
let allChannels = [];
let selectedChannels = [];
let autoRefreshInterval = null;
let currentVisualizationType = 'time';

// 初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// 应用初始化函数
async function initializeApp() {
    // 创建应用UI
    createAppUI();
    
    // 加载数据源
    await loadDataSources();
    
    // 初始化事件监听器
    initEventListeners();
    
    // 加载初始可视化
    refreshVisualization();
}

// 创建应用UI
function createAppUI() {
    const appContainer = document.getElementById('app');
    
    // 创建控制面板
    const controlsHTML = `
        <div class="controls">
            <div class="data-source-info" id="data-source-panel">
                <div>
                    <h3>数据源</h3>
                    <select id="data-source-selector">
                        <option value="">加载中...</option>
                    </select>
                </div>
            </div>
            
            <div class="tabs">
                <div class="tab active" data-tab="time">时域</div>
                <div class="tab" data-tab="frequency">频域</div>
                <div class="tab" data-tab="topographic">地形图</div>
                <div class="tab" data-tab="combined">组合视图</div>
            </div>
            
            <div class="tab-content active" id="time-tab">
                <div class="control-group">
                    <label>时间窗口 (秒)</label>
                    <div class="slider-container">
                        <span class="slider-value">1</span>
                        <input type="range" id="time-window" min="1" max="10" step="1" value="5">
                        <span class="slider-value">10</span>
                    </div>
                    <span id="time-window-value">5.0</span> 秒
                </div>
                
                <div class="control-group">
                    <label>基线校正</label>
                    <input type="checkbox" id="baseline-correction" checked>
                </div>
            </div>
            
            <div class="tab-content" id="frequency-tab">
                <div class="control-group">
                    <label>频率范围 (Hz)</label>
                    <div class="slider-container">
                        <input type="number" id="freq-min" min="0" max="50" step="1" value="0" style="width: 60px">
                        <input type="range" id="freq-range" min="0" max="100" step="1" value="50" style="margin: 0 10px">
                        <input type="number" id="freq-max" min="1" max="100" step="1" value="50" style="width: 60px">
                    </div>
                </div>
            </div>
            
            <div class="tab-content" id="topographic-tab">
                <div class="control-group">
                    <label>频带选择</label>
                    <select id="freq-band">
                        <option value="delta">Delta (0.5-4 Hz)</option>
                        <option value="theta">Theta (4-8 Hz)</option>
                        <option value="alpha" selected>Alpha (8-13 Hz)</option>
                        <option value="beta">Beta (13-30 Hz)</option>
                        <option value="gamma">Gamma (30-45 Hz)</option>
                    </select>
                </div>
            </div>
            
            <div class="tab-content" id="combined-tab">
                <div class="control-group">
                    <label>时间窗口 (秒)</label>
                    <div class="slider-container">
                        <span class="slider-value">1</span>
                        <input type="range" id="combined-time-window" min="1" max="10" step="1" value="5">
                        <span class="slider-value">10</span>
                    </div>
                    <span id="combined-time-window-value">5.0</span> 秒
                </div>
                
                <div class="control-group">
                    <label>频带选择</label>
                    <select id="combined-freq-band">
                        <option value="delta">Delta (0.5-4 Hz)</option>
                        <option value="theta">Theta (4-8 Hz)</option>
                        <option value="alpha" selected>Alpha (8-13 Hz)</option>
                        <option value="beta">Beta (13-30 Hz)</option>
                        <option value="gamma">Gamma (30-45 Hz)</option>
                    </select>
                </div>
            </div>
            
            <div class="control-group">
                <label>通道选择</label>
                <div class="channel-selector" id="channel-selector">
                    <!-- 通道选择器将在这里动态生成 -->
                    <div>加载中...</div>
                </div>
            </div>
            
            <div class="refresh-controls">
                <button id="refresh-btn">刷新可视化</button>
                <div class="auto-refresh">
                    <input type="checkbox" id="auto-refresh">
                    <label for="auto-refresh">自动刷新</label>
                    <select id="refresh-interval">
                        <option value="1000">1秒</option>
                        <option value="2000" selected>2秒</option>
                        <option value="5000">5秒</option>
                        <option value="10000">10秒</option>
                    </select>
                </div>
            </div>
        </div>
    `;
    
    // 创建可视化面板
    const visualizationHTML = `
        <div class="visualization-panel">
            <div class="viz-card">
                <h3 id="viz-title">EEG 可视化</h3>
                <div class="viz-container">
                    <img id="viz-image" class="viz-image" src="" alt="EEG Visualization">
                    <div class="overlay-controls">
                        <button class="overlay-btn" id="download-btn">下载</button>
                    </div>
                </div>
                <div class="status" id="status-message">准备就绪</div>
            </div>
        </div>
    `;
    
    // 组合UI
    appContainer.innerHTML = controlsHTML + visualizationHTML;
}

// 加载数据源
async function loadDataSources() {
    try {
        const response = await fetch('/api/data_sources');
        const data = await response.json();
        
        if (data.success) {
            dataSources = data.sources;
            activeSource = data.active_source;
            
            // 更新数据源选择器
            updateDataSourceSelector();
            
            // 如果有数据源，加载第一个数据源的通道
            if (dataSources.length > 0) {
                // 从数据源获取实际的通道列表
                const sourceData = dataSources.find(source => source.id === activeSource);
                if (sourceData && sourceData.channel_names) {
                    allChannels = sourceData.channel_names;
                } else {
                    // 如果没有通道信息，使用空数组
                    allChannels = [];
                }
                selectedChannels = [...allChannels]; // 默认选择所有通道
                updateChannelSelector();
            }
        } else {
            showStatusMessage('加载数据源失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('加载数据源出错:', error);
        showStatusMessage('加载数据源出错: ' + error.message, 'error');
    }
}

// 更新数据源选择器
function updateDataSourceSelector() {
    const selector = document.getElementById('data-source-selector');
    selector.innerHTML = '';
    
    dataSources.forEach(source => {
        const option = document.createElement('option');
        option.value = source.id;
        option.textContent = `${source.name} (${source.subject}) - ${source.channel_count}通道`;
        selector.appendChild(option);
    });
    
    // 设置当前活动数据源
    if (activeSource) {
        selector.value = activeSource;
    }
}

// 更新通道选择器
function updateChannelSelector() {
    const container = document.getElementById('channel-selector');
    container.innerHTML = '';
    
    // 添加全选/取消全选按钮
    const selectAllContainer = document.createElement('div');
    selectAllContainer.className = 'channel-checkbox';
    
    const selectAllCheckbox = document.createElement('input');
    selectAllCheckbox.type = 'checkbox';
    selectAllCheckbox.id = 'select-all-channels';
    selectAllCheckbox.checked = selectedChannels.length === allChannels.length;
    
    const selectAllLabel = document.createElement('label');
    selectAllLabel.htmlFor = 'select-all-channels';
    selectAllLabel.textContent = '全选/取消全选';
    
    selectAllContainer.appendChild(selectAllCheckbox);
    selectAllContainer.appendChild(selectAllLabel);
    container.appendChild(selectAllContainer);
    
    // 添加通道复选框
    allChannels.forEach(channel => {
        const channelContainer = document.createElement('div');
        channelContainer.className = 'channel-checkbox';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `channel-${channel}`;
        checkbox.value = channel;
        checkbox.checked = selectedChannels.includes(channel);
        
        const label = document.createElement('label');
        label.htmlFor = `channel-${channel}`;
        label.textContent = channel;
        
        channelContainer.appendChild(checkbox);
        channelContainer.appendChild(label);
        container.appendChild(channelContainer);
    });
    
    // 添加全选/取消全选事件
    document.getElementById('select-all-channels').addEventListener('change', function() {
        const checked = this.checked;
        document.querySelectorAll('#channel-selector input[type="checkbox"]').forEach(checkbox => {
            if (checkbox.id !== 'select-all-channels') {
                checkbox.checked = checked;
            }
        });
        
        // 更新选中的通道
        updateSelectedChannels();
    });
    
    // 添加通道选择事件
    document.querySelectorAll('#channel-selector input[type="checkbox"]').forEach(checkbox => {
        if (checkbox.id !== 'select-all-channels') {
            checkbox.addEventListener('change', function() {
                updateSelectedChannels();
                
                // 更新全选按钮状态
                const allChecked = selectedChannels.length === allChannels.length;
                document.getElementById('select-all-channels').checked = allChecked;
            });
        }
    });
}

// 更新选中的通道
function updateSelectedChannels() {
    selectedChannels = [];
    document.querySelectorAll('#channel-selector input[type="checkbox"]').forEach(checkbox => {
        if (checkbox.id !== 'select-all-channels' && checkbox.checked) {
            selectedChannels.push(checkbox.value);
        }
    });
}

// 初始化事件监听器
function initEventListeners() {
    // 数据源选择事件
    document.getElementById('data-source-selector').addEventListener('change', function() {
        activeSource = this.value;
        refreshVisualization();
    });
    
    // 标签切换事件
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            // 更新活动标签
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // 更新活动内容
            const tabId = this.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById(`${tabId}-tab`).classList.add('active');
            
            // 更新当前可视化类型
            currentVisualizationType = tabId;
            
            // 刷新可视化
            refreshVisualization();
        });
    });
    
    // 时间窗口滑块事件
    document.getElementById('time-window').addEventListener('input', function() {
        document.getElementById('time-window-value').textContent = this.value;
    });
    
    document.getElementById('combined-time-window').addEventListener('input', function() {
        document.getElementById('combined-time-window-value').textContent = this.value;
    });
    
    // 频率范围滑块事件
    document.getElementById('freq-range').addEventListener('input', function() {
        document.getElementById('freq-max').value = this.value;
    });
    
    document.getElementById('freq-min').addEventListener('input', function() {
        if (parseInt(this.value) >= parseInt(document.getElementById('freq-max').value)) {
            document.getElementById('freq-max').value = parseInt(this.value) + 1;
            document.getElementById('freq-range').value = document.getElementById('freq-max').value;
        }
    });
    
    document.getElementById('freq-max').addEventListener('input', function() {
        document.getElementById('freq-range').value = this.value;
        if (parseInt(this.value) <= parseInt(document.getElementById('freq-min').value)) {
            document.getElementById('freq-min').value = parseInt(this.value) - 1;
        }
    });
    
    // 刷新按钮事件
    document.getElementById('refresh-btn').addEventListener('click', function() {
        refreshVisualization();
    });
    
    // 自动刷新事件
    document.getElementById('auto-refresh').addEventListener('change', function() {
        if (this.checked) {
            const interval = parseInt(document.getElementById('refresh-interval').value);
            autoRefreshInterval = setInterval(refreshVisualization, interval);
        } else {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
    });
    
    // 刷新间隔事件
    document.getElementById('refresh-interval').addEventListener('change', function() {
        if (document.getElementById('auto-refresh').checked) {
            clearInterval(autoRefreshInterval);
            const interval = parseInt(this.value);
            autoRefreshInterval = setInterval(refreshVisualization, interval);
        }
    });
    
    // 下载按钮事件
    document.getElementById('download-btn').addEventListener('click', function() {
        const img = document.getElementById('viz-image');
        if (img.src) {
            const a = document.createElement('a');
            a.href = img.src;
            a.download = `eeg_${currentVisualizationType}_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
            a.click();
        }
    });
}

// 刷新可视化
async function refreshVisualization(preloadedData = null) {
    if (!activeSource) {
        showStatusMessage('请选择数据源', 'warning');
        return;
    }
    
    if (selectedChannels.length === 0) {
        showStatusMessage('请至少选择一个通道', 'warning');
        return;
    }
    
    // 如果是WebSocket推送的数据更新，可以直接使用预加载的数据
    if (preloadedData) {
        updateVisualizationWithData(preloadedData);
        return;
    }
    
    showStatusMessage('正在加载可视化...', 'info');
    const startTime = performance.now();
    
    try {
        // 准备请求参数
        const params = {
            source_id: activeSource, // 添加数据源ID参数
            visualization_type: currentVisualizationType,
            channels: selectedChannels,
            source_id: activeSource
        };
        
        // 根据可视化类型添加特定参数
        if (currentVisualizationType === 'time') {
            params.time_window = parseFloat(document.getElementById('time-window').value);
            params.baseline_correction = document.getElementById('baseline-correction').checked;
        } else if (currentVisualizationType === 'frequency') {
            params.freq_min = parseInt(document.getElementById('freq-min').value);
            params.freq_max = parseInt(document.getElementById('freq-max').value);
        } else if (currentVisualizationType === 'topographic') {
            params.freq_band = document.getElementById('freq-band').value;
        } else if (currentVisualizationType === 'combined') {
            params.time_window = parseFloat(document.getElementById('combined-time-window').value);
            params.freq_band = document.getElementById('combined-freq-band').value;
        }
        
        // 发送请求
        const response = await fetch('/api/visualize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateVisualizationWithData(data);
            
            // 计算并显示响应时间
            const endTime = performance.now();
            const responseTime = (endTime - startTime).toFixed(2);
            showStatusMessage(`可视化已更新 (${responseTime}ms)`, 'success');
        } else {
            showStatusMessage('可视化失败: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('可视化请求出错:', error);
        showStatusMessage('可视化请求出错: ' + error.message, 'error');
    }
}

// 显示状态消息
function showStatusMessage(message, type = 'info') {
    const statusElement = document.getElementById('status-message');
    statusElement.textContent = message;
    
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

// 更新可视化数据
function updateVisualizationWithData(data) {
    // 更新可视化图像
    if (data.image) {
        document.getElementById('viz-image').src = 'data:image/png;base64,' + data.image;
    }
    
    // 更新标题
    let title = '';
    if (currentVisualizationType === 'time') {
        title = '时域可视化';
    } else if (currentVisualizationType === 'frequency') {
        title = '频域可视化';
    } else if (currentVisualizationType === 'topographic') {
        title = '脑地形图可视化';
    } else if (currentVisualizationType === 'combined') {
        title = '组合可视化';
    }
    document.getElementById('viz-title').textContent = title;
    
    // 如果有延迟信息，显示在状态栏
    if (data.latency) {
        const latencyElement = document.getElementById('latency-info');
        if (!latencyElement) {
            const newLatencyElement = document.createElement('div');
            newLatencyElement.id = 'latency-info';
            newLatencyElement.className = 'status';
            document.querySelector('.viz-card').appendChild(newLatencyElement);
        }
        document.getElementById('latency-info').textContent = `端到端延迟: ${data.latency.toFixed(2)}ms`;
    }
    
    // 如果有响应时间信息，显示在状态消息中
    if (data.response_time) {
        showStatusMessage(`可视化已更新 (${data.response_time})`, 'success');
    }
}