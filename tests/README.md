# EEG数据发送和接收测试模块

本测试模块用于验证EEG数据发送器和接收器之间的连接功能，确保数据能够正确传输。

## 修改说明

为了解决发送端和接收端连接建立问题，我们进行了以下改进：

1. 在`sender.py`中添加了直接指定接收器地址的参数：
   - `receiver_host`: 接收器主机地址
   - `receiver_metadata_port`: 接收器元数据端口
   - `receiver_data_port`: 接收器数据端口

2. 改进了连接逻辑，增加了对本地地址（localhost/127.0.0.1）的支持

3. 创建了完整的测试模块，使用pytest进行单元测试

## 测试模块使用方法

### 安装依赖

确保已安装pytest：

```bash
pip install pytest
```

### 运行测试

在项目根目录下运行：

```bash
python -m pytest tests/test_sender_receiver.py -v
```

或者直接运行测试文件：

```bash
python tests/test_sender_receiver.py
```

## 测试内容

测试模块包含以下测试用例：

1. `test_sender_initialization`: 测试发送器初始化
2. `test_setup_sockets`: 测试套接字设置
3. `test_generate_payload_id`: 测试生成负载ID
4. `test_prepare_metadata`: 测试准备元数据
5. `test_metadata_connection`: 测试元数据连接
6. `test_data_streaming`: 测试数据流传输

## 使用修改后的发送器

现在可以通过命令行参数直接指定接收器地址：

```bash
python src/sender/sender.py --receiver-host 127.0.0.1 --receiver-metadata-port 9000 --receiver-data-port 9001 --subject-id 01 --task resting
```

## 故障排除

如果连接仍然存在问题，请检查：

1. 确保接收器已经启动并正在监听指定的端口
2. 检查防火墙设置，确保允许指定端口的通信
3. 如果在不同机器上运行，确保网络连接正常
4. 使用`--receiver-host`参数直接指定接收器的IP地址

## 模拟测试

测试模块包含一个`MockReceiver`类，可以模拟接收器行为进行测试，无需启动实际的接收器服务。这对于开发和调试非常有用。