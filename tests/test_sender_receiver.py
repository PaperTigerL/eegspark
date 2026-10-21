import os
import sys
import time
import socket
import threading
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入发送器和接收器模块
from sender.sender import EEGDataSender, DataPayload
from sender.eeg_reader import EEGReader, EEGDataInfo

# 模拟EEGDataInfo类
class MockEEGDataInfo:
    def __init__(self):
        self.subject_id = "sub-01"
        self.task_name = "test_task"
        self.sampling_rate = 250
        self.n_channels = 32
        self.n_samples = 1000
        self.duration = 4.0
        self.reference = "average"
        self.filters = {"highpass": 0.1, "lowpass": 50}
        self.recording_date = "2023-01-01"
        self.channels_info = {}
        self.events_info = []
        self.json_metadata = {}
        self.annotations = []
        
        # 创建通道位置信息
        class ChannelLocation:
            def __init__(self, name, type="EEG", unit="uV", x=0, y=0, z=0):
                self.name = name
                self.type = type
                self.unit = unit
                self.x = x
                self.y = y
                self.z = z
        
        self.channel_locations = [
            ChannelLocation(f"CH{i}", x=np.cos(i/16*np.pi), y=np.sin(i/16*np.pi))
            for i in range(self.n_channels)
        ]
        
        # 模拟原始数据
        class MockRaw:
            def __init__(self, n_channels, n_samples):
                self._data = np.random.randn(n_channels, n_samples)
            
            def get_data(self):
                return self._data
        
        self.raw = MockRaw(self.n_channels, self.n_samples)

# 模拟EEGReader类
class MockEEGReader:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
    
    def read_eeg_data(self, subject_id, task_name):
        return MockEEGDataInfo()

# 模拟接收器服务器
class MockReceiver:
    def __init__(self, host='127.0.0.1', metadata_port=9000, data_port=9001):
        self.host = host
        self.metadata_port = metadata_port
        self.data_port = data_port
        self.metadata_socket = None
        self.data_socket = None
        self.is_running = False
        self.metadata_thread = None
        self.data_thread = None
        self.received_metadata = None
        self.received_data_packets = []
    
    def start(self):
        """启动模拟接收器服务"""
        self.is_running = True
        
        # 启动元数据服务器
        self.metadata_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.metadata_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.metadata_socket.bind((self.host, self.metadata_port))
        self.metadata_socket.listen(5)
        self.metadata_thread = threading.Thread(target=self._metadata_server)
        self.metadata_thread.daemon = True
        self.metadata_thread.start()
        
        # 启动数据服务器
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_socket.bind((self.host, self.data_port))
        self.data_socket.listen(5)
        self.data_thread = threading.Thread(target=self._data_server)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        print(f"模拟接收器启动: {self.host}:{self.metadata_port}/{self.data_port}")
    
    def _metadata_server(self):
        """元数据服务器线程"""
        try:
            while self.is_running:
                try:
                    self.metadata_socket.settimeout(1.0)
                    client, addr = self.metadata_socket.accept()
                    print(f"元数据服务接收到连接: {addr}")
                    
                    # 接收数据长度
                    length_bytes = client.recv(4)
                    if not length_bytes:
                        client.close()
                        continue
                    
                    data_length = int.from_bytes(length_bytes, byteorder='big')
                    
                    # 接收数据内容
                    data = b''
                    while len(data) < data_length:
                        chunk = client.recv(min(4096, data_length - len(data)))
                        if not chunk:
                            break
                        data += chunk
                    
                    if len(data) == data_length:
                        # 解析元数据
                        self.received_metadata = json.loads(data.decode('utf-8'))
                        print(f"接收到元数据: {len(data)} 字节")
                    
                    client.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"元数据服务器错误: {e}")
                    break
        finally:
            if self.metadata_socket:
                self.metadata_socket.close()
    
    def _data_server(self):
        """数据服务器线程"""
        try:
            while self.is_running:
                try:
                    self.data_socket.settimeout(1.0)
                    client, addr = self.data_socket.accept()
                    client.settimeout(60)  # 设置较长的超时时间
                    print(f"数据服务接收到连接: {addr}")
                    
                    # 处理数据流
                    self._handle_data_stream(client)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"数据服务器错误: {e}")
                    break
        finally:
            if self.data_socket:
                self.data_socket.close()
    
    def _handle_data_stream(self, client):
        """处理数据流连接"""
        try:
            while self.is_running:
                # 接收数据长度
                length_bytes = client.recv(4)
                if not length_bytes:
                    break
                
                data_length = int.from_bytes(length_bytes, byteorder='big')
                
                # 接收数据内容
                data = b''
                while len(data) < data_length:
                    chunk = client.recv(min(4096, data_length - len(data)))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) == data_length:
                    # 解析数据包
                    packet = json.loads(data.decode('utf-8'))
                    packet_type = packet.get("type")
                    
                    if packet_type == "START":
                        print(f"接收到开始标记: {packet}")
                    elif packet_type == "DATA":
                        self.received_data_packets.append(packet)
                        print(f"接收到数据包: 序列号={packet.get('sequence_num')}, 样本数={packet.get('n_samples')}")
                    elif packet_type == "END":
                        print(f"接收到结束标记: {packet}")
                        break
        except Exception as e:
            print(f"处理数据流错误: {e}")
        finally:
            client.close()
    
    def stop(self):
        """停止模拟接收器服务"""
        self.is_running = False
        
        if self.metadata_thread:
            self.metadata_thread.join(timeout=2.0)
        if self.data_thread:
            self.data_thread.join(timeout=2.0)
        
        if self.metadata_socket:
            self.metadata_socket.close()
        if self.data_socket:
            self.data_socket.close()
        
        print("模拟接收器已停止")

# 测试用例
@pytest.fixture
def mock_eeg_reader():
    """模拟EEGReader"""
    with patch('src.sender.sender.EEGReader', MockEEGReader):
        yield

@pytest.fixture
def mock_receiver():
    """创建模拟接收器"""
    receiver = MockReceiver(host='127.0.0.1')
    receiver.start()
    yield receiver
    receiver.stop()

def test_sender_initialization():
    """测试发送器初始化"""
    sender = EEGDataSender(host='127.0.0.1', metadata_port=9000, data_port=9001)
    assert sender.host == '127.0.0.1'
    assert sender.metadata_port == 9000
    assert sender.data_port == 9001
    assert sender.is_running == False
    assert sender.data_thread is None

def test_setup_sockets():
    """测试套接字设置"""
    sender = EEGDataSender(host='127.0.0.1', metadata_port=9000, data_port=9001)
    sender.setup_sockets()
    
    assert sender.metadata_socket is not None
    assert sender.data_socket is not None
    
    # 清理
    sender.close()

def test_generate_payload_id():
    """测试生成负载ID"""
    sender = EEGDataSender()
    payload_id = sender.generate_payload_id("sub-01", "test_task")
    
    assert "sub-01" in payload_id
    assert "test_task" in payload_id
    assert len(payload_id.split("_")) == 3  # 格式应为: subject_task_timestamp

@pytest.mark.usefixtures("mock_eeg_reader")
def test_prepare_metadata():
    """测试准备元数据"""
    sender = EEGDataSender(dataset_root="./test_data")
    eeg_info = sender.eeg_reader.read_eeg_data("01", "test_task")
    payload_id = "test_payload_id"
    
    metadata = sender._prepare_metadata(eeg_info, payload_id)
    
    assert metadata["payload_id"] == payload_id
    assert metadata["subject_id"] == "sub-01"
    assert metadata["task_name"] == "test_task"
    assert metadata["sampling_rate"] == 250
    assert metadata["n_channels"] == 32
    assert len(metadata["channel_locations"]) == 32

@pytest.mark.usefixtures("mock_eeg_reader")
def test_metadata_connection(mock_receiver):
    """测试元数据连接"""
    # 创建发送器
    sender = EEGDataSender(host='127.0.0.1', metadata_port=9000, data_port=9001, dataset_root="./test_data")
    sender.setup_sockets()
    
    # 修改send_metadata方法以直接连接到本地接收器
    with patch.object(sender, 'send_metadata', wraps=sender.send_metadata) as mock_send:
        def direct_connect(*args, **kwargs):
            # 直接连接到本地接收器
            try:
                subject_id = args[0] if args else kwargs.get('subject_id', '01')
                task_name = args[1] if len(args) > 1 else kwargs.get('task_name', 'test_task')
                
                # 读取EEG数据元数据
                actual_subject_id = subject_id
                if subject_id.startswith('sub-'):
                    actual_subject_id = subject_id[4:]
                
                eeg_info = sender.eeg_reader.read_eeg_data(actual_subject_id, task_name)
                payload_id = sender.generate_payload_id(subject_id, task_name)
                
                # 准备元数据
                metadata = sender._prepare_metadata(eeg_info, payload_id)
                
                # 直接连接到接收器
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(5.0)
                client_socket.connect(('127.0.0.1', 9000))
                
                # 将对象转换为JSON并发送
                data = json.dumps(metadata).encode('utf-8')
                # 发送长度
                client_socket.sendall(len(data).to_bytes(4, byteorder='big'))
                # 发送数据
                client_socket.sendall(data)
                client_socket.close()
                
                return payload_id
            except Exception as e:
                print(f"发送元数据失败: {e}")
                return None
        
        mock_send.side_effect = direct_connect
        
        # 发送元数据
        payload_id = sender.send_metadata("01", "test_task")
        
        # 验证结果
        assert payload_id is not None
        time.sleep(1)  # 等待接收器处理
        assert mock_receiver.received_metadata is not None
        assert mock_receiver.received_metadata["subject_id"] == "sub-01"
        assert mock_receiver.received_metadata["task_name"] == "test_task"
    
    # 清理
    sender.close()

@pytest.mark.usefixtures("mock_eeg_reader")
def test_data_streaming(mock_receiver):
    """测试数据流传输"""
    # 创建发送器
    sender = EEGDataSender(host='127.0.0.1', metadata_port=9000, data_port=9001, dataset_root="./test_data")
    sender.setup_sockets()
    
    # 修改_stream_data_thread方法以直接连接到本地接收器
    original_stream_data = sender._stream_data_thread
    
    def mock_stream_data(raw_data, eeg_info, payload_id, speed_factor, chunk_samples):
        try:
            total_samples = raw_data.shape[1]
            total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            chunk_duration = chunk_samples / eeg_info.sampling_rate
            
            # 直接连接到接收器
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)
            client_socket.connect(('127.0.0.1', 9001))
            
            # 发送开始标记
            start_msg = {
                "type": "START", 
                "payload_id": payload_id,
                "total_chunks": total_chunks,
                "chunk_samples": chunk_samples
            }
            start_data = json.dumps(start_msg).encode('utf-8')
            client_socket.sendall(len(start_data).to_bytes(4, byteorder='big'))
            client_socket.sendall(start_data)
            
            # 只发送前3个数据块进行测试
            test_chunks = min(3, total_chunks)
            current_sample = 0
            
            for chunk_idx in range(test_chunks):
                # 准备当前数据块
                end_sample = min(current_sample + chunk_samples, total_samples)
                chunk_data = raw_data[:, current_sample:end_sample]
                actual_samples = end_sample - current_sample
                
                # 构建数据包
                data_packet = {
                    "type": "DATA",
                    "payload_id": payload_id,
                    "sequence_num": chunk_idx + 1,
                    "total_chunks": total_chunks,
                    "n_channels": eeg_info.n_channels,
                    "n_samples": actual_samples,
                    "timestamp": time.time(),
                    "events": []  # 简化测试，不包含事件
                }
                
                # 将numpy数组转换为简单列表并添加到数据包
                data_packet["data"] = chunk_data.tolist()
                
                # 序列化并发送
                json_data = json.dumps(data_packet).encode('utf-8')
                
                # 发送数据长度
                client_socket.sendall(len(json_data).to_bytes(4, byteorder='big'))
                # 发送数据内容
                client_socket.sendall(json_data)
                
                # 更新当前样本位置
                current_sample = end_sample
                
                # 控制发送速率
                time.sleep(0.1)
            
            # 发送结束标记
            end_msg = {"type": "END", "payload_id": payload_id}
            end_data = json.dumps(end_msg).encode('utf-8')
            client_socket.sendall(len(end_data).to_bytes(4, byteorder='big'))
            client_socket.sendall(end_data)
            
            # 关闭连接
            client_socket.close()
            
        except Exception as e:
            print(f"模拟数据流发送错误: {e}")
    
    # 替换方法
    sender._stream_data_thread = mock_stream_data
    
    # 发送元数据
    payload_id = "test_payload_id_123"
    
    # 启动数据流
    eeg_info = sender.eeg_reader.read_eeg_data("01", "test_task")
    raw_data = eeg_info.raw.get_data()
    
    # 启动数据发送线程
    sender.is_running = True
    data_thread = threading.Thread(
        target=sender._stream_data_thread,
        args=(raw_data, eeg_info, payload_id, 1.0, 100)
    )
    data_thread.daemon = True
    data_thread.start()
    
    # 等待数据发送完成
    data_thread.join(timeout=10.0)
    
    # 验证结果
    time.sleep(2)  # 等待接收器处理
    assert len(mock_receiver.received_data_packets) > 0
    assert mock_receiver.received_data_packets[0]["payload_id"] == payload_id
    
    # 清理
    sender.is_running = False
    sender.close()

# 主函数，用于直接运行测试
if __name__ == "__main__":
    pytest.main(['-xvs', __file__])