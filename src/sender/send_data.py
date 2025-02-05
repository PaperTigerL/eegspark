import socket
import json
import os
import numpy as np
import pandas as pd
import time
import threading

class EEGDataLoader:
    def __init__(self, data_dir, subject_id='01', task_name='resting'):
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.task_name = task_name
        self.eeg_data = None
        self.metadata = {}
        self.chunk_size = None  # 添加chunk_size属性
        self.load_data()
    
    def load_data(self):
        """加载EEG数据"""
        eeg_dir = os.path.join(self.data_dir, f'sub-{self.subject_id}', 'eeg')
        
        # 1. 读取元数据和通道信息
        vhdr_path = os.path.join(eeg_dir, f"sub-{self.subject_id}_task-{self.task_name}_eeg.vhdr")
        channels = []
        sampling_rate = None
        
        with open(vhdr_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'SamplingInterval=' in line:
                    sampling_rate = 1000000/float(line.split('=')[1])
                    # 根据采样率计算chunk_size: 采样率的1/10秒数据
                    self.chunk_size = int(sampling_rate / 10)
                if line.startswith('Ch') and '=' in line:
                    ch_info = line.split('=')[1].split(',')
                    channels.append({
                        'name': ch_info[0],
                        'unit': ch_info[3] if len(ch_info) > 3 else 'µV',
                        'type': 'EEG',
                        'scaling': float(ch_info[2]) if len(ch_info) > 2 else 1.0
                    })
        
        # 2. 读取EEG数据
        eeg_path = os.path.join(eeg_dir, f"sub-{self.subject_id}_task-{self.task_name}_eeg.eeg")
        raw_data = np.fromfile(eeg_path, dtype=np.float32)
        n_channels = len(channels)
        n_samples = len(raw_data) // n_channels
        self.eeg_data = raw_data.reshape(n_samples, n_channels)
        
        # 应用缩放因子
        for i, ch in enumerate(channels):
            self.eeg_data[:,i] *= ch['scaling']
        
        # 构建元数据
        self.metadata = {
            'subject_id': self.subject_id,
            'task_name': self.task_name,
            'sampling_rate': float(sampling_rate),
            'n_channels': n_channels,
            'n_samples': n_samples,
            'duration': n_samples / sampling_rate,
            'data_range': [float(self.eeg_data.min()), float(self.eeg_data.max())],
            'channels': channels
        }

class MetadataServer:
    def __init__(self, data_dir, port=9001):
        self.data_dir = data_dir
        self.port = port
        self.running = False
        
    def start(self):
        """启动元数据服务器"""
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
    def _run_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.port))
        sock.listen(1)
        print(f'元数据服务已启动在端口 {self.port}')
        
        while self.running:
            try:
                conn, addr = sock.accept()
                print(f'元数据服务收到连接: {addr}')
                self._handle_client(conn)
            except Exception as e:
                print(f'元数据服务错误: {e}')
            finally:
                if 'conn' in locals():
                    conn.close()
                    
    def _handle_client(self, conn):
        try:
            metadata = self._load_metadata()
            conn.send(json.dumps(metadata).encode() + b'\n')
        except Exception as e:
            print(f'处理元数据请求时出错: {e}')
            
    def _load_metadata(self):
        # 加载元数据逻辑...
        return EEGDataLoader(self.data_dir).metadata

class EEGDataStream:
    def __init__(self, data_dir, port=9000):
        self.data_dir = data_dir
        self.port = port
        self.loader = EEGDataLoader(data_dir)
        
    def start_streaming(self):
        """开始数据流传输"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.port))
        sock.listen(1)
        print(f'EEG数据流服务已启动在端口 {self.port}')
        
        while True:
            try:
                conn, addr = sock.accept()
                print(f'数据流服务收到连接: {addr}')
                self._stream_data(conn)
            except Exception as e:
                print(f'数据流服务错误: {e}')
            finally:
                if 'conn' in locals():
                    conn.close()
                    
    def _stream_data(self, conn):
        """发送EEG数据流 - 时间-范围值数据对"""
        try:
            chunk_size = self.loader.chunk_size
            n_channels = self.loader.metadata['n_channels']
            total_chunks = self.loader.eeg_data.shape[0] // chunk_size
            sampling_rate = self.loader.metadata['sampling_rate']
            
            #print(f"\n[发送配置] 通道数: {n_channels}, 采样率: {sampling_rate}Hz")
            
            for chunk_idx in range(total_chunks):
                for ch in range(n_channels):
                    start_idx = chunk_idx * chunk_size
                    end_idx = start_idx + chunk_size
                    time_points = np.arange(start_idx, end_idx) / sampling_rate
                    voltage_data = self.loader.eeg_data[start_idx:end_idx, ch]
                    
                    # 将电压范围值作为一个数据点
                    range_data = voltage_data  # 保持原始范围值
                    
                    # 创建时间-范围值数据对
                    data_pairs = np.column_stack((time_points, range_data)).astype(np.float32)
                    
                    # 构建数据包头
                    header = f"EEG:{self.loader.subject_id}:{chunk_idx}:{ch}".ljust(32)
                    header_bytes = header.encode('ascii')
                    
                    # 数据长度标记
                    data_bytes = data_pairs.tobytes()
                    length_bytes = len(data_bytes).to_bytes(4, byteorder='big')
                    
                    # 发送数据包
                    packet = header_bytes + length_bytes + data_bytes
                    conn.sendall(packet)
                    
                    
                
                time.sleep(chunk_size / sampling_rate)
                
        except Exception as e:
            print(f'[错误] 数据流发送失败: {e}')

def main():
    data_dir = os.environ.get('DATA_DIR', '/app/data/ds005779')
    
    # 启动元数据服务
    metadata_server = MetadataServer(data_dir)
    metadata_server.start()
    
    # 启动数据流服务
    stream_server = EEGDataStream(data_dir)
    stream_server.start_streaming()

if __name__ == '__main__':
    main()