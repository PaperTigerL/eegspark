import os
import socket
import json
import numpy as np
from typing import Dict, List, Optional
# 修改导入方式
from common.eegrdd import EEGRDD

class MetadataReceiver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metadata = None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    def start(self, host='data-sender', port=9001):
        """启动元数据接收服务"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"连接元数据服务 {host}:{port}")
            sock.connect((host, port))
            
            metadata = self._receive_metadata(sock)
            if metadata:
                self._save_metadata(metadata)
                print("元数据已保存")
                return metadata
                
        except Exception as e:
            print(f"元数据接收错误: {e}")
        finally:
            sock.close()
            
    def _receive_metadata(self, sock):
        """接收元数据"""
        buffer = b""
        while True:
            chunk = sock.recv(1024)
            if not chunk:
                return None
                
            buffer += chunk
            if b"\n" in buffer:
                # 解析JSON数据
                try:
                    metadata = json.loads(buffer.decode('utf-8').strip())
                    return metadata
                except json.JSONDecodeError as e:
                    print(f"元数据JSON解析错误: {e}")
                    return None
                    
    def _save_metadata(self, metadata):
        """保存元数据到文件"""
        json_path = os.path.join(self.save_dir, 
                               f"sub-{metadata['subject_id']}_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

class EEGDataReceiver:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.metadata = None
        self.processors = []
        self.chunk_size = 500
        
    def add_processor(self, processor):
        """添加数据处理器"""
        self.processors.append(processor)
        
    def start(self, metadata: Dict, host='data-sender', port=9000):
        self.metadata = metadata
        self.host = host
        self.port = port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            self._receive_data(sock)
        except Exception as e:
            print(f"[错误] 连接失败: {e}")
        finally:
            sock.close()

    def _receive_data(self, sock: socket.socket) -> None:
        HEADER_SIZE = 32
        try:
            while True:
                header_bytes = sock.recv(HEADER_SIZE)
                if not header_bytes:
                    break
                    
                try:
                    header = header_bytes.decode('ascii').strip()
                    parts = header.split(':')
                    
                    if len(parts) != 4 or parts[0] != 'EEG':
                        continue
                        
                    _, subject_id, chunk_idx, channel = parts
                    chunk_idx = int(chunk_idx)
                    channel = int(channel)
                    
                    length_bytes = sock.recv(4)
                    data_size = int.from_bytes(length_bytes, byteorder='big')
                    data_bytes = self._read_exact(sock, data_size)
                    
                    data_array = np.frombuffer(data_bytes, dtype=np.float32)
                    data_array = data_array.reshape(-1, 2)
                    time_points = data_array[:, 0]
                    voltage_data = data_array[:, 1]
                    
                    eegrdd = EEGRDD(
                        subject_id=subject_id,
                        time_points=time_points,
                        voltage_data=voltage_data,
                        sampling_rate=self.metadata['sampling_rate'],
                        channel=channel,
                        chunk_idx=chunk_idx
                    )
                    
                    for processor in self.processors:
                        processor.process(eegrdd)
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"[错误] 连接中断: {e}")
            
    def _read_exact(self, sock, size):
        """精确读取指定字节数的数据"""
        data = b''
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                raise ConnectionError("连接断开")
            data += packet
        return data

    def _create_eegrdd_realtime(self, data: np.ndarray) -> None:
        try:
            eegrdd = EEGRDD.from_metadata(self.metadata, data)
            self.eegrdd_list.append(eegrdd)
            # 实时处理新生成的EEGRDD
            for processor in self.processors:
                processor.process(eegrdd)
        except Exception as e:
            print(f"[错误] 创建或处理EEGRDD失败: {e}")  # 保留错误输出

    def get_latest_eegrdd(self) -> Optional[EEGRDD]:
        """获取最新的EEGRDD实例"""
        return self.eegrdd_list[-1] if self.eegrdd_list else None
