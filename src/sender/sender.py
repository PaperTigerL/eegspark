import os
import time
import socket
import threading
import json
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
import argparse
from eeg_reader import EEGReader, EEGDataInfo


@dataclass
class DataPayload:
    """数据传输负载结构"""
    payload_id: str          # 唯一标识符，用于关联元数据和数据
    subject_id: str          # 受试者ID
    task_name: str           # 任务名称
    timestamp: float         # 发送时间戳
    payload_type: str        # 'metadata'或'eeg_data'
    content: Any             # 实际内容(元数据或EEG数据)
    sequence_num: int = 0    # 序列号(用于分块传输)
    total_chunks: int = 1    # 总块数


class EEGDataSender:
    """EEG数据发送器"""
    def __init__(self, host: str = '0.0.0.0', 
                 metadata_port: int = 9000, 
                 data_port: int = 9001, 
                 dataset_root: str = "/app/data/ds005779"):
        """初始化EEG数据发送器
        
        Args:
            host: 监听主机地址
            metadata_port: 元数据服务端口
            data_port: 数据服务端口
            dataset_root: 数据集根目录路径
        """
        self.host = host
        self.metadata_port = metadata_port
        self.data_port = data_port
        self.dataset_root = dataset_root
        self.eeg_reader = EEGReader(dataset_root)
        self.metadata_socket = None
        self.data_socket = None
        self.is_running = False
        self.data_thread = None
    
    def setup_sockets(self):
        """设置Socket连接"""
        # 设置元数据socket
        self.metadata_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.metadata_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.metadata_socket.bind((self.host, self.metadata_port))
        self.metadata_socket.listen(5)
        print(f"元数据服务监听 {self.host}:{self.metadata_port}")
        
        # 设置数据socket
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.data_socket.bind((self.host, self.data_port))
        self.data_socket.listen(5)
        print(f"数据服务监听 {self.host}:{self.data_port}")
    
    def generate_payload_id(self, subject_id: str, task_name: str) -> str:
        """生成唯一的数据负载ID"""
        timestamp = int(time.time())
        return f"{subject_id}_{task_name}_{timestamp}"
    
    def send_metadata(self, subject_id: str, task_name: str):
        """发送EEG元数据
        
        Args:
            subject_id: 受试者ID
            task_name: 任务名称
        """
        try:
            # 读取EEG数据元数据
            print(f"正在读取受试者 {subject_id} 的 {task_name} 任务元数据...")
            eeg_info = self.eeg_reader.read_eeg_data(subject_id, task_name)
            payload_id = self.generate_payload_id(subject_id, task_name)
            
            # 准备元数据
            metadata = self._prepare_metadata(eeg_info, payload_id)
            
            # 等待连接并发送元数据
            print("等待接收端连接元数据服务...")
            client, addr = self.metadata_socket.accept()
            print(f"元数据服务接收到连接：{addr}")
            
            # 将对象转换为JSON并发送
            data = json.dumps(metadata).encode('utf-8')
            # 发送长度
            client.sendall(len(data).to_bytes(4, byteorder='big'))
            # 发送数据
            client.sendall(data)
            client.close()
            
            print(f"元数据已发送: {len(data)} 字节")
            return payload_id
            
        except Exception as e:
            print(f"发送元数据失败: {e}")
            return None
    
    def _prepare_metadata(self, eeg_info: EEGDataInfo, payload_id: str) -> Dict:
        """准备要发送的元数据"""
        # 从EEGDataInfo中提取所有不包含原始数据的信息
        metadata = {
            "payload_id": payload_id,
            "subject_id": eeg_info.subject_id,
            "task_name": eeg_info.task_name,
            "sampling_rate": eeg_info.sampling_rate,
            "n_channels": eeg_info.n_channels,
            "n_samples": eeg_info.n_samples,
            "duration": eeg_info.duration,
            "reference": eeg_info.reference,
            "filters": eeg_info.filters,
            # 直接使用字符串，不要再调用strftime
            "recording_date": eeg_info.recording_date,  # 已经在EEGReader中处理为字符串
            
            # 添加安全检查，如果属性不存在则使用默认值
            "channels_info": getattr(eeg_info, 'channels_info', {}),
            "events_info": getattr(eeg_info, 'events_info', []),
            "json_metadata": getattr(eeg_info, 'json_metadata', {}),
            
            "channel_names": [ch.name for ch in eeg_info.channel_locations],
            # 添加通道类型和单位信息
            "channel_types": [ch.type for ch in eeg_info.channel_locations],
            "channel_units": [ch.unit for ch in eeg_info.channel_locations],
            "channel_locations": [
                {"name": ch.name, "type": ch.type, "unit": ch.unit, "x": ch.x, "y": ch.y, "z": ch.z} 
                for ch in eeg_info.channel_locations
            ],
            "annotations": eeg_info.annotations
        }
        return metadata
    
    def start_streaming_data(self, subject_id: str, task_name: str, 
                           payload_id: str, speed_factor: float = 1.0,
                           chunk_samples: int = 250):
        """开始流式发送EEG数据
        
        Args:
            subject_id: 受试者ID 
            task_name: 任务名称
            payload_id: 数据负载ID，用于关联元数据
            speed_factor: 速度因子（>1加速，<1减速）
            chunk_samples: 每次发送的样本数
        """
        if self.data_thread and self.data_thread.is_alive():
            print("已有数据流在发送中，请先停止")
            return
        
        try:
            # 读取EEG数据
            eeg_info = self.eeg_reader.read_eeg_data(subject_id, task_name)
            
            # 获取原始数据
            raw_data = eeg_info.raw.get_data()
            
            # 启动数据发送线程
            self.is_running = True
            self.data_thread = threading.Thread(
                target=self._stream_data_thread,
                args=(raw_data, eeg_info, payload_id, speed_factor, chunk_samples)
            )
            self.data_thread.daemon = True
            self.data_thread.start()
            
            print(f"开始发送数据流: 受试者 {subject_id}, 任务 {task_name}")
            return True
            
        except Exception as e:
            print(f"启动数据流失败: {e}")
            return False
    
    def _stream_data_thread(self, raw_data: np.ndarray, eeg_info: EEGDataInfo, 
                           payload_id: str, speed_factor: float, chunk_samples: int):
        """数据流发送线程
        
        Args:
            raw_data: EEG原始数据数组
            eeg_info: EEG数据信息
            payload_id: 数据负载ID
            speed_factor: 速度因子
            chunk_samples: 每次发送的样本数
        """
        try:
            total_samples = raw_data.shape[1]
            total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            chunk_duration = chunk_samples / eeg_info.sampling_rate
            
            print(f"数据流参数: {eeg_info.n_channels}通道, {total_samples}样本, "
                  f"采样率{eeg_info.sampling_rate}Hz, {total_chunks}数据块")
            print(f"速度因子: {speed_factor}x (块大小={chunk_samples}样本, "
                  f"实际间隔={chunk_duration/speed_factor:.3f}秒)")
            
            # 等待连接
            print("等待接收端连接数据服务...")
            client, addr = self.data_socket.accept()
            client.settimeout(60)  # 设置超时
            print(f"数据服务接收到连接：{addr}")
            
            # 发送开始标记
            start_msg = {
                "type": "START", 
                "payload_id": payload_id,
                "total_chunks": total_chunks,
                "chunk_samples": chunk_samples
            }
            start_data = json.dumps(start_msg).encode('utf-8')
            client.sendall(len(start_data).to_bytes(4, byteorder='big'))
            client.sendall(start_data)
            
            # 逐块发送数据
            current_sample = 0
            
            for chunk_idx in range(total_chunks):
                if not self.is_running:
                    break
                    
                start_time = time.time()
                
                # 准备当前数据块
                end_sample = min(current_sample + chunk_samples, total_samples)
                chunk_data = raw_data[:, current_sample:end_sample]
                actual_samples = end_sample - current_sample
                
                # 计算当前数据块的时间范围（秒）
                start_time_sec = current_sample / eeg_info.sampling_rate
                end_time_sec = end_sample / eeg_info.sampling_rate
                
                # 筛选当前时间窗口内的事件
                chunk_events = []
                for event in eeg_info.annotations:
                    event_time = event['onset']
                    # 如果事件发生在当前数据块的时间范围内，则添加到当前数据包
                    if start_time_sec <= event_time < end_time_sec:
                        chunk_events.append(event)
                
                # 构建数据包
                data_packet = {
                    "type": "DATA",
                    "payload_id": payload_id,
                    "sequence_num": chunk_idx + 1,
                    "total_chunks": total_chunks,
                    "n_channels": eeg_info.n_channels,
                    "n_samples": actual_samples,
                    "timestamp": time.time(),
                    "events": chunk_events  # 只包含当前时间窗口的事件
                }
                
                # 将numpy数组转换为简单列表并添加到数据包
                data_packet["data"] = chunk_data.tolist()
                
                # 序列化并发送
                json_data = json.dumps(data_packet).encode('utf-8')
                
                try:
                    # 发送数据长度
                    client.sendall(len(json_data).to_bytes(4, byteorder='big'))
                    # 发送数据内容
                    client.sendall(json_data)
                    
                    print(f"发送数据块 {chunk_idx+1}/{total_chunks}: "
                          f"{actual_samples}样本, {len(json_data)}字节")
                    
                    # 更新当前样本位置
                    current_sample = end_sample
                    
                    # 控制发送速率以模拟实时数据生成
                    elapsed = time.time() - start_time
                    sleep_time = (chunk_duration / speed_factor) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                except socket.timeout:
                    print("发送超时，接收端可能断开连接")
                    break
                except ConnectionError:
                    print("连接断开")
                    break
                    
            # 发送结束标记
            if self.is_running:
                end_msg = {"type": "END", "payload_id": payload_id}
                end_data = json.dumps(end_msg).encode('utf-8')
                try:
                    client.sendall(len(end_data).to_bytes(4, byteorder='big'))
                    client.sendall(end_data)
                    print("数据流发送完成")
                except:
                    print("发送结束标记失败")
                    
            # 关闭连接
            try:
                client.close()
            except:
                pass
                
        except Exception as e:
            print(f"数据流发送错误: {e}")
        finally:
            self.is_running = False
    
    def stop_streaming(self):
        """停止数据流发送"""
        self.is_running = False
        if self.data_thread:
            self.data_thread.join(timeout=2.0)
            print("数据流发送已停止")
    
    def close(self):
        """关闭发送器"""
        self.stop_streaming()
        if self.metadata_socket:
            self.metadata_socket.close()
        if self.data_socket:
            self.data_socket.close()
        print("发送器已关闭")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EEG数据发送器')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--metadata-port', type=int, default=9000, help='元数据端口')
    parser.add_argument('--data-port', type=int, default=9001, help='数据端口')
    parser.add_argument('--subject-id', default=os.environ.get('SUBJECT_ID', '01'), help='受试者ID')
    parser.add_argument('--task', default=os.environ.get('TASK_NAME', 'resting'), help='任务名称')
    parser.add_argument('--data-dir', default=os.environ.get('DATA_DIR', '/app/data/ds005779'), help='数据目录')
    parser.add_argument('--speed', type=float, default=2.0, help='数据发送速度因子')
    parser.add_argument('--chunk-size', type=int, default=500, help='每次发送的样本数')
    parser.add_argument('--simulate', action='store_true', help='模拟实时实验数据')
    
    args = parser.parse_args()
    
    try:
        print(f"=== EEG数据发送器 - 受试者 {args.subject_id} ===")
        
        # 创建发送器
        sender = EEGDataSender(
            host=args.host, 
            metadata_port=args.metadata_port, 
            data_port=args.data_port, 
            dataset_root=args.data_dir
        )
        
        # 设置socket
        sender.setup_sockets()
        
        print(f"准备发送受试者 {args.subject_id} 的 {args.task} 任务数据")
        
        # 发送元数据
        payload_id = sender.send_metadata(args.subject_id, args.task)
        
        if payload_id:
            # 开始发送数据流
            sender.start_streaming_data(
                args.subject_id, 
                args.task, 
                payload_id,
                args.speed,
                args.chunk_size
            )
            
            # 等待数据发送完成或中断
            try:
                while sender.is_running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n接收到中断，停止发送")
                sender.stop_streaming()
                
        # 关闭发送器
        sender.close()
        
    except Exception as e:
        print(f"错误: {e}")
    except KeyboardInterrupt:
        print("\n程序被中断")


if __name__ == "__main__":
    main()
