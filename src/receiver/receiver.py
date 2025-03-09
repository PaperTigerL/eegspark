import os
import pickle
import time
import socket
import json
import threading
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
from typing import Dict, List, Any, Tuple
import logging
from pyspark import SparkContext, RDD
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import *
import sys
from datetime import datetime  # 添加这一行导入datetime

# 添加common目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'common'))
from eegrdd import EEGRDD
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processor'))
import os
import sys
# 添加处理器目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '/processor')
# 注释掉这些行
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processor'))
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)) + '/processor')
# from processor.pipeline import EEGProcessingPipeline


# 设置日志
logging.basicConfig(level=logging.WARNING, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SparkEEGReceiver')

class SparkEEGReceiver:
    """Spark EEG数据接收器 - 专注于数据接收和转换"""
    def __init__(self, spark, data_sources, save_dir='./received_data'):
        """初始化EEG接收器"""
        self.spark = spark
        self.data_sources = data_sources
        self.save_dir = save_dir
        self.is_running = False
        self.metadata_connections = {}
        self.data_connections = {}
        self.latest_eegrdd = None
        self.received_count = 0
        self.stats = {'packets_received': 0, 'bytes_received': 0}
        
        # 流式处理回调 - 新增
        self.data_callback = None
        
        self.receiver_threads = []  # 添加这一行
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
    
    def set_data_callback(self, callback):
        """设置数据处理回调函数"""
        self.data_callback = callback
        logger.info("已设置数据处理回调函数")
    
    def get_stats(self):
        """获取接收器统计信息"""
        return self.stats
    
    def get_received_count(self):
        """获取已接收的数据包数量"""
        return self.received_count
    
    def start_receiving(self):
        """启动数据接收处理"""
        self.is_running = True
        logger.info("启动EEG数据接收处理")
        
        # 为每个数据源创建接收线程
        for subject_id, meta_host, meta_port, data_host, data_port in self.data_sources:
            thread = threading.Thread(
                target=self._receive_data_for_subject,
                args=(subject_id, meta_host, meta_port, data_host, data_port),
                daemon=True
            )
            self.receiver_threads.append(thread)
            thread.start()
            logger.info(f"已启动受试者 {subject_id} 的数据接收线程")
    
    def _receive_data_for_subject(self, subject_id, meta_host, meta_port, data_host, data_port):
        """接收指定受试者的数据"""
        try:
            logger.info(f"尝试连接元数据服务: {meta_host}:{meta_port}")
            meta_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            meta_sock.connect((meta_host, meta_port))
            self.metadata_connections[subject_id] = meta_sock
            logger.info(f"已连接到元数据服务: {meta_host}:{meta_port}")
            
            # 接收元数据
            metadata_json = self._receive_metadata(meta_sock)
            metadata = json.loads(metadata_json)
            logger.info(f"接收到元数据: 受试者 {subject_id}, 采样率 {metadata.get('sampling_rate')}")
            
            # 连接数据服务
            logger.info(f"尝试连接数据服务: {data_host}:{data_port}")
            data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            data_sock.connect((data_host, data_port))
            self.data_connections[subject_id] = data_sock
            logger.info(f"已连接到数据服务: {data_host}:{data_port}")
            
            # 持续接收数据
            while self.is_running:
                # 接收数据包大小
                size_bytes = data_sock.recv(4)
                if not size_bytes:
                    logger.warning(f"数据连接关闭: 受试者 {subject_id}")
                    break
                    
                data_size = int.from_bytes(size_bytes, byteorder='big')
                logger.info(f"准备接收数据包: {data_size} 字节")
                
                # 接收实际数据
                data_buffer = bytearray()
                buffer_size = 8192
                while len(data_buffer) < data_size:
                    chunk = data_sock.recv(min(buffer_size, data_size - len(data_buffer)))
                    if not chunk:
                        break
                    data_buffer.extend(chunk)
                
                # 解析JSON数据包
                try:
                    json_data = data_buffer.decode('utf-8')
                    data_packet = json.loads(json_data)
                    packet_type = data_packet.get('type')
                    
                    logger.info(f"接收到数据包: 类型={packet_type}, 序列号={data_packet.get('sequence_num', 0)}")
                    
                    # 处理不同类型的数据包
                    if packet_type == 'START':
                        logger.info(f"开始接收数据流: {data_packet}")
                        continue
                    elif packet_type == 'END':
                        logger.info(f"数据流接收完成: {data_packet}")
                        break
                    elif packet_type == 'DATA':
                        # 从数据包中提取EEG数据
                        raw_data = np.array(data_packet['data'])
                        n_channels = data_packet['n_channels']
                        n_samples = data_packet['n_samples']
                        
                        # 验证数据形状
                        if raw_data.shape != (n_channels, n_samples):
                            logger.warning(f"数据形状不符合预期: {raw_data.shape} vs 期望 ({n_channels}, {n_samples})")
                            # 尝试重塑数据
                            if len(raw_data) == n_channels * n_samples:
                                raw_data = raw_data.reshape(n_channels, n_samples)
                        
                        logger.info(f"接收到EEG数据: 形状 {raw_data.shape}")
                        
                        # 处理数据
                        # 更新元数据，确保与当前数据包匹配
                        packet_metadata = metadata.copy()
                        packet_metadata.update({
                            'subject_id': data_packet.get('subject_id', subject_id),
                            'task_name': data_packet.get('task_name', packet_metadata.get('task_name', '')),
                            'n_channels': n_channels,
                            'n_samples': n_samples,
                            # 添加事件信息，如果数据包中包含
                            'events_info': data_packet.get('events', [])
                        })
                        
                        # 调试输出事件信息
                        if 'events' in data_packet:
                            logger.info(f"数据包包含事件信息: {len(data_packet['events'])}个事件")
                        else:
                            logger.warning("数据包不包含事件信息")
                        
                        self._process_received_data(subject_id, packet_metadata, raw_data)
                    else:
                        logger.warning(f"未知数据包类型: {packet_type}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {e}")
                except Exception as e:
                    logger.error(f"处理数据包时出错: {e}", exc_info=True)
                
        except ConnectionRefusedError:
            logger.error(f"连接被拒绝: {meta_host}:{meta_port} 或 {data_host}:{data_port}")
        except Exception as e:
            logger.error(f"接收数据时出错: {e}", exc_info=True)
        finally:
            # 关闭连接
            if subject_id in self.metadata_connections:
                self.metadata_connections[subject_id].close()
                del self.metadata_connections[subject_id]
            if subject_id in self.data_connections:
                self.data_connections[subject_id].close()
                del self.data_connections[subject_id]
    
    def _receive_metadata(self, sock):
        """从套接字接收元数据JSON"""
        try:
            # 接收消息长度（4字节整数）
            length_bytes = sock.recv(4)
            if not length_bytes:
                return None
                
            # 解码消息长度
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # 接收完整消息
            metadata_buffer = b''
            remaining = message_length
            
            while remaining > 0:
                chunk = sock.recv(min(remaining, 4096))
                if not chunk:
                    break
                    
                metadata_buffer += chunk
                remaining -= len(chunk)
            
            # 解码JSON
            metadata_json = metadata_buffer.decode('utf-8')
            return metadata_json
            
        except Exception as e:
            logger.error(f"接收元数据出错: {e}", exc_info=True)
            return None
    
    def _process_received_data(self, subject_id, metadata, eeg_data):
        """处理接收到的EEG数据 - 仅保存并通知回调，不进行处理"""
        try:
            # 创建EEGRDD对象
            metadata['subject_id'] = subject_id
            
            # 添加调试输出
            logger.info(f"开始创建EEGRDD对象，metadata keys: {list(metadata.keys())}")
            if 'channel_locations' in metadata:
                logger.info(f"处理前：元数据中channel_locations长度: {len(metadata['channel_locations'])}")
                
                # 检查数据维度和通道名称是否匹配
                if eeg_data.shape[0] != len(metadata['channel_locations']):
                    logger.warning(f"数据维度({eeg_data.shape[0]})与通道名称数量({len(metadata['channel_locations'])})不匹配")

            # 创建EEGRDD对象
            eeg_rdd = EEGRDD.from_metadata_and_data(metadata, eeg_data)

            # 创建后检查
            logger.info(f"创建EEGRDD后，channel_locations长度: {len(eeg_rdd.channel_locations) if hasattr(eeg_rdd, 'channel_locations') else 0}")
            self.latest_eegrdd = eeg_rdd
            
            # 更新统计信息
            self.received_count += 1
            
            # 保存数据
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            save_path = os.path.join(self.save_dir, f"{subject_id}_{timestamp}.npz")
            np.savez(save_path, 
                    eeg_data=eeg_data, 
                    metadata={key: value for key, value in metadata.items() 
                            if isinstance(value, (str, int, float, bool, list))})
            
            logger.info(f"接收到的数据已保存: {save_path}")
            
            # 触发回调，但确保不包含处理逻辑
            if self.data_callback and callable(self.data_callback):
                logger.info(f"触发数据回调: 受试者 {subject_id}")
                try:
                    self.data_callback(subject_id, eeg_rdd)
                except Exception as e:
                    logger.error(f"数据回调执行失败: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"处理数据时出错: {e}", exc_info=True)
    
    def get_latest_data(self):
        """获取最新接收到的EEGRDD对象"""
        return self.latest_eegrdd
    
    def stop(self):
        """停止接收数据"""
        self.is_running = False
        
        # 关闭所有连接
        for subject_id, conn in self.metadata_connections.items():
            try:
                conn.close()
            except:
                pass
                
        for subject_id, conn in self.data_connections.items():
            try:
                conn.close()
            except:
                pass
                
        # 等待所有线程结束
        for thread in self.receiver_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
                
        logger.info("数据接收已停止")

