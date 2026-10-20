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
    def __init__(self, spark, data_sources, save_dir='./received_data', buffer_duration=3.0, min_data_duration=2.0):
        """初始化EEG接收器
        
        Args:
            spark: SparkSession对象
            data_sources: 数据源列表
            save_dir: 数据保存目录
            buffer_duration: 数据缓冲区持续时间(秒)
            min_data_duration: 最小数据长度要求(秒)
        """
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
        
        # 数据缓冲设置
        self.buffer_duration = buffer_duration  # 缓冲区持续时间(秒)
        self.min_data_duration = min_data_duration  # 最小数据长度要求(秒)
        self.data_buffers = {}  # 按受试者ID存储的数据缓冲区
        self.metadata_buffers = {}  # 按受试者ID存储的元数据缓冲区
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 新增日志
        try:
            logger.info(f"[SparkEEGReceiver.__init__] 传入SparkSession: id={id(spark)}, appName={getattr(spark.sparkContext, 'appName', '未知')}")
        except Exception as e:
            logger.warning(f"[SparkEEGReceiver.__init__] 记录SparkSession日志失败: {e}")
    
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
        """处理接收到的EEG数据 - 优化版，实现更灵活的数据累积策略"""
        try:
            # 更新统计信息
            self.received_count += 1
            self.stats['packets_received'] += 1
            self.stats['bytes_received'] += eeg_data.size * eeg_data.itemsize
            
            # 保存原始数据
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            save_path = os.path.join(self.save_dir, f"{subject_id}_{timestamp}.npz")
            np.savez(save_path, 
                    eeg_data=eeg_data, 
                    metadata={key: value for key, value in metadata.items() 
                            if isinstance(value, (str, int, float, bool, list))})
            
            logger.info(f"接收到的数据已保存: {save_path}")
            
            # 数据缓冲处理 - 优化版
            sampling_rate = metadata.get('sampling_rate', 250)  # 默认采样率250Hz
            
            # 初始化缓冲区（如果不存在）
            if subject_id not in self.data_buffers:
                self.data_buffers[subject_id] = []
                self.metadata_buffers[subject_id] = metadata.copy()
                logger.info(f"为受试者 {subject_id} 创建新的数据缓冲区")
            
            # 添加数据到缓冲区
            self.data_buffers[subject_id].append(eeg_data)
            
            # 计算当前缓冲区数据长度（秒）
            total_samples = sum(data.shape[1] for data in self.data_buffers[subject_id])
            buffer_duration = total_samples / sampling_rate
            
            logger.info(f"受试者 {subject_id} 的数据缓冲区: {len(self.data_buffers[subject_id])}个数据包, {buffer_duration:.2f}秒")
            
            # 优化的数据处理策略：
            # 1. 如果单个数据包长度已经足够，直接处理
            # 2. 如果缓冲区累积数据长度达到要求，则处理
            # 3. 保持更灵活的重叠策略
            
            # 检查单个数据包是否足够长
            current_duration = eeg_data.shape[1] / sampling_rate
            process_now = False
            
            if current_duration >= self.min_data_duration:
                # 单个数据包足够长，可以直接处理
                logger.info(f"当前数据包长度({current_duration:.2f}秒)已满足最小要求，直接处理")
                process_now = True
                # 使用当前数据包
                process_data = eeg_data
                process_metadata = metadata.copy()
                process_metadata.update({
                    'subject_id': subject_id,
                    'n_samples': process_data.shape[1],
                    'duration': current_duration
                })
            elif buffer_duration >= self.min_data_duration:
                # 缓冲区累积数据长度达到要求
                logger.info(f"缓冲区累积数据长度({buffer_duration:.2f}秒)已满足最小要求，处理合并数据")
                process_now = True
                # 合并缓冲区数据
                process_data = np.concatenate(self.data_buffers[subject_id], axis=1)
                process_metadata = self.metadata_buffers[subject_id].copy()
                process_metadata.update({
                    'subject_id': subject_id,
                    'n_samples': process_data.shape[1],
                    'duration': buffer_duration
                })
            
            if process_now:
                # 处理数据
                logger.info(f"处理数据，形状: {process_data.shape}")
                
                # 创建EEGRDD对象
                logger.info(f"开始创建EEGRDD对象，metadata keys: {list(process_metadata.keys())}")
                eeg_rdd = EEGRDD.from_metadata_and_data(process_metadata, process_data)
                
                # 创建后检查
                logger.info(f"创建EEGRDD后，channel_locations长度: {len(eeg_rdd.channel_locations) if hasattr(eeg_rdd, 'channel_locations') else 0}")
                self.latest_eegrdd = eeg_rdd
                
                # 触发回调
                if self.data_callback and callable(self.data_callback):
                    logger.info(f"触发数据回调: 受试者 {subject_id}, 数据长度: {process_metadata['duration']:.2f}秒")
                    try:
                        self.data_callback(subject_id, eeg_rdd)
                    except Exception as e:
                        logger.error(f"数据回调执行失败: {e}", exc_info=True)
                
                # 更新缓冲区 - 优化的重叠策略
                overlap_duration = min(self.buffer_duration * 0.2, 0.5)  # 动态重叠时间，最大0.5秒
                overlap_samples = int(sampling_rate * overlap_duration)
                
                if process_data.shape[1] > overlap_samples:
                    # 保留最新的数据作为重叠
                    self.data_buffers[subject_id] = [process_data[:, -overlap_samples:]]
                    logger.info(f"保留{overlap_samples}个样本点({overlap_duration:.2f}秒)作为重叠")
                else:
                    # 如果数据太短，清空缓冲区重新开始
                    self.data_buffers[subject_id] = []
                    logger.info(f"数据太短，清空受试者 {subject_id} 的数据缓冲区")
                
                # 更新元数据缓冲区
                self.metadata_buffers[subject_id] = metadata.copy()
            
            # 缓冲区管理 - 防止缓冲区过大
            if buffer_duration > self.buffer_duration * 1.5:
                # 如果缓冲区过大，只保留最新的数据
                keep_duration = self.buffer_duration
                keep_samples = int(keep_duration * sampling_rate)
                
                # 计算需要保留的数据包
                new_buffer = []
                current_samples = 0
                
                # 从后向前遍历数据包，直到达到所需的样本数
                for data in reversed(self.data_buffers[subject_id]):
                    if current_samples >= keep_samples:
                        break
                    new_buffer.insert(0, data)
                    current_samples += data.shape[1]
                
                # 更新缓冲区
                self.data_buffers[subject_id] = new_buffer
                logger.info(f"缓冲区过大，已清理至{current_samples}个样本点({current_samples/sampling_rate:.2f}秒)")
        except Exception as e:
            logger.error(f"处理接收数据时出错: {e}", exc_info=True)

