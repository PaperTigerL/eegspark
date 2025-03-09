import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EEGRDD:
    """EEG数据类 - 增强版，支持更多元数据"""
    subject_id: str
    voltage_data: np.ndarray   # 电压值数组
    sampling_rate: float
    
    # 基本属性
    payload_id: str = None     # 数据包ID
    task_name: str = "unknown" # 任务名称
    block_id: str = None       # 数据块ID
    n_channels: int = 0        # 通道数
    n_samples: int = 0         # 样本数
    duration: float = 0.0      # 持续时间(秒)
    reference: str = None      # 参考方式
    
    # 通道信息
    channel_names: List[str] = field(default_factory=list)  # 通道名称
    channel_types: List[str] = field(default_factory=list)  # 通道类型
    channel_units: List[str] = field(default_factory=list)  # 通道单位
    channel_locations: List[Dict[str, Any]] = field(default_factory=list)  # 通道位置
    
    # 事件信息
    events: List[Dict[str, Any]] = field(default_factory=list)  # 事件标记
    
    # 其他属性
    time_points: np.ndarray = None  # 时间点数组
    data_range: List[float] = None  # 数据范围
    annotations: List[Dict] = field(default_factory=list)  # 注释
    json_metadata: Dict = field(default_factory=dict)  # 原始JSON元数据

    def __post_init__(self):
        # 确保数据范围存在
        if self.data_range is None and self.voltage_data is not None:
            self.data_range = [float(self.voltage_data.min()), float(self.voltage_data.max())]
        
        # 确保通道数和样本数正确
        if self.voltage_data is not None:
            if self.n_channels == 0 and len(self.voltage_data.shape) > 0:
                self.n_channels = self.voltage_data.shape[0]
            if self.n_samples == 0 and len(self.voltage_data.shape) > 1:
                self.n_samples = self.voltage_data.shape[1]

    @classmethod
    def from_metadata(cls, metadata: dict, data: np.ndarray):
        """从元数据和数据创建实例 - 基本版本"""
        return cls(
            subject_id=metadata['subject_id'],
            task_name=metadata.get('task_name', 'unknown'),
            sampling_rate=metadata['sampling_rate'],
            n_channels=metadata['n_channels'],
            n_samples=metadata['n_samples'],
            duration=metadata.get('duration', 0.0),
            data_range=metadata.get('data_range'),
            voltage_data=data
        )
    
    @classmethod
    def from_metadata_and_data(cls, metadata: dict, data: np.ndarray):
        """从元数据和数据创建实例 - 增强版本，处理更多元数据"""
        # 创建基本对象
        eegrdd = cls(
            subject_id=metadata['subject_id'],
            voltage_data=data,
            sampling_rate=metadata['sampling_rate'],
            payload_id=metadata.get('payload_id'),
            task_name=metadata.get('task_name', 'unknown'),
            block_id=metadata.get('block_id'),
            n_channels=metadata['n_channels'],
            n_samples=metadata['n_samples'],
            duration=metadata.get('duration', 0.0),
            reference=metadata.get('reference')
        )
        
        # 处理通道信息
        if 'channel_names' in metadata:
            eegrdd.channel_names = metadata['channel_names']
        
        # 处理通道类型 - 优先使用直接提供的channel_types，否则从channels_info中提取
        if 'channel_types' in metadata and metadata['channel_types']:
            eegrdd.channel_types = metadata['channel_types']
        elif 'channels_info' in metadata and metadata['channels_info']:
            channels_info = metadata['channels_info']
            if isinstance(channels_info, dict) and 'type' in channels_info:
                eegrdd.channel_types = [channels_info['type']] * eegrdd.n_channels
            elif isinstance(channels_info, list) and len(channels_info) > 0:
                eegrdd.channel_types = [ch.get('type', 'EEG') for ch in channels_info]
        
        # 处理通道单位 - 优先使用直接提供的channel_units，否则从channels_info中提取
        if 'channel_units' in metadata and metadata['channel_units']:
            eegrdd.channel_units = metadata['channel_units']
        elif 'channels_info' in metadata and metadata['channels_info']:
            channels_info = metadata['channels_info']
            if isinstance(channels_info, dict) and 'units' in channels_info:
                eegrdd.channel_units = [channels_info['units']] * eegrdd.n_channels
            elif isinstance(channels_info, list) and len(channels_info) > 0:
                eegrdd.channel_units = [ch.get('units', 'uV') for ch in channels_info]
        
        # 处理通道位置
        if 'channel_locations' in metadata:
            eegrdd.channel_locations = metadata['channel_locations']
        
        # 处理事件信息 - 优先使用annotations字段
        if 'annotations' in metadata and metadata['annotations']:
            eegrdd.events = metadata['annotations']
            eegrdd.annotations = metadata['annotations']  # 同时设置annotations字段
        elif 'events_info' in metadata:
            # 即使events_info是空列表也要设置，确保events字段被正确初始化
            eegrdd.events = metadata['events_info']
            eegrdd.annotations = metadata['events_info']  # 同时设置annotations字段
            # 添加调试日志
            import logging
            logger = logging.getLogger('EEGRDD')
            logger.info(f"从events_info设置events字段: {len(metadata['events_info'])}个事件")
            if len(metadata['events_info']) > 0:
                logger.info(f"事件示例: {metadata['events_info'][0]}")
            else:
                logger.info("events_info为空列表")
        
        # 保存原始JSON元数据
        if 'json_metadata' in metadata:
            eegrdd.json_metadata = metadata['json_metadata']
        
        return eegrdd
        
    def to_json(self) -> str:
        """导出元数据为JSON"""
        metadata = {
            'payload_id': self.payload_id,
            'subject_id': self.subject_id,
            'task_name': self.task_name,
            'block_id': self.block_id,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'duration': self.duration,
            'reference': self.reference,
            'channel_names': self.channel_names,
            'channel_types': self.channel_types,
            'channel_units': self.channel_units,
            'data_range': self.data_range
        }
        return json.dumps(metadata, indent=4)
        
    def get_channel_data(self, channel_idx: int) -> np.ndarray:
        """获取指定通道的数据"""
        if self.voltage_data is None or channel_idx >= self.n_channels:
            return None
        return self.voltage_data[channel_idx]
        
    def get_metadata(self) -> dict:
        """获取元数据"""
        return {
            'payload_id': self.payload_id,
            'subject_id': self.subject_id,
            'task_name': self.task_name,
            'block_id': self.block_id,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'duration': self.duration,
            'reference': self.reference,
            'channel_names': self.channel_names,
            'channel_types': self.channel_types,
            'channel_units': self.channel_units,
            'channel_locations': self.channel_locations,
            'events': self.events,
            'data_range': self.data_range
        }