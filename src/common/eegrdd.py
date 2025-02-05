import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EEGRDD:
    """EEG数据类 - 时间-电压数据对"""
    subject_id: str
    time_points: np.ndarray    # 时间点数组
    voltage_data: np.ndarray   # 电压值数组
    sampling_rate: float
    channel: int = None
    chunk_idx: int = None
    task_name: str = "unknown"
    n_channels: int = 0
    n_samples: int = 0
    duration: float = 0.0
    data_range: List[float] = None
    channels: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.data_range is None:
            self.data_range = [float(self.voltage_data.min()), float(self.voltage_data.max())]
        if self.channels is None:
            self.channels = []

    @classmethod
    def from_metadata(cls, metadata: dict, data: np.ndarray):
        """从元数据和数据创建实例"""
        return cls(
            subject_id=metadata['subject_id'],
            task_name=metadata['task_name'],
            sampling_rate=metadata['sampling_rate'],
            n_channels=metadata['n_channels'],
            n_samples=metadata['n_samples'],
            duration=metadata['duration'],
            data_range=metadata['data_range'],
            channels=metadata['channels'],
            voltage_data=data
        )
        
    def to_json(self) -> str:
        """导出元数据为JSON"""
        metadata = {
            'subject_id': self.subject_id,
            'task_name': self.task_name,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'duration': self.duration,
            'data_range': self.data_range,
            'channels': self.channels
        }
        return json.dumps(metadata, indent=4)
        
    def get_channel_data(self, channel_idx: int) -> np.ndarray:
        """获取指定通道的数据"""
        return self.voltage_data[:, channel_idx]
        
    def get_metadata(self) -> dict:
        """获取元数据"""
        return {
            'subject_id': self.subject_id,
            'task_name': self.task_name,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'duration': self.duration,
            'data_range': self.data_range,
            'channels': self.channels
        }