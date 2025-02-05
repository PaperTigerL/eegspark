import numpy as np
from scipy import signal
from typing import List
from .features import EEGFeatureExtractor

class EEGPreprocessor:
    """EEG数据预处理类"""
    
    @staticmethod
    def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float):
        """带通滤波"""
        # 数据校验
        if not isinstance(data, np.ndarray) or data.size == 0:
            return np.array([])
            
        # 限制数据范围，避免溢出
        data = np.clip(data, -1e6, 1e6)
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        data = data.astype(np.float64)
        
        # 参数验证
        nyq = 0.5 * fs
        if lowcut >= nyq or highcut >= nyq:
            print(f"[警告] 截止频率({lowcut}, {highcut})超出奈奎斯特频率({nyq})")
            return data
            
        try:
            # 减小滤波器阶数
            order = min(int(0.02 * fs), len(data) // 3)  # 降低order值
            if order % 2 == 0:
                order += 1
                
            # 使用简单的butterworth滤波器
            b, a = signal.butter(4, [lowcut/nyq, highcut/nyq], btype='band')
            filtered = signal.filtfilt(b, a, data)
            
            return filtered.astype(np.float32)
            
        except Exception as e:
            print(f"[错误] 滤波处理失败: {e}")
            return data
    
    @staticmethod
    def normalize(data: np.ndarray):
        """归一化处理"""
        try:
            # 处理极端值
            data = np.clip(data, -1e6, 1e6)
            mean = np.nanmean(data)
            std = np.nanstd(data)
            
            # 避免除零错误
            if std == 0 or np.isnan(std):
                return np.zeros_like(data)
                
            return ((data - mean) / std).astype(np.float32)
            
        except Exception as e:
            print(f"[错误] 归一化处理失败: {e}")
            return np.zeros_like(data)
        
    @staticmethod
    def downsample(data: np.ndarray, factor: int):
        """降采样
        Args:
            data: 输入数据
            factor: 降采样因子
        """
        return data[::factor]

class EEGProcessor:
    """EEG数据处理类"""
    def __init__(self):
        self.preprocessor = EEGPreprocessor()
        
    def process(self, eegrdd):
        try:
            # 只进行基础预处理
            filtered_data = self.preprocessor.bandpass_filter(
                eegrdd.eeg_data,
                lowcut=1.0,
                highcut=50.0,
                fs=eegrdd.sampling_rate
            )
            
            # 添加处理后的数据到EEGRDD
            eegrdd.processed_data = filtered_data
            return filtered_data
            
        except Exception as e:
            print(f"[错误] 数据处理失败: {e}")
            return []