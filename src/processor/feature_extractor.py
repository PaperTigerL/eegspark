import numpy as np
import logging
from datetime import datetime

# 设置日志
logger = logging.getLogger('EEGFeatureExtractor')

class EEGFeatureExtractor:
    """EEG特征提取器
    
    提取各种EEG信号特征，包括时域和频域特征
    """
    def __init__(self, features=None, use_gpu=False):
        """初始化特征提取器
        
        Args:
            features: 要提取的特征列表，如果为None则使用默认特征
            use_gpu: 是否使用GPU加速
        """
        # 默认提取的特征
        self.features = features or [
            'mean', 'std', 'kurtosis', 'skewness', 'hjorth_mobility', 
            'hjorth_complexity', 'band_power_ratio'
        ]
        self.use_gpu = use_gpu
        logger.info("EEG特征提取器初始化完成")
    
    def extract_time_features(self, signal):
        """提取时域特征
        
        Args:
            signal: 一维信号数组
            
        Returns:
            特征字典
        """
        try:
            # 计算基本统计特征
            mean = np.mean(signal)
            std = np.std(signal)
            
            # 计算峰度和偏度
            signal_norm = signal - mean
            if std > 0:
                kurtosis = (np.mean(signal_norm**4) / (std**4)) - 3
                skewness = np.mean(signal_norm**3) / (std**3)
            else:
                kurtosis = 0
                skewness = 0
                
            # 计算Hjorth参数
            diff1 = np.diff(signal)
            diff2 = np.diff(diff1)
            
            var0 = np.var(signal)
            var1 = np.var(diff1)
            var2 = np.var(diff2)
            
            mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
            complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 and var1 > 0 else 0
            
            return {
                'mean': float(mean),
                'std': float(std),
                'kurtosis': float(kurtosis),
                'skewness': float(skewness),
                'hjorth_mobility': float(mobility),
                'hjorth_complexity': float(complexity)
            }
        except Exception as e:
            logger.error(f"提取时域特征错误: {e}")
            return {}
    
    def extract_frequency_features(self, signal, sampling_rate, bands=None):
        """提取频域特征
        
        Args:
            signal: 一维信号数组
            sampling_rate: 采样率
            bands: 频带定义，如果为None则使用默认频带
            
        Returns:
            特征字典
        """
        try:
            # 默认频带
            if bands is None:
                bands = {
                    'delta': (0.5, 4.0),
                    'theta': (4.0, 8.0),
                    'alpha': (8.0, 13.0),
                    'beta': (13.0, 30.0),
                    'gamma': (30.0, 100.0)
                }
            
            # 计算功率谱
            from scipy import signal as sp_signal
            freqs, psd = sp_signal.welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)))
            
            # 计算各频带能量
            band_powers = {}
            for band_name, (low_freq, high_freq) in bands.items():
                # 找到频带对应的索引
                idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                # 计算频带能量
                if np.any(idx_band):
                    band_powers[band_name] = np.mean(psd[idx_band])
                else:
                    band_powers[band_name] = 0
            
            # 计算频带能量比
            total_power = np.sum(list(band_powers.values()))
            band_power_ratio = {}
            if total_power > 0:
                for band_name, power in band_powers.items():
                    band_power_ratio[f"{band_name}_ratio"] = power / total_power
            
            # 合并特征
            features = {}
            features.update(band_powers)
            features.update(band_power_ratio)
            
            return features
        except Exception as e:
            logger.error(f"提取频域特征错误: {e}")
            return {}