import numpy as np
from scipy import signal

class EEGFeatureExtractor:
    """EEG特征提取类"""
    
    @staticmethod
    def extract_time_features(data: np.ndarray):
        """提取时域特征"""
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'peak_to_peak': np.ptp(data),
            'zero_crossing_rate': np.sum(np.diff(np.signbit(data)))
        }
        return features