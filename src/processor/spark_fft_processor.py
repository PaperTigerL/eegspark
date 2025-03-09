from processor.spark_processor import SparkProcessor
import numpy as np
import logging
from scipy import signal
from datetime import datetime

logger = logging.getLogger('SparkFFTProcessor')

class SparkFFTProcessor(SparkProcessor):
    """FFT频谱分析处理器 - Spark优化版"""
    
    def __init__(self, window_size=1024, step=None, use_gpu=False):
        super().__init__(use_gpu)
        self.window_size = window_size
        self.step = step if step is not None else window_size // 2
        
        # 标准EEG频段定义（单位：Hz）
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def process(self, eegrdd):
        """对EEG数据进行频谱分析"""
        try:
            # 获取当前活动的SparkSession
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            
            # 使用经过滤波的数据(如果存在)，否则使用原始数据
            if hasattr(eegrdd, 'filtered_data') and eegrdd.filtered_data is not None:
                data = eegrdd.filtered_data
            else:
                data = eegrdd.voltage_data
            
            fs = eegrdd.sampling_rate
            n_channels = data.shape[0]
            
            # 根据采样率动态调整窗口大小，确保频率分辨率适合不同采样率
            # 对于较高采样率，使用更大的窗口以获得更好的频率分辨率
            if fs > 1000:
                actual_window_size = min(self.window_size * 2, data.shape[1] // 2)
                actual_step = actual_window_size // 2
            elif fs < 250:
                actual_window_size = max(self.window_size // 2, 128)  # 确保至少128点
                actual_step = actual_window_size // 2
            else:
                actual_window_size = self.window_size
                actual_step = self.step
            
            # 准备窗口数据
            windows = []
            for start in range(0, data.shape[1] - actual_window_size + 1, actual_step):
                end = start + actual_window_size
                for ch in range(n_channels):
                    # (通道索引, 窗口索引, 窗口数据, 采样率)
                    windows.append((ch, start//actual_step, data[ch, start:end], fs))
            
            # 转换为RDD并并行处理
            windows_rdd = spark.sparkContext.parallelize(windows)
            
            # 定义FFT计算函数
            def compute_fft(window_data):
                ch_idx, window_idx, signal, fs = window_data
                from scipy import signal as sp_signal
                import numpy as np
                
                # 应用汉宁窗函数 - 符合国际标准的频谱分析
                windowed = signal * sp_signal.windows.hann(len(signal))
                
                # 计算FFT
                fft = np.abs(np.fft.rfft(windowed))
                
                # 计算频率值
                freqs = np.fft.rfftfreq(len(signal), 1.0/fs)
                
                return (ch_idx, window_idx, fft, freqs)
            
            # 并行计算FFT
            fft_rdd = windows_rdd.map(compute_fft)
            
            # 计算各频段的能量
            def compute_band_power(fft_data):
                ch_idx, window_idx, fft, freqs = fft_data
                import numpy as np
                
                band_powers = {}
                for band_name, (low, high) in self.frequency_bands.items():
                    # 找到频段对应的索引范围
                    band_mask = (freqs >= low) & (freqs <= high)
                    if np.any(band_mask):
                        # 计算绝对功率 - 符合国际标准的功率计算方法
                        power = np.sum(fft[band_mask]**2) / len(freqs)
                        band_powers[band_name] = power
                    else:
                        # 确保所有频带都有值，即使没有对应的频率点
                        band_powers[band_name] = 0.0
                        
                return (ch_idx, window_idx, band_powers)
            
            # 计算频段能量
            band_powers_rdd = fft_rdd.map(compute_band_power)
            
            # 按通道分组并计算平均频段能量
            channel_bands = band_powers_rdd.map(
                lambda x: (x[0], x[2])  # (ch_idx, band_powers)
            ).groupByKey().mapValues(
                lambda powers: {
                    band: np.mean([p.get(band, 0) for p in powers])
                    for band in self.frequency_bands
                }
            ).collect()
            
            # 整理结果到字典中
            band_powers = {band: np.zeros(n_channels) for band in self.frequency_bands}
            for ch_idx, ch_powers in channel_bands:
                for band, power in ch_powers.items():
                    band_powers[band][ch_idx] = power
            
            # 计算相对功率 - 符合国际标准
            total_power = np.zeros(n_channels)
            for band in band_powers:
                total_power += band_powers[band]
            
            # 添加相对功率
            relative_powers = {}
            for band in band_powers:
                relative_powers[f"{band}_relative"] = np.zeros(n_channels)
                for ch in range(n_channels):
                    if total_power[ch] > 0:
                        relative_powers[f"{band}_relative"][ch] = band_powers[band][ch] / total_power[ch]
            
            # 合并绝对功率和相对功率
            band_powers.update(relative_powers)
            
            # 将频谱信息保存到EEGRDD中
            eegrdd.band_powers = band_powers
            
            # 保存频率轴信息 - 使用最后一个窗口的频率值
            sample_freqs = fft_rdd.map(lambda x: x[3]).take(1)[0] if fft_rdd.count() > 0 else np.array([])
            eegrdd.fft_freqs = sample_freqs
            
            # 添加处理历史记录
            if not hasattr(eegrdd, 'processing_history'):
                eegrdd.processing_history = []
            
            eegrdd.processing_history.append({
                'operation': 'fft_analysis',
                'window_size': actual_window_size,
                'step': actual_step,
                'sampling_rate': fs,
                'timestamp': datetime.now().isoformat()
            })
            
            return eegrdd
            
        except Exception as e:
            logger.error(f"FFT分析错误: {e}")
            return eegrdd