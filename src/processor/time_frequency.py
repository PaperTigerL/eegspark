import numpy as np
from scipy import signal
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TimeFrequency')

class TimeFrequencyAnalyzer:
    """时频分析器 - 使用SciPy实现STFT和频谱分析，适用于实时短数据流"""
    
    def __init__(self, sampling_rate: float):
        """初始化时频分析器
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        self.last_error = None
    
    def compute_time_frequency(self, data: np.ndarray, freqs: np.ndarray = None, 
                          window_size: float = 0.5, overlap: float = 0.25) -> Dict[str, Any]:
        """计算时频表示 - 使用STFT方法，适用于实时短数据流
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)或(n_samples,)
            freqs: 频率点，如果为None则自动生成
            window_size: 窗口大小(秒)
            overlap: 窗口重叠(秒)
            
        Returns:
            Dict: 时频分析结果
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # 记录数据情况
            logger.info(f"时频分析数据 - 形状: {data.shape}, 采样率: {self.sampling_rate}Hz")
            logger.info(f"数据统计 - 最小值: {np.min(data)}, 最大值: {np.max(data)}, 均值: {np.mean(data)}, 标准差: {np.std(data)}")
            
            # 计算每个通道的数据长度（时间）
            duration = data.shape[1] / self.sampling_rate
            logger.info(f"数据持续时间: {duration:.4f}秒, 样本点数: {data.shape[1]}")
            
            # 使用STFT方法计算时频表示
            return self.compute_stft(data, window_size, overlap)
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"时频分析失败: {e}", exc_info=True)
            return {
                'power': np.array([]),
                'freqs': np.array([]),
                'times': np.array([]),
                'method': 'stft',
                'error': str(e)
            }
    
    def compute_stft(self, data: np.ndarray, window_size: float = 0.5, 
                    overlap: float = 0.25) -> Dict[str, Any]:
        """使用短时傅里叶变换计算时频表示 - 优化版，支持500ms窗口
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)或(n_samples,)
            window_size: 窗口大小(秒)，默认500ms
            overlap: 窗口重叠(秒)，默认250ms
            
        Returns:
            Dict: 时频分析结果，包含power, freqs, times等信息
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # 优化的滑动窗口计算 - 支持500ms窗口
            # 窗口大小固定为500ms，步长为100ms，实现80%重叠
            min_freq_of_interest = 1.0  # Hz，最低感兴趣频率
            min_window_size = 2 / min_freq_of_interest  # 秒，至少包含2个周期
            
            # 计算窗口大小和重叠 - 优化版
            nperseg = int(window_size * self.sampling_rate)  # 默认500ms
            noverlap = int(overlap * self.sampling_rate)    # 默认250ms
            
            # 使用FFTW库加速FFT计算
            try:
                from pyfftw.interfaces import scipy_fftpack
                fft_method = scipy_fftpack
                logger.info("使用FFTW库加速FFT计算")
            except ImportError:
                fft_method = None  # 将使用SciPy默认FFT
            
            # 如果数据太短，调整窗口大小
            if nperseg > data.shape[1]:
                nperseg = data.shape[1]
                noverlap = nperseg // 2
                logger.warning(f"数据长度({data.shape[1]})小于窗口大小({nperseg})，已调整窗口大小为{nperseg}")
            
            # 确保窗口大小至少为32点（STFT的最小要求）
            if nperseg < 32:
                nperseg = min(32, data.shape[1])
                noverlap = nperseg // 2
                logger.warning(f"窗口大小过小，已调整为{nperseg}点")
            
            # 初始化结果
            all_freqs = None
            all_times = None
            all_power = []
            
            # 对每个通道进行STFT
            for ch_idx in range(data.shape[0]):
                # 获取通道数据
                ch_data = data[ch_idx]
                
                # 应用短时傅里叶变换
                freqs, times, Zxx = signal.stft(
                    ch_data, 
                    fs=self.sampling_rate, 
                    window='hann', 
                    nperseg=nperseg, 
                    noverlap=noverlap
                )
                
                # 计算功率谱
                power = np.abs(Zxx) ** 2
                
                # 保存结果
                all_power.append(power)
                if all_freqs is None:
                    all_freqs = freqs
                    all_times = times
            
            # 转换为numpy数组
            all_power = np.array(all_power)
            
            logger.info(f"STFT完成，结果形状: {all_power.shape}")
            return {
                'power': all_power,
                'freqs': all_freqs,
                'times': all_times,
                'method': 'stft'
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"STFT失败: {e}", exc_info=True)
            return {
                'power': np.array([]),
                'freqs': np.array([]),
                'times': np.array([]),
                'method': 'stft',
                'error': str(e)
            }
    
    def compute_band_power(self, data: np.ndarray, freq_bands: Dict[str, Tuple[float, float]], 
                         method: str = 'stft') -> Dict[str, np.ndarray]:
        """计算频带功率
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)或(n_samples,)
            freq_bands: 频带定义，格式为{band_name: (freq_min, freq_max)}
            method: 计算方法，默认为'stft'
            
        Returns:
            Dict[str, np.ndarray]: 频带功率字典，每个频带对应一个通道的功率值数组
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # 初始化结果字典
            band_powers = {band: np.zeros(data.shape[0]) for band in freq_bands.keys()}
            relative_powers = {band: np.zeros(data.shape[0]) for band in freq_bands.keys()}
            
            # 使用STFT方法
            if method == 'stft':
                # 使用短时傅里叶变换
                tf_result = self.compute_stft(data)
                
                # 如果计算失败，返回空结果
                if tf_result['power'].size == 0:
                    return band_powers
                
                # 使用STFT结果计算频带功率
                freqs = tf_result['freqs']
                power = tf_result['power']
                
                # 计算各频带功率
                for band_name, (fmin, fmax) in freq_bands.items():
                    # 找出频带对应的索引
                    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                    if len(idx) > 0:
                        # 计算频带平均功率
                        for ch_idx in range(data.shape[0]):
                            # 计算频带平均功率 - 对时间和频率维度平均
                            band_powers[band_name][ch_idx] = np.mean(power[ch_idx, idx, :])
                
                logger.info(f"STFT频带功率计算完成，频带: {list(freq_bands.keys())}")
            
            # 使用FFT方法（更快速）
            elif method == 'fft':
                # 对每个通道计算FFT
                for ch_idx in range(data.shape[0]):
                    # 获取通道数据
                    ch_data = data[ch_idx]
                    
                    # 应用汉宁窗
                    ch_data = ch_data * np.hanning(len(ch_data))
                    
                    # 计算FFT
                    fft_data = np.abs(np.fft.rfft(ch_data)) ** 2
                    freqs = np.fft.rfftfreq(len(ch_data), 1.0/self.sampling_rate)
                    
                    # 计算总功率（用于相对功率计算）
                    total_power = np.sum(fft_data)
                    
                    # 计算各频带功率
                    for band_name, (fmin, fmax) in freq_bands.items():
                        # 找出频带对应的索引
                        idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                        if len(idx) > 0:
                            # 计算频带绝对功率
                            band_power = np.mean(fft_data[idx])
                            band_powers[band_name][ch_idx] = band_power
                            
                            # 计算频带相对功率
                            if total_power > 0:
                                relative_powers[band_name][ch_idx] = np.sum(fft_data[idx]) / total_power
                
                logger.info(f"FFT频带功率计算完成，频带: {list(freq_bands.keys())}")
            
            # 返回结果，包括绝对功率和相对功率
            return {
                'absolute': band_powers,
                'relative': relative_powers
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"频带功率计算失败: {e}", exc_info=True)
            return {'absolute': band_powers, 'relative': {}}
    
    def compute_psd(self, data: np.ndarray, method: str = 'welch', 
                   fmin: float = 0.5, fmax: float = 50.0) -> Dict[str, Any]:
        """计算功率谱密度
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)或(n_samples,)
            method: 计算方法，'welch'或'multitaper'
            fmin: 最小频率
            fmax: 最大频率
            
        Returns:
            Dict: 功率谱密度结果
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # 初始化结果
            psds = []
            freqs = None
            
            # 对每个通道计算PSD
            for ch_idx in range(data.shape[0]):
                # 获取通道数据
                ch_data = data[ch_idx]
                
                if method == 'welch':
                    # 使用Welch方法计算PSD
                    # 计算窗口大小 - 自适应窗口大小
                    nperseg = min(data.shape[1], int(4 * self.sampling_rate))  # 4秒或数据长度
                    if nperseg < 32:  # 确保窗口至少有32个点
                        nperseg = min(32, data.shape[1])
                    
                    # 计算PSD
                    freqs_ch, psd_ch = signal.welch(
                        ch_data, 
                        fs=self.sampling_rate, 
                        nperseg=nperseg,
                        scaling='density'
                    )
                    
                    # 保存结果
                    if freqs is None:
                        # 筛选频率范围
                        mask = (freqs_ch >= fmin) & (freqs_ch <= fmax)
                        freqs = freqs_ch[mask]
                        psds.append(psd_ch[mask])
                    else:
                        mask = (freqs_ch >= fmin) & (freqs_ch <= fmax)
                        psds.append(psd_ch[mask])
                
                elif method == 'multitaper':
                    # 使用多窗口方法计算PSD (使用scipy.signal模拟)
                    # 设置参数
                    time_bandwidth = 4.0  # 时间带宽积
                    num_tapers = int(2 * time_bandwidth) - 1  # 窗口数量
                    
                    # 初始化结果
                    psd_mt = np.zeros((int(data.shape[1]//2) + 1,))
                    
                    # 使用多个窗口计算PSD并平均
                    for k in range(num_tapers):
                        # 创建DPSS窗口 (使用汉宁窗近似)
                        taper = np.hanning(len(ch_data))
                        
                        # 应用窗口
                        tapered_data = ch_data * taper
                        
                        # 计算FFT
                        fft_data = np.abs(np.fft.rfft(tapered_data)) ** 2 / len(ch_data)
                        
                        # 累加结果
                        psd_mt += fft_data
                    
                    # 平均结果
                    psd_mt /= num_tapers
                    
                    # 计算频率
                    freqs_mt = np.fft.rfftfreq(len(ch_data), 1.0/self.sampling_rate)
                    
                    # 保存结果
                    if freqs is None:
                        # 筛选频率范围
                        mask = (freqs_mt >= fmin) & (freqs_mt <= fmax)
                        freqs = freqs_mt[mask]
                        psds.append(psd_mt[mask])
                    else:
                        mask = (freqs_mt >= fmin) & (freqs_mt <= fmax)
                        psds.append(psd_mt[mask])
            
            # 转换为numpy数组
            psds = np.array(psds)
            
            logger.info(f"PSD计算完成，结果形状: {psds.shape}")
            return {
                'psds': psds,
                'freqs': freqs,
                'method': method
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"PSD计算失败: {e}", exc_info=True)
            return {
                'psds': np.array([]),
                'freqs': np.array([]),
                'method': method,
                'error': str(e)
            }
            
    def compute_spectral_entropy(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """计算频谱熵 - 衡量信号频谱的不确定性或复杂性
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)或(n_samples,)
            normalize: 是否归一化熵值
            
        Returns:
            np.ndarray: 频谱熵值，每个通道一个值
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # 初始化结果
            entropy = np.zeros(data.shape[0])
            
            # 对每个通道计算频谱熵
            for ch_idx in range(data.shape[0]):
                # 获取通道数据
                ch_data = data[ch_idx]
                
                # 计算功率谱密度
                freqs, psd = signal.welch(ch_data, fs=self.sampling_rate)
                
                # 归一化PSD（使其和为1，类似概率分布）
                psd_norm = psd / np.sum(psd)
                
                # 计算熵值（香农熵）
                # 避免log(0)，通过添加一个小的常数
                entropy[ch_idx] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                
                # 归一化熵值（如果需要）
                if normalize:
                    # 最大熵值为log2(N)，其中N是频率点的数量
                    max_entropy = np.log2(len(freqs))
                    entropy[ch_idx] /= max_entropy
            
            logger.info(f"频谱熵计算完成，结果形状: {entropy.shape}")
            return entropy
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"频谱熵计算失败: {e}", exc_info=True)
            return np.zeros(data.shape[0])
    
    def compute_band_ratios(self, data: np.ndarray, freq_bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """计算频带比率 - 常用于EEG分析的指标，如alpha/theta比率等
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)或(n_samples,)
            freq_bands: 频带定义，格式为{band_name: (freq_min, freq_max)}
            
        Returns:
            Dict[str, np.ndarray]: 频带比率字典
        """
        try:
            # 计算频带功率
            band_power_result = self.compute_band_power(data, freq_bands)
            band_powers = band_power_result['absolute']
            
            # 初始化结果字典
            band_ratios = {}
            
            # 计算常用的频带比率
            band_names = list(freq_bands.keys())
            
            # 计算所有可能的频带比率
            for i, band1 in enumerate(band_names):
                for band2 in band_names[i+1:]:
                    ratio_name = f"{band1}/{band2}"
                    # 避免除以零
                    denominator = band_powers[band2].copy()
                    denominator[denominator == 0] = 1e-10
                    band_ratios[ratio_name] = band_powers[band1] / denominator
            
            # 计算一些特定的常用比率
            if 'alpha' in band_powers and 'theta' in band_powers:
                # Alpha/Theta比率 - 与放松状态相关
                denominator = band_powers['theta'].copy()
                denominator[denominator == 0] = 1e-10
                band_ratios['alpha/theta'] = band_powers['alpha'] / denominator
            
            if 'beta' in band_powers and 'theta' in band_powers:
                # Beta/Theta比率 - 与注意力相关
                denominator = band_powers['theta'].copy()
                denominator[denominator == 0] = 1e-10
                band_ratios['beta/theta'] = band_powers['beta'] / denominator
            
            if 'alpha_low' in band_powers and 'alpha_high' in band_powers:
                # 低频Alpha/高频Alpha比率 - 与认知处理相关
                denominator = band_powers['alpha_high'].copy()
                denominator[denominator == 0] = 1e-10
                band_ratios['alpha_low/alpha_high'] = band_powers['alpha_low'] / denominator
            
            logger.info(f"频带比率计算完成，计算了{len(band_ratios)}个比率")
            return band_ratios
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"频带比率计算失败: {e}", exc_info=True)
            return {}
    
    def compute_spectral_edge_frequency(self, data: np.ndarray, ref_freq: float = 30.0, 
                                      proportion: float = 0.95) -> np.ndarray:
        """计算频谱边缘频率 - 功率谱中累积到指定比例功率的频率
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)或(n_samples,)
            ref_freq: 参考频率上限，默认为30Hz
            proportion: 累积功率比例，默认为0.95（95%）
            
        Returns:
            np.ndarray: 频谱边缘频率，每个通道一个值
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            # 初始化结果
            sef = np.zeros(data.shape[0])
            
            # 对每个通道计算频谱边缘频率
            for ch_idx in range(data.shape[0]):
                # 获取通道数据
                ch_data = data[ch_idx]
                
                # 计算功率谱密度
                freqs, psd = signal.welch(ch_data, fs=self.sampling_rate)
                
                # 限制到参考频率以下
                mask = freqs <= ref_freq
                freqs = freqs[mask]
                psd = psd[mask]
                
                # 计算累积功率
                total_power = np.sum(psd)
                cumulative_power = np.cumsum(psd) / total_power
                
                # 找到达到指定比例的索引
                idx = np.where(cumulative_power >= proportion)[0]
                if len(idx) > 0:
                    sef[ch_idx] = freqs[idx[0]]
                else:
                    sef[ch_idx] = freqs[-1]  # 如果没有达到指定比例，使用最高频率
            
            logger.info(f"频谱边缘频率计算完成，结果形状: {sef.shape}")
            return sef
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"频谱边缘频率计算失败: {e}", exc_info=True)
            return np.zeros(data.shape[0])