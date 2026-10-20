import numpy as np
import mne
from scipy import signal
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Connectivity')

class ConnectivityAnalyzer:
    """EEG连接性分析器 - 实现功能连接和有效连接分析"""
    
    def __init__(self, sampling_rate: float):
        """初始化连接性分析器
        
        Args:
            sampling_rate: 采样率(Hz)
        """
        self.sampling_rate = sampling_rate
        self.last_error = None
    
    def compute_coherence(self, data: np.ndarray, fmin: float = 0, fmax: float = None, 
                         window_size: float = 2.0, overlap: float = 0.5) -> Dict[str, Any]:
        """计算通道间相干性
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)
            fmin: 最小频率(Hz)
            fmax: 最大频率(Hz)，如果为None则使用奈奎斯特频率
            window_size: 窗口大小(秒)
            overlap: 窗口重叠(秒)
            
        Returns:
            Dict: 相干性分析结果，包含coherence, freqs, channel_pairs等信息
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            if data.shape[0] < 2:
                logger.warning("计算相干性需要至少两个通道")
                return {
                    'coherence': np.array([]),
                    'freqs': np.array([]),
                    'channel_pairs': [],
                    'error': "计算相干性需要至少两个通道"
                }
            
            # 计算窗口大小和重叠
            nperseg = int(window_size * self.sampling_rate)
            noverlap = int(overlap * self.sampling_rate)
            
            # 如果数据太短，调整窗口大小
            if nperseg > data.shape[1]:
                nperseg = data.shape[1]
                noverlap = nperseg // 2
                logger.warning(f"数据长度({data.shape[1]})小于窗口大小({nperseg})，已调整窗口大小为{nperseg}")
            
            # 确保窗口大小至少为32点
            if nperseg < 32:
                nperseg = min(32, data.shape[1])
                noverlap = nperseg // 2
                logger.warning(f"窗口大小过小，已调整为{nperseg}点")
            
            # 设置最大频率
            if fmax is None:
                fmax = self.sampling_rate / 2  # 奈奎斯特频率
            
            # 计算所有通道对的相干性
            n_channels = data.shape[0]
            channel_pairs = []
            coherence_matrices = []
            
            # 计算频率分辨率
            freq_resolution = self.sampling_rate / nperseg
            logger.info(f"频率分辨率: {freq_resolution:.2f}Hz")
            
            # 计算相干性
            freqs, coherence = signal.coherence(
                data[0], data[1], 
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # 筛选频率范围
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            freqs = freqs[freq_mask]
            
            # 初始化相干性矩阵 (频率 x 通道 x 通道)
            coherence_matrix = np.zeros((len(freqs), n_channels, n_channels))
            
            # 计算所有通道对的相干性
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    channel_pairs.append((i, j))
                    _, coh = signal.coherence(
                        data[i], data[j], 
                        fs=self.sampling_rate,
                        nperseg=nperseg,
                        noverlap=noverlap
                    )
                    coherence_matrix[:, i, j] = coh[freq_mask]
                    coherence_matrix[:, j, i] = coh[freq_mask]  # 对称矩阵
            
            # 对角线设为1
            for i in range(n_channels):
                coherence_matrix[:, i, i] = 1.0
            
            return {
                'coherence': coherence_matrix,
                'freqs': freqs,
                'channel_pairs': channel_pairs,
                'method': 'coherence'
            }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"计算相干性失败: {e}", exc_info=True)
            return {
                'coherence': np.array([]),
                'freqs': np.array([]),
                'channel_pairs': [],
                'method': 'coherence',
                'error': str(e)
            }
    
    def compute_pli(self, data: np.ndarray, fmin: float = 0, fmax: float = None,
                   window_size: float = 2.0, overlap: float = 0.5) -> Dict[str, Any]:
        """计算相位锁定值(Phase Locking Index)
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)
            fmin: 最小频率(Hz)
            fmax: 最大频率(Hz)，如果为None则使用奈奎斯特频率
            window_size: 窗口大小(秒)
            overlap: 窗口重叠(秒)
            
        Returns:
            Dict: PLI分析结果，包含pli, freqs, channel_pairs等信息
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            if data.shape[0] < 2:
                logger.warning("计算PLI需要至少两个通道")
                return {
                    'pli': np.array([]),
                    'freqs': np.array([]),
                    'channel_pairs': [],
                    'error': "计算PLI需要至少两个通道"
                }
            
            # 使用MNE的PLI实现
            # 首先创建MNE的数据结构
            ch_names = [f'ch{i}' for i in range(data.shape[0])]
            info = mne.create_info(ch_names=ch_names, sfreq=self.sampling_rate, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # 设置最大频率
            if fmax is None:
                fmax = self.sampling_rate / 2  # 奈奎斯特频率
            
            # 计算PLI
            try:
                # 使用MNE的spectral_connectivity函数
                from mne.connectivity import spectral_connectivity
                
                # 将连续数据转换为epochs
                n_samples = data.shape[1]
                n_epochs = max(1, n_samples // int(window_size * self.sampling_rate))
                epoch_length = n_samples // n_epochs
                
                epochs_data = np.zeros((n_epochs, data.shape[0], epoch_length))
                for i in range(n_epochs):
                    start = i * epoch_length
                    end = start + epoch_length
                    if end <= n_samples:
                        epochs_data[i] = data[:, start:end]
                
                # 创建epochs对象
                epochs = mne.EpochsArray(epochs_data, info)
                
                # 计算PLI
                con, freqs, times, _, _ = spectral_connectivity(
                    epochs, method='pli', mode='multitaper',
                    sfreq=self.sampling_rate, fmin=fmin, fmax=fmax,
                    faverage=True, mt_adaptive=True
                )
                
                # 重塑结果
                n_channels = data.shape[0]
                n_freqs = len(freqs)
                pli_matrix = np.zeros((n_freqs, n_channels, n_channels))
                
                # 填充PLI矩阵
                idx = 0
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        pli_matrix[:, i, j] = con[idx, :, 0]
                        pli_matrix[:, j, i] = con[idx, :, 0]  # 对称矩阵
                        idx += 1
                
                # 对角线设为1
                for i in range(n_channels):
                    pli_matrix[:, i, i] = 1.0
                
                # 创建通道对列表
                channel_pairs = []
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        channel_pairs.append((i, j))
                
                return {
                    'pli': pli_matrix,
                    'freqs': freqs,
                    'channel_pairs': channel_pairs,
                    'method': 'pli'
                }
                
            except ImportError:
                # 如果MNE的connectivity模块不可用，使用简化的PLI实现
                logger.warning("MNE connectivity模块不可用，使用简化的PLI实现")
                
                # 使用Hilbert变换计算相位
                analytic_signal = signal.hilbert(data, axis=1)
                phase = np.angle(analytic_signal)
                
                # 计算所有通道对的PLI
                n_channels = data.shape[0]
                channel_pairs = []
                
                # 初始化PLI矩阵 (通道 x 通道)
                pli_matrix = np.zeros((1, n_channels, n_channels))
                
                # 计算所有通道对的PLI
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        channel_pairs.append((i, j))
                        # 计算相位差
                        phase_diff = phase[i] - phase[j]
                        # 计算相位差的符号
                        sign = np.sign(np.sin(phase_diff))
                        # 计算PLI
                        pli = np.abs(np.mean(sign))
                        pli_matrix[0, i, j] = pli
                        pli_matrix[0, j, i] = pli  # 对称矩阵
                
                # 对角线设为1
                for i in range(n_channels):
                    pli_matrix[0, i, i] = 1.0
                
                return {
                    'pli': pli_matrix,
                    'freqs': np.array([0]),  # 简化实现不区分频率
                    'channel_pairs': channel_pairs,
                    'method': 'pli_simplified'
                }
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"计算PLI失败: {e}", exc_info=True)
            return {
                'pli': np.array([]),
                'freqs': np.array([]),
                'channel_pairs': [],
                'method': 'pli',
                'error': str(e)
            }
    
    def compute_granger_causality(self, data: np.ndarray, max_lag: int = 10) -> Dict[str, Any]:
        """计算Granger因果关系
        
        Args:
            data: 数据数组，形状为(n_channels, n_samples)
            max_lag: 最大滞后阶数
            
        Returns:
            Dict: Granger因果关系分析结果
        """
        try:
            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)
                
            if data.shape[0] < 2:
                logger.warning("计算Granger因果关系需要至少两个通道")
                return {
                    'granger': np.array([]),
                    'channel_pairs': [],
                    'error': "计算Granger因果关系需要至少两个通道"
                }
            
            # 尝试导入statsmodels
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                from statsmodels.tsa.api import VAR
                
                # 转置数据以适应statsmodels的输入格式 (samples, channels)
                data_t = data.T
                
                # 创建VAR模型
                model = VAR(data_t)
                
                # 拟合模型
                results = model.fit(maxlags=max_lag)
                
                # 获取通道数
                n_channels = data.shape[0]
                
                # 初始化Granger因果关系矩阵 (通道 x 通道)
                granger_matrix = np.zeros((n_channels, n_channels))
                
                # 计算所有通道对的Granger因果关系
                channel_pairs = []
                for i in range(n_channels):
                    for j in range(n_channels):
                        if i != j:
                            channel_pairs.append((i, j))
                            # 执行Granger因果关系测试
                            test_result = grangercausalitytests(data_t[:, [i, j]], maxlag=max_lag, verbose=False)
                            
                            # 获取最小的p值作为Granger因果性度量
                            min_p_value = float('inf')
                            for lag in range(1, max_lag + 1):
                                if lag in test_result:
                                    # 使用F检验的p值
                                    p_value = test_result[lag][0]['ssr_ftest'][1]
                                    if p_value < min_p_value:
                                        min_p_value = p_value
                            
                            # 将p值转换为因果强度（1-p值）
                            granger_matrix[i, j] = 1.0 - min_p_value
                
                # 创建通道对列表
                channel_pairs = [(i, j) for i in range(n_channels) for j in range(n_channels) if i != j]
                
                return {
                    'granger': granger_matrix,
                    'channel_pairs': channel_pairs,
                    'method': 'granger_causality'
                }
                
            except ImportError:
                logger.warning("statsmodels模块不可用，无法计算Granger因果关系")
                return {
                    'granger': np.array([]),
                    'channel_pairs': [],
                    'method': 'granger_causality',
                    'error': "statsmodels模块不可用，无法计算Granger因果关系"
                }
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"计算Granger因果关系失败: {e}", exc_info=True)
            return {
                'granger': np.array([]),
                'channel_pairs': [],
                'method': 'granger_causality',
                'error': str(e)
            }