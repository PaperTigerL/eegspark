import os
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import mne
from mne.preprocessing import ICA
from scipy import signal as sp_signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings

# 设置日志
logger = logging.getLogger('EEGPipeline')
# 忽略MNE的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

class EEGProcessingPipeline:
    """标准EEG数据处理流水线
    
    符合国际标准的EEG数据处理流程，包括：
    1. 预处理（带通滤波、去伪迹）
    2. 频谱分析（频带能量）
    3. 连通性分析
    4. 特征提取
    5. 结果可视化
    """
    
    def __init__(self, save_dir: str = './results', use_gpu: bool = False, visualizer=None):
        """初始化EEG处理流水线"""
        self.save_dir = save_dir
        self.use_gpu = use_gpu
        self.visualizer = visualizer
        
        # 明确初始化处理器列表
        self.processors = []
        
        # 频带定义
        self.frequency_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0)
        }
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建子目录
        self.plots_dir = os.path.join(save_dir, 'plots')
        self.models_dir = os.path.join(save_dir, 'models')
        self.reports_dir = os.path.join(save_dir, 'reports')
        
        for directory in [self.plots_dir, self.models_dir, self.reports_dir]:
            os.makedirs(directory, exist_ok=True)

        # 初始化特征提取器
        from .feature_extractor import EEGFeatureExtractor
        self.feature_extractor = EEGFeatureExtractor()
        self.processors.append(('feature', self.feature_extractor))
        logger.info("特征提取器添加成功")
        
        # 添加事件处理器 - 使用异常处理避免导入问题
        try:
            from .event_processor import EventProcessor
            self.event_processor = EventProcessor(use_gpu=use_gpu)
            self.processors.append(('event', self.event_processor))
            logger.info("事件处理器添加成功")
        except ImportError as e:
            logger.warning(f"无法导入事件处理器: {e}")
        
        logger.info("EEG处理流水线初始化完成")
    def process(self, eeg_obj):
        """处理EEG数据"""
        try:
            # 在处理开始前检查并转换通道位置
            if hasattr(eeg_obj, 'channel_locations') and eeg_obj.channel_locations:
                # 检查是否是字典格式，转换为坐标列表
                if len(eeg_obj.channel_locations) > 0 and isinstance(eeg_obj.channel_locations[0], dict) and 'x' in eeg_obj.channel_locations[0]:
                    eeg_obj.channel_locations = [
                        [loc['x'], loc['y']] 
                        for loc in eeg_obj.channel_locations
                    ]
            
            # 保留原始voltage_data的副本
            original_voltage_data = None
            if hasattr(eeg_obj, 'voltage_data') and eeg_obj.voltage_data is not None:
                original_voltage_data = eeg_obj.voltage_data.copy()
            
            # 执行处理流程
            logger.info(f"开始处理受试者 {eeg_obj.subject_id} 的EEG数据")
            eeg_obj = self._preprocess(eeg_obj)
            eeg_obj = self._frequency_analysis(eeg_obj)
            eeg_obj = self._connectivity_analysis(eeg_obj)
            
            # 添加新的分析方法
            if hasattr(eeg_obj, 'events') and eeg_obj.events is not None:
                eeg_obj = self._erp_analysis(eeg_obj)
                logger.info("事件相关电位分析完成")
            
            eeg_obj = self._time_frequency_analysis(eeg_obj)
            logger.info("时频分析完成")
            
            # 如果有通道位置信息，进行源定位分析
            if hasattr(eeg_obj, 'channel_locations') and len(eeg_obj.channel_locations) > 0:
                try:
                    eeg_obj = self._source_localization(eeg_obj)
                    logger.info("源定位分析完成")
                except Exception as e:
                    logger.warning(f"源定位分析失败: {e}")
            
            # 特征提取
            eeg_obj = self._extract_features(eeg_obj)
            
            # 如果有标签数据，进行机器学习分类
            if hasattr(eeg_obj, 'labels') and eeg_obj.labels is not None:
                try:
                    eeg_obj = self._apply_machine_learning(eeg_obj)
                    logger.info("机器学习分类完成")
                except Exception as e:
                    logger.warning(f"机器学习分类失败: {e}")
            
            # 生成摘要报告
            eeg_obj = self._generate_summary(eeg_obj)
            
            # 确保voltage_data属性存在
            if not hasattr(eeg_obj, 'voltage_data') or eeg_obj.voltage_data is None:
                if original_voltage_data is not None:
                    eeg_obj.voltage_data = original_voltage_data
            
            return eeg_obj
            
        except Exception as e:
            logger.error(f"EEG处理流程出错: {e}", exc_info=True)
            return eeg_obj
    
    def _preprocess(self, eeg_obj):
        """符合国际标准的EEG预处理
        
        包括:
        - 带通滤波 (0.5-100Hz)
        - 陷波滤波 (50/60Hz)
        - ICA去伪迹处理（眼电、肌电等）
        - 坏通道检测与插值
        - 基线校正
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            预处理后的EEG对象
        """
        try:
            # 检查原始数据是否存在
            if not hasattr(eeg_obj, 'voltage_data') or eeg_obj.voltage_data is None:
                logger.error("无法进行预处理：缺少电压数据")
                return eeg_obj
                
            # 获取采样率
            fs = eeg_obj.sampling_rate
            
            # 对每个通道进行滤波
            n_channels = eeg_obj.voltage_data.shape[0]
            filtered_data = np.zeros_like(eeg_obj.voltage_data)
            
            # 处理每个通道，避免一次性处理所有数据导致内存问题
            for ch_idx in range(n_channels):
                # 带通滤波
                channel_data = eeg_obj.voltage_data[ch_idx]
                filtered_channel = self._bandpass_filter(channel_data, 0.5, 100.0, fs)
                
                # 陷波滤波（电源噪声）
                filtered_channel = self._notch_filter(filtered_channel, 50.0, fs)  # 或60Hz，取决于地区
                
                # 存储滤波结果
                filtered_data[ch_idx] = filtered_channel
            
            # 更新对象，保留原始数据
            eeg_obj.filtered_data = filtered_data
            
            # 应用ICA去伪迹
            if hasattr(eeg_obj, 'channel_names') and len(eeg_obj.channel_names) > 0:
                try:
                    # 尝试使用MNE进行ICA处理
                    eeg_obj = self._apply_ica(eeg_obj)
                    logger.info("ICA去伪迹处理完成")
                except Exception as e:
                    logger.warning(f"ICA去伪迹处理失败: {e}")
            
            # 坏通道检测与插值
            if hasattr(eeg_obj, 'channel_names') and hasattr(eeg_obj, 'channel_locations'):
                try:
                    eeg_obj = self._detect_bad_channels(eeg_obj)
                    logger.info("坏通道检测与插值完成")
                except Exception as e:
                    logger.warning(f"坏通道检测失败: {e}")
            
            # 基线校正
            try:
                eeg_obj = self._baseline_correction(eeg_obj)
                logger.info("基线校正完成")
            except Exception as e:
                logger.warning(f"基线校正失败: {e}")
            
            # 确保voltage_data也被更新，这样后续处理和可视化都能使用
            eeg_obj.voltage_data = eeg_obj.filtered_data
            
            # 记录预处理信息
            eeg_obj.preprocessing_info = {
                'bandpass_filter': {'low': 0.5, 'high': 100.0},
                'notch_filter': {'freq': 50.0},
                'ica_applied': hasattr(eeg_obj, 'ica_components'),
                'bad_channels_detected': hasattr(eeg_obj, 'bad_channels'),
                'baseline_corrected': True,
                'timestamp': datetime.now().isoformat()
            }
            
            return eeg_obj
            
        except Exception as e:
            logger.error(f"EEG预处理错误: {e}", exc_info=True)
            return eeg_obj
    
    def _bandpass_filter(self, signal, low_freq, high_freq, fs):
        """带通滤波
        
        Args:
            signal: 信号数据
            low_freq: 低频截止频率
            high_freq: 高频截止频率
            fs: 采样率
            
        Returns:
            滤波后的信号
        """
        try:
            # 设计滤波器
            nyq = 0.5 * fs
            low = low_freq / nyq
            high = high_freq / nyq
            b, a = sp_signal.butter(4, [low, high], btype='band')
            # 应用滤波器
            filtered = sp_signal.filtfilt(b, a, signal)
            return filtered
        except Exception as e:
            logger.error(f"带通滤波错误: {e}")
            return signal
    
    def _notch_filter(self, signal, notch_freq, fs):
        """陷波滤波
        
        Args:
            signal: 信号数据
            notch_freq: 陷波频率
            fs: 采样率
            
        Returns:
            滤波后的信号
        """
        try:
            # 设计陷波滤波器
            nyq = 0.5 * fs
            quality_factor = 30.0  # 品质因子
            b, a = sp_signal.iirnotch(notch_freq / nyq, quality_factor)
            # 应用滤波器
            filtered = sp_signal.filtfilt(b, a, signal)
            return filtered
        except Exception as e:
            logger.error(f"陷波滤波错误: {e}")
            return signal
            
    def _apply_ica(self, eeg_obj):
        """应用ICA去伪迹
        
        使用独立成分分析(ICA)去除眼电、肌电等伪迹
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            处理后的EEG对象
        """
        try:
            # 获取数据和通道信息
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            ch_names = eeg_obj.channel_names
            fs = eeg_obj.sampling_rate
            
            # 创建MNE Raw对象
            info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # 应用ICA
            ica = ICA(n_components=min(15, len(ch_names)-1), random_state=42)
            ica.fit(raw)
            
            # 自动检测和移除眼电伪迹
            # 注意：这里使用简化的方法，实际应用中可能需要更复杂的方法
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            if eog_indices:
                ica.exclude = eog_indices
                logger.info(f"检测到 {len(eog_indices)} 个眼电伪迹成分")
            
            # 应用ICA去伪迹
            ica_raw = raw.copy()
            ica.apply(ica_raw)
            
            # 更新EEG对象
            eeg_obj.filtered_data = ica_raw.get_data()
            eeg_obj.ica_components = ica.get_components()
            eeg_obj.ica_excluded = ica.exclude
            
            # 保存ICA图
            if hasattr(self, 'plots_dir'):
                fig = ica.plot_components()
                fig_path = os.path.join(self.plots_dir, f"{eeg_obj.subject_id}_ica_components.png")
                plt.savefig(fig_path)
                plt.close(fig)
                
                fig = ica.plot_sources(raw)
                fig_path = os.path.join(self.plots_dir, f"{eeg_obj.subject_id}_ica_sources.png")
                plt.savefig(fig_path)
                plt.close(fig)
            
            return eeg_obj
        except Exception as e:
            logger.error(f"ICA去伪迹处理错误: {e}", exc_info=True)
            return eeg_obj
            
    def _detect_bad_channels(self, eeg_obj):
        """检测和插值坏通道
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            处理后的EEG对象
        """
        try:
            # 获取数据和通道信息
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            ch_names = eeg_obj.channel_names
            fs = eeg_obj.sampling_rate
            
            # 创建MNE Raw对象
            info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # 设置通道位置
            if hasattr(eeg_obj, 'channel_locations') and eeg_obj.channel_locations:
                # 创建montage
                pos = np.array(eeg_obj.channel_locations)
                if pos.shape[1] == 2:  # 如果只有x,y坐标，添加z=0
                    pos = np.hstack([pos, np.zeros((pos.shape[0], 1))])
                montage = mne.channels.make_dig_montage(
                    ch_pos={ch: pos[i] for i, ch in enumerate(ch_names)},
                    coord_frame='head'
                )
                raw.set_montage(montage)
            
            # 检测坏通道
            bad_channels = mne.preprocessing.find_bad_channels_maxwell(raw)
            if not bad_channels:  # 如果Maxwell方法没有找到，尝试其他方法
                # 使用统计方法检测异常通道
                data_std = np.std(data, axis=1)
                mean_std = np.mean(data_std)
                std_std = np.std(data_std)
                threshold = mean_std + 2 * std_std  # 2个标准差作为阈值
                
                bad_indices = np.where(data_std > threshold)[0]
                bad_channels = [ch_names[i] for i in bad_indices]
            
            # 标记坏通道
            if bad_channels:
                logger.info(f"检测到 {len(bad_channels)} 个坏通道: {bad_channels}")
                raw.info['bads'] = bad_channels
                
                # 插值坏通道
                if len(bad_channels) < len(ch_names) * 0.2:  # 如果坏通道不超过20%
                    raw_interp = raw.copy()
                    raw_interp.interpolate_bads()
                    
                    # 更新EEG对象
                    eeg_obj.filtered_data = raw_interp.get_data()
                    eeg_obj.bad_channels = bad_channels
                    eeg_obj.interpolated = True
                else:
                    logger.warning(f"坏通道过多 ({len(bad_channels)}), 跳过插值")
                    eeg_obj.bad_channels = bad_channels
                    eeg_obj.interpolated = False
            else:
                logger.info("未检测到坏通道")
                eeg_obj.bad_channels = []
                eeg_obj.interpolated = False
            
            return eeg_obj
        except Exception as e:
            logger.error(f"坏通道检测错误: {e}", exc_info=True)
            return eeg_obj
            
    def _baseline_correction(self, eeg_obj):
        """基线校正
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            处理后的EEG对象
        """
        try:
            # 获取数据
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            
            # 计算每个通道的基线（前10%的数据点）
            n_samples = data.shape[1]
            baseline_end = int(n_samples * 0.1)
            baseline = np.mean(data[:, :baseline_end], axis=1, keepdims=True)
            
            # 减去基线
            corrected_data = data - baseline
            
            # 更新EEG对象
            eeg_obj.filtered_data = corrected_data
            
            return eeg_obj
        except Exception as e:
            logger.error(f"基线校正错误: {e}", exc_info=True)
            return eeg_obj
    
    def _frequency_analysis(self, eeg_obj):
        """频域分析 - 计算频带能量
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            添加了频带能量的EEG对象
        """
        try:
            # 使用预处理后的数据，如果没有就用原始数据
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            
            if data is None:
                logger.error("无法进行频域分析：缺少数据")
                return eeg_obj
            
            # 初始化频带能量字典
            eeg_obj.band_powers = {}
            
            # 对每个通道计算频谱
            for ch_idx in range(data.shape[0]):
                # 计算功率谱密度
                freqs, psd = self._compute_psd(data[ch_idx], eeg_obj.sampling_rate)
                
                # 计算频带能量
                band_power = self._compute_band_power(freqs, psd)
                
                # 将每个频带的能量存储到对应的字典中
                for band_name, power_info in band_power.items():
                    if band_name not in eeg_obj.band_powers:
                        eeg_obj.band_powers[band_name] = np.zeros(data.shape[0])
                    eeg_obj.band_powers[band_name][ch_idx] = power_info['absolute']
            
            # 确保voltage_data属性存在
            if not hasattr(eeg_obj, 'voltage_data') or eeg_obj.voltage_data is None:
                if hasattr(eeg_obj, 'filtered_data') and eeg_obj.filtered_data is not None:
                    eeg_obj.voltage_data = eeg_obj.filtered_data
            
            return eeg_obj
            
        except Exception as e:
            logger.error(f"频域分析错误: {e}")
            return eeg_obj
    
    def _compute_psd(self, signal, fs):
        """计算功率谱密度
        
        Args:
            signal: 信号数据
            fs: 采样率
            
        Returns:
            freqs: 频率数组
            psd: 功率谱密度
        """
        # 使用Welch方法计算功率谱密度，更稳定
        freqs, psd = sp_signal.welch(signal, fs, nperseg=min(256, len(signal)//4), scaling='density')
        return freqs, psd
        
    def _erp_analysis(self, eeg_obj):
        """事件相关电位(ERP)分析
        
        对事件相关数据进行时间锁定平均，提取ERP波形
        
        Args:
            eeg_obj: EEG对象，必须包含events属性
            
        Returns:
            处理后的EEG对象
        """
        try:
            # 检查事件数据是否存在
            if not hasattr(eeg_obj, 'events') or eeg_obj.events is None:
                logger.warning("无法进行ERP分析：缺少事件数据")
                return eeg_obj
                
            # 获取数据和事件
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            events = eeg_obj.events
            fs = eeg_obj.sampling_rate
            
            # 初始化ERP结果字典
            eeg_obj.erp_data = {}
            
            # 对每种事件类型进行处理
            unique_event_types = np.unique([event['type'] for event in events if 'type' in event])
            
            for event_type in unique_event_types:
                # 提取该类型的所有事件
                type_events = [event for event in events if event.get('type') == event_type]
                
                # 提取事件时间点
                event_samples = [int(event['latency'] * fs) for event in type_events if 'latency' in event]
                
                # 定义ERP时间窗口（事件前200ms到事件后800ms）
                pre_samples = int(0.2 * fs)  # 200ms
                post_samples = int(0.8 * fs)  # 800ms
                
                # 提取每个事件的数据片段
                epochs = []
                for sample in event_samples:
                    if sample >= pre_samples and sample + post_samples < data.shape[1]:
                        epoch = data[:, sample-pre_samples:sample+post_samples]
                        epochs.append(epoch)
                
                if epochs:
                    # 计算ERP（时间锁定平均）
                    erp = np.mean(epochs, axis=0)
                    
                    # 创建时间轴
                    times = np.linspace(-0.2, 0.8, erp.shape[1])
                    
                    # 存储ERP数据
                    eeg_obj.erp_data[event_type] = {
                        'erp': erp,
                        'times': times,
                        'n_trials': len(epochs)
                    }
                    
                    # 绘制ERP图
                    if hasattr(self, 'plots_dir'):
                        plt.figure(figsize=(10, 6))
                        for ch_idx in range(min(9, erp.shape[0])):
                            plt.subplot(3, 3, ch_idx+1)
                            plt.plot(times, erp[ch_idx])
                            if hasattr(eeg_obj, 'channel_names'):
                                plt.title(eeg_obj.channel_names[ch_idx])
                            plt.axvline(x=0, color='r', linestyle='--')  # 事件发生时间
                            plt.xlabel('Time (s)')
                            plt.ylabel('Amplitude (μV)')
                        plt.tight_layout()
                        plt.suptitle(f'ERP for Event Type: {event_type} (n={len(epochs)})')
                        plt.savefig(os.path.join(self.plots_dir, f"{eeg_obj.subject_id}_erp_{event_type}.png"))
                        plt.close()
            
            return eeg_obj
        except Exception as e:
            logger.error(f"ERP分析错误: {e}", exc_info=True)
            return eeg_obj
            
    def _time_frequency_analysis(self, eeg_obj):
        """时频分析
        
        使用小波变换进行时频分析，计算不同频带随时间的能量变化
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            处理后的EEG对象
        """
        try:
            # 获取数据
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            fs = eeg_obj.sampling_rate
            
            # 初始化时频结果字典
            eeg_obj.time_frequency = {}
            
            # 对每个通道进行时频分析
            for ch_idx in range(min(4, data.shape[0])):  # 限制处理通道数，避免计算量过大
                channel_data = data[ch_idx]
                
                # 定义频率范围
                freqs = np.linspace(1, min(100, fs/2-1), 50)  # 1-100Hz，50个频率点
                
                # 使用连续小波变换
                n_cycles = freqs / 2.  # 频率越高，周期数越多
                power = mne.time_frequency.tfr_array_morlet(channel_data[np.newaxis, np.newaxis, :], 
                                                          sfreq=fs, freqs=freqs, 
                                                          n_cycles=n_cycles, output='power')
                
                # 存储结果
                ch_name = eeg_obj.channel_names[ch_idx] if hasattr(eeg_obj, 'channel_names') else f"ch{ch_idx}"
                eeg_obj.time_frequency[ch_name] = {
                    'power': power[0, 0],  # 去除多余维度
                    'freqs': freqs,
                    'times': np.arange(channel_data.shape[0]) / fs
                }
                
                # 绘制时频图
                if hasattr(self, 'plots_dir'):
                    plt.figure(figsize=(10, 6))
                    plt.imshow(np.log10(power[0, 0]), 
                              aspect='auto', 
                              origin='lower',
                              extent=[0, channel_data.shape[0]/fs, freqs[0], freqs[-1]])
                    plt.colorbar(label='Log Power')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Frequency (Hz)')
                    plt.title(f'Time-Frequency Analysis: {ch_name}')
                    plt.savefig(os.path.join(self.plots_dir, f"{eeg_obj.subject_id}_tf_{ch_name}.png"))
                    plt.close()
            
            return eeg_obj
        except Exception as e:
            logger.error(f"时频分析错误: {e}", exc_info=True)
            return eeg_obj
            
    def _source_localization(self, eeg_obj):
        """源定位分析
        
        使用MNE进行EEG源定位，定位脑电信号的来源
        
        Args:
            eeg_obj: EEG对象，必须包含channel_locations属性
            
        Returns:
            处理后的EEG对象
        """
        try:
            # 检查通道位置信息是否存在
            if not hasattr(eeg_obj, 'channel_locations') or not eeg_obj.channel_locations:
                logger.warning("无法进行源定位：缺少通道位置信息")
                return eeg_obj
                
            # 获取数据和通道信息
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            ch_names = eeg_obj.channel_names if hasattr(eeg_obj, 'channel_names') else [f"ch{i}" for i in range(data.shape[0])]
            fs = eeg_obj.sampling_rate
            
            # 创建MNE Raw对象
            info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # 设置通道位置
            pos = np.array(eeg_obj.channel_locations)
            if pos.shape[1] == 2:  # 如果只有x,y坐标，添加z=0
                pos = np.hstack([pos, np.zeros((pos.shape[0], 1))])
            montage = mne.channels.make_dig_montage(
                ch_pos={ch: pos[i] for i, ch in enumerate(ch_names)},
                coord_frame='head'
            )
            raw.set_montage(montage)
            
            # 使用标准头模型
            sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.085)  # 单位：米
            
            # 计算前向模型
            src = mne.setup_volume_source_space(sphere=sphere, pos=10.0)
            fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere)
            
            # 计算噪声协方差矩阵
            noise_cov = mne.compute_raw_covariance(raw, method='shrunk')
            
            # 使用eLORETA方法进行源定位
            inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2)
            stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0/9.0, method='eLORETA')
            
            # 存储源定位结果
            eeg_obj.source_data = {
                'data': stc.data,
                'vertices': stc.vertices,
                'times': stc.times
            }
            
            # 绘制源定位结果
            if hasattr(self, 'plots_dir'):
                # 找到最大活动时间点
                peak_time_idx = np.argmax(np.sum(stc.data**2, axis=0))
                peak_time = stc.times[peak_time_idx]
                
                # 绘制该时间点的源定位结果
                brain = stc.plot(subjects_dir=None, hemi='both', initial_time=peak_time,
                                surface='white', views=['lat', 'med'])
                brain.save_image(os.path.join(self.plots_dir, f"{eeg_obj.subject_id}_source_localization.png"))
            
            return eeg_obj
        except Exception as e:
            logger.error(f"源定位分析错误: {e}", exc_info=True)
            return eeg_obj
    
    def _compute_band_power(self, freqs, psd):
        """计算各频带能量
        
        Args:
            freqs: 频率数组
            psd: 功率谱密度
            
        Returns:
            band_power: 频带能量字典
        """
        band_power = {}
        
        # 计算各频带能量
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # 找到频带对应的索引
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # 计算频带能量
            if np.any(idx_band):
                # 绝对能量
                absolute_power = np.mean(psd[idx_band])
                
                # 相对能量（占总能量的比例）
                total_power = np.sum(psd)
                relative_power = absolute_power / total_power if total_power > 0 else 0
                
                band_power[band_name] = {
                    'absolute': float(absolute_power),
                    'relative': float(relative_power)
                }
            else:
                band_power[band_name] = {'absolute': 0.0, 'relative': 0.0}
        
        return band_power
    
    def _connectivity_analysis(self, eeg_obj):
        """连通性分析
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            添加了连通性指标的EEG对象
        """
        try:
            # 使用预处理后的数据
            data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
            
            if data is None or data.shape[0] < 2:
                logger.warning("无法进行连通性分析：通道数不足或数据缺失")
                return eeg_obj
            
            # 计算通道间相关性
            correlation_matrix = np.corrcoef(data)
            
            # 存储连通性指标
            eeg_obj.connectivity = {
                'correlation': correlation_matrix
            }
            
            return eeg_obj
            
        except Exception as e:
            logger.error(f"连通性分析错误: {e}", exc_info=True)
            return eeg_obj
    
    def _extract_features(self, eeg_obj):
        """特征提取
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            添加了特征的EEG对象
        """
        try:
            # 使用特征提取器提取特征
            if hasattr(self, 'feature_extractor'):
                # 使用预处理后的数据
                data = eeg_obj.filtered_data if hasattr(eeg_obj, 'filtered_data') else eeg_obj.voltage_data
                
                if data is None:
                    logger.error("无法进行特征提取：缺少数据")
                    return eeg_obj
                
                # 提取时域特征
                time_features = self.feature_extractor.extract_time_features(data)
                
                # 提取频域特征
                freq_features = {}
                if hasattr(eeg_obj, 'band_powers') and eeg_obj.band_powers:
                    freq_features = self.feature_extractor.extract_frequency_features(eeg_obj.band_powers, eeg_obj.sampling_rate)
                
                # 提取连通性特征
                conn_features = {}
                if hasattr(eeg_obj, 'connectivity') and eeg_obj.connectivity:
                    conn_features = self.feature_extractor.extract_connectivity_features(eeg_obj.connectivity)
                
                # 提取ERP特征（如果有ERP数据）
                erp_features = {}
                if hasattr(eeg_obj, 'erp_data') and eeg_obj.erp_data:
                    erp_features = self._extract_erp_features(eeg_obj.erp_data)
                
                # 提取时频特征（如果有时频数据）
                tf_features = {}
                if hasattr(eeg_obj, 'time_frequency') and eeg_obj.time_frequency:
                    tf_features = self._extract_time_frequency_features(eeg_obj.time_frequency)
                
                # 合并所有特征
                eeg_obj.features = {
                    'time_domain': time_features,
                    'frequency_domain': freq_features,
                    'connectivity': conn_features,
                    'erp': erp_features,
                    'time_frequency': tf_features
                }
                
                # 创建特征向量（用于机器学习）
                feature_vector = self._create_feature_vector(eeg_obj.features)
                eeg_obj.feature_vector = feature_vector
            
            return eeg_obj
            
        except Exception as e:
            logger.error(f"特征提取错误: {e}", exc_info=True)
            return eeg_obj
            
    def _extract_erp_features(self, erp_data):
        """从ERP数据中提取特征
        
        Args:
            erp_data: ERP数据字典
            
        Returns:
            ERP特征字典
        """
        features = {}
        
        for event_type, data in erp_data.items():
            erp = data['erp']
            times = data['times']
            
            # 提取P300特征（假设在250-500ms之间）
            p300_window = np.logical_and(times >= 0.25, times <= 0.5)
            if np.any(p300_window):
                # 对每个通道提取P300振幅和潜伏期
                p300_features = {}
                for ch_idx in range(erp.shape[0]):
                    ch_erp = erp[ch_idx, p300_window]
                    p300_amp = np.max(ch_erp)
                    p300_lat_idx = np.argmax(ch_erp)
                    p300_lat = times[p300_window][p300_lat_idx]
                    
                    p300_features[f'ch{ch_idx}'] = {
                        'amplitude': float(p300_amp),
                        'latency': float(p300_lat)
                    }
                
                features[f'{event_type}_p300'] = p300_features
            
            # 提取N200特征（假设在150-250ms之间）
            n200_window = np.logical_and(times >= 0.15, times <= 0.25)
            if np.any(n200_window):
                # 对每个通道提取N200振幅和潜伏期
                n200_features = {}
                for ch_idx in range(erp.shape[0]):
                    ch_erp = erp[ch_idx, n200_window]
                    n200_amp = np.min(ch_erp)  # N200是负波
                    n200_lat_idx = np.argmin(ch_erp)
                    n200_lat = times[n200_window][n200_lat_idx]
                    
                    n200_features[f'ch{ch_idx}'] = {
                        'amplitude': float(n200_amp),
                        'latency': float(n200_lat)
                    }
                
                features[f'{event_type}_n200'] = n200_features
        
        return features
    
    def _extract_time_frequency_features(self, tf_data):
        """从时频数据中提取特征
        
        Args:
            tf_data: 时频数据字典
            
        Returns:
            时频特征字典
        """
        features = {}
        
        for ch_name, data in tf_data.items():
            power = data['power']
            freqs = data['freqs']
            
            # 计算各频带的平均能量
            band_features = {}
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # 找到频带对应的索引
                band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                if np.any(band_idx):
                    # 计算频带平均能量随时间的变化
                    band_power = np.mean(power[band_idx, :], axis=0)
                    
                    # 提取特征：平均能量、最大能量、能量变化率
                    band_features[band_name] = {
                        'mean_power': float(np.mean(band_power)),
                        'max_power': float(np.max(band_power)),
                        'power_variance': float(np.var(band_power))
                    }
            
            features[ch_name] = band_features
        
        return features
    
    def _create_feature_vector(self, features_dict):
        """将特征字典转换为特征向量（用于机器学习）
        
        Args:
            features_dict: 特征字典
            
        Returns:
            特征向量和特征名称列表
        """
        feature_vector = []
        feature_names = []
        
        # 处理时域特征
        if 'time_domain' in features_dict and features_dict['time_domain']:
            for feature_name, value in features_dict['time_domain'].items():
                if isinstance(value, (int, float, np.number)):
                    feature_vector.append(float(value))
                    feature_names.append(f'time_{feature_name}')
                elif isinstance(value, (list, np.ndarray)):
                    for i, v in enumerate(value):
                        feature_vector.append(float(v))
                        feature_names.append(f'time_{feature_name}_{i}')
        
        # 处理频域特征
        if 'frequency_domain' in features_dict and features_dict['frequency_domain']:
            for band, band_features in features_dict['frequency_domain'].items():
                for feature_name, value in band_features.items():
                    if isinstance(value, (int, float, np.number)):
                        feature_vector.append(float(value))
                        feature_names.append(f'freq_{band}_{feature_name}')
        
        # 处理连通性特征
        if 'connectivity' in features_dict and features_dict['connectivity']:
            for conn_type, conn_features in features_dict['connectivity'].items():
                if isinstance(conn_features, (int, float, np.number)):
                    feature_vector.append(float(conn_features))
                    feature_names.append(f'conn_{conn_type}')
                elif isinstance(conn_features, dict):
                    for feature_name, value in conn_features.items():
                        if isinstance(value, (int, float, np.number)):
                            feature_vector.append(float(value))
                            feature_names.append(f'conn_{conn_type}_{feature_name}')
        
        return {'vector': np.array(feature_vector), 'names': feature_names}
    def _apply_machine_learning(self, eeg_obj):
        """应用机器学习分类
        
        使用机器学习算法对EEG特征进行分类
        
        Args:
            eeg_obj: EEG对象，必须包含feature_vector和labels属性
            
        Returns:
            处理后的EEG对象
        """
        try:
            # 检查特征向量和标签是否存在
            if not hasattr(eeg_obj, 'feature_vector') or not hasattr(eeg_obj, 'labels'):
                logger.warning("无法进行机器学习分类：缺少特征向量或标签")
                return eeg_obj
            
            # 获取特征向量和标签
            X = eeg_obj.feature_vector['vector']
            y = np.array(eeg_obj.labels)
            
            # 初始化分类结果字典
            eeg_obj.ml_results = {}
            
            # 标准化特征
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 应用多种分类器
            classifiers = {
                'svm': SVC(kernel='rbf', probability=True),
                'random_forest': RandomForestClassifier(n_estimators=100)
            }
            
            for clf_name, clf in classifiers.items():
                # 使用交叉验证评估分类器
                cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
                
                # 在全部数据上训练分类器
                clf.fit(X_scaled, y)
                
                # 预测概率
                y_pred = clf.predict(X_scaled)
                if hasattr(clf, 'predict_proba'):
                    y_prob = clf.predict_proba(X_scaled)
                else:
                    y_prob = None
                
                # 计算准确率
                accuracy = np.mean(y_pred == y)
                
                # 存储结果
                eeg_obj.ml_results[clf_name] = {
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                    'accuracy': float(accuracy),
                    'predictions': y_pred.tolist(),
                    'probabilities': y_prob.tolist() if y_prob is not None else None
                }
                
                # 保存模型
                if hasattr(self, 'models_dir'):
                    import joblib
                    model_path = os.path.join(self.models_dir, f"{eeg_obj.subject_id}_{clf_name}_model.pkl")
                    joblib.dump(clf, model_path)
                    
                    # 保存特征缩放器
                    scaler_path = os.path.join(self.models_dir, f"{eeg_obj.subject_id}_scaler.pkl")
                    joblib.dump(scaler, scaler_path)
                    
                    # 保存特征名称
                    feature_names_path = os.path.join(self.models_dir, f"{eeg_obj.subject_id}_feature_names.json")
                    with open(feature_names_path, 'w') as f:
                        import json
                        json.dump(eeg_obj.feature_vector['names'], f)
            
            # 绘制分类结果
            if hasattr(self, 'plots_dir'):
                plt.figure(figsize=(10, 6))
                for i, (clf_name, results) in enumerate(eeg_obj.ml_results.items()):
                    plt.bar(i, results['cv_mean'], yerr=results['cv_std'], capsize=10)
                plt.xticks(range(len(eeg_obj.ml_results)), list(eeg_obj.ml_results.keys()))
                plt.ylabel('Cross-Validation Accuracy')
                plt.title('Classification Performance')
                plt.ylim(0, 1)
                plt.savefig(os.path.join(self.plots_dir, f"{eeg_obj.subject_id}_classification_results.png"))
                plt.close()
            
            return eeg_obj
        except Exception as e:
            logger.error(f"机器学习分类错误: {e}", exc_info=True)
            return eeg_obj
            
    def _generate_summary(self, eeg_obj):
        """生成分析摘要
        
        Args:
            eeg_obj: EEG对象
            
        Returns:
            添加了摘要的EEG对象
        """
        try:
            # 初始化摘要字典
            summary = {
                'subject_id': eeg_obj.subject_id if hasattr(eeg_obj, 'subject_id') else 'unknown',
                'recording_date': datetime.now().strftime('%Y-%m-%d'),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': 0,
                'channels': 0,
                'band_power_summary': {},
                'feature_summary': {},
                'abnormality_index': 0.0
            }
            
            # 添加基本信息
            if hasattr(eeg_obj, 'voltage_data') and eeg_obj.voltage_data is not None:
                summary['channels'] = eeg_obj.voltage_data.shape[0]
                summary['duration'] = eeg_obj.voltage_data.shape[1] / eeg_obj.sampling_rate if hasattr(eeg_obj, 'sampling_rate') else 0
            
            # 添加频带能量摘要
            if hasattr(eeg_obj, 'band_powers') and eeg_obj.band_powers:
                for band_name, powers in eeg_obj.band_powers.items():
                    summary['band_power_summary'][band_name] = {
                        'mean': float(np.mean(powers)),
                        'std': float(np.std(powers)),
                        'max': float(np.max(powers)),
                        'min': float(np.min(powers))
                    }
            
            # 添加特征摘要
            if hasattr(eeg_obj, 'features') and eeg_obj.features:
                for domain, features in eeg_obj.features.items():
                    if features and isinstance(features, dict):
                        summary['feature_summary'][domain] = len(features)
            
            # 添加机器学习结果摘要
            if hasattr(eeg_obj, 'ml_results') and eeg_obj.ml_results:
                summary['ml_results'] = {}
                for clf_name, results in eeg_obj.ml_results.items():
                    summary['ml_results'][clf_name] = {
                        'accuracy': results['accuracy'],
                        'cv_mean': results['cv_mean']
                    }
            
            # 保存摘要
            eeg_obj.summary = summary
            
            # 保存摘要到文件
            if hasattr(eeg_obj, 'subject_id'):
                summary_path = os.path.join(self.save_dir, f"{eeg_obj.subject_id}_summary.json")
                try:
                    import json
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        # 将NumPy数组转换为列表以便JSON序列化
                        json_summary = self._convert_numpy_to_list(summary)
                        json.dump(json_summary, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    logger.error(f"保存摘要文件失败: {e}")
            
            return eeg_obj
            
        except Exception as e:
            logger.error(f"生成摘要错误: {e}", exc_info=True)
            return eeg_obj
    
    def _convert_numpy_to_list(self, obj):
        """将NumPy数组转换为列表，以便JSON序列化
        
        Args:
            obj: 包含NumPy数组的对象
            
        Returns:
            转换后的对象
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj