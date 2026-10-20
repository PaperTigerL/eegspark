import os
import numpy as np
import mne
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pyspark import RDD
from common.eegrdd import EEGRDD

@dataclass
class ProcessingConfig:
    """EEG处理配置"""
    # 滤波设置
    filter_lowpass: float = 40.0  # 低通滤波频率
    filter_highpass: float = 1.0   # 高通滤波频率
    filter_notch: float = 50.0     # 陷波滤波频率
    
    # 去伪迹设置
    artifact_threshold: float = 100.0  # 伪迹阈值(μV)
    ica_components: int = 20          # ICA组件数量
    
    # 时频分析设置
    freq_bands: Dict[str, Tuple[float, float]] = None  # 频带定义
    
    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 40)
            }

class EEGProcessor:
    """EEG数据处理器 - 结合MNE和Spark"""
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """初始化处理器
        
        Args:
            config: 处理配置，如果为None则使用默认配置
        """
        self.config = config or ProcessingConfig()
        self._montage = None
        self._load_standard_montage()
    
    def _load_standard_montage(self):
        """加载标准10-20系统电极位置"""
        try:
            self._montage = mne.channels.make_standard_montage('standard_1020')
        except Exception as e:
            print(f"警告: 无法加载标准电极位置: {e}")
    
    def eegrdd_to_mne(self, eegrdd: EEGRDD) -> mne.io.RawArray:
        """将EEGRDD转换为MNE Raw对象 - 增强版，处理各种边缘情况
        
        Args:
            eegrdd: EEGRDD对象
            
        Returns:
            mne.io.RawArray: MNE Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        # 首先验证电压数据
        if eegrdd.voltage_data is None:
            logger.error("电压数据为空，无法创建MNE对象")
            raise ValueError("电压数据为空，无法创建MNE对象")
            
        # 验证通道名称
        if not eegrdd.channel_names or len(eegrdd.channel_names) == 0:
            logger.warning("通道名称列表为空，创建默认通道名")
            eegrdd.channel_names = [f"Ch{i+1}" for i in range(eegrdd.voltage_data.shape[0])]
        
        # 确保通道名称数量与数据通道数匹配
        if eegrdd.voltage_data.shape[0] != len(eegrdd.channel_names):
            logger.warning(f"通道数量不匹配: 数据通道({eegrdd.voltage_data.shape[0]}) ≠ 名称数量({len(eegrdd.channel_names)})")
            
            # 调整channel_names以匹配数据维度
            if eegrdd.voltage_data.shape[0] > len(eegrdd.channel_names):
                # 如果数据通道数多于名称数量，添加默认通道名
                additional_channels = eegrdd.voltage_data.shape[0] - len(eegrdd.channel_names)
                for i in range(additional_channels):
                    eegrdd.channel_names.append(f"Ch{len(eegrdd.channel_names)+1}")
            else:
                # 如果数据通道数少于名称数量，截断通道名列表
                eegrdd.channel_names = eegrdd.channel_names[:eegrdd.voltage_data.shape[0]]
        
        # 创建通道信息
        ch_types = ['eeg'] * len(eegrdd.channel_names)  # 默认全部为EEG通道
        
        # 如果有通道类型信息，则使用实际的类型
        if eegrdd.channel_types:
            # 确保通道类型列表长度与通道名称列表长度一致
            if len(eegrdd.channel_types) != len(eegrdd.channel_names):
                # 调整通道类型列表长度
                if len(eegrdd.channel_types) > len(eegrdd.channel_names):
                    eegrdd.channel_types = eegrdd.channel_types[:len(eegrdd.channel_names)]
                else:
                    # 扩展通道类型列表，使用'eeg'作为默认类型
                    eegrdd.channel_types.extend(['eeg'] * (len(eegrdd.channel_names) - len(eegrdd.channel_types)))
            
            # 转换为小写并确保所有类型都是有效的
            valid_types = ['eeg', 'ecg', 'emg', 'eog', 'misc', 'stim', 'resp', 'bio']
            ch_types = []
            for ch_type in eegrdd.channel_types:
                if not ch_type:  # 处理None或空字符串
                    ch_types.append('eeg')
                    continue
                    
                ch_type_lower = ch_type.lower()
                if ch_type_lower not in valid_types:
                    ch_types.append('misc')
                else:
                    ch_types.append(ch_type_lower)
        
        # 创建MNE info对象
        try:
            logger.info(f"创建MNE info对象: {len(eegrdd.channel_names)}个通道, 采样率{eegrdd.sampling_rate}Hz")
            info = mne.create_info(
                ch_names=eegrdd.channel_names,
                sfreq=eegrdd.sampling_rate,
                ch_types=ch_types
            )
        except Exception as e:
            logger.error(f"创建MNE info对象失败: {e}")
            # 尝试使用简化的通道类型
            info = mne.create_info(
                ch_names=eegrdd.channel_names,
                sfreq=eegrdd.sampling_rate,
                ch_types=['eeg'] * len(eegrdd.channel_names)
            )
        
        # 创建Raw对象
        raw = mne.io.RawArray(eegrdd.voltage_data, info)
        
        # 添加事件标注
        if eegrdd.events and len(eegrdd.events) > 0:
            logger.info(f"处理事件标注: {len(eegrdd.events)}个事件")
            # 创建单个Annotations对象，而不是多个
            onset = []
            duration = []
            description = []
            for event in eegrdd.events:
                try:
                    onset.append(float(event.get('onset', 0)))
                    duration.append(float(event.get('duration', 0)))
                    description.append(str(event.get('type', 'unknown')))
                except Exception as e:
                    logger.warning(f"处理事件标注时出错: {e}, 跳过此事件")
            
            if onset:  # 如果有有效事件
                try:
                    # 创建单个Annotations对象
                    annotations = mne.Annotations(onset=onset, duration=duration, description=description)
                    raw.set_annotations(annotations)
                    logger.info(f"成功设置{len(onset)}个事件标注")
                except Exception as e:
                    logger.error(f"设置事件标注时出错: {e}")
        
        return raw
    
    def apply_montage(self, raw, eegrdd):
        """应用电极位置信息
        
        Args:
            raw: MNE Raw对象
            eegrdd: EEGRDD对象
            
        Returns:
            mne.io.Raw: 应用了电极位置的Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        # 如果没有通道位置信息，使用标准10-20系统
        if not eegrdd.channel_locations or len(eegrdd.channel_locations) == 0:
            if self._montage:
                try:
                    raw.set_montage(self._montage)
                    logger.info("应用标准10-20系统电极位置")
                except Exception as e:
                    logger.warning(f"应用标准电极位置时出错: {e}")
            return raw
            
        # 使用自定义电极位置
        try:
            # 创建通道位置字典
            dig_ch_pos = {}
            for ch_name, loc in zip(eegrdd.channel_names, eegrdd.channel_locations):
                if not loc:  # 跳过空位置
                    dig_ch_pos[ch_name] = [0.0, 0.0, 0.0]
                elif isinstance(loc, dict):  # 字典格式
                    if 'x' in loc and 'y' in loc and 'z' in loc:
                        dig_ch_pos[ch_name] = [loc['x'], loc['y'], loc['z']]
                    else:
                        dig_ch_pos[ch_name] = [0.0, 0.0, 0.0]
                else:  # 非字典格式
                    dig_ch_pos[ch_name] = [0.0, 0.0, 0.0]
            
            # 创建自定义montage
            montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos)
            raw.set_montage(montage)
            logger.info("应用自定义电极位置")
            return raw
        except Exception as e:
            logger.warning(f"应用自定义电极位置时出错: {e}")
            return raw
    
    def mne_to_eegrdd(self, raw: mne.io.Raw, original_eegrdd: EEGRDD) -> EEGRDD:
        """将MNE Raw对象转换回EEGRDD
        
        Args:
            raw: MNE Raw对象
            original_eegrdd: 原始EEGRDD对象（用于保留元数据）
            
        Returns:
            EEGRDD: 新的EEGRDD对象
        """
        # 获取数据
        data = raw.get_data()
        
        # 获取事件标注
        events = []
        if raw.annotations is not None:
            for annot in raw.annotations:
                events.append({
                    'onset': float(annot['onset']),
                    'duration': float(annot['duration']),
                    'type': str(annot['description'])
                })
        
        # 创建新的EEGRDD对象
        return EEGRDD(
            subject_id=original_eegrdd.subject_id,
            voltage_data=data,
            sampling_rate=raw.info['sfreq'],
            payload_id=original_eegrdd.payload_id,
            task_name=original_eegrdd.task_name,
            block_id=original_eegrdd.block_id,
            n_channels=len(raw.ch_names),
            n_samples=len(raw.times),
            duration=len(raw.times) / raw.info['sfreq'],
            reference=original_eegrdd.reference,
            channel_names=raw.ch_names,
            channel_types=[ch_type.upper() for ch_type in raw.get_channel_types()],
            channel_units=original_eegrdd.channel_units,
            channel_locations=original_eegrdd.channel_locations,
            events=events,
            annotations=events
        )
    
    def process_eegrdd(self, eegrdd: EEGRDD) -> EEGRDD:
        """处理单个EEGRDD对象 - 简化版，专注于格式转换和MNE处理
        
        Args:
            eegrdd: 输入的EEGRDD对象
            
        Returns:
            EEGRDD: 处理后的EEGRDD对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        # 记录输入EEGRDD的关键信息
        logger.info(f"接收到EEGRDD数据: subject_id={eegrdd.subject_id}, 数据形状: {eegrdd.voltage_data.shape if eegrdd.voltage_data is not None else 'None'}")
        
        # 检查通道信息是否匹配
        if eegrdd.voltage_data is not None and eegrdd.voltage_data.shape[0] != len(eegrdd.channel_names):
            logger.warning(f"通道数量不匹配: 数据通道({eegrdd.voltage_data.shape[0]}) ≠ 名称数量({len(eegrdd.channel_names)})")
            
            # 调整channel_names以匹配数据维度
            if eegrdd.voltage_data.shape[0] > len(eegrdd.channel_names):
                # 如果数据通道数多于名称数量，添加默认通道名
                additional_channels = eegrdd.voltage_data.shape[0] - len(eegrdd.channel_names)
                for i in range(additional_channels):
                    eegrdd.channel_names.append(f"Ch{len(eegrdd.channel_names)+1}")
                logger.info(f"已添加{additional_channels}个默认通道名")
            else:
                # 如果数据通道数少于名称数量，截断通道名列表
                eegrdd.channel_names = eegrdd.channel_names[:eegrdd.voltage_data.shape[0]]
                logger.info(f"已截断通道名列表至{eegrdd.voltage_data.shape[0]}个")
        
        # 检查通道类型是否匹配 - 简化日志
        if eegrdd.channel_types and len(eegrdd.channel_types) != len(eegrdd.channel_names):
            # 调整通道类型列表长度
            if len(eegrdd.channel_types) > len(eegrdd.channel_names):
                eegrdd.channel_types = eegrdd.channel_types[:len(eegrdd.channel_names)]
            else:
                # 扩展通道类型列表，使用'EEG'作为默认类型
                eegrdd.channel_types.extend(['EEG'] * (len(eegrdd.channel_names) - len(eegrdd.channel_types)))
        
        # 检查通道位置是否匹配 - 简化日志
        if eegrdd.channel_locations and len(eegrdd.channel_locations) != len(eegrdd.channel_names):
            # 调整通道位置列表长度
            if len(eegrdd.channel_locations) > len(eegrdd.channel_names):
                eegrdd.channel_locations = eegrdd.channel_locations[:len(eegrdd.channel_names)]
            else:
                # 扩展通道位置列表，使用空字典作为默认位置
                eegrdd.channel_locations.extend([{'x': 0.0, 'y': 0.0, 'z': 0.0}] * 
                                              (len(eegrdd.channel_names) - len(eegrdd.channel_locations)))
        
        # 转换为MNE对象并进行处理
        try:
            logger.info("开始转换为MNE格式并进行处理...")
            raw = self.eegrdd_to_mne(eegrdd)
            
            # 尝试进行MNE处理 - 取消注释以启用实际处理
            # 1. 滤波处理
            raw_filtered = self.filter_data(raw)
            logger.info("滤波处理完成")
            
            # 2. 设置参考
            raw_referenced = self.set_reference(raw_filtered)
            logger.info("参考设置完成")
            
            # 3. 去除伪迹
            raw_clean = self.remove_artifacts_auto(raw_referenced)
            logger.info("伪迹去除完成")
            
            # 4. 频谱分析
            band_powers = self.compute_band_power(raw_clean)
            logger.info("频带功率计算完成")
            
            # 5. 连接性分析
            connectivity = self.compute_connectivity(raw_clean, method='wpli')
            logger.info("连接性分析完成")
            
            # 6. 时频分析
            tf_results = self.compute_time_frequency(raw_clean, method='wavelet')
            logger.info("时频分析完成")
            
            # 7. 事件相关电位分析（如果有事件标注）
            erp_results = {}
            if raw_clean.annotations is not None and len(raw_clean.annotations) > 0:
                erp_results = self.compute_erp(raw_clean)
                logger.info("ERP分析完成")
            
            # 8. 源定位分析（如果有通道位置信息）
            source_results = {}
            if raw_clean.info['dig']:
                source_results = self.compute_source_localization(raw_clean)
                logger.info("源定位分析完成")
            
            # 转换回EEGRDD
            processed_eeg = self.mne_to_eegrdd(raw_clean, eegrdd)
            
            # 添加处理结果
            processed_eeg.band_powers = band_powers
            processed_eeg.connectivity = connectivity
            processed_eeg.time_frequency = tf_results
            processed_eeg.erp = erp_results
            processed_eeg.source = source_results
            
            logger.info("MNE处理完成并转换回EEGRDD格式")
            
            return processed_eeg
            
        except Exception as e:
            logger.error(f"MNE处理出错: {e}", exc_info=True)
            # 出错时返回原始数据
            return eegrdd
            logger.info("滤波处理完成")
            
            # 2. 设置参考
            raw_referenced = self.set_reference(raw_filtered)
            logger.info("参考设置完成")
            
            # 3. 去除伪迹 - 可选，取消注释启用
            # raw_clean = self.remove_artifacts_auto(raw_referenced)
            # logger.info("伪迹去除完成")
            
            # 转换回EEGRDD
            processed_eeg = self.mne_to_eegrdd(raw_referenced, eegrdd)
            logger.info("MNE处理完成并转换回EEGRDD格式")
            
            return processed_eeg
            
        except Exception as e:
            logger.error(f"MNE处理出错: {e}", exc_info=True)
            # 出错时返回原始数据
            return eegrdd
    
    def process_eegrdd_batch(self, eegrdd_rdd: RDD) -> RDD:
        """批量处理EEGRDD对象
        
        Args:
            eegrdd_rdd: EEGRDD对象的RDD
            
        Returns:
            RDD: 处理后的EEGRDD对象RDD
        """
        return eegrdd_rdd.map(self.process_eegrdd)
    
    def extract_features(self, eegrdd: EEGRDD) -> Dict[str, Any]:
        """从EEGRDD中提取特征
        
        Args:
            eegrdd: EEGRDD对象
            
        Returns:
            Dict: 特征字典
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始提取特征: subject_id={eegrdd.subject_id}, task_name={eegrdd.task_name}")
        
        # 转换为MNE对象
        raw = self.eegrdd_to_mne(eegrdd)
        
        # 计算功率谱密度 - 使用新版MNE API
        logger.info("计算功率谱密度...")
        spectrum = raw.compute_psd()
        psds = spectrum.get_data()
        freqs = spectrum.freqs
        logger.info(f"功率谱密度计算完成，频率范围: {freqs[0]:.1f}-{freqs[-1]:.1f}Hz")
        
        # 计算各频带的平均功率
        band_powers = {}
        for band_name, (fmin, fmax) in self.config.freq_bands.items():
            logger.info(f"计算{band_name}频带({fmin}-{fmax}Hz)的平均功率...")
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psds[:, freq_mask], axis=1)
            band_powers[band_name] = band_power.tolist()
            logger.info(f"{band_name}频带功率计算完成")
        
        logger.info("特征提取完成")
        # 返回特征字典
        return {
            'subject_id': eegrdd.subject_id,
            'task_name': eegrdd.task_name,
            'band_powers': band_powers,
            'channels': eegrdd.channel_names
        }
        
    def filter_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """应用滤波器处理EEG数据
        
        Args:
            raw: MNE Raw对象
            
        Returns:
            mne.io.Raw: 滤波后的Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始滤波处理: {self.config.filter_highpass}-{self.config.filter_lowpass}Hz")
        
        # 应用带通滤波
        try:
            raw_filtered = raw.copy()
            raw_filtered.filter(l_freq=self.config.filter_highpass, h_freq=self.config.filter_lowpass)
        except Exception as e:
            logger.error(f"带通滤波失败: {e}")
            return raw
        
        # 应用陷波滤波（去除电源干扰）
        if self.config.filter_notch > 0:
            try:
                raw_filtered.notch_filter(freqs=[self.config.filter_notch])
            except Exception as e:
                logger.error(f"陷波滤波失败: {e}")
        
        return raw_filtered
    
    def set_reference(self, raw: mne.io.Raw, reference='average') -> mne.io.Raw:
        """设置EEG参考方式
        
        Args:
            raw: MNE Raw对象
            reference: 参考方式，可以是'average'或特定通道名称列表
            
        Returns:
            mne.io.Raw: 重参考后的Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"设置参考: {reference}")
        
        try:
            raw_referenced = raw.copy()
            raw_referenced.set_eeg_reference(ref_channels=reference)
            return raw_referenced
        except Exception as e:
            logger.error(f"设置参考失败: {e}")
            return raw
    
    def remove_artifacts_auto(self, raw: mne.io.Raw) -> mne.io.Raw:
        """自动去除伪迹
        
        Args:
            raw: MNE Raw对象
            
        Returns:
            mne.io.Raw: 去伪迹后的Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info("开始自动去除伪迹...")
        
        # 复制原始数据
        raw_clean = raw.copy()
        
        # 1. 去除大幅值伪迹
        logger.info(f"去除大幅值伪迹，阈值: {self.config.artifact_threshold}μV")
        try:
            # 获取数据
            data = raw_clean.get_data()
            
            # 标记超过阈值的数据点
            mask = np.abs(data) > self.config.artifact_threshold
            n_artifacts = np.sum(mask)
            
            if n_artifacts > 0:
                # 将伪迹替换为NaN，后续可以插值
                data[mask] = np.nan
                logger.info(f"检测到{n_artifacts}个伪迹数据点，已替换为NaN")
                
                # 使用插值替换NaN值
                from scipy import interpolate
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx]
                    nan_indices = np.isnan(ch_data)
                    if np.any(nan_indices):
                        non_nan_indices = ~nan_indices
                        if np.sum(non_nan_indices) > 0:  # 确保有足够的非NaN点进行插值
                            f = interpolate.interp1d(
                                np.where(non_nan_indices)[0], 
                                ch_data[non_nan_indices],
                                kind='linear', 
                                bounds_error=False, 
                                fill_value="extrapolate"
                            )
                            ch_data[nan_indices] = f(np.where(nan_indices)[0])
                
                # 更新Raw对象的数据
                raw_clean._data = data
                logger.info("已完成伪迹插值")
            else:
                logger.info("未检测到超过阈值的伪迹")
                
        except Exception as e:
            logger.error(f"去除大幅值伪迹失败: {e}")
        
        # 2. 应用ICA去除眼电和肌电伪迹
        if self.config.ica_components > 0:
            logger.info(f"应用ICA去除伪迹，组件数量: {self.config.ica_components}")
            try:
                # 创建ICA对象
                ica = mne.preprocessing.ICA(
                    n_components=self.config.ica_components,
                    random_state=42
                )
                
                # 应用ICA
                ica.fit(raw_clean)
                logger.info("ICA拟合完成")
                
                # 自动检测眼电伪迹组件
                try:
                    eog_indices, eog_scores = ica.find_bads_eog(raw_clean)
                    if eog_indices:
                        logger.info(f"检测到{len(eog_indices)}个眼电伪迹组件: {eog_indices}")
                        ica.exclude = eog_indices
                except Exception as e:
                    logger.warning(f"自动检测眼电伪迹失败: {e}")
                
                # 应用ICA去除伪迹组件
                ica.apply(raw_clean)
                logger.info("ICA伪迹去除完成")
                
            except Exception as e:
                logger.error(f"ICA去除伪迹失败: {e}")
        
        logger.info("自动去除伪迹完成")
        return raw_clean
    
    def compute_psd(self, raw: mne.io.Raw, fmin=0, fmax=100, method='welch') -> Tuple[np.ndarray, np.ndarray]:
        """计算功率谱密度
        
        Args:
            raw: MNE Raw对象
            fmin: 最小频率
            fmax: 最大频率
            method: 计算方法，'welch'或'multitaper'
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (psds, freqs) - 功率谱密度和对应频率
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始计算功率谱密度，方法: {method}, 频率范围: {fmin}-{fmax}Hz")
        
        try:
            if method == 'welch':
                spectrum = raw.compute_psd()
                psds = spectrum.get_data()
                freqs = spectrum.freqs
                logger.info("Welch方法功率谱密度计算完成")
            elif method == 'multitaper':
                psds, freqs = mne.time_frequency.psd_multitaper(
                    raw, fmin=fmin, fmax=fmax
                )
                logger.info("Multitaper方法功率谱密度计算完成")
            else:
                logger.error(f"不支持的功率谱密度计算方法: {method}")
                return None, None
                
            logger.info(f"功率谱密度计算完成，频率点数: {len(freqs)}")
            return psds, freqs
            
        except Exception as e:
            logger.error(f"计算功率谱密度失败: {e}")
            return None, None
    
    def compute_band_power(self, raw: mne.io.Raw) -> Dict[str, np.ndarray]:
        """计算各频带功率
        
        Args:
            raw: MNE Raw对象
            
        Returns:
            Dict[str, np.ndarray]: 各频带的功率
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info("开始计算频带功率...")
        
        try:
            # 计算功率谱密度
            psds, freqs = self.compute_psd(raw)
            if psds is None or freqs is None:
                logger.error("无法计算频带功率，功率谱密度计算失败")
                return {}
            
            # 计算各频带的平均功率
            band_powers = {}
            for band_name, (fmin, fmax) in self.config.freq_bands.items():
                logger.info(f"计算{band_name}频带({fmin}-{fmax}Hz)的平均功率...")
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                if np.sum(freq_mask) > 0:  # 确保频带内有频率点
                    band_power = np.mean(psds[:, freq_mask], axis=1)
                    band_powers[band_name] = band_power
                    logger.info(f"{band_name}频带功率计算完成")
                else:
                    logger.warning(f"{band_name}频带({fmin}-{fmax}Hz)内没有频率点，跳过")
            
            logger.info("频带功率计算完成")
            return band_powers
            
        except Exception as e:
            logger.error(f"计算频带功率失败: {e}")
            return {}
    
    def compute_connectivity(self, raw: mne.io.Raw, method='wpli', fmin=8, fmax=13) -> np.ndarray:
        """计算脑连接性
        
        Args:
            raw: MNE Raw对象
            method: 连接性计算方法，可选'coh'(相干性)、'plv'(相位锁定值)、'wpli'(加权相位滞后指数)
            fmin: 最小频率
            fmax: 最大频率
            
        Returns:
            np.ndarray: 连接性矩阵
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始计算脑连接性，方法: {method}, 频率范围: {fmin}-{fmax}Hz")
        
        try:
            # 准备数据
            raw_copy = raw.copy()
            
            # 计算连接性
            # 创建epochs以进行连接性分析
            # 如果没有事件，创建虚拟事件
            if len(raw_copy.annotations) == 0:
                # 创建虚拟事件，每秒一个
                onsets = np.arange(0, raw_copy.times[-1], 1.0)
                durations = np.zeros_like(onsets)
                descriptions = ['event'] * len(onsets)
                raw_copy.set_annotations(mne.Annotations(onsets, durations, descriptions))
            
            # 从注释创建事件
            events, event_id = mne.events_from_annotations(raw_copy)
            
            # 创建epochs，使用更宽松的阈值
            reject = None  # 不使用自动拒绝
            epochs = mne.Epochs(raw_copy, events, event_id=event_id, tmin=0, tmax=1.0,
                              baseline=None, preload=True, reject=reject)
            
            if method == 'coh':
                # 计算相干性
                from mne.connectivity import spectral_connectivity_epochs
                con = spectral_connectivity_epochs(
                    epochs, method='coh', mode='multitaper', 
                    fmin=fmin, fmax=fmax, faverage=True
                )
                logger.info("相干性计算完成")
                return con.get_data()
                
            elif method == 'plv':
                # 计算相位锁定值
                from mne.connectivity import spectral_connectivity_epochs
                con = spectral_connectivity_epochs(
                    epochs, method='plv', mode='multitaper',
                    fmin=fmin, fmax=fmax, faverage=True
                )
                logger.info("相位锁定值计算完成")
                return con.get_data()
                
            elif method == 'wpli':
                # 计算加权相位滞后指数
                from mne.connectivity import spectral_connectivity_epochs
                con = spectral_connectivity_epochs(
                    epochs, method='wpli', mode='multitaper',
                    fmin=fmin, fmax=fmax, faverage=True
                )
                logger.info("加权相位滞后指数计算完成")
                return con
                
            else:
                logger.error(f"不支持的连接性计算方法: {method}")
                return None
                
        except Exception as e:
            logger.error(f"计算脑连接性失败: {e}")
            return None
    
    def compute_time_frequency(self, raw: mne.io.Raw, method='wavelet', freqs=None, n_cycles=None) -> Dict[str, Any]:
        """计算时频分析
        
        Args:
            raw: MNE Raw对象
            method: 时频分析方法，可选'wavelet'(小波变换)、'stft'(短时傅里叶变换)
            freqs: 频率点，如果为None则自动生成
            n_cycles: 小波周期数，如果为None则自动设置
            
        Returns:
            Dict: 时频分析结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始时频分析，方法: {method}")
        
        try:
            # 准备数据
            raw_copy = raw.copy()
            
            # 设置默认参数 - 针对实时短数据流优化
            if freqs is None:
                freqs = np.arange(4, 41, 1)  # 4-40Hz，提高最低频率以适应短数据
            
            if n_cycles is None:
                # 减小n_cycles值以适应短数据流
                n_cycles = np.ones_like(freqs) * 3  # 固定为3个周期，适合实时短数据
            
            # 计算时频表示
            if method == 'wavelet':
                # 使用Morlet小波变换
                from mne.time_frequency import tfr_array_morlet
                
                # 获取数据
                data = raw_copy.get_data()
                sfreq = raw_copy.info['sfreq']
                
                # 记录数据形状
                logger.info(f"时频分析数据形状: {data.shape}")
                
                # 重新组织数据形状以符合tfr_array_morlet的要求
                # tfr_array_morlet期望输入形状为(n_epochs, n_chans, n_times)
                # 而我们的数据形状为(n_chans, n_times)
                # 因此需要添加一个epoch维度
                data = data.reshape(1, *data.shape)  # 变为(1, n_chans, n_times)
                
                # 计算小波变换
                logger.info(f"使用Morlet小波变换计算时频表示，频率点数: {len(freqs)}，重塑后数据形状: {data.shape}")
                tfr = tfr_array_morlet(data, sfreq, freqs, n_cycles=n_cycles, output='power')
                
                # 如果结果是(1, n_chans, n_freqs, n_times)，去掉第一个维度
                if tfr.shape[0] == 1:
                    tfr = tfr[0]
                
                logger.info("小波变换时频分析完成")
                return {
                    'tfr': tfr,
                    'freqs': freqs,
                    'times': raw_copy.times,
                    'method': 'wavelet'
                }
                
            elif method == 'stft':
                # 使用短时傅里叶变换
                from scipy import signal
                
                # 获取数据
                data = raw_copy.get_data()
                sfreq = raw_copy.info['sfreq']
                
                # 计算STFT
                logger.info("使用短时傅里叶变换计算时频表示")
                n_fft = int(sfreq * 2)  # 2秒窗口
                n_overlap = int(n_fft * 0.5)  # 50%重叠
                
                # 为每个通道计算STFT
                tfr = []
                for ch_idx in range(data.shape[0]):
                    f, t, Zxx = signal.stft(data[ch_idx], fs=sfreq, nperseg=n_fft, noverlap=n_overlap)
                    tfr.append(np.abs(Zxx)**2)  # 功率谱
                
                tfr = np.array(tfr)
                
                logger.info("短时傅里叶变换时频分析完成")
                return {
                    'tfr': tfr,
                    'freqs': f,
                    'times': t,
                    'method': 'stft'
                }
                
            else:
                logger.error(f"不支持的时频分析方法: {method}")
                return {}
                
        except Exception as e:
            logger.error(f"时频分析失败: {e}", exc_info=True)
            return {}
    
    def compute_erp(self, raw: mne.io.Raw, event_id=None, tmin=-0.2, tmax=1.0) -> Dict[str, Any]:
        """计算事件相关电位(ERP)
        
        Args:
            raw: MNE Raw对象，必须包含事件标注
            event_id: 事件ID字典，如果为None则自动从标注中提取
            tmin: 事件前时间窗口(秒)
            tmax: 事件后时间窗口(秒)
            
        Returns:
            Dict: ERP分析结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始ERP分析，时间窗口: {tmin}s到{tmax}s")
        
        try:
            # 检查是否有事件标注
            if raw.annotations is None or len(raw.annotations) == 0:
                logger.error("无法进行ERP分析：缺少事件标注")
                return {}
            
            # 从标注中提取事件
            events, event_dict = mne.events_from_annotations(raw)
            
            if events.shape[0] == 0:
                logger.error("无法从标注中提取事件")
                return {}
                
            logger.info(f"从标注中提取到{events.shape[0]}个事件")
            
            # 如果没有提供event_id，使用提取的event_dict
            if event_id is None:
                event_id = event_dict
                
            logger.info(f"事件类型: {event_id}")
            
            # 创建Epochs对象 - 使用更宽松的阈值设置
            # 设置适合EEG数据的阈值，±100μV
            reject = {'eeg': 100e-6}  # 100 μV，单位为伏特
            # 或者可以使用百分比阈值
            # reject = None
            # reject_by_annotation = True
            
            epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                              baseline=(tmin, 0), preload=True, reject=reject)
            
            logger.info(f"创建了{len(epochs)}个epoch")
            
            # 计算ERP
            evoked_dict = {}
            for event_name in event_id.keys():
                try:
                    evoked = epochs[event_name].average()
                    evoked_dict[event_name] = evoked
                    logger.info(f"计算了'{event_name}'事件的ERP")
                except Exception as e:
                    logger.warning(f"计算'{event_name}'事件的ERP失败: {e}")
            
            logger.info("ERP分析完成")
            return {
                'epochs': epochs,
                'evoked': evoked_dict,
                'event_id': event_id
            }
            
        except Exception as e:
            logger.error(f"ERP分析失败: {e}", exc_info=True)
            return {}
    
    def compute_source_localization(self, raw: mne.io.Raw, method='eLORETA') -> Dict[str, Any]:
        """计算源定位
        
        Args:
            raw: MNE Raw对象，必须包含通道位置信息
            method: 源定位方法，可选'eLORETA'、'MNE'、'dSPM'
            
        Returns:
            Dict: 源定位结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始源定位分析，方法: {method}")
        
        try:
            # 检查是否有通道位置信息
            if not raw.info['dig']:
                logger.error("无法进行源定位：缺少通道位置信息")
                return {}
            
            # 使用标准头模型
            sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.085)  # 单位：米
            
            # 创建源空间
            src = mne.setup_volume_source_space(sphere=sphere, pos=10.0)
            
            # 计算前向模型
            fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
            
            # 计算噪声协方差矩阵
            logger.info("开始计算噪声协方差矩阵...")
            try:
                # 检查是否有事件标注
                if raw.annotations is not None and len(raw.annotations) > 0:
                    logger.info(f"使用现有的{len(raw.annotations)}个事件标注计算协方差矩阵")
                    noise_cov = mne.compute_raw_covariance(raw, method='shrunk')
                else:
                    # 如果没有事件标注，创建固定长度事件
                    logger.info("没有事件标注，创建固定长度事件")
                    # 记录数据长度
                    data_len = len(raw.times) / raw.info['sfreq']
                    logger.info(f"数据长度: {data_len}秒")
                    
                    # 创建固定长度事件，每2秒一个事件
                    events = mne.make_fixed_length_events(raw, duration=2.0)
                    logger.info(f"创建了{len(events)}个固定长度事件")
                    
                    # 使用这些事件创建epochs
                    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
                    logger.info(f"创建了{len(epochs)}个epoch")
                    
                    # 使用epochs计算协方差矩阵
                    noise_cov = mne.compute_covariance(epochs, method='shrunk')
            except Exception as e:
                logger.error(f"计算噪声协方差矩阵失败: {e}，使用空协方差矩阵")
                # 创建一个空的协方差矩阵
                noise_cov = mne.Covariance(data=np.eye(len(raw.ch_names)), names=raw.ch_names)
            
            # 计算逆算子
            inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2)
            
            # 应用逆算子
            if method == 'eLORETA':
                stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0/9.0, method='eLORETA')
            elif method == 'MNE':
                stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0/9.0, method='MNE')
            elif method == 'dSPM':
                stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0/9.0, method='dSPM')
            else:
                logger.error(f"不支持的源定位方法: {method}")
                return {}
            
            logger.info("源定位分析完成")
            return {
                'stc': stc,
                'src': src,
                'fwd': fwd,
                'inv': inv,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"源定位分析失败: {e}", exc_info=True)
            return {}
    
    def compute_statistics(self, data1, data2, test_type='ttest') -> Dict[str, Any]:
        """计算统计分析
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            test_type: 统计检验类型，可选'ttest'、'wilcoxon'、'permutation'
            
        Returns:
            Dict: 统计分析结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始统计分析，检验类型: {test_type}")
        
        try:
            from scipy import stats
            
            if test_type == 'ttest':
                # 配对t检验
                t_val, p_val = stats.ttest_rel(data1, data2)
                logger.info("配对t检验完成")
                return {
                    'test_type': 'ttest',
                    't_value': t_val,
                    'p_value': p_val
                }
                
            elif test_type == 'wilcoxon':
                # Wilcoxon符号秩检验
                w_val, p_val = stats.wilcoxon(data1, data2)
                logger.info("Wilcoxon符号秩检验完成")
                return {
                    'test_type': 'wilcoxon',
                    'w_value': w_val,
                    'p_value': p_val
                }
                
            elif test_type == 'permutation':
                # 排列检验
                from mne.stats import permutation_t_test
                t_obs, p_val, h0 = permutation_t_test([data1, data2], n_permutations=1000)
                logger.info("排列检验完成")
                return {
                    'test_type': 'permutation',
                    't_observed': t_obs,
                    'p_value': p_val,
                    'h0': h0
                }
                
            else:
                logger.error(f"不支持的统计检验类型: {test_type}")
                return {}
                
        except Exception as e:
            logger.error(f"统计分析失败: {e}", exc_info=True)
            return {}
    
    def process_eegrdd(self, eegrdd: EEGRDD) -> EEGRDD:
        """处理单个EEGRDD对象 - 增强版，包含更多MNE处理功能
        
        Args:
            eegrdd: 输入的EEGRDD对象
            
        Returns:
            EEGRDD: 处理后的EEGRDD对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        # 记录输入EEGRDD的关键信息
        logger.info(f"接收到EEGRDD数据: subject_id={eegrdd.subject_id}, 数据形状: {eegrdd.voltage_data.shape if eegrdd.voltage_data is not None else 'None'}")
        
        # 检查通道信息是否匹配
        if eegrdd.voltage_data is not None and eegrdd.voltage_data.shape[0] != len(eegrdd.channel_names):
            logger.warning(f"通道数量不匹配: 数据通道({eegrdd.voltage_data.shape[0]}) ≠ 名称数量({len(eegrdd.channel_names)})")
            
            # 调整channel_names以匹配数据维度
            if eegrdd.voltage_data.shape[0] > len(eegrdd.channel_names):
                # 如果数据通道数多于名称数量，添加默认通道名
                additional_channels = eegrdd.voltage_data.shape[0] - len(eegrdd.channel_names)
                for i in range(additional_channels):
                    eegrdd.channel_names.append(f"Ch{len(eegrdd.channel_names)+1}")
                logger.info(f"已添加{additional_channels}个默认通道名")
            else:
                # 如果数据通道数少于名称数量，截断通道名列表
                eegrdd.channel_names = eegrdd.channel_names[:eegrdd.voltage_data.shape[0]]
                logger.info(f"已截断通道名列表至{eegrdd.voltage_data.shape[0]}个")
        
        # 检查通道类型是否匹配 - 简化日志
        if eegrdd.channel_types and len(eegrdd.channel_types) != len(eegrdd.channel_names):
            # 调整通道类型列表长度
            if len(eegrdd.channel_types) > len(eegrdd.channel_names):
                eegrdd.channel_types = eegrdd.channel_types[:len(eegrdd.channel_names)]
            else:
                # 扩展通道类型列表，使用'EEG'作为默认类型
                eegrdd.channel_types.extend(['EEG'] * (len(eegrdd.channel_names) - len(eegrdd.channel_types)))
        
        # 检查通道位置是否匹配 - 简化日志
        if eegrdd.channel_locations and len(eegrdd.channel_locations) != len(eegrdd.channel_names):
            # 调整通道位置列表长度
            if len(eegrdd.channel_locations) > len(eegrdd.channel_names):
                eegrdd.channel_locations = eegrdd.channel_locations[:len(eegrdd.channel_names)]
            else:
                # 扩展通道位置列表，使用空字典作为默认位置
                eegrdd.channel_locations.extend([{'x': 0.0, 'y': 0.0, 'z': 0.0}] * 
                                              (len(eegrdd.channel_names) - len(eegrdd.channel_locations)))
        
        # 转换为MNE对象并进行处理
        try:
            logger.info("开始转换为MNE格式并进行处理...")
            raw = self.eegrdd_to_mne(eegrdd)
            
            # 应用电极位置
            raw = self.apply_montage(raw, eegrdd)
            
            # 1. 滤波处理
            raw_filtered = self.filter_data(raw)
            logger.info("滤波处理完成")
            
            # 2. 设置参考
            raw_referenced = self.set_reference(raw_filtered)
            logger.info("参考设置完成")
            
            # 3. 去除伪迹
            raw_clean = self.remove_artifacts_auto(raw_referenced)
            logger.info("伪迹去除完成")
            
            # 4. 频谱分析
            band_powers = self.compute_band_power(raw_clean)
            logger.info("频带功率计算完成")
            
            # 5. 连接性分析
            connectivity = self.compute_connectivity(raw_clean, method='wpli')
            logger.info("连接性分析完成")
            
            # 6. 时频分析
            tf_results = self.compute_time_frequency(raw_clean, method='wavelet')
            logger.info("时频分析完成")
            
            # 7. 事件相关电位分析（如果有事件标注）
            erp_results = {}
            if raw_clean.annotations is not None and len(raw_clean.annotations) > 0:
                erp_results = self.compute_erp(raw_clean)
                logger.info("ERP分析完成")
            
            # 8. 源定位分析（如果有通道位置信息）
            source_results = {}
            if raw_clean.info['dig']:
                source_results = self.compute_source_localization(raw_clean)
                logger.info("源定位分析完成")
            
            # 转换回EEGRDD
            processed_eeg = self.mne_to_eegrdd(raw_clean, eegrdd)
            
            # 添加处理结果
            processed_eeg.band_powers = band_powers
            processed_eeg.connectivity = connectivity
            processed_eeg.time_frequency = tf_results
            processed_eeg.erp = erp_results
            processed_eeg.source = source_results
            
            logger.info("MNE处理完成并转换回EEGRDD格式")
            
            return processed_eeg
            
        except Exception as e:
            logger.error(f"MNE处理出错: {e}", exc_info=True)
            # 出错时返回原始数据
            return eegrdd
            logger.info("滤波处理完成")
            
            # 2. 设置参考
            raw_referenced = self.set_reference(raw_filtered)
            logger.info("参考设置完成")
            
            # 3. 去除伪迹
            raw_clean = self.remove_artifacts_auto(raw_referenced)
            logger.info("伪迹去除完成")
            
            # 4. 频谱分析
            band_powers = self.compute_band_power(raw_clean)
            logger.info("频带功率计算完成")
            
            # 5. 连接性分析
            connectivity = self.compute_connectivity(raw_clean, method='wpli')
            logger.info("连接性分析完成")
            
            # 6. 时频分析
            tf_results = self.compute_time_frequency(raw_clean, method='wavelet')
            logger.info("时频分析完成")
            
            # 7. 事件相关电位分析（如果有事件标注）
            erp_results = {}
            if raw_clean.annotations is not None and len(raw_clean.annotations) > 0:
                erp_results = self.compute_erp(raw_clean)
                logger.info("ERP分析完成")
            
            # 8. 源定位分析（如果有通道位置信息）
            source_results = {}
            if raw_clean.info['dig']:
                source_results = self.compute_source_localization(raw_clean)
                logger.info("源定位分析完成")
            
            # 转换回EEGRDD
            processed_eeg = self.mne_to_eegrdd(raw_clean, eegrdd)
            
            # 添加处理结果
            processed_eeg.band_powers = band_powers
            processed_eeg.connectivity = connectivity
            processed_eeg.time_frequency = tf_results
            processed_eeg.erp = erp_results
            processed_eeg.source = source_results
            
            logger.info("MNE处理完成并转换回EEGRDD格式")
            
            return processed_eeg
            
        except Exception as e:
            logger.error(f"MNE处理出错: {e}", exc_info=True)
            # 出错时返回原始数据
            return eegrdd
    
    def process_eegrdd_batch(self, eegrdd_rdd: RDD) -> RDD:
        """批量处理EEGRDD对象
        
        Args:
            eegrdd_rdd: EEGRDD对象的RDD
            
        Returns:
            RDD: 处理后的EEGRDD对象RDD
        """
        return eegrdd_rdd.map(self.process_eegrdd)
    
    def extract_features(self, eegrdd: EEGRDD) -> Dict[str, Any]:
        """从EEGRDD中提取特征
        
        Args:
            eegrdd: EEGRDD对象
            
        Returns:
            Dict: 特征字典
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始提取特征: subject_id={eegrdd.subject_id}, task_name={eegrdd.task_name}")
        
        # 转换为MNE对象
        raw = self.eegrdd_to_mne(eegrdd)
        
        # 计算功率谱密度 - 使用新版MNE API
        logger.info("计算功率谱密度...")
        spectrum = raw.compute_psd()
        psds = spectrum.get_data()
        freqs = spectrum.freqs
        logger.info(f"功率谱密度计算完成，频率范围: {freqs[0]:.1f}-{freqs[-1]:.1f}Hz")
        
        # 计算各频带的平均功率
        band_powers = {}
        for band_name, (fmin, fmax) in self.config.freq_bands.items():
            logger.info(f"计算{band_name}频带({fmin}-{fmax}Hz)的平均功率...")
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psds[:, freq_mask], axis=1)
            band_powers[band_name] = band_power.tolist()
            logger.info(f"{band_name}频带功率计算完成")
        
        logger.info("特征提取完成")
        # 返回特征字典
        return {
            'subject_id': eegrdd.subject_id,
            'task_name': eegrdd.task_name,
            'band_powers': band_powers,
            'channels': eegrdd.channel_names
        }
        
    def filter_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """应用滤波器处理EEG数据
        
        Args:
            raw: MNE Raw对象
            
        Returns:
            mne.io.Raw: 滤波后的Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始滤波处理: {self.config.filter_highpass}-{self.config.filter_lowpass}Hz")
        
        # 应用带通滤波
        try:
            raw_filtered = raw.copy()
            raw_filtered.filter(l_freq=self.config.filter_highpass, h_freq=self.config.filter_lowpass)
        except Exception as e:
            logger.error(f"带通滤波失败: {e}")
            return raw
        
        # 应用陷波滤波（去除电源干扰）
        if self.config.filter_notch > 0:
            try:
                raw_filtered.notch_filter(freqs=[self.config.filter_notch])
            except Exception as e:
                logger.error(f"陷波滤波失败: {e}")
        
        return raw_filtered
    
    def set_reference(self, raw: mne.io.Raw, reference='average') -> mne.io.Raw:
        """设置EEG参考方式
        
        Args:
            raw: MNE Raw对象
            reference: 参考方式，可以是'average'或特定通道名称列表
            
        Returns:
            mne.io.Raw: 重参考后的Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"设置参考: {reference}")
        
        try:
            raw_referenced = raw.copy()
            raw_referenced.set_eeg_reference(ref_channels=reference)
            return raw_referenced
        except Exception as e:
            logger.error(f"设置参考失败: {e}")
            return raw
    
    def remove_artifacts_auto(self, raw: mne.io.Raw) -> mne.io.Raw:
        """自动去除伪迹
        
        Args:
            raw: MNE Raw对象
            
        Returns:
            mne.io.Raw: 去伪迹后的Raw对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info("开始自动去除伪迹...")
        
        # 复制原始数据
        raw_clean = raw.copy()
        
        # 1. 去除大幅值伪迹
        logger.info(f"去除大幅值伪迹，阈值: {self.config.artifact_threshold}μV")
        try:
            # 获取数据
            data = raw_clean.get_data()
            
            # 标记超过阈值的数据点
            mask = np.abs(data) > self.config.artifact_threshold
            n_artifacts = np.sum(mask)
            
            if n_artifacts > 0:
                # 将伪迹替换为NaN，后续可以插值
                data[mask] = np.nan
                logger.info(f"检测到{n_artifacts}个伪迹数据点，已替换为NaN")
                
                # 使用插值替换NaN值
                from scipy import interpolate
                for ch_idx in range(data.shape[0]):
                    ch_data = data[ch_idx]
                    nan_indices = np.isnan(ch_data)
                    if np.any(nan_indices):
                        non_nan_indices = ~nan_indices
                        if np.sum(non_nan_indices) > 0:  # 确保有足够的非NaN点进行插值
                            f = interpolate.interp1d(
                                np.where(non_nan_indices)[0], 
                                ch_data[non_nan_indices],
                                kind='linear', 
                                bounds_error=False, 
                                fill_value="extrapolate"
                            )
                            ch_data[nan_indices] = f(np.where(nan_indices)[0])
                
                # 更新Raw对象的数据
                raw_clean._data = data
                logger.info("已完成伪迹插值")
            else:
                logger.info("未检测到超过阈值的伪迹")
                
        except Exception as e:
            logger.error(f"去除大幅值伪迹失败: {e}")
        
        # 2. 应用ICA去除眼电和肌电伪迹
        if self.config.ica_components > 0:
            logger.info(f"应用ICA去除伪迹，组件数量: {self.config.ica_components}")
            try:
                # 创建ICA对象
                ica = mne.preprocessing.ICA(
                    n_components=self.config.ica_components,
                    random_state=42
                )
                
                # 应用ICA
                ica.fit(raw_clean)
                logger.info("ICA拟合完成")
                
                # 自动检测眼电伪迹组件
                try:
                    eog_indices, eog_scores = ica.find_bads_eog(raw_clean)
                    if eog_indices:
                        logger.info(f"检测到{len(eog_indices)}个眼电伪迹组件: {eog_indices}")
                        ica.exclude = eog_indices
                except Exception as e:
                    logger.warning(f"自动检测眼电伪迹失败: {e}")
                
                # 应用ICA去除伪迹组件
                ica.apply(raw_clean)
                logger.info("ICA伪迹去除完成")
                
            except Exception as e:
                logger.error(f"ICA去除伪迹失败: {e}")
        
        logger.info("自动去除伪迹完成")
        return raw_clean
    
    def compute_psd(self, raw: mne.io.Raw, fmin=0, fmax=100, method='welch') -> Tuple[np.ndarray, np.ndarray]:
        """计算功率谱密度
        
        Args:
            raw: MNE Raw对象
            fmin: 最小频率
            fmax: 最大频率
            method: 计算方法，'welch'或'multitaper'
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (psds, freqs) - 功率谱密度和对应频率
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始计算功率谱密度，方法: {method}, 频率范围: {fmin}-{fmax}Hz")
        
        try:
            if method == 'welch':
                spectrum = raw.compute_psd()
                psds = spectrum.get_data()
                freqs = spectrum.freqs
                logger.info("Welch方法功率谱密度计算完成")
            elif method == 'multitaper':
                psds, freqs = mne.time_frequency.psd_multitaper(
                    raw, fmin=fmin, fmax=fmax
                )
                logger.info("Multitaper方法功率谱密度计算完成")
            else:
                logger.error(f"不支持的功率谱密度计算方法: {method}")
                return None, None
                
            logger.info(f"功率谱密度计算完成，频率点数: {len(freqs)}")
            return psds, freqs
            
        except Exception as e:
            logger.error(f"计算功率谱密度失败: {e}")
            return None, None
    
    def compute_band_power(self, raw: mne.io.Raw) -> Dict[str, np.ndarray]:
        """计算各频带功率
        
        Args:
            raw: MNE Raw对象
            
        Returns:
            Dict[str, np.ndarray]: 各频带的功率
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info("开始计算频带功率...")
        
        try:
            # 计算功率谱密度
            psds, freqs = self.compute_psd(raw)
            if psds is None or freqs is None:
                logger.error("无法计算频带功率，功率谱密度计算失败")
                return {}
            
            # 计算各频带的平均功率
            band_powers = {}
            for band_name, (fmin, fmax) in self.config.freq_bands.items():
                logger.info(f"计算{band_name}频带({fmin}-{fmax}Hz)的平均功率...")
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                if np.sum(freq_mask) > 0:  # 确保频带内有频率点
                    band_power = np.mean(psds[:, freq_mask], axis=1)
                    band_powers[band_name] = band_power
                    logger.info(f"{band_name}频带功率计算完成")
                else:
                    logger.warning(f"{band_name}频带({fmin}-{fmax}Hz)内没有频率点，跳过")
            
            logger.info("频带功率计算完成")
            return band_powers
            
        except Exception as e:
            logger.error(f"计算频带功率失败: {e}")
            return {}
    
    def compute_connectivity(self, raw: mne.io.Raw, method='wpli', fmin=8, fmax=13) -> np.ndarray:
        """计算脑连接性
        
        Args:
            raw: MNE Raw对象
            method: 连接性计算方法，可选'coh'(相干性)、'plv'(相位锁定值)、'wpli'(加权相位滞后指数)
            fmin: 最小频率
            fmax: 最大频率
            
        Returns:
            np.ndarray: 连接性矩阵
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始计算脑连接性，方法: {method}, 频率范围: {fmin}-{fmax}Hz")
        
        try:
            # 准备数据
            raw_copy = raw.copy()
            
            # 计算连接性
            # 创建epochs以进行连接性分析
            # 如果没有事件，创建虚拟事件
            if len(raw_copy.annotations) == 0:
                # 创建虚拟事件，每秒一个
                onsets = np.arange(0, raw_copy.times[-1], 1.0)
                durations = np.zeros_like(onsets)
                descriptions = ['event'] * len(onsets)
                raw_copy.set_annotations(mne.Annotations(onsets, durations, descriptions))
            
            # 从注释创建事件
            events, event_id = mne.events_from_annotations(raw_copy)
            
            # 创建epochs，使用更宽松的阈值
            reject = None  # 不使用自动拒绝
            epochs = mne.Epochs(raw_copy, events, event_id=event_id, tmin=0, tmax=1.0,
                              baseline=None, preload=True, reject=reject)
            
            if method == 'coh':
                # 计算相干性
                from mne.connectivity import spectral_connectivity_epochs
                con = spectral_connectivity_epochs(
                    epochs, method='coh', mode='multitaper', 
                    fmin=fmin, fmax=fmax, faverage=True
                )
                logger.info("相干性计算完成")
                return con.get_data()
                
            elif method == 'plv':
                # 计算相位锁定值
                from mne.connectivity import spectral_connectivity_epochs
                con = spectral_connectivity_epochs(
                    epochs, method='plv', mode='multitaper',
                    fmin=fmin, fmax=fmax, faverage=True
                )
                logger.info("相位锁定值计算完成")
                return con.get_data()
                
            elif method == 'wpli':
                # 计算加权相位滞后指数
                from mne.connectivity import spectral_connectivity_epochs
                con = spectral_connectivity_epochs(
                    epochs, method='wpli', mode='multitaper',
                    fmin=fmin, fmax=fmax, faverage=True
                )
                logger.info("加权相位滞后指数计算完成")
                return con
                
            else:
                logger.error(f"不支持的连接性计算方法: {method}")
                return None
                
        except Exception as e:
            logger.error(f"计算脑连接性失败: {e}")
            return None
    
    def compute_time_frequency(self, raw: mne.io.Raw, method='wavelet', freqs=None, n_cycles=None) -> Dict[str, Any]:
        """计算时频分析
        
        Args:
            raw: MNE Raw对象
            method: 时频分析方法，可选'wavelet'(小波变换)、'stft'(短时傅里叶变换)
            freqs: 频率点，如果为None则自动生成
            n_cycles: 小波周期数，如果为None则自动设置
            
        Returns:
            Dict: 时频分析结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始时频分析，方法: {method}")
        
        try:
            # 准备数据
            raw_copy = raw.copy()
            
            # 设置默认参数 - 针对实时短数据流优化
            if freqs is None:
                freqs = np.arange(4, 41, 1)  # 4-40Hz，提高最低频率以适应短数据
            
            if n_cycles is None:
                # 减小n_cycles值以适应短数据流
                n_cycles = np.ones_like(freqs) * 3  # 固定为3个周期，适合实时短数据
            
            # 计算时频表示
            if method == 'wavelet':
                # 使用Morlet小波变换
                from mne.time_frequency import tfr_array_morlet
                
                # 获取数据
                data = raw_copy.get_data()
                sfreq = raw_copy.info['sfreq']
                
                # 记录数据形状
                logger.info(f"时频分析数据形状: {data.shape}")
                
                # 重新组织数据形状以符合tfr_array_morlet的要求
                # tfr_array_morlet期望输入形状为(n_epochs, n_chans, n_times)
                # 而我们的数据形状为(n_chans, n_times)
                # 因此需要添加一个epoch维度
                data = data.reshape(1, *data.shape)  # 变为(1, n_chans, n_times)
                
                # 计算小波变换
                logger.info(f"使用Morlet小波变换计算时频表示，频率点数: {len(freqs)}，重塑后数据形状: {data.shape}")
                tfr = tfr_array_morlet(data, sfreq, freqs, n_cycles=n_cycles, output='power')
                
                # 如果结果是(1, n_chans, n_freqs, n_times)，去掉第一个维度
                if tfr.shape[0] == 1:
                    tfr = tfr[0]
                
                logger.info("小波变换时频分析完成")
                return {
                    'tfr': tfr,
                    'freqs': freqs,
                    'times': raw_copy.times,
                    'method': 'wavelet'
                }
                
            elif method == 'stft':
                # 使用短时傅里叶变换
                from scipy import signal
                
                # 获取数据
                data = raw_copy.get_data()
                sfreq = raw_copy.info['sfreq']
                
                # 计算STFT
                logger.info("使用短时傅里叶变换计算时频表示")
                n_fft = int(sfreq * 2)  # 2秒窗口
                n_overlap = int(n_fft * 0.5)  # 50%重叠
                
                # 为每个通道计算STFT
                tfr = []
                for ch_idx in range(data.shape[0]):
                    f, t, Zxx = signal.stft(data[ch_idx], fs=sfreq, nperseg=n_fft, noverlap=n_overlap)
                    tfr.append(np.abs(Zxx)**2)  # 功率谱
                
                tfr = np.array(tfr)
                
                logger.info("短时傅里叶变换时频分析完成")
                return {
                    'tfr': tfr,
                    'freqs': f,
                    'times': t,
                    'method': 'stft'
                }
                
            else:
                logger.error(f"不支持的时频分析方法: {method}")
                return {}
                
        except Exception as e:
            logger.error(f"时频分析失败: {e}", exc_info=True)
            return {}
    
    def compute_erp(self, raw: mne.io.Raw, event_id=None, tmin=-0.2, tmax=1.0) -> Dict[str, Any]:
        """计算事件相关电位(ERP)
        
        Args:
            raw: MNE Raw对象，必须包含事件标注
            event_id: 事件ID字典，如果为None则自动从标注中提取
            tmin: 事件前时间窗口(秒)
            tmax: 事件后时间窗口(秒)
            
        Returns:
            Dict: ERP分析结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始ERP分析，时间窗口: {tmin}s到{tmax}s")
        
        try:
            # 检查是否有事件标注
            if raw.annotations is None or len(raw.annotations) == 0:
                logger.error("无法进行ERP分析：缺少事件标注")
                return {}
            
            # 从标注中提取事件
            events, event_dict = mne.events_from_annotations(raw)
            
            if events.shape[0] == 0:
                logger.error("无法从标注中提取事件")
                return {}
                
            logger.info(f"从标注中提取到{events.shape[0]}个事件")
            
            # 如果没有提供event_id，使用提取的event_dict
            if event_id is None:
                event_id = event_dict
                
            logger.info(f"事件类型: {event_id}")
            
            # 创建Epochs对象 - 使用更宽松的阈值设置
            # 设置适合EEG数据的阈值，±100μV
            reject = {'eeg': 100e-6}  # 100 μV，单位为伏特
            # 或者可以使用百分比阈值
            # reject = None
            # reject_by_annotation = True
            
            epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                              baseline=(tmin, 0), preload=True, reject=reject)
            
            logger.info(f"创建了{len(epochs)}个epoch")
            
            # 计算ERP
            evoked_dict = {}
            for event_name in event_id.keys():
                try:
                    evoked = epochs[event_name].average()
                    evoked_dict[event_name] = evoked
                    logger.info(f"计算了'{event_name}'事件的ERP")
                except Exception as e:
                    logger.warning(f"计算'{event_name}'事件的ERP失败: {e}")
            
            logger.info("ERP分析完成")
            return {
                'epochs': epochs,
                'evoked': evoked_dict,
                'event_id': event_id
            }
            
        except Exception as e:
            logger.error(f"ERP分析失败: {e}", exc_info=True)
            return {}
    
    def compute_source_localization(self, raw: mne.io.Raw, method='eLORETA') -> Dict[str, Any]:
        """计算源定位
        
        Args:
            raw: MNE Raw对象，必须包含通道位置信息
            method: 源定位方法，可选'eLORETA'、'MNE'、'dSPM'
            
        Returns:
            Dict: 源定位结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始源定位分析，方法: {method}")
        
        try:
            # 检查是否有通道位置信息
            if not raw.info['dig']:
                logger.error("无法进行源定位：缺少通道位置信息")
                return {}
            
            # 使用标准头模型
            sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.085)  # 单位：米
            
            # 创建源空间
            src = mne.setup_volume_source_space(sphere=sphere, pos=10.0)
            
            # 计算前向模型
            fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
            
            # 计算噪声协方差矩阵
            logger.info("开始计算噪声协方差矩阵...")
            try:
                # 检查是否有事件标注
                if raw.annotations is not None and len(raw.annotations) > 0:
                    logger.info(f"使用现有的{len(raw.annotations)}个事件标注计算协方差矩阵")
                    noise_cov = mne.compute_raw_covariance(raw, method='shrunk')
                else:
                    # 如果没有事件标注，创建固定长度事件
                    logger.info("没有事件标注，创建固定长度事件")
                    # 记录数据长度
                    data_len = len(raw.times) / raw.info['sfreq']
                    logger.info(f"数据长度: {data_len}秒")
                    
                    # 创建固定长度事件，每2秒一个事件
                    events = mne.make_fixed_length_events(raw, duration=2.0)
                    logger.info(f"创建了{len(events)}个固定长度事件")
                    
                    # 使用这些事件创建epochs
                    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True)
                    logger.info(f"创建了{len(epochs)}个epoch")
                    
                    # 使用epochs计算协方差矩阵
                    noise_cov = mne.compute_covariance(epochs, method='shrunk')
            except Exception as e:
                logger.error(f"计算噪声协方差矩阵失败: {e}，使用空协方差矩阵")
                # 创建一个空的协方差矩阵
                noise_cov = mne.Covariance(data=np.eye(len(raw.ch_names)), names=raw.ch_names)
            
            # 计算逆算子
            inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2)
            
            # 应用逆算子
            if method == 'eLORETA':
                stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0/9.0, method='eLORETA')
            elif method == 'MNE':
                stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0/9.0, method='MNE')
            elif method == 'dSPM':
                stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0/9.0, method='dSPM')
            else:
                logger.error(f"不支持的源定位方法: {method}")
                return {}
            
            logger.info("源定位分析完成")
            return {
                'stc': stc,
                'src': src,
                'fwd': fwd,
                'inv': inv,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"源定位分析失败: {e}", exc_info=True)
            return {}
    
    def compute_statistics(self, data1, data2, test_type='ttest') -> Dict[str, Any]:
        """计算统计分析
        
        Args:
            data1: 第一组数据
            data2: 第二组数据
            test_type: 统计检验类型，可选'ttest'、'wilcoxon'、'permutation'
            
        Returns:
            Dict: 统计分析结果
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        logger.info(f"开始统计分析，检验类型: {test_type}")
        
        try:
            from scipy import stats
            
            if test_type == 'ttest':
                # 配对t检验
                t_val, p_val = stats.ttest_rel(data1, data2)
                logger.info("配对t检验完成")
                return {
                    'test_type': 'ttest',
                    't_value': t_val,
                    'p_value': p_val
                }
                
            elif test_type == 'wilcoxon':
                # Wilcoxon符号秩检验
                w_val, p_val = stats.wilcoxon(data1, data2)
                logger.info("Wilcoxon符号秩检验完成")
                return {
                    'test_type': 'wilcoxon',
                    'w_value': w_val,
                    'p_value': p_val
                }
                
            elif test_type == 'permutation':
                # 排列检验
                from mne.stats import permutation_t_test
                t_obs, p_val, h0 = permutation_t_test([data1, data2], n_permutations=1000)
                logger.info("排列检验完成")
                return {
                    'test_type': 'permutation',
                    't_observed': t_obs,
                    'p_value': p_val,
                    'h0': h0
                }
                
            else:
                logger.error(f"不支持的统计检验类型: {test_type}")
                return {}
                
        except Exception as e:
            logger.error(f"统计分析失败: {e}", exc_info=True)
            return {}
    
    def process_eegrdd(self, eegrdd: EEGRDD) -> EEGRDD:
        """处理单个EEGRDD对象 - 增强版，包含更多MNE处理功能
        
        Args:
            eegrdd: 输入的EEGRDD对象
            
        Returns:
            EEGRDD: 处理后的EEGRDD对象
        """
        import logging
        logger = logging.getLogger('EEGProcessor')
        
        # 记录输入EEGRDD的关键信息
        logger.info(f"接收到EEGRDD数据: subject_id={eegrdd.subject_id}, 数据形状: {eegrdd.voltage_data.shape if eegrdd.voltage_data is not None else 'None'}")
        
        # 检查通道信息是否匹配
        if eegrdd.voltage_data is not None and eegrdd.voltage_data.shape[0] != len(eegrdd.channel_names):
            logger.warning(f"通道数量不匹配: 数据通道({eegrdd.voltage_data.shape[0]}) ≠ 名称数量({len(eegrdd.channel_names)})")
            
            # 调整channel_names以匹配数据维度
            if eegrdd.voltage_data.shape[0] > len(eegrdd.channel_names):
                # 如果数据通道数多于名称数量，添加默认通道名
                additional_channels = eegrdd.voltage_data.shape[0] - len(eegrdd.channel_names)
                for i in range(additional_channels):
                    eegrdd.channel_names.append(f"Ch{len(eegrdd.channel_names)+1}")
                logger.info(f"已添加{additional_channels}个默认通道名")
            else:
                # 如果数据通道数少于名称数量，截断通道名列表
                eegrdd.channel_names = eegrdd.channel_names[:eegrdd.voltage_data.shape[0]]
                logger.info(f"已截断通道名列表至{eegrdd.voltage_data.shape[0]}个")
        
        # 检查通道类型是否匹配 - 简化日志
        if eegrdd.channel_types and len(eegrdd.channel_types) != len(eegrdd.channel_names):
            # 调整通道类型列表长度
            if len(eegrdd.channel_types) > len(eegrdd.channel_names):
                eegrdd.channel_types = eegrdd.channel_types[:len(eegrdd.channel_names)]
            else:
                # 扩展通道类型列表，使用'EEG'作为默认类型
                eegrdd.channel_types.extend(['EEG'] * (len(eegrdd.channel_names) - len(eegrdd.channel_types)))
        
        # 检查通道位置是否匹配 - 简化日志
        if eegrdd.channel_locations and len(eegrdd.channel_locations) != len(eegrdd.channel_names):
            # 调整通道位置列表长度
            if len(eegrdd.channel_locations) > len(eegrdd.channel_names):
                eegrdd.channel_locations = eegrdd.channel_locations[:len(eegrdd.channel_names)]
            else:
                # 扩展通道位置列表，使用空字典作为默认位置
                eegrdd.channel_locations.extend([{'x': 0.0, 'y': 0.0, 'z': 0.0}] * 
                                              (len(eegrdd.channel_names) - len(eegrdd.channel_locations)))
        
        # 转换为MNE对象并进行处理
        try:
            logger.info("开始转换为MNE格式并进行处理...")
            raw = self.eegrdd_to_mne(eegrdd)
            
            # 应用电极位置
            raw = self.apply_montage(raw, eegrdd)
            
            # 1. 滤波处理
            raw_filtered = self.filter_data(raw)
            logger.info("滤波处理完成")
            
            # 2. 设置参考
            raw_referenced = self.set_reference(raw_filtered)
            logger.info("参考设置完成")
            
            # 3. 去除伪迹
            raw_clean = self.remove_artifacts_auto(raw_referenced)
            logger.info("伪迹去除完成")
            
            # 4. 频谱分析
            band_powers = self.compute_band_power(raw_clean)
            logger.info("频带功率计算完成")
            
            # 5. 连接性分析
            connectivity = self.compute_connectivity(raw_clean, method='wpli')
            logger.info("连接性分析完成")
            
            # 6. 时频分析
            tf_results = self.compute_time_frequency(raw_clean, method='wavelet')
            logger.info("时频分析完成")
            
            # 7. 事件相关电位分析（如果有事件标注）
            erp_results = {}
            if raw_clean.annotations is not None and len(raw_clean.annotations) > 0:
                erp_results = self.compute_erp(raw_clean)
                logger.info("ERP分析完成")
            
            # 8. 源定位分析（如果有通道位置信息）
            source_results = {}
            if raw_clean.info['dig']:
                source_results = self.compute_source_localization(raw_clean)
                logger.info("源定位分析完成")
            
            # 转换回EEGRDD
            processed_eeg = self.mne_to_eegrdd(raw_clean, eegrdd)
            
            # 添加处理结果
            processed_eeg.band_powers = band_powers
            processed_eeg.connectivity = connectivity
            processed_eeg.time_frequency = tf_results
            processed_eeg.erp = erp_results
            processed_eeg.source = source_results
            
            logger.info("MNE处理完成并转换回EEGRDD格式")
            
            return processed_eeg
            
        except Exception as e:
            logger.error(f"MNE处理出错: {e}", exc_info=True)
            # 出错时返回原始数据
            return eegrdd