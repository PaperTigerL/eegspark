import os
import numpy as np
import mne
import pywt
import logging
import time  # 确保导入time模块
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from common.eegrdd import EEGRDD

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RealtimeEEGProcessor")


@dataclass
class RealtimeProcessingConfig:
    """实时EEG处理配置 - 优化版"""

    # 滤波设置
    filter_lowpass: float = 45.0  # 低通滤波频率，提高以保留更多高频信息
    filter_highpass: float = 0.5  # 高通滤波频率，降低以保留更多低频信息
    filter_notch: float = 50.0  # 陷波滤波频率

    # 去伪迹设置
    artifact_threshold: float = 150.0  # 伪迹阈值(μV)，提高以减少误判
    use_ica: bool = False  # 禁用ICA以提高实时处理速度
    ica_components: int = 0  # ICA组件数量设为0，确保不会执行ICA

    # 时频分析设置
    freq_bands: Dict[str, Tuple[float, float]] = None  # 频带定义

    # 数据缓冲设置
    buffer_duration: float = 3.0  # 缓冲区持续时间(秒)
    min_data_duration: float = 1.0  # 最小数据长度要求(秒)，降低以适应短数据
    overlap_duration: float = 0.5  # 数据重叠时间(秒)

    # 时频分析设置
    use_stft: bool = True  # 是否使用短时傅里叶变换
    stft_window_size: float = 0.5  # 短时傅里叶变换窗口大小(秒)
    stft_overlap: float = 0.25  # 短时傅里叶变换窗口重叠(秒)
    use_wavelet: bool = False  # 是否使用小波变换，默认不使用

    # 实时处理设置
    adaptive_filtering: bool = True  # 是否使用自适应滤波
    use_short_time_fft: bool = True  # 是否使用短时傅里叶变换

    # 超时设置
    max_processing_time: float = 0.5  # 最大处理时间(秒)，超过此时间将中断处理
    enable_timeout_detection: bool = True  # 是否启用超时检测

    def __post_init__(self):
        if self.freq_bands is None:
            self.freq_bands = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha_low": (8, 10),
                "alpha_high": (10, 13),
                "beta_low": (13, 20),
                "beta_high": (20, 30),
                "gamma": (30, 45),
            }


class RealtimeEEGProcessor:
    """实时EEG数据处理器 - 使用PyWavelets和MNE优化短时间序列处理"""

    def __init__(self, config: Optional[RealtimeProcessingConfig] = None):
        """初始化处理器

        Args:
            config: 处理配置，如果为None则使用默认配置
        """
        self.config = config or RealtimeProcessingConfig()
        self._montage = None
        self._load_standard_montage()

        # 初始化数据缓冲区
        self._data_buffer = {}
        self._buffer_metadata = {}

        # 初始化处理状态
        self._processing_state = {}

    def _load_standard_montage(self):
        """加载标准10-20系统电极位置"""
        try:
            self._montage = mne.channels.make_standard_montage("standard_1020")
        except Exception as e:
            logger.warning(f"无法加载标准电极位置: {e}")

    def eegrdd_to_mne(self, eegrdd: EEGRDD) -> mne.io.RawArray:
        """将EEGRDD转换为MNE Raw对象 - 增强版，处理各种边缘情况

        Args:
            eegrdd: EEGRDD对象

        Returns:
            mne.io.RawArray: MNE Raw对象
        """
        # 首先验证电压数据
        if eegrdd.voltage_data is None:
            logger.error("电压数据为空，无法创建MNE对象")
            raise ValueError("电压数据为空，无法创建MNE对象")

        # 验证通道名称
        if not eegrdd.channel_names or len(eegrdd.channel_names) == 0:
            logger.warning("通道名称列表为空，创建默认通道名")
            eegrdd.channel_names = [
                f"Ch{i+1}" for i in range(eegrdd.voltage_data.shape[0])
            ]

        # 确保通道名称数量与数据通道数匹配
        if eegrdd.voltage_data.shape[0] != len(eegrdd.channel_names):
            logger.warning(
                f"通道数量不匹配: 数据通道({eegrdd.voltage_data.shape[0]}) ≠ 名称数量({len(eegrdd.channel_names)})"
            )

            # 调整channel_names以匹配数据维度
            if eegrdd.voltage_data.shape[0] > len(eegrdd.channel_names):
                # 如果数据通道数多于名称数量，添加默认通道名
                additional_channels = eegrdd.voltage_data.shape[0] - len(
                    eegrdd.channel_names
                )
                for i in range(additional_channels):
                    eegrdd.channel_names.append(f"Ch{len(eegrdd.channel_names)+1}")
            else:
                # 如果数据通道数少于名称数量，截断通道名列表
                eegrdd.channel_names = eegrdd.channel_names[
                    : eegrdd.voltage_data.shape[0]
                ]

        # 创建通道信息
        ch_types = ["eeg"] * len(eegrdd.channel_names)  # 默认全部为EEG通道

        # 如果有通道类型信息，则使用实际的类型
        if eegrdd.channel_types:
            # 确保通道类型列表长度与通道名称列表长度一致
            if len(eegrdd.channel_types) != len(eegrdd.channel_names):
                # 调整通道类型列表长度
                if len(eegrdd.channel_types) > len(eegrdd.channel_names):
                    eegrdd.channel_types = eegrdd.channel_types[
                        : len(eegrdd.channel_names)
                    ]
                else:
                    # 扩展通道类型列表，使用'eeg'作为默认类型
                    eegrdd.channel_types.extend(
                        ["eeg"]
                        * (len(eegrdd.channel_names) - len(eegrdd.channel_types))
                    )

            # 转换为小写并确保所有类型都是有效的
            valid_types = ["eeg", "ecg", "emg", "eog", "misc", "stim", "resp", "bio"]
            ch_types = []
            for ch_type in eegrdd.channel_types:
                if not ch_type:  # 处理None或空字符串
                    ch_types.append("eeg")
                    continue

                ch_type_lower = ch_type.lower()
                if ch_type_lower not in valid_types:
                    ch_types.append("misc")
                else:
                    ch_types.append(ch_type_lower)

        # 创建MNE info对象
        try:
            logger.info(
                f"创建MNE info对象: {len(eegrdd.channel_names)}个通道, 采样率{eegrdd.sampling_rate}Hz"
            )
            info = mne.create_info(
                ch_names=eegrdd.channel_names,
                sfreq=eegrdd.sampling_rate,
                ch_types=ch_types,
            )
        except Exception as e:
            logger.error(f"创建MNE info对象失败: {e}")
            # 尝试使用简化的通道类型
            info = mne.create_info(
                ch_names=eegrdd.channel_names,
                sfreq=eegrdd.sampling_rate,
                ch_types=["eeg"] * len(eegrdd.channel_names),
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
                    onset.append(float(event.get("onset", 0)))
                    duration.append(float(event.get("duration", 0)))
                    description.append(str(event.get("type", "unknown")))
                except Exception as e:
                    logger.warning(f"处理事件标注时出错: {e}, 跳过此事件")

            if onset:  # 如果有有效事件
                try:
                    # 创建单个Annotations对象
                    annotations = mne.Annotations(
                        onset=onset, duration=duration, description=description
                    )
                    raw.set_annotations(annotations)
                    logger.info(f"成功设置{len(onset)}个事件标注")
                except Exception as e:
                    logger.error(f"设置事件标注时出错: {e}")

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
                events.append(
                    {
                        "onset": float(annot["onset"]),
                        "duration": float(annot["duration"]),
                        "type": str(annot["description"]),
                    }
                )

        # 创建新的EEGRDD对象，保留原始元数据
        new_eegrdd = EEGRDD(
            subject_id=original_eegrdd.subject_id,
            voltage_data=data,
            sampling_rate=original_eegrdd.sampling_rate,
            payload_id=original_eegrdd.payload_id,
            task_name=original_eegrdd.task_name,
            block_id=original_eegrdd.block_id,
            n_channels=data.shape[0],
            n_samples=data.shape[1],
            duration=data.shape[1] / original_eegrdd.sampling_rate,
            reference=original_eegrdd.reference,
            channel_names=original_eegrdd.channel_names,
            channel_types=original_eegrdd.channel_types,
            channel_units=original_eegrdd.channel_units,
            channel_locations=original_eegrdd.channel_locations,
            events=events,
        )

        return new_eegrdd

    def add_to_buffer(self, subject_id: str, eegrdd: EEGRDD) -> bool:
        """将数据添加到缓冲区

        Args:
            subject_id: 受试者ID
            eegrdd: EEGRDD对象

        Returns:
            bool: 是否达到处理条件
        """
        sampling_rate = eegrdd.sampling_rate

        # 初始化缓冲区（如果不存在）
        if subject_id not in self._data_buffer:
            self._data_buffer[subject_id] = []
            self._buffer_metadata[subject_id] = eegrdd.get_metadata()
            logger.info(f"为受试者 {subject_id} 创建新的数据缓冲区")

        # 添加数据到缓冲区
        self._data_buffer[subject_id].append(eegrdd.voltage_data)

        # 计算当前缓冲区数据长度（秒）
        total_samples = sum(data.shape[1] for data in self._data_buffer[subject_id])
        buffer_duration = total_samples / sampling_rate

        logger.info(
            f"受试者 {subject_id} 的数据缓冲区: {len(self._data_buffer[subject_id])}个数据包, {buffer_duration:.2f}秒"
        )

        # 检查单个数据包是否足够长
        current_duration = eegrdd.voltage_data.shape[1] / sampling_rate

        # 判断是否达到处理条件
        if current_duration >= self.config.min_data_duration:
            # 单个数据包足够长
            logger.info(f"当前数据包长度({current_duration:.2f}秒)已满足最小要求")
            return True
        elif buffer_duration >= self.config.min_data_duration:
            # 缓冲区累积数据长度达到要求
            logger.info(f"缓冲区累积数据长度({buffer_duration:.2f}秒)已满足最小要求")
            return True

        # 未达到处理条件
        return False

    def get_buffer_data(self, subject_id: str) -> Tuple[np.ndarray, Dict]:
        """获取缓冲区数据

        Args:
            subject_id: 受试者ID

        Returns:
            Tuple[np.ndarray, Dict]: 合并后的数据和元数据
        """
        if (
            subject_id not in self._data_buffer
            or len(self._data_buffer[subject_id]) == 0
        ):
            logger.warning(f"受试者 {subject_id} 的数据缓冲区为空")
            return None, None

        # 合并数据
        data = np.concatenate(self._data_buffer[subject_id], axis=1)
        metadata = self._buffer_metadata[subject_id].copy()

        # 更新元数据
        metadata.update(
            {
                "n_samples": data.shape[1],
                "duration": data.shape[1] / metadata["sampling_rate"],
            }
        )

        return data, metadata

    def clear_buffer(self, subject_id: str, keep_overlap: bool = True):
        """清空缓冲区

        Args:
            subject_id: 受试者ID
            keep_overlap: 是否保留重叠部分
        """
        if subject_id not in self._data_buffer:
            return

        if not keep_overlap:
            # 完全清空缓冲区
            self._data_buffer[subject_id] = []
            logger.info(f"已清空受试者 {subject_id} 的数据缓冲区")
            return

        # 保留重叠部分
        if len(self._data_buffer[subject_id]) > 0:
            last_data = self._data_buffer[subject_id][-1]
            sampling_rate = self._buffer_metadata[subject_id]["sampling_rate"]
            overlap_samples = int(sampling_rate * self.config.overlap_duration)

            if last_data.shape[1] > overlap_samples:
                # 保留最后一个数据包的重叠部分
                self._data_buffer[subject_id] = [last_data[:, -overlap_samples:]]
                logger.info(
                    f"保留{overlap_samples}个样本点({self.config.overlap_duration:.2f}秒)作为重叠"
                )
            else:
                # 如果最后一个数据包太短，直接保留
                self._data_buffer[subject_id] = [last_data]
                logger.info(
                    f"保留最后一个数据包({last_data.shape[1]/sampling_rate:.2f}秒)作为重叠"
                )

    def process_eegrdd(self, eegrdd: EEGRDD, start_time: float = None) -> EEGRDD:
        """处理EEGRDD对象 - 实时优化版，带有超时检测机制

        Args:
            eegrdd: EEGRDD对象
            start_time: 处理开始时间，用于超时检测

        Returns:
            EEGRDD: 处理后的EEGRDD对象
        """
        try:
            # 如果未提供开始时间，使用当前时间
            if start_time is None:
                start_time = time.time()

            # 转换为MNE对象
            raw = self.eegrdd_to_mne(eegrdd)

            # 超时检测
            if self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0:
                    logger.warning(
                        f"转换为MNE对象后已超时，已消耗{elapsed_time:.3f}秒，跳过后续处理"
                    )
                    return eegrdd  # 超时直接返回原始数据

            # 应用滤波 - 传递开始时间用于超时检测
            raw = self._apply_filtering(raw, start_time)

            # 超时检测
            if self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0:
                    logger.warning(
                        f"滤波后已超时，已消耗{elapsed_time:.3f}秒，跳过去伪迹"
                    )
                    # 转换回EEGRDD并返回
                    return self.mne_to_eegrdd(raw, eegrdd)

            # 去除伪迹 - 如果剩余时间不足，可能会跳过
            if not self.config.enable_timeout_detection or remaining_time > 0.1:
                raw = self._remove_artifacts(raw, start_time)
            else:
                logger.warning(f"剩余时间不足({remaining_time:.3f}秒)，跳过去伪迹处理")

            # 转换回EEGRDD
            processed_eegrdd = self.mne_to_eegrdd(raw, eegrdd)

            return processed_eegrdd

        except Exception as e:
            logger.error(f"处理EEGRDD对象时出错: {e}", exc_info=True)
            return eegrdd  # 出错时返回原始数据

    def _apply_filtering(self, raw: mne.io.Raw, start_time: float = None) -> mne.io.Raw:
        """应用滤波 - 优化版，适用于短时间序列，带有超时检测

        Args:
            raw: MNE Raw对象
            start_time: 处理开始时间，用于超时检测

        Returns:
            mne.io.Raw: 滤波后的Raw对象
        """
        try:
            # 超时检测
            if start_time is not None and self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0:
                    logger.warning(
                        f"滤波前已超时，已消耗{elapsed_time:.3f}秒，跳过滤波"
                    )
                    return raw  # 超时直接返回原始数据

                logger.info(f"滤波前剩余处理时间: {remaining_time:.3f}秒")

                # 根据剩余时间调整滤波策略
                if remaining_time < 0.2:  # 如果剩余时间很少，使用超简化滤波
                    logger.warning(f"剩余时间不足，使用超简化滤波")
                    # 只应用最基本的低通滤波，使用最短的滤波器长度
                    try:
                        # 使用scipy的信号处理函数代替MNE的滤波功能
                        from scipy import signal

                        # 获取原始数据
                        data = raw.get_data()
                        filtered_data = np.zeros_like(data)

                        # 设计简单的低通滤波器
                        nyquist = raw.info["sfreq"] / 2.0
                        h_freq = (
                            self.config.filter_lowpass / nyquist
                            if self.config.filter_lowpass
                            else None
                        )

                        if h_freq:
                            # 使用最短的滤波器长度
                            n_samples = len(raw)
                            # 自适应滤波器长度，确保不超过信号长度的1/3
                            filter_order = min(31, max(15, int(n_samples / 3)))
                            logger.info(f"超简化滤波使用滤波器长度: {filter_order}")
                            b = signal.firwin(filter_order, h_freq)

                            # 对每个通道应用滤波
                            for ch_idx in range(data.shape[0]):
                                # 使用lfilter代替filtfilt，速度更快但相位不为零
                                filtered_data[ch_idx] = signal.lfilter(
                                    b, 1.0, data[ch_idx]
                                )
                        else:
                            # 无滤波
                            filtered_data = data

                        # 创建新的Raw对象
                        raw_filtered = mne.io.RawArray(filtered_data, raw.info.copy())

                        # 复制原始对象的其他属性
                        if hasattr(raw, "annotations") and raw.annotations is not None:
                            raw_filtered.set_annotations(raw.annotations)

                        return raw_filtered
                    except Exception as e:
                        logger.error(f"超简化滤波失败: {e}")
                        return raw  # 出错时返回原始数据

            # 检查数据长度是否足够
            data_duration = len(raw) / raw.info["sfreq"]
            min_duration = (
                1.0 / self.config.filter_highpass * 3
            )  # 至少包含3个最低频率周期

            if data_duration < min_duration:
                logger.warning(
                    f"数据长度({data_duration:.2f}秒)不足以进行{self.config.filter_highpass}Hz高通滤波，使用自适应滤波"
                )
                # 对于短数据，使用更高的高通滤波频率
                adjusted_highpass = max(
                    self.config.filter_highpass, 1.0 / data_duration
                )
                logger.info(
                    f"调整高通滤波频率: {self.config.filter_highpass} -> {adjusted_highpass}Hz"
                )

                # 应用自适应滤波
                if self.config.adaptive_filtering:
                    # 计算适合当前数据长度的滤波器长度
                    # 滤波器长度应该小于数据长度的1/3，对于短时间序列使用更短的滤波器长度
                    n_samples = len(raw)
                    # 限制最大滤波器长度，不超过信号长度的1/3，且不超过1秒的采样点数
                    max_filter_length = min(
                        int(n_samples / 3), int(raw.info["sfreq"] * 1.0)
                    )
                    # 对于短时间序列，进一步限制滤波器长度
                    if n_samples < 100:  # 如果样本点数很少
                        max_filter_length = min(
                            max_filter_length, 31
                        )  # 使用更短的滤波器长度

                    logger.info(f"使用自适应滤波器长度: {max_filter_length} 样本点")

                    # 超时检测
                    if start_time is not None and self.config.enable_timeout_detection:
                        elapsed_time = time.time() - start_time
                        remaining_time = self.config.max_processing_time - elapsed_time

                        if remaining_time <= 0.1:  # 如果剩余时间极少，进一步简化
                            max_filter_length = min(
                                max_filter_length, 31
                            )  # 使用更短的滤波器长度
                            logger.warning(
                                f"剩余时间极少，进一步缩短滤波器长度: {max_filter_length}"
                            )
                        elif (
                            n_samples < 100
                        ):  # 对于短时间序列，无论剩余时间如何，都使用更短的滤波器长度
                            max_filter_length = min(
                                max_filter_length, 31
                            )  # 使用更短的滤波器长度
                            logger.warning(
                                f"短时间序列，使用更短的滤波器长度: {max_filter_length}"
                            )

                    # 使用scipy的信号处理函数代替MNE的滤波功能
                    from scipy import signal

                    # 获取原始数据
                    data = raw.get_data()
                    filtered_data = np.zeros_like(data)

                    # 对每个通道应用滤波
                    for ch_idx in range(data.shape[0]):
                        # 设计FIR滤波器
                        nyquist = raw.info["sfreq"] / 2.0
                        h_freq = (
                            self.config.filter_lowpass / nyquist
                            if self.config.filter_lowpass
                            else None
                        )
                        l_freq = (
                            adjusted_highpass / nyquist if adjusted_highpass else None
                        )

                        # 设计滤波器
                        if h_freq and l_freq:
                            # 带通滤波器
                            b = signal.firwin(
                                max_filter_length, [l_freq, h_freq], pass_zero=False
                            )
                        elif h_freq:
                            # 低通滤波器
                            b = signal.firwin(max_filter_length, h_freq)
                        elif l_freq:
                            # 高通滤波器 - 使用频谱反转技术
                            b = signal.firwin(
                                max_filter_length, l_freq, pass_zero=False
                            )
                        else:
                            # 无滤波
                            filtered_data[ch_idx] = data[ch_idx]
                            continue

                        # 超时检测 - 每处理几个通道检查一次
                        if (
                            start_time is not None
                            and self.config.enable_timeout_detection
                            and ch_idx % 4 == 0
                        ):  # 每4个通道检查一次
                            elapsed_time = time.time() - start_time
                            remaining_time = (
                                self.config.max_processing_time - elapsed_time
                            )

                            if remaining_time <= 0:
                                logger.warning(
                                    f"通道滤波中超时，已处理{ch_idx}/{data.shape[0]}个通道，返回部分处理结果"
                                )
                                # 复制已处理的通道，未处理的通道使用原始数据
                                for i in range(ch_idx, data.shape[0]):
                                    filtered_data[i] = data[i]
                                break

                        # 应用滤波器 - 使用lfilter代替filtfilt，速度更快
                        filtered_data[ch_idx] = signal.lfilter(b, 1.0, data[ch_idx])

                    # 创建新的Raw对象
                    raw_filtered = mne.io.RawArray(filtered_data, raw.info.copy())

                    # 复制原始对象的其他属性
                    if hasattr(raw, "annotations") and raw.annotations is not None:
                        raw_filtered.set_annotations(raw.annotations)

                    return raw_filtered

                else:
                    # 使用MNE的滤波功能，但调整参数以适应短数据
                    logger.info(f"使用MNE标准滤波，高通频率调整为{adjusted_highpass}Hz")

                    # 超时检测
                    if start_time is not None and self.config.enable_timeout_detection:
                        elapsed_time = time.time() - start_time
                        remaining_time = self.config.max_processing_time - elapsed_time

                        if (
                            remaining_time <= 0.2
                        ):  # 如果剩余时间不多，使用更快的滤波方法
                            logger.warning(f"剩余时间不足，使用快速滤波方法")
                            # 使用iir_params='auto'可以加快滤波速度
                            return raw.filter(
                                l_freq=adjusted_highpass,
                                h_freq=self.config.filter_lowpass,
                                picks="all",
                                method="iir",
                                iir_params="auto",
                                verbose=False,
                            )

                    # 标准滤波
                    return raw.filter(
                        l_freq=adjusted_highpass,
                        h_freq=self.config.filter_lowpass,
                        picks="all",
                        verbose=False,
                    )
            else:
                # 标准滤波
                logger.info(f"数据长度({data_duration:.2f}秒)足够，使用标准滤波")

                # 超时检测
                if start_time is not None and self.config.enable_timeout_detection:
                    elapsed_time = time.time() - start_time
                    remaining_time = self.config.max_processing_time - elapsed_time

                    if remaining_time <= 0:
                        logger.warning(
                            f"滤波前已超时，已消耗{elapsed_time:.3f}秒，跳过滤波"
                        )
                        return raw  # 超时直接返回原始数据

                    if remaining_time <= 0.2:  # 如果剩余时间不多，使用更快的滤波方法
                        logger.warning(f"剩余时间不足，使用快速滤波方法")
                        # 使用iir_params='auto'可以加快滤波速度
                        return raw.filter(
                            l_freq=self.config.filter_highpass,
                            h_freq=self.config.filter_lowpass,
                            picks="all",
                            method="iir",
                            iir_params="auto",
                            verbose=False,
                        )

                # 标准滤波
                return raw.filter(
                    l_freq=self.config.filter_highpass,
                    h_freq=self.config.filter_lowpass,
                    picks="all",
                    verbose=False,
                )

        except Exception as e:
            logger.error(f"应用滤波时出错: {e}", exc_info=True)
            return raw  # 出错时返回原始数据

    def _remove_artifacts(
        self, raw: mne.io.Raw, start_time: float = None
    ) -> mne.io.Raw:
        """去除伪迹 - 优化版，使用CLEEGN模型进行伪迹检测和去除

        Args:
            raw: MNE Raw对象
            start_time: 处理开始时间，用于超时检测

        Returns:
            mne.io.Raw: 去除伪迹后的Raw对象
        """
        try:
            # 超时检测
            if start_time is not None and self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0:
                    logger.warning(
                        f"去伪迹前已超时，已消耗{elapsed_time:.3f}秒，跳过去伪迹"
                    )
                    return raw  # 超时直接返回原始数据

                logger.info(f"去伪迹前剩余处理时间: {remaining_time:.3f}秒")

                # 根据剩余时间调整去伪迹策略
                if remaining_time < 0.2:  # 如果剩余时间很少，跳过复杂处理
                    logger.warning(f"剩余时间不足，仅执行基本阈值去伪迹")
                    # 使用简单阈值去伪迹
                    return self._threshold_artifact_removal(raw, start_time)

            # 强制禁用ICA以提高实时处理速度
            if self.config.use_ica:  # 如果配置中启用了ICA，覆盖它
                self.config.use_ica = False
                self.config.ica_components = 0
                logger.warning("强制禁用ICA以提高实时处理速度和稳定性")

            # 复制原始数据
            raw_clean = raw.copy()

            # 尝试使用CLEEGN模型进行伪迹检测和去除
            try:
                # 导入CLEEGN模型
                from processor.cleegn_model import get_cleegn_instance

                # 获取数据
                data = raw_clean.get_data()

                # 获取采样率
                sampling_rate = raw_clean.info["sfreq"]

                # 获取通道数
                n_channels = data.shape[0]

                # 获取CLEEGN模型实例
                cleegn = get_cleegn_instance(
                    sampling_rate=sampling_rate,
                    n_channels=n_channels,
                    window_size=0.5,  # 使用0.5秒窗口
                    use_gpu=False,
                )  # 默认不使用GPU

                # 使用CLEEGN模型检测和去除伪迹
                start_cleegn = time.time()
                cleaned_data, artifacts = cleegn.remove_artifacts(data)
                cleegn_time = time.time() - start_cleegn

                # 记录处理时间和结果
                artifact_count = np.sum(artifacts)
                logger.info(
                    f"CLEEGN模型检测到{artifact_count}个通道存在伪迹，处理耗时: {cleegn_time*1000:.2f}ms"
                )

                # 更新数据
                raw_clean._data = cleaned_data

                # 如果CLEEGN模型成功处理，直接返回结果
                return raw_clean

            except Exception as e:
                logger.warning(f"CLEEGN模型处理失败: {e}，回退到传统阈值方法")
                # 回退到传统阈值方法
                return self._threshold_artifact_removal(raw, start_time)

        except Exception as e:
            logger.error(f"去除伪迹时出错: {e}", exc_info=True)
            return raw  # 出错时返回原始数据

    def _threshold_artifact_removal(
        self, raw: mne.io.Raw, start_time: float = None
    ) -> mne.io.Raw:
        """使用传统阈值方法去除伪迹 (备用方法)

        Args:
            raw: MNE Raw对象
            start_time: 处理开始时间，用于超时检测

        Returns:
            mne.io.Raw: 去除伪迹后的Raw对象
        """
        # 复制原始数据
        raw_clean = raw.copy()

        # 简单阈值去伪迹
        if self.config.artifact_threshold > 0:
            # 获取数据
            data = raw_clean.get_data()

            # 计算阈值（微伏）
            threshold = self.config.artifact_threshold * 1e-6  # 转换为伏特

            # 检测超过阈值的数据点
            artifacts = np.abs(data) > threshold
            artifact_count = np.sum(artifacts)

            if artifact_count > 0:
                # 超时检测
                if start_time is not None and self.config.enable_timeout_detection:
                    elapsed_time = time.time() - start_time
                    remaining_time = self.config.max_processing_time - elapsed_time

                    if remaining_time <= 0:
                        logger.warning(
                            f"检测到{artifact_count}个伪迹点，但已超时，跳过处理"
                        )
                        return raw_clean

                # 计算每个通道的伪迹比例
                channel_artifact_ratio = np.mean(artifacts, axis=1)

                # 对于伪迹比例低的通道，使用插值替换伪迹
                for ch_idx in range(data.shape[0]):
                    # 超时检测 - 每处理几个通道检查一次
                    if (
                        start_time is not None
                        and self.config.enable_timeout_detection
                        and ch_idx % 4 == 0
                    ):  # 每4个通道检查一次
                        elapsed_time = time.time() - start_time
                        remaining_time = self.config.max_processing_time - elapsed_time

                        if remaining_time <= 0:
                            logger.warning(
                                f"通道去伪迹中超时，已处理{ch_idx}/{data.shape[0]}个通道，返回部分处理结果"
                            )
                            break

                    if channel_artifact_ratio[ch_idx] < 0.1:  # 伪迹比例小于10%
                        # 找出该通道的伪迹点
                        bad_indices = np.where(artifacts[ch_idx])[0]

                        if len(bad_indices) > 0:
                            # 使用线性插值替换伪迹
                            for idx in bad_indices:
                                # 超时检测 - 如果伪迹点太多，定期检查
                                if (
                                    start_time is not None
                                    and self.config.enable_timeout_detection
                                    and idx % 100 == 0
                                    and len(bad_indices) > 200
                                ):
                                    elapsed_time = time.time() - start_time
                                    remaining_time = (
                                        self.config.max_processing_time - elapsed_time
                                    )

                                    if remaining_time <= 0:
                                        logger.warning(
                                            f"伪迹点处理中超时，已处理{idx}/{len(bad_indices)}个点，中断处理"
                                        )
                                        break

                                # 找到前后非伪迹点
                                left_idx = idx - 1
                                while left_idx >= 0 and artifacts[ch_idx, left_idx]:
                                    left_idx -= 1

                                right_idx = idx + 1
                                while (
                                    right_idx < data.shape[1]
                                    and artifacts[ch_idx, right_idx]
                                ):
                                    right_idx += 1

                                # 如果找到了前后非伪迹点，进行插值
                                if left_idx >= 0 and right_idx < data.shape[1]:
                                    # 线性插值
                                    data[ch_idx, idx] = data[ch_idx, left_idx] + (
                                        data[ch_idx, right_idx] - data[ch_idx, left_idx]
                                    ) * (idx - left_idx) / (right_idx - left_idx)
                                elif left_idx >= 0:
                                    # 使用左侧值
                                    data[ch_idx, idx] = data[ch_idx, left_idx]
                                elif right_idx < data.shape[1]:
                                    # 使用右侧值
                                    data[ch_idx, idx] = data[ch_idx, right_idx]

                # 更新数据
                raw_clean._data = data
                logger.info(f"已处理{artifact_count}个伪迹点")

        return raw_clean

    def extract_features(
        self, eegrdd: EEGRDD, start_time: float = None
    ) -> Dict[str, Any]:
        """提取特征 - 优化版，使用小波变换或短时傅里叶变换，带有超时检测

        Args:
            eegrdd: EEGRDD对象
            start_time: 处理开始时间，用于超时检测

        Returns:
            Dict[str, Any]: 特征字典
        """
        try:
            # 如果未提供开始时间，使用当前时间
            if start_time is None:
                start_time = time.time()

            # 初始化特征字典
            features = {
                "subject_id": eegrdd.subject_id,
                "task_name": eegrdd.task_name,
                "timestamp": time.time(),
                "band_powers": {},
                "complexity": {},
            }

            # 获取数据
            data = eegrdd.voltage_data
            sampling_rate = eegrdd.sampling_rate

            # 超时检测
            if self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0:
                    logger.warning(
                        f"特征提取前已超时，已消耗{elapsed_time:.3f}秒，返回基本特征"
                    )
                    return features  # 超时直接返回基本特征

                logger.info(f"特征提取前剩余处理时间: {remaining_time:.3f}秒")

            # 根据剩余时间选择特征提取方法
            if self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time < 0.2:  # 如果剩余时间很少，使用最简单的FFT方法
                    logger.warning(
                        f"剩余时间不足({remaining_time:.3f}秒)，使用简化的FFT方法"
                    )
                    band_powers = self._compute_fft_band_powers(data, sampling_rate)
                elif (
                    remaining_time < 0.3 and self.config.use_wavelet
                ):  # 如果剩余时间较少且原本使用小波变换，改用STFT
                    logger.warning(
                        f"剩余时间较少({remaining_time:.3f}秒)，改用STFT代替小波变换"
                    )
                    band_powers = self._compute_stft_band_powers(data, sampling_rate)
                else:  # 使用原定方法
                    if self.config.use_wavelet:
                        # 使用小波变换计算频带能量
                        band_powers = self._compute_wavelet_band_powers(
                            data, sampling_rate
                        )
                    elif self.config.use_short_time_fft:
                        # 使用短时傅里叶变换计算频带能量
                        band_powers = self._compute_stft_band_powers(
                            data, sampling_rate
                        )
                    else:
                        # 使用标准FFT计算频带能量
                        band_powers = self._compute_fft_band_powers(data, sampling_rate)
            else:  # 不检测超时，使用原定方法
                if self.config.use_wavelet:
                    # 使用小波变换计算频带能量
                    band_powers = self._compute_wavelet_band_powers(data, sampling_rate)
                elif self.config.use_short_time_fft:
                    # 使用短时傅里叶变换计算频带能量
                    band_powers = self._compute_stft_band_powers(data, sampling_rate)
                else:
                    # 使用标准FFT计算频带能量
                    band_powers = self._compute_fft_band_powers(data, sampling_rate)

            features["band_powers"] = band_powers

            # 超时检测 - 在计算复杂度前检查
            if self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0.1:  # 如果剩余时间很少，跳过复杂度计算
                    logger.warning(
                        f"剩余时间不足({remaining_time:.3f}秒)，跳过复杂度计算"
                    )
                    return features

            # 计算信号复杂度指标
            for ch_idx, ch_name in enumerate(eegrdd.channel_names):
                # 超时检测 - 每处理几个通道检查一次
                if (
                    self.config.enable_timeout_detection and ch_idx % 4 == 0
                ):  # 每4个通道检查一次
                    elapsed_time = time.time() - start_time
                    remaining_time = self.config.max_processing_time - elapsed_time

                    if remaining_time <= 0:
                        logger.warning(
                            f"复杂度计算中超时，已处理{ch_idx}/{len(eegrdd.channel_names)}个通道，返回部分结果"
                        )
                        break

                # 计算样本熵（简化版）
                features["complexity"][ch_name] = self._compute_approximate_entropy(
                    data[ch_idx]
                )

            # 计算频带能量比
            try:
                band_power_ratio = self._compute_band_power_ratio(
                    features["band_powers"]
                )
                features["band_power_ratio"] = band_power_ratio
            except Exception as e:
                logger.warning(f"频带能量比计算失败: {e}")

            # 超时检测
            if self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0.1:  # 如果剩余时间很少，跳过频谱熵计算
                    logger.warning(
                        f"剩余时间不足({remaining_time:.3f}秒)，跳过频谱熵计算"
                    )
                    return features

            # 计算频谱熵
            try:
                spectral_entropy = self._compute_spectral_entropy(data, sampling_rate)
                features["spectral_entropy"] = spectral_entropy
            except Exception as e:
                logger.warning(f"频谱熵计算失败: {e}")

            return features

        except Exception as e:
            logger.error(f"提取特征时出错: {e}", exc_info=True)
            # 返回基本特征
            return {
                "subject_id": eegrdd.subject_id,
                "task_name": eegrdd.task_name,
                "timestamp": time.time(),
                "band_powers": {},
                "complexity": {},
            }

    def _compute_wavelet_band_powers(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, float]:
        """使用小波变换计算频带能量 - 已注释，用于分析小波变换所需的数据情况

        Args:
            data: 数据数组，形状为(n_channels, n_samples)
            sampling_rate: 采样率

        Returns:
            Dict[str, float]: 频带能量字典
        """
        # 记录数据情况，用于分析小波变换所需的最小数据长度
        logger.info(f"小波变换数据分析 - 形状: {data.shape}, 采样率: {sampling_rate}Hz")
        logger.info(
            f"数据统计 - 最小值: {np.min(data)}, 最大值: {np.max(data)}, 均值: {np.mean(data)}, 标准差: {np.std(data)}"
        )

        # 计算每个通道的数据长度（时间）
        duration = data.shape[1] / sampling_rate
        logger.info(f"数据持续时间: {duration:.4f}秒, 样本点数: {data.shape[1]}")

        # 分析小波变换所需的最小数据长度
        min_required_samples = 64  # 当前代码中的最小要求
        min_required_duration = min_required_samples / sampling_rate
        logger.info(
            f"小波变换当前最小要求: {min_required_samples}样本点 ({min_required_duration:.4f}秒)"
        )

        # 分析可用的尺度范围
        max_possible_scales = data.shape[1] // 4  # 当前代码中的限制
        logger.info(
            f"可用的最大尺度数: {max_possible_scales} (当前设置: {self.config.wavelet_scales})"
        )

        # 分析理论上可分析的最低频率
        lowest_analyzable_freq = sampling_rate / data.shape[1]
        logger.info(f"理论上可分析的最低频率: {lowest_analyzable_freq:.2f}Hz")

        # 检查是否满足小波变换的最小数据要求
        if data.shape[1] < min_required_samples:
            logger.warning(
                f"数据长度({data.shape[1]})不满足小波变换最小要求({min_required_samples})，将使用FFT方法代替"
            )

        # 暂时注释掉小波变换的实现，改用FFT方法
        logger.info("小波变换已暂时注释，使用FFT方法代替")
        return self._compute_fft_band_powers(data, sampling_rate)  # 使用FFT方法代替

    def _compute_stft_band_powers(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, float]:
        """使用短时傅里叶变换计算频带能量 - 适用于非平稳信号

        Args:
            data: 数据数组，形状为(n_channels, n_samples)
            sampling_rate: 采样率

        Returns:
            Dict[str, float]: 频带能量字典
        """
        try:
            from scipy import signal

            # 初始化结果字典
            band_powers = {band: 0.0 for band in self.config.freq_bands.keys()}

            # 检查数据长度是否足够
            if data.shape[1] < 64:  # 如果数据太短，使用简化方法
                logger.warning(
                    f"数据长度过短({data.shape[1]}样本点)，使用简化的FFT方法"
                )
                return self._compute_fft_band_powers(
                    data, sampling_rate
                )  # 回退到FFT方法

            # 自适应窗口大小 - 根据数据长度调整
            window_size = min(
                int(self.config.stft_window_size * sampling_rate), data.shape[1] // 2
            )
            if window_size < 16:  # 窗口太小，无法进行有效的频谱分析
                window_size = min(data.shape[1] // 2, 16)  # 确保至少有16个点

            # 自适应重叠 - 确保至少有2个窗口
            overlap = min(
                int(self.config.stft_overlap * sampling_rate), window_size // 2
            )

            logger.info(f"使用自适应STFT: 窗口大小={window_size}点, 重叠={overlap}点")

            # 对每个通道进行STFT
            for ch_idx in range(data.shape[0]):
                # 获取通道数据
                ch_data = data[ch_idx]

                # 数据预处理 - 去均值
                ch_data = ch_data - np.mean(ch_data)

                try:
                    # 应用短时傅里叶变换
                    f, t, Zxx = signal.stft(
                        ch_data,
                        fs=sampling_rate,
                        window="hann",
                        nperseg=window_size,
                        noverlap=overlap,
                        detrend="constant",  # 去除直流分量
                        boundary="zeros",  # 边界处理方式
                    )

                    # 计算功率谱密度
                    psd = np.mean(np.abs(Zxx) ** 2, axis=1)

                    # 计算各频带能量
                    for band_name, (
                        band_min,
                        band_max,
                    ) in self.config.freq_bands.items():
                        # 找出频带对应的索引
                        idx_min = (
                            np.argmax(f >= band_min)
                            if band_min <= f[-1]
                            else len(f) - 1
                        )
                        idx_max = (
                            np.argmax(f >= band_max)
                            if band_max <= f[-1]
                            else len(f) - 1
                        )

                        # 计算频带能量
                        if idx_min < idx_max:
                            band_energy = np.sum(psd[idx_min : idx_max + 1])
                            band_powers[band_name] += (
                                band_energy / data.shape[0]
                            )  # 平均到通道数
                except Exception as e:
                    logger.warning(f"STFT计算失败: {e}，使用FFT方法")
                    # 使用FFT作为备选方法
                    try:
                        # 应用汉宁窗
                        windowed_data = ch_data * np.hanning(len(ch_data))

                        # 计算FFT
                        fft_data = np.abs(np.fft.rfft(windowed_data)) ** 2
                        freqs = np.fft.rfftfreq(len(windowed_data), 1.0 / sampling_rate)

                        # 计算各频带能量
                        for band_name, (
                            band_min,
                            band_max,
                        ) in self.config.freq_bands.items():
                            # 找出频带对应的索引
                            idx_min = (
                                np.argmax(freqs >= band_min)
                                if band_min <= freqs[-1]
                                else len(freqs) - 1
                            )
                            idx_max = (
                                np.argmax(freqs >= band_max)
                                if band_max <= freqs[-1]
                                else len(freqs) - 1
                            )

                            # 计算频带能量
                            if idx_min < idx_max:
                                band_energy = np.sum(fft_data[idx_min : idx_max + 1])
                                band_powers[band_name] += (
                                    band_energy / data.shape[0]
                                )  # 平均到通道数
                    except Exception as e:
                        logger.error(f"FFT计算也失败: {e}")
                        # 如果所有方法都失败，返回空结果
                        continue

            # 归一化处理
            total_power = sum(band_powers.values())
            if total_power > 0:
                for band in band_powers:
                    band_powers[band] /= total_power

            return band_powers

        except Exception as e:
            logger.error(f"计算STFT频带能量时出错: {e}", exc_info=True)
            # 出错时回退到FFT方法
            return self._compute_fft_band_powers(data, sampling_rate)

    def _compute_fft_band_powers(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, float]:
        """使用FFT计算频带能量 - 优化版，适合短时间序列

        Args:
            data: 数据数组，形状为(n_channels, n_samples)
            sampling_rate: 采样率

        Returns:
            Dict[str, float]: 频带能量字典
        """
        try:
            # 初始化结果字典
            band_powers = {band: 0.0 for band in self.config.freq_bands.keys()}

            # 检查数据长度
            if data.shape[1] < 8:  # 数据太短，无法进行有效分析
                logger.warning(f"数据长度过短({data.shape[1]}样本点)，无法进行频谱分析")
                # 返回均匀分布的能量
                total = 1.0 / len(band_powers)
                return {band: total for band in self.config.freq_bands.keys()}

            # 对每个通道进行FFT
            for ch_idx in range(data.shape[0]):
                # 获取通道数据
                ch_data = data[ch_idx]

                # 数据预处理 - 去均值和线性趋势
                ch_data = ch_data - np.mean(ch_data)
                if len(ch_data) > 3:  # 至少需要3个点才能去线性趋势
                    try:
                        from scipy import signal

                        ch_data = signal.detrend(ch_data)
                    except Exception:
                        pass  # 如果detrend失败，继续使用原始数据

                # 应用窗函数 - 对于短时间序列，使用更适合的窗函数
                if len(ch_data) < 32:
                    # 对于非常短的序列，使用矩形窗或Tukey窗
                    try:
                        from scipy import signal

                        window = signal.tukey(
                            len(ch_data), alpha=0.25
                        )  # 较小的alpha值使窗函数更接近矩形窗
                        ch_data = ch_data * window
                    except Exception:
                        # 如果失败，使用简单的矩形窗（即不应用窗函数）
                        pass
                else:
                    # 对于较长序列，使用汉宁窗
                    ch_data = ch_data * np.hanning(len(ch_data))

                # 计算FFT - 使用零填充提高频率分辨率
                n_fft = max(
                    256, 2 ** (int(np.log2(len(ch_data))) + 1)
                )  # 使用2的幂次方提高FFT效率
                try:
                    fft_data = np.abs(np.fft.rfft(ch_data, n=n_fft)) ** 2
                    freqs = np.fft.rfftfreq(n_fft, 1.0 / sampling_rate)
                except Exception as e:
                    logger.warning(f"FFT计算失败: {e}，尝试使用较小的FFT长度")
                    # 如果失败，尝试使用较小的FFT长度
                    n_fft = max(64, 2 ** int(np.log2(len(ch_data))))
                    fft_data = np.abs(np.fft.rfft(ch_data, n=n_fft)) ** 2
                    freqs = np.fft.rfftfreq(n_fft, 1.0 / sampling_rate)

                # 计算各频带能量 - 使用更精确的频带边界检测
                for band_name, (band_min, band_max) in self.config.freq_bands.items():
                    # 找出频带对应的索引 - 改进的方法
                    idx_band = np.where((freqs >= band_min) & (freqs <= band_max))[0]

                    if len(idx_band) > 0:
                        # 计算频带能量
                        band_energy = np.sum(fft_data[idx_band])
                        band_powers[band_name] += (
                            band_energy / data.shape[0]
                        )  # 平均到通道数
                    else:
                        # 如果没有找到对应频带，记录警告
                        logger.warning(
                            f"频带 {band_name} ({band_min}-{band_max}Hz) 在当前频率分辨率下未找到对应点"
                        )

            # 归一化处理
            total_power = sum(band_powers.values())
            if total_power > 0:
                for band in band_powers:
                    band_powers[band] /= total_power
            else:
                # 如果总能量为0，使用均匀分布
                total = 1.0 / len(band_powers)
                for band in band_powers:
                    band_powers[band] = total

            return band_powers

        except Exception as e:
            logger.error(f"计算FFT频带能量时出错: {e}", exc_info=True)
            # 出错时返回均匀分布的能量
            total = 1.0 / len(self.config.freq_bands)
            return {band: total for band in self.config.freq_bands.keys()}

    def _compute_band_power_ratio(
        self, band_powers: Dict[str, float]
    ) -> Dict[str, float]:
        """计算频带能量比

        Args:
            band_powers: 频带能量字典

        Returns:
            Dict[str, float]: 频带能量比字典
        """
        # 初始化结果字典
        band_power_ratio = {}

        # 计算总能量
        total_power = sum(band_powers.values())

        # 避免除以零
        if total_power <= 0:
            return {band: 0.0 for band in band_powers.keys()}

        # 计算各频带能量占比
        for band, power in band_powers.items():
            band_power_ratio[band] = power / total_power

        # 计算常用的频带比值
        if (
            "theta" in band_powers
            and "alpha_low" in band_powers
            and "alpha_high" in band_powers
        ):
            alpha = band_powers["alpha_low"] + band_powers["alpha_high"]
            band_power_ratio["theta_alpha_ratio"] = (
                band_powers["theta"] / alpha if alpha > 0 else 0
            )

        if (
            "alpha_low" in band_powers
            and "alpha_high" in band_powers
            and "beta_low" in band_powers
            and "beta_high" in band_powers
        ):
            alpha = band_powers["alpha_low"] + band_powers["alpha_high"]
            beta = band_powers["beta_low"] + band_powers["beta_high"]
            band_power_ratio["alpha_beta_ratio"] = alpha / beta if beta > 0 else 0

        return band_power_ratio

    def _compute_spectral_entropy(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, float]:
        """计算频谱熵

        Args:
            data: 数据数组，形状为(n_channels, n_samples)
            sampling_rate: 采样率

        Returns:
            Dict[str, float]: 频谱熵字典，每个通道一个值
        """
        from scipy import signal
        import numpy as np

        # 初始化结果字典
        spectral_entropy = {}

        # 对每个通道计算频谱熵
        for ch_idx in range(data.shape[0]):
            # 获取通道数据
            ch_data = data[ch_idx]

            # 计算功率谱密度
            f, psd = signal.welch(
                ch_data,
                fs=sampling_rate,
                nperseg=min(256, len(ch_data)),
                scaling="density",
            )

            # 归一化PSD
            psd_norm = psd / np.sum(psd)

            # 计算熵值 (Shannon熵)
            entropy = -np.sum(
                psd_norm * np.log2(psd_norm + 1e-10)
            )  # 添加小值避免log(0)

            # 存储结果
            spectral_entropy[f"ch{ch_idx+1}"] = entropy

        return spectral_entropy

    def _compute_approximate_entropy(
        self, data: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        """计算近似熵 - 简化版，用于评估信号复杂度

        Args:
            data: 一维数据数组
            m: 嵌入维度
            r: 相似性阈值（标准差的比例）

        Returns:
            float: 近似熵值
        """
        try:
            # 数据标准化
            data = (data - np.mean(data)) / np.std(data)
            r = r * np.std(data)

            # 如果数据太短，返回默认值
            if len(data) < 100:
                return 0.0

            # 计算近似熵
            def _phi(m):
                # 创建模板向量
                templates = np.array(
                    [data[i : i + m] for i in range(len(data) - m + 1)]
                )
                # 计算模板间距离
                distances = np.max(
                    np.abs(templates[:, np.newaxis] - templates[np.newaxis, :]), axis=2
                )
                # 计算相似模板比例
                count = np.sum(distances <= r, axis=1) / (len(data) - m + 1)
                # 返回对数平均值
                return np.mean(np.log(count))

            # 计算两个嵌入维度的差异
            return abs(_phi(m) - _phi(m + 1))

        except Exception as e:
            logger.error(f"计算近似熵时出错: {e}", exc_info=True)
            return 0.0

    def process_realtime(
        self, subject_id: str, eegrdd: EEGRDD
    ) -> Tuple[EEGRDD, Dict[str, Any]]:
        """实时处理EEG数据，带有超时检测机制

        Args:
            subject_id: 受试者ID
            eegrdd: EEGRDD对象

        Returns:
            Tuple[EEGRDD, Dict[str, Any]]: 处理后的EEGRDD对象和特征
        """
        try:
            # 记录开始时间，用于超时检测
            start_time = time.time()

            # 添加到缓冲区
            should_process = self.add_to_buffer(subject_id, eegrdd)

            if not should_process:
                # 数据长度不足，返回原始数据
                logger.info(f"数据长度不足，跳过处理")
                return eegrdd, None

            # 获取缓冲区数据
            buffer_data, buffer_metadata = self.get_buffer_data(subject_id)

            if buffer_data is None:
                # 缓冲区为空，返回原始数据
                return eegrdd, None

            # 创建临时EEGRDD对象
            buffer_eegrdd = EEGRDD(
                subject_id=subject_id,
                voltage_data=buffer_data,
                sampling_rate=buffer_metadata["sampling_rate"],
                payload_id=buffer_metadata.get("payload_id"),
                task_name=buffer_metadata.get("task_name", "unknown"),
                block_id=buffer_metadata.get("block_id"),
                n_channels=buffer_data.shape[0],
                n_samples=buffer_data.shape[1],
                duration=buffer_data.shape[1] / buffer_metadata["sampling_rate"],
                reference=buffer_metadata.get("reference"),
                channel_names=buffer_metadata.get("channel_names", []),
                channel_types=buffer_metadata.get("channel_types", []),
                channel_units=buffer_metadata.get("channel_units", []),
                channel_locations=buffer_metadata.get("channel_locations", []),
                events=buffer_metadata.get("events", []),
            )

            # 检查是否启用超时检测
            if self.config.enable_timeout_detection:
                # 计算已经消耗的时间
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0:
                    logger.warning(
                        f"处理超时，已消耗{elapsed_time:.3f}秒，跳过处理返回原始数据"
                    )
                    # 清空缓冲区，保留重叠部分
                    self.clear_buffer(subject_id, keep_overlap=True)
                    return eegrdd, None

                logger.info(f"剩余处理时间: {remaining_time:.3f}秒")

                # 根据剩余时间调整处理策略
                if remaining_time < 0.2:  # 如果剩余时间很少，使用简化处理
                    logger.warning(f"剩余时间不足，使用简化处理流程")
                    # 简化处理：只应用基本滤波，跳过复杂的特征提取
                    try:
                        # 转换为MNE对象
                        raw = self.eegrdd_to_mne(buffer_eegrdd)

                        # 应用简化滤波 - 只使用低通滤波，使用更快的IIR滤波器
                        raw = raw.copy().filter(
                            l_freq=None,
                            h_freq=self.config.filter_lowpass,
                            method="iir",
                            iir_params="auto",  # 使用自动参数，速度更快
                            verbose=False,
                        )

                        # 转换回EEGRDD
                        processed_eegrdd = self.mne_to_eegrdd(raw, buffer_eegrdd)

                        # 清空缓冲区，保留重叠部分
                        self.clear_buffer(subject_id, keep_overlap=True)

                        # 计算最基本的特征（只计算频带能量，使用最快的FFT方法）
                        basic_features = {
                            "subject_id": processed_eegrdd.subject_id,
                            "task_name": processed_eegrdd.task_name,
                            "timestamp": time.time(),
                            "band_powers": self._compute_fft_band_powers(
                                processed_eegrdd.voltage_data,
                                processed_eegrdd.sampling_rate,
                            ),
                            "connectivity": {},
                            "complexity": {},
                        }

                        return processed_eegrdd, basic_features
                    except Exception as e:
                        logger.error(f"简化处理失败: {e}", exc_info=True)
                        # 清空缓冲区，保留重叠部分
                        self.clear_buffer(subject_id, keep_overlap=True)
                        return eegrdd, None

            # 处理数据
            processed_eegrdd = self.process_eegrdd(buffer_eegrdd)

            # 再次检查超时
            if self.config.enable_timeout_detection:
                elapsed_time = time.time() - start_time
                remaining_time = self.config.max_processing_time - elapsed_time

                if remaining_time <= 0:
                    logger.warning(
                        f"特征提取前已超时，已消耗{elapsed_time:.3f}秒，跳过特征提取"
                    )
                    # 清空缓冲区，保留重叠部分
                    self.clear_buffer(subject_id, keep_overlap=True)
                    return processed_eegrdd, None

            # 提取特征
            features = self.extract_features(processed_eegrdd)

            # 清空缓冲区，保留重叠部分
            self.clear_buffer(subject_id, keep_overlap=True)

            # 记录总处理时间
            total_time = time.time() - start_time
            logger.info(f"总处理时间: {total_time:.3f}秒")

            return processed_eegrdd, features

        except Exception as e:
            logger.error(f"实时处理EEG数据时出错: {e}", exc_info=True)
            # 出错时也要清空缓冲区
            self.clear_buffer(subject_id, keep_overlap=True)
            return eegrdd, None


import time  # 添加time模块导入，用于时间戳生成
