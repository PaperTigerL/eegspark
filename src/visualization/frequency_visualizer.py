import numpy as np
import matplotlib
from matplotlib.figure import Figure
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import time
from scipy import signal

from .base_visualizer import BaseVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FrequencyVisualizer")


class FrequencyVisualizer(BaseVisualizer):
    """EEG频域可视化类 - 专门处理频域分析的可视化"""

    def __init__(self, sampling_rate: float = 5000.0, reference: str = "FCz"):
        """初始化频域可视化器

        Args:
            sampling_rate: 采样率 (Hz)
            reference: 参考电极
        """
        super().__init__(sampling_rate, reference)

    def plot_frequency_domain(
        self,
        data: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        freq_range: Tuple[float, float] = (0, 50),
        selected_channels: Optional[List[str]] = None,
        show_bands: bool = True,
        method: str = "welch",  # 'fft', 'welch', 或 'multitaper'
        log_scale: bool = False,
        normalize: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        source_id: Optional[str] = None,
    ) -> str:
        """绘制频域图，支持多种频谱分析方法

        Args:
            data: 形状为(n_channels, n_samples)的数据数组，如果为None，则使用source_id的数据
            channel_names: 通道名称列表，如果为None，则使用source_id的通道名称
            freq_range: 频率范围 (Hz)
            selected_channels: 要显示的通道列表，如果为None，则显示所有通道
            show_bands: 是否显示频带标记
            method: 频谱分析方法 ('fft', 'welch', 或 'multitaper')
            log_scale: 是否使用对数刻度
            normalize: 是否归一化功率谱
            figsize: 图形大小
            source_id: 数据源标识符，如果为None，则使用活动数据源

        Returns:
            str: Base64编码的图像
        """
        # 如果没有提供数据和通道名称，从数据源获取
        if data is None or channel_names is None:
            source_data = self.get_data_source(source_id)
            if source_data is None:
                logger.error(f"没有可用于频域绘图的数据源")
                return ""

            if data is None:
                data = source_data["data"]
            if channel_names is None:
                channel_names = source_data["channel_names"]

            # 使用数据源的采样率
            source_sampling_rate = source_data.get("sampling_rate", self.sampling_rate)
        else:
            source_sampling_rate = self.sampling_rate
            source_data = None

        # 获取数据源ID用于显示
        display_source_id = source_id or self.active_source or "未知数据源"

        # 生成缓存键
        cache_key = self._generate_cache_key(
            "frequency_domain",
            data_shape=data.shape if data is not None else None,
            data_hash=(
                (
                    hash(data.tobytes())
                    if data.size < 10000
                    else hash(data[:, ::10].tobytes())
                )
                if data is not None
                else None
            ),
            channel_names=tuple(channel_names) if channel_names is not None else None,
            freq_range=freq_range,
            selected_channels=tuple(selected_channels) if selected_channels else None,
            show_bands=show_bands,
            method=method,
            log_scale=log_scale,
            normalize=normalize,
            figsize=figsize,
            source_id=display_source_id,
        )

        # 检查缓存
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # 记录开始时间，用于性能分析
        start_time_perf = time.time()

        try:
            # 确保数据是二维的
            if data is not None and data.ndim == 1:
                data = data.reshape(1, -1)

            # 创建数据的副本，避免修改原始数据
            data_copy = None
            if data is not None:
                data_copy = data.copy()

            # 过滤选择的通道
            if (
                selected_channels is not None
                and len(selected_channels) > 0
                and channel_names is not None
            ):
                selected_indices = [
                    i for i, ch in enumerate(channel_names) if ch in selected_channels
                ]
                if selected_indices:  # 确保有选择的通道
                    data_copy = data_copy[selected_indices]
                    filtered_channel_names = [
                        channel_names[i] for i in selected_indices
                    ]
                else:
                    # 如果没有匹配的通道，使用所有通道
                    filtered_channel_names = channel_names
            else:
                filtered_channel_names = channel_names

            # 创建图像
            fig = Figure(figsize=figsize, dpi=80)
            canvas = fig.canvas

            # 计算通道数量
            n_channels = (
                len(filtered_channel_names)
                if filtered_channel_names
                else data_copy.shape[0]
            )

            # 创建子图布局
            if n_channels <= 4:
                # 对于少量通道，使用单行布局
                n_rows, n_cols = 1, n_channels
            else:
                # 对于多个通道，使用多行布局
                n_cols = min(4, n_channels)  # 最多4列
                n_rows = (n_channels + n_cols - 1) // n_cols  # 向上取整

            # 计算频谱
            freqs_list = []
            psd_list = []

            for i in range(data_copy.shape[0]):
                channel_data = data_copy[i]

                if method == "fft":
                    # 使用FFT计算功率谱
                    n = len(channel_data)
                    fft_data = np.fft.rfft(channel_data * np.hanning(n))
                    freqs = np.fft.rfftfreq(n, d=1.0 / source_sampling_rate)
                    psd = np.abs(fft_data) ** 2 / n
                elif method == "multitaper":
                    try:
                        from mne.time_frequency import psd_array_multitaper

                        # 使用MNE的多窗口方法
                        psd, freqs = psd_array_multitaper(
                            channel_data,
                            source_sampling_rate,
                            fmin=freq_range[0],
                            fmax=freq_range[1],
                            adaptive=True,
                            normalization="full",
                        )
                    except ImportError:
                        # 如果MNE不可用，回退到Welch方法
                        logger.warning("MNE导入失败，回退到Welch方法")
                        freqs, psd = signal.welch(
                            channel_data,
                            fs=source_sampling_rate,
                            nperseg=min(1024, len(channel_data) // 8),
                            noverlap=min(512, len(channel_data) // 16),
                        )
                else:  # 默认使用Welch方法
                    # 使用Welch方法计算功率谱密度
                    freqs, psd = signal.welch(
                        channel_data,
                        fs=source_sampling_rate,
                        nperseg=min(1024, len(channel_data) // 8),
                        noverlap=min(512, len(channel_data) // 16),
                    )

                # 过滤频率范围
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                freqs = freqs[mask]
                psd = psd[mask]

                # 归一化（如果需要）
                if normalize and np.max(psd) > 0:
                    psd = psd / np.max(psd)

                freqs_list.append(freqs)
                psd_list.append(psd)

            # 绘制每个通道的频谱
            for i in range(n_channels):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)

                # 获取通道颜色
                if filtered_channel_names and i < len(filtered_channel_names):
                    channel_color = self._get_channel_color(filtered_channel_names[i])
                else:
                    channel_color = "gray"

                # 绘制功率谱
                ax.plot(freqs_list[i], psd_list[i], color=channel_color, linewidth=1.5)

                # 如果启用了频带显示，添加频带标记
                if show_bands:
                    y_max = np.max(psd_list[i])
                    y_min = np.min(psd_list[i])
                    height = y_max - y_min

                    for band_name, (band_min, band_max) in self.freq_bands.items():
                        # 检查频带是否在显示范围内
                        if band_max >= freq_range[0] and band_min <= freq_range[1]:
                            # 调整频带范围到显示范围
                            plot_min = max(band_min, freq_range[0])
                            plot_max = min(band_max, freq_range[1])

                            # 获取频带颜色
                            band_color = self.band_colors.get(band_name, "gray")

                            # 添加频带填充
                            ax.axvspan(plot_min, plot_max, alpha=0.2, color=band_color)

                            # 添加频带标签（仅在第一个子图中）
                            if i == 0:
                                # 在频带中心添加标签
                                band_center = (plot_min + plot_max) / 2
                                ax.text(
                                    band_center,
                                    y_max + height * 0.05,
                                    band_name,
                                    ha="center",
                                    va="bottom",
                                    fontsize=8,
                                    color=band_color,
                                )

                # 设置坐标轴
                ax.set_xlim(freq_range)
                if log_scale:
                    ax.set_yscale("log")

                # 设置标签
                if i % n_cols == 0:  # 仅在每行的第一个子图中添加y轴标签
                    if normalize:
                        ax.set_ylabel("归一化功率")
                    else:
                        ax.set_ylabel("功率 (μV²/Hz)")

                if i >= n_channels - n_cols:  # 仅在最后一行添加x轴标签
                    ax.set_xlabel("频率 (Hz)")

                # 设置标题
                if filtered_channel_names and i < len(filtered_channel_names):
                    ax.set_title(filtered_channel_names[i], fontsize=10)
                else:
                    ax.set_title(f"通道 {i+1}", fontsize=10)

            # 设置总标题
            fig.suptitle(
                f"EEG频域分析 - {method.upper()} - 数据源: {display_source_id}",
                fontsize=12,
            )

            # 优化图形外观
            self._optimize_figure_for_web(fig)

            # 转换为base64
            img_base64 = self._figure_to_base64(fig)

            # 缓存结果
            self._cache_result(cache_key, img_base64)

            # 记录处理时间
            process_time = time.time() - start_time_perf
            logger.debug(f"频域绘图完成，处理时间: {process_time:.3f}秒")

            return img_base64

        except Exception as e:
            logger.error(f"频域绘图出错: {e}", exc_info=True)
            # 创建错误图像
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, f"绘图错误: {str(e)}", ha="center", va="center", fontsize=12
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return self._figure_to_base64(fig)

    def calculate_band_powers(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, np.ndarray]:
        """计算各频带的功率

        Args:
            data: 形状为(n_channels, n_samples)的EEG数据
            sampling_rate: 采样率 (Hz)

        Returns:
            Dict[str, np.ndarray]: 各频带的功率，键为频带名称，值为形状为(n_channels,)的功率数组
        """
        n_channels = data.shape[0]
        band_powers = {}

        # 对每个通道计算功率谱
        for i in range(n_channels):
            # 使用Welch方法计算功率谱密度
            freqs, psd = signal.welch(
                data[i],
                fs=sampling_rate,
                nperseg=min(1024, data.shape[1] // 8),
                noverlap=min(512, data.shape[1] // 16),
            )

            # 计算各频带的功率
            for band_name, (band_min, band_max) in self.freq_bands.items():
                # 找到频带范围内的索引
                idx_band = np.logical_and(freqs >= band_min, freqs <= band_max)

                # 计算频带功率（平均值）
                if np.any(idx_band):
                    if band_name not in band_powers:
                        band_powers[band_name] = np.zeros(n_channels)

                    band_powers[band_name][i] = np.mean(psd[idx_band])

        return band_powers
