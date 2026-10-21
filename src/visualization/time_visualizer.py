import numpy as np
import matplotlib
from matplotlib.figure import Figure
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import time

from .base_visualizer import BaseVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TimeVisualizer")


class TimeVisualizer(BaseVisualizer):
    """EEG时域可视化类 - 专门处理时域波形的可视化"""

    def __init__(self, sampling_rate: float = 5000.0, reference: str = "FCz"):
        """初始化时域可视化器

        Args:
            sampling_rate: 采样率 (Hz)
            reference: 参考电极
        """
        super().__init__(sampling_rate, reference)

        # 时域特定设置
        self.sliding_window = 0.5  # 500ms滑动窗口
        self.sliding_step = 0.1  # 100ms步长，实现80%重叠

    def plot_time_domain(
        self,
        data: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        start_time: float = 0,
        duration: float = 5.0,
        baseline_correction: bool = True,
        baseline_window: Tuple[float, float] = (-0.2, 0),
        voltage_scale: Optional[Tuple[float, float]] = None,
        time_markers: Optional[List[Dict[str, Any]]] = None,
        sliding_step: float = 1.0,
        selected_channels: Optional[List[str]] = None,
        show_ms_precision: bool = True,  # 显示毫秒级精度
        show_artifacts: bool = True,  # 显示伪迹标记
        show_erp_average: bool = False,  # 显示ERP平均
        figsize: Tuple[int, int] = (12, 8),
        source_id: Optional[str] = None,
    ) -> str:
        """绘制时域波形，支持增强的实时可视化功能

        Args:
            data: 形状为(n_channels, n_samples)的数据数组，如果为None，则使用source_id的数据
            channel_names: 通道名称列表，如果为None，则使用source_id的通道名称
            start_time: 起始时间（秒）
            duration: 显示持续时间（秒）
            baseline_correction: 是否执行基线校正
            baseline_window: 基线校正窗口（秒）
            voltage_scale: 可选的手动电压刻度，格式为(min_μV, max_μV)，如果为None，则使用自动缩放
            time_markers: 时间标记列表，格式为[{'time': float, 'label': str, 'color': str}]
            sliding_step: 滑动窗口可视化的步长（秒）
            selected_channels: 要显示的通道列表，如果为None，则显示所有通道
            show_ms_precision: 是否在时间轴上显示毫秒精度
            show_artifacts: 是否检测并标记伪迹
            show_erp_average: 是否显示事件相关电位平均值
            figsize: 图形大小
            source_id: 数据源标识符，如果为None，则使用活动数据源

        Returns:
            str: Base64编码的图像
        """
        # 如果没有提供数据和通道名称，从数据源获取
        if data is None or channel_names is None:
            source_data = self.get_data_source(source_id)
            if source_data is None:
                logger.error(f"没有可用于时域绘图的数据源")
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
            "time_domain",
            data_shape=data.shape if data is not None else None,
            data_hash=(
                (
                    hash(data.tobytes())
                    if data.size < 10000
                    else hash(data[:, ::10].tobytes())
                )
                if data is not None
                else None
            ),  # 降采样以加快哈希计算
            channel_names=tuple(channel_names) if channel_names is not None else None,
            start_time=start_time,
            duration=duration,
            baseline_correction=baseline_correction,
            baseline_window=baseline_window,
            voltage_scale=voltage_scale,
            sliding_step=sliding_step,
            time_markers=str(time_markers) if time_markers else None,
            figsize=figsize,
            source_id=display_source_id,  # 添加数据源ID到缓存键
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

            # 计算时间轴
            n_samples = data_copy.shape[1] if data_copy is not None else 0
            times = np.arange(n_samples) / source_sampling_rate

            # 计算显示范围
            start_idx = int(start_time * source_sampling_rate)
            end_idx = min(
                int((start_time + duration) * source_sampling_rate), n_samples
            )

            # 基线校正
            if baseline_correction:
                baseline_start = int(baseline_window[0] * source_sampling_rate)
                baseline_end = int(baseline_window[1] * source_sampling_rate)

                # 确保索引有效
                baseline_start = max(0, baseline_start)
                baseline_end = min(n_samples, baseline_end)

                if baseline_end > baseline_start:
                    # 计算基线均值
                    baseline_mean = np.mean(
                        data_copy[:, baseline_start:baseline_end], axis=1, keepdims=True
                    )
                    # 减去基线均值
                    data_copy = data_copy - baseline_mean
                    logger.debug(f"已应用基线校正，窗口: {baseline_window}s")

            # 创建图像，针对Web显示进行优化
            fig = Figure(figsize=figsize, dpi=80)  # 降低DPI以减小文件大小
            canvas = fig.canvas
            ax = fig.add_subplot(111)

            # 计算显示范围内的数据，避免处理不必要的数据点
            display_data = data_copy[:, start_idx:end_idx]
            display_times = times[start_idx:end_idx]

            # 优化降采样策略，确保毫秒级精度显示
            # 计算每毫秒的采样点数
            ms_resolution = 1  # 目标毫秒分辨率
            points_per_ms = source_sampling_rate / 1000  # 每毫秒的采样点数

            # 根据显示需求确定最佳降采样因子
            if show_ms_precision:
                # 确保至少保留1ms分辨率的数据点 - 对于捕捉瞬态事件（如癫痫样放电）至关重要
                max_downsample = max(
                    1, int(points_per_ms / ms_resolution)
                )  # 最大降采样因子
                # 限制最大降采样因子，确保不会丢失毫秒级事件
                max_downsample = min(max_downsample, int(source_sampling_rate / 1000))
                logger.debug(
                    f"毫秒级精度模式启用，每毫秒至少保留{1/max_downsample:.2f}个数据点"
                )
            else:
                # 非毫秒精度模式，可以更激进地降采样
                max_downsample = max(1, int(points_per_ms / 10))  # 每10ms至少一个点

            # 根据显示窗口大小和屏幕分辨率优化数据点数量
            # 假设屏幕宽度为1000像素，每像素最多显示一个数据点
            target_points = min(
                2000, int(duration * 1000)
            )  # 每秒最多1000个点，确保毫秒级分辨率

            if len(display_times) > target_points:
                # 计算最佳降采样因子，平衡性能和精度
                downsample_factor = min(
                    len(display_times) // target_points + 1, max_downsample
                )
                display_data = display_data[:, ::downsample_factor]
                display_times = display_times[::downsample_factor]
                logger.debug(
                    f"应用降采样，因子: {downsample_factor}，保留{1000/downsample_factor:.1f}点/秒，确保{ms_resolution}ms时间分辨率"
                )

            # 过滤选择的通道
            if (
                selected_channels is not None
                and len(selected_channels) > 0
                and channel_names is not None
            ):
                # 确保channel_names不为None再使用enumerate
                selected_indices = [
                    i for i, ch in enumerate(channel_names) if ch in selected_channels
                ]
                if selected_indices:  # 确保有选择的通道
                    display_data = display_data[selected_indices]
                    filtered_channel_names = [
                        channel_names[i] for i in selected_indices
                    ]
                else:
                    # 如果没有匹配的通道，使用所有通道
                    filtered_channel_names = channel_names
            else:
                filtered_channel_names = channel_names

            # 检测伪迹
            artifacts = None
            if show_artifacts and display_data is not None:
                artifacts = self._detect_artifacts(display_data)

            # 计算电压缩放
            if voltage_scale is None:
                # 自动缩放，使用数据的95%分位数
                if display_data is not None and display_data.size > 0:
                    v_min = np.percentile(display_data, 2.5)
                    v_max = np.percentile(display_data, 97.5)
                    # 确保有一定的范围，避免平直线
                    if v_max - v_min < 10:
                        v_center = (v_max + v_min) / 2
                        v_min = v_center - 5
                        v_max = v_center + 5
                else:
                    v_min, v_max = -100, 100  # 默认范围
            else:
                v_min, v_max = voltage_scale

            # 计算通道间距
            n_channels_to_plot = (
                len(filtered_channel_names)
                if filtered_channel_names
                else display_data.shape[0]
            )
            channel_spacing = (v_max - v_min) * 1.2  # 通道间距比电压范围大20%

            # 绘制EEG波形
            if display_data is not None and display_data.size > 0:
                for i in range(display_data.shape[0]):
                    # 获取通道颜色
                    if filtered_channel_names and i < len(filtered_channel_names):
                        channel_color = self._get_channel_color(
                            filtered_channel_names[i]
                        )
                    else:
                        channel_color = "gray"

                    # 计算通道偏移
                    offset = -i * channel_spacing

                    # 绘制波形
                    ax.plot(
                        display_times,
                        display_data[i] + offset,
                        color=channel_color,
                        linewidth=0.8,
                    )

                    # 如果启用了伪迹检测，标记伪迹
                    if artifacts is not None and i < artifacts.shape[0]:
                        artifact_times = display_times[artifacts[i]]
                        artifact_values = display_data[i, artifacts[i]] + offset
                        if len(artifact_times) > 0:
                            ax.scatter(
                                artifact_times,
                                artifact_values,
                                color="red",
                                s=10,
                                marker="x",
                                alpha=0.7,
                            )

                    # 添加通道标签
                    if filtered_channel_names and i < len(filtered_channel_names):
                        ax.text(
                            display_times[0] - 0.1,
                            offset,
                            filtered_channel_names[i],
                            ha="right",
                            va="center",
                            fontsize=8,
                            color=channel_color,
                        )

            # 添加时间标记
            if time_markers:
                for marker in time_markers:
                    marker_time = marker.get("time", 0)
                    marker_label = marker.get("label", "")
                    marker_color = marker.get("color", "red")

                    if start_time <= marker_time <= start_time + duration:
                        ax.axvline(
                            x=marker_time, color=marker_color, linestyle="--", alpha=0.7
                        )
                        ax.text(
                            marker_time,
                            v_max,
                            marker_label,
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color=marker_color,
                            bbox=dict(
                                facecolor="white", alpha=0.7, edgecolor="none", pad=1
                            ),
                        )

            # 设置坐标轴
            ax.set_xlim(display_times[0], display_times[-1])
            ax.set_ylim(-(n_channels_to_plot - 1) * channel_spacing - v_max, v_max)

            # 设置标签
            ax.set_xlabel("时间 (秒)")
            ax.set_ylabel("电压 (μV)")

            # 设置标题，包含数据源信息
            ax.set_title(f"EEG时域波形 - 数据源: {display_source_id}")

            # 添加电压刻度参考线
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

            # 优化图形外观
            self._optimize_figure_for_web(fig)

            # 转换为base64
            img_base64 = self._figure_to_base64(fig)

            # 缓存结果
            self._cache_result(cache_key, img_base64)

            # 记录处理时间
            process_time = time.time() - start_time_perf
            logger.debug(f"时域绘图完成，处理时间: {process_time:.3f}秒")

            return img_base64

        except Exception as e:
            logger.error(f"时域绘图出错: {e}", exc_info=True)
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
