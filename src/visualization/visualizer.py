import os
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import time

# 导入专门的可视化器
from .base_visualizer import BaseVisualizer
from .time_visualizer import TimeVisualizer
from .frequency_visualizer import FrequencyVisualizer
from .topographic_visualizer import TopographicVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# logger = logging.getLogger("EEGVisualizer")


class EEGVisualizer:
    """EEG数据可视化器 - 支持时域、频域和空间域（拓扑图）可视化，具有多数据源支持

    此类整合了专门的可视化器类，提供统一的接口。
    """

    def __init__(self, sampling_rate: float = 5000.0, reference: str = "FCz"):
        """初始化可视化器

        Args:
            sampling_rate: 采样率 (Hz)
            reference: 参考电极
        """
        # 初始化专门的可视化器
        self.base_visualizer = BaseVisualizer(sampling_rate, reference)
        self.time_visualizer = TimeVisualizer(sampling_rate, reference)
        self.frequency_visualizer = FrequencyVisualizer(sampling_rate, reference)
        self.topographic_visualizer = TopographicVisualizer(sampling_rate, reference)

        # 共享设置
        self.sampling_rate = sampling_rate
        self.reference = reference

  

    # 数据源管理方法 - 委托给基础可视化器

    def add_data_source(
        self,
        source_id: str,
        data: np.ndarray,
        channel_names: List[str],
        sampling_rate: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """添加新的数据源到可视化器

        Args:
            source_id: 数据源的唯一标识符
            data: 数据数组，形状为 (n_channels, n_samples)
            channel_names: 通道名称列表
            sampling_rate: 此数据源的采样率（如果与默认值不同）
            metadata: 数据源的附加元数据
        """
        # 添加到所有可视化器
        self.base_visualizer.add_data_source(
            source_id, data, channel_names, sampling_rate, metadata
        )
        self.time_visualizer.add_data_source(
            source_id, data, channel_names, sampling_rate, metadata
        )
        self.frequency_visualizer.add_data_source(
            source_id, data, channel_names, sampling_rate, metadata
        )
        self.topographic_visualizer.add_data_source(
            source_id, data, channel_names, sampling_rate, metadata
        )

    def remove_data_source(self, source_id: str) -> bool:
        """移除数据源

        Args:
            source_id: 要移除的数据源标识符

        Returns:
            bool: 如果成功移除则为True，否则为False
        """
        # 从所有可视化器中移除
        result = self.base_visualizer.remove_data_source(source_id)
        self.time_visualizer.remove_data_source(source_id)
        self.frequency_visualizer.remove_data_source(source_id)
        self.topographic_visualizer.remove_data_source(source_id)
        return result

    def set_active_source(self, source_id: str) -> bool:
        """设置活动数据源

        Args:
            source_id: 要设置为活动的数据源标识符

        Returns:
            bool: 如果成功设置则为True，否则为False
        """
        # 在所有可视化器中设置
        result = self.base_visualizer.set_active_source(source_id)
        self.time_visualizer.set_active_source(source_id)
        self.frequency_visualizer.set_active_source(source_id)
        self.topographic_visualizer.set_active_source(source_id)
        return result

    def get_data_source(self, source_id: Optional[str] = None) -> Optional[Dict]:
        """获取数据源

        Args:
            source_id: 要获取的数据源标识符，如果为None，则返回活动数据源

        Returns:
            Optional[Dict]: 数据源信息，如果未找到则为None
        """
        return self.base_visualizer.get_data_source(source_id)

    def get_data_source_ids(self) -> List[str]:
        """获取所有数据源标识符

        Returns:
            List[str]: 数据源标识符列表
        """
        return self.base_visualizer.get_data_source_ids()
        
    def update_visualization(self, source_id: str) -> None:
        """更新可视化并触发WebSocket数据更新
        
        Args:
            source_id: 要更新的数据源ID
        """
        # 调用各子可视化器的更新方法
        self.time_visualizer.update_visualization(source_id)
        self.frequency_visualizer.update_visualization(source_id)
        self.topographic_visualizer.update_visualization(source_id)
        
        # 获取数据源信息
        data_source = self.get_data_source(source_id)
        if data_source:
            # 触发WebSocket数据更新事件
            data = {
                'source_id': source_id,
                'data': data_source['data'].tolist(),
                'channel_names': data_source['channel_names'],
                'sampling_rate': data_source['sampling_rate'],
                'timestamp': time.time()
            }
            self.socketio.emit('data_update', data)

    # 可视化方法 - 委托给专门的可视化器

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
        show_ms_precision: bool = True,
        show_artifacts: bool = True,
        show_erp_average: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        source_id: Optional[str] = None,
    ) -> str:
        """绘制时域波形

        委托给TimeVisualizer处理
        """
        return self.time_visualizer.plot_time_domain(
            data,
            channel_names,
            start_time,
            duration,
            baseline_correction,
            baseline_window,
            voltage_scale,
            time_markers,
            sliding_step,
            selected_channels,
            show_ms_precision,
            show_artifacts,
            show_erp_average,
            figsize,
            source_id,
        )

    def plot_frequency_domain(
        self,
        data: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        freq_range: Tuple[float, float] = (0, 50),
        selected_channels: Optional[List[str]] = None,
        show_bands: bool = True,
        method: str = "welch",
        log_scale: bool = False,
        normalize: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        source_id: Optional[str] = None,
    ) -> str:
        """绘制频域图

        委托给FrequencyVisualizer处理
        """
        return self.frequency_visualizer.plot_frequency_domain(
            data,
            channel_names,
            freq_range,
            selected_channels,
            show_bands,
            method,
            log_scale,
            normalize,
            figsize,
            source_id,
        )

    def plot_topographic_map(
        self,
        data: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        time_window: Optional[Tuple[float, float]] = None,
        freq_band: Optional[str] = None,
        selected_time: Optional[float] = None,
        interpolation: str = "cubic",
        contours: int = 6,
        colormap: str = "RdBu_r",
        show_names: bool = True,
        show_sensors: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        source_id: Optional[str] = None,
    ) -> str:
        """绘制EEG拓扑图

        委托给TopographicVisualizer处理
        """
        return self.topographic_visualizer.plot_topographic_map(
            data,
            channel_names,
            time_window,
            freq_band,
            selected_time,
            interpolation,
            contours,
            colormap,
            show_names,
            show_sensors,
            figsize,
            source_id,
        )

    def plot_impedance_map(
        self,
        impedance_values: np.ndarray,
        channel_names: List[str],
        figsize: Tuple[int, int] = (10, 8),
    ) -> str:
        """绘制电极阻抗拓扑图

        委托给TopographicVisualizer处理
        """
        return self.topographic_visualizer.plot_impedance_map(
            impedance_values, channel_names, figsize
        )

    def plot_combined_visualization(
        self,
        data: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        start_time: float = 0,
        duration: float = 5.0,
        selected_channels: Optional[List[str]] = None,
        freq_range: Tuple[float, float] = (0, 50),
        show_topography: bool = True,
        figsize: Tuple[int, int] = (15, 10),
        source_id: Optional[str] = None,
    ) -> str:
        """创建组合可视化，包括时域、频域和拓扑图

        Args:
            data: 形状为(n_channels, n_samples)的数据数组，如果为None，则使用source_id的数据
            channel_names: 通道名称列表，如果为None，则使用source_id的通道名称
            start_time: 起始时间（秒）
            duration: 显示持续时间（秒）
            selected_channels: 要显示的通道列表，如果为None，则显示所有通道
            freq_range: 频率范围 (Hz)
            show_topography: 是否显示拓扑图
            figsize: 图形大小
            source_id: 数据源标识符，如果为None，则使用活动数据源

        Returns:
            str: Base64编码的图像
        """
        # 获取数据源
        if data is None or channel_names is None:
            source_data = self.get_data_source(source_id)
            if source_data is None:
              
                return ""

            if data is None:
                data = source_data["data"]
            if channel_names is None:
                channel_names = source_data["channel_names"]

        # 生成缓存键
        cache_key = self.base_visualizer._generate_cache_key(
            "combined_visualization",
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
            start_time=start_time,
            duration=duration,
            selected_channels=tuple(selected_channels) if selected_channels else None,
            freq_range=freq_range,
            show_topography=show_topography,
            figsize=figsize,
            source_id=source_id or self.base_visualizer.active_source,
        )

        # 检查缓存
        cached_result = self.base_visualizer._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # 创建组合图像
            import matplotlib
            from matplotlib.figure import Figure
            import matplotlib.gridspec as gridspec

            fig = Figure(figsize=figsize, dpi=80)
            canvas = fig.canvas

            # 创建网格布局
            if show_topography:
                # 3个面板：时域、频域、拓扑图
                gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
                ax_time = fig.add_subplot(gs[0, :])
                ax_freq = fig.add_subplot(gs[1, 0])
                ax_topo = fig.add_subplot(gs[1, 1])
            else:
                # 2个面板：时域、频域
                gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1])
                ax_time = fig.add_subplot(gs[0])
                ax_freq = fig.add_subplot(gs[1])

            # 获取时域图
            time_img = self.plot_time_domain(
                data,
                channel_names,
                start_time,
                duration,
                selected_channels=selected_channels,
                figsize=(figsize[0], figsize[1] * 0.6),
                source_id=source_id,
            )

            # 获取频域图
            freq_img = self.plot_frequency_domain(
                data,
                channel_names,
                freq_range,
                selected_channels=selected_channels,
                figsize=(figsize[0] * 0.5, figsize[1] * 0.3),
                source_id=source_id,
            )

            # 如果需要，获取拓扑图
            if show_topography:
                # 使用中点时间创建拓扑图
                mid_time = start_time + duration / 2
                topo_img = self.plot_topographic_map(
                    data,
                    channel_names,
                    selected_time=mid_time,
                    figsize=(figsize[0] * 0.5, figsize[1] * 0.3),
                    source_id=source_id,
                )

            # 将单独的图像组合成一个图像
            # 注意：这里我们不直接使用生成的图像，而是重新绘制组合图像
            # 这是因为直接组合base64图像比较复杂，而且可能导致图像质量下降

            # 设置标题
            fig.suptitle(
                f'EEG组合可视化 - 数据源: {source_id or self.base_visualizer.active_source or "未知"}',
                fontsize=14,
            )

            # 优化图形外观
            self.base_visualizer._optimize_figure_for_web(fig)

            # 转换为base64
            img_base64 = self.base_visualizer._figure_to_base64(fig)

            # 缓存结果
            self.base_visualizer._cache_result(cache_key, img_base64)

            return img_base64

        except Exception as e:
           
            # 创建错误图像
            from matplotlib.figure import Figure

            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5, f"绘图错误: {str(e)}", ha="center", va="center", fontsize=12
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return self.base_visualizer._figure_to_base64(fig)

    # 辅助方法

    def calculate_band_powers(
        self, data: np.ndarray, sampling_rate: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """计算各频带的功率

        委托给FrequencyVisualizer处理

        Args:
            data: 形状为(n_channels, n_samples)的EEG数据
            sampling_rate: 采样率 (Hz)，如果为None，则使用默认采样率

        Returns:
            Dict[str, np.ndarray]: 各频带的功率，键为频带名称，值为形状为(n_channels,)的功率数组
        """
        return self.frequency_visualizer.calculate_band_powers(
            data, sampling_rate or self.sampling_rate
        )
