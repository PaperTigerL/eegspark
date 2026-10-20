import numpy as np
import matplotlib
from matplotlib.figure import Figure
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import mne
from mne.viz import plot_topomap

from .base_visualizer import BaseVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TopographicVisualizer')

class TopographicVisualizer(BaseVisualizer):
    """EEG拓扑图可视化类 - 专门处理空间域拓扑图的可视化"""
    
    def __init__(self, sampling_rate: float = 5000.0, reference: str = 'FCz'):
        """初始化拓扑图可视化器
        
        Args:
            sampling_rate: 采样率 (Hz)
            reference: 参考电极
        """
        super().__init__(sampling_rate, reference)
    
    def plot_topographic_map(self, data: Optional[np.ndarray] = None, channel_names: Optional[List[str]] = None,
                            time_window: Optional[Tuple[float, float]] = None,
                            freq_band: Optional[str] = None,
                            selected_time: Optional[float] = None,
                            interpolation: str = 'cubic',
                            contours: int = 6,
                            colormap: str = 'RdBu_r',
                            show_names: bool = True,
                            show_sensors: bool = True,
                            figsize: Tuple[int, int] = (10, 8),
                            source_id: Optional[str] = None) -> str:
        """绘制EEG拓扑图
        
        Args:
            data: 形状为(n_channels, n_samples)的数据数组，如果为None，则使用source_id的数据
            channel_names: 通道名称列表，如果为None，则使用source_id的通道名称
            time_window: 时间窗口 (秒)，用于平均数据
            freq_band: 频带名称，如果提供，则绘制该频带的功率拓扑图
            selected_time: 选定的时间点 (秒)，如果提供，则绘制该时间点的拓扑图
            interpolation: 插值方法
            contours: 等高线数量
            colormap: 颜色映射
            show_names: 是否显示通道名称
            show_sensors: 是否显示传感器位置
            figsize: 图形大小
            source_id: 数据源标识符，如果为None，则使用活动数据源
            
        Returns:
            str: Base64编码的图像
        """
        # 如果没有提供数据和通道名称，从数据源获取
        if data is None or channel_names is None:
            source_data = self.get_data_source(source_id)
            if source_data is None:
                logger.error(f"没有可用于拓扑图的数据源")
                return ""
                
            if data is None:
                data = source_data['data']
            if channel_names is None:
                channel_names = source_data['channel_names']
            
            # 使用数据源的采样率
            source_sampling_rate = source_data.get('sampling_rate', self.sampling_rate)
        else:
            source_sampling_rate = self.sampling_rate
            source_data = None
        
        # 获取数据源ID用于显示
        display_source_id = source_id or self.active_source or "未知数据源"
        
        # 生成缓存键
        cache_key = self._generate_cache_key(
            'topographic_map',
            data_shape=data.shape if data is not None else None,
            data_hash=(hash(data.tobytes()) if data.size < 10000 else hash(data[:, ::10].tobytes())) if data is not None else None,
            channel_names=tuple(channel_names) if channel_names is not None else None,
            time_window=time_window,
            freq_band=freq_band,
            selected_time=selected_time,
            interpolation=interpolation,
            contours=contours,
            colormap=colormap,
            show_names=show_names,
            show_sensors=show_sensors,
            figsize=figsize,
            source_id=display_source_id
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
            
            # 创建图像
            fig = Figure(figsize=figsize, dpi=80)
            canvas = fig.canvas
            
            # 准备拓扑图数据
            if freq_band is not None and freq_band in self.freq_bands:
                # 频带功率拓扑图
                from .frequency_visualizer import FrequencyVisualizer
                freq_viz = FrequencyVisualizer(source_sampling_rate, self.reference)
                band_powers = freq_viz.calculate_band_powers(data_copy, source_sampling_rate)
                
                if freq_band in band_powers:
                    topo_data = band_powers[freq_band]
                    title = f'{freq_band.capitalize()} 频带功率拓扑图 ({self.freq_bands[freq_band][0]}-{self.freq_bands[freq_band][1]} Hz)'
                else:
                    logger.warning(f"未找到频带 {freq_band} 的功率数据")
                    topo_data = np.zeros(data_copy.shape[0])
                    title = f'频带功率拓扑图 (无数据)'
            elif selected_time is not None:
                # 特定时间点的拓扑图
                time_idx = int(selected_time * source_sampling_rate)
                if 0 <= time_idx < data_copy.shape[1]:
                    topo_data = data_copy[:, time_idx]
                    title = f'时间点 {selected_time:.3f}s 的电压拓扑图'
                else:
                    logger.warning(f"时间点 {selected_time}s 超出数据范围")
                    topo_data = np.zeros(data_copy.shape[0])
                    title = f'时间点拓扑图 (无数据)'
            elif time_window is not None:
                # 时间窗口平均的拓扑图
                start_idx = int(time_window[0] * source_sampling_rate)
                end_idx = int(time_window[1] * source_sampling_rate)
                
                # 确保索引在有效范围内
                start_idx = max(0, start_idx)
                end_idx = min(data_copy.shape[1], end_idx)
                
                if start_idx < end_idx:
                    topo_data = np.mean(data_copy[:, start_idx:end_idx], axis=1)
                    title = f'时间窗口 {time_window[0]:.3f}s - {time_window[1]:.3f}s 的平均电压拓扑图'
                else:
                    logger.warning(f"时间窗口 {time_window} 无效或超出数据范围")
                    topo_data = np.zeros(data_copy.shape[0])
                    title = f'时间窗口拓扑图 (无数据)'
            else:
                # 默认使用全部数据的平均值
                topo_data = np.mean(data_copy, axis=1)
                title = f'全数据平均电压拓扑图'
            
            # 获取电极位置
            if self.montage is None:
                logger.warning("未加载标准电极位置，尝试重新加载")
                self._load_standard_montage()
            
            if self.montage is not None:
                # 创建MNE info对象
                info = mne.create_info(ch_names=channel_names, sfreq=source_sampling_rate, ch_types=['eeg'] * len(channel_names))
                
                # 设置蒙太奇
                try:
                    # 尝试设置蒙太奇
                    info.set_montage(self.montage)
                    
                    # 获取电极位置
                    pos = []
                    for ch_name in channel_names:
                        if ch_name in info.ch_names:
                            ch_idx = info.ch_names.index(ch_name)
                            pos.append(info['chs'][ch_idx]['loc'][:2])  # 只取前两个坐标 (x, y)
                        else:
                            # 如果通道名称不在蒙太奇中，使用默认位置
                            pos.append([0, 0])
                    
                    pos = np.array(pos)
                    
                    # 绘制拓扑图
                    ax = fig.add_subplot(111)
                    im, cn = plot_topomap(topo_data, pos, axes=ax, show=False,
                                        cmap=colormap, contours=contours,
                                        interpolation=interpolation,
                                        show_names=show_names if show_names else None,
                                        names=channel_names if show_names else None,
                                        show_sensors=show_sensors)
                    
                    # 添加颜色条
                    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                    if freq_band is not None:
                        cbar.set_label('功率 (μV²/Hz)')
                    else:
                        cbar.set_label('电压 (μV)')
                    
                    # 设置标题
                    ax.set_title(f'{title}\n数据源: {display_source_id}')
                    
                except Exception as e:
                    logger.error(f"设置蒙太奇或绘制拓扑图失败: {e}")
                    # 创建错误信息
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"拓扑图绘制失败: {str(e)}\n请确保通道名称与标准10-20系统兼容", 
                            ha='center', va='center', fontsize=12, wrap=True)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
            else:
                # 如果没有蒙太奇，创建错误信息
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "无法加载标准电极位置\n请确保MNE库正确安装", 
                        ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            # 优化图形外观
            self._optimize_figure_for_web(fig)
            
            # 转换为base64
            img_base64 = self._figure_to_base64(fig)
            
            # 缓存结果
            self._cache_result(cache_key, img_base64)
            
            # 记录处理时间
            process_time = time.time() - start_time_perf
            logger.debug(f"拓扑图绘制完成，处理时间: {process_time:.3f}秒")
            
            return img_base64
            
        except Exception as e:
            logger.error(f"拓扑图绘制出错: {e}", exc_info=True)
            # 创建错误图像
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"绘图错误: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return self._figure_to_base64(fig)
    
    def plot_impedance_map(self, impedance_values: np.ndarray, channel_names: List[str],
                          figsize: Tuple[int, int] = (10, 8)) -> str:
        """绘制电极阻抗拓扑图
        
        Args:
            impedance_values: 阻抗值数组 (kΩ)
            channel_names: 通道名称列表
            figsize: 图形大小
            
        Returns:
            str: Base64编码的图像
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(
            'impedance_map',
            impedance_values=tuple(impedance_values),
            channel_names=tuple(channel_names),
            figsize=figsize
        )
        
        # 检查缓存
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # 创建图像
            fig = Figure(figsize=figsize, dpi=80)
            canvas = fig.canvas
            
            # 获取电极位置
            if self.montage is None:
                logger.warning("未加载标准电极位置，尝试重新加载")
                self._load_standard_montage()
            
            if self.montage is not None:
                # 创建MNE info对象
                info = mne.create_info(ch_names=channel_names, sfreq=self.sampling_rate, ch_types=['eeg'] * len(channel_names))
                
                # 设置蒙太奇
                try:
                    # 尝试设置蒙太奇
                    info.set_montage(self.montage)
                    
                    # 获取电极位置
                    pos = []
                    for ch_name in channel_names:
                        if ch_name in info.ch_names:
                            ch_idx = info.ch_names.index(ch_name)
                            pos.append(info['chs'][ch_idx]['loc'][:2])  # 只取前两个坐标 (x, y)
                        else:
                            # 如果通道名称不在蒙太奇中，使用默认位置
                            pos.append([0, 0])
                    
                    pos = np.array(pos)
                    
                    # 创建颜色映射函数，根据阻抗值分配颜色
                    def get_impedance_colors(values):
                        colors = np.zeros((len(values), 4))  # RGBA颜色
                        for i, val in enumerate(values):
                            if val <= self.impedance_thresholds['excellent']:
                                colors[i] = [0, 0.8, 0, 1]  # 绿色 - 优秀
                            elif val <= self.impedance_thresholds['good']:
                                colors[i] = [0.9, 0.9, 0, 1]  # 黄色 - 良好
                            elif val <= self.impedance_thresholds['acceptable']:
                                colors[i] = [1, 0.5, 0, 1]  # 橙色 - 可接受
                            else:
                                colors[i] = [0.8, 0, 0, 1]  # 红色 - 较差
                        return colors
                    
                    # 获取颜色
                    sensor_colors = get_impedance_colors(impedance_values)
                    
                    # 绘制拓扑图
                    ax = fig.add_subplot(111)
                    im, cn = plot_topomap(impedance_values, pos, axes=ax, show=False,
                                        cmap='RdYlGn_r', contours=0,  # 使用红黄绿色映射，反转以使低阻抗为绿色
                                        interpolation='cubic',
                                        show_names=True, names=channel_names,
                                        show_sensors=True, sensor_colors=sensor_colors)
                    
                    # 添加颜色条
                    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('阻抗 (kΩ)')
                    
                    # 添加阻抗阈值标记
                    for threshold_name, threshold_value in sorted(self.impedance_thresholds.items(), key=lambda x: x[1]):
                        if threshold_name != 'poor':  # 跳过'poor'，因为它是无穷大
                            cbar.ax.axhline(y=threshold_value, color='black', linestyle='--', linewidth=0.5)
                            cbar.ax.text(1.5, threshold_value, f'{threshold_name} ({threshold_value} kΩ)', 
                                        va='center', ha='left', fontsize=8, transform=cbar.ax.transData)
                    
                    # 设置标题
                    ax.set_title('电极阻抗拓扑图')
                    
                except Exception as e:
                    logger.error(f"设置蒙太奇或绘制阻抗图失败: {e}")
                    # 创建错误信息
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f"阻抗图绘制失败: {str(e)}\n请确保通道名称与标准10-20系统兼容", 
                            ha='center', va='center', fontsize=12, wrap=True)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
            else:
                # 如果没有蒙太奇，创建错误信息
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "无法加载标准电极位置\n请确保MNE库正确安装", 
                        ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            # 优化图形外观
            self._optimize_figure_for_web(fig)
            
            # 转换为base64
            img_base64 = self._figure_to_base64(fig)
            
            # 缓存结果
            self._cache_result(cache_key, img_base64)
            
            return img_base64
            
        except Exception as e:
            logger.error(f"阻抗图绘制出错: {e}", exc_info=True)
            # 创建错误图像
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"绘图错误: {str(e)}", ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return self._figure_to_base64(fig)