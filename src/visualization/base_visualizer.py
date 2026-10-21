import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import mne
import hashlib
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# logger = logging.getLogger("EEGVisualizerBase")


class BaseVisualizer:
    """EEG可视化基础类 - 提供共享功能和数据管理"""

    def __init__(self, sampling_rate: float = 5000.0, reference: str = "FCz"):
        """初始化可视化器基础类

        Args:
            sampling_rate: 采样率 (Hz)
            reference: 参考电极
        """
        self.sampling_rate = sampling_rate
        self.reference = reference
        self.montage = None
        self._load_standard_montage()

        # 添加多数据源支持
        self.data_sources = {}
        self.active_source = None

        # 添加图像缓存字典
        self.image_cache = {}
        self.cache_size_limit = 50  # 最大缓存数量
        self.cache_ttl = 300  # 缓存有效期（秒）

        # 设置电极阻抗阈值（kΩ）
        self.impedance_thresholds = {
            "excellent": 5,  # <5kΩ 为优秀（绿色）
            "good": 20,  # <20kΩ 为良好（黄色）
            "acceptable": 50,  # <50kΩ 为可接受（橙色）
            "poor": float("inf"),  # >50kΩ 为较差（红色）
        }

        # 伪迹检测阈值
        self.artifact_thresholds = {
            "amplitude": 100,  # 微伏，超过此值视为幅值异常
            "gradient": 50,  # 微伏/样本，超过此值视为梯度异常
            "flatline": 0.5,  # 微伏，低于此值视为平直线
            "line_noise": 10,  # 工频噪声阈值（dB）
        }

        # 设置字体支持国际字符
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
        matplotlib.rcParams["axes.unicode_minus"] = False

        # 设置颜色映射
        self.cmap = "RdBu_r"  # 红蓝色映射，适合EEG数据

        # 设置频带定义 - 按国际标准频段
        self.freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

        # 设置频带颜色 - 统一颜色方案
        self.band_colors = {
            "delta": "#EBEBD3",  # 浅灰色
            "theta": "#F4D35E",  # 黄色
            "alpha": "#EE964B",  # 橙色
            "beta": "#0D3B66",  # 深蓝色
            "gamma": "#F95738",  # 红色
        }

        # 设置通道分组 - 基于10-20系统
        self.channel_groups = {
            "frontal": [
                "Fp1",
                "Fp2",
                "F3",
                "F4",
                "F7",
                "F8",
                "Fz",
                "Fpz",
                "AFz",
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "F1",
                "F2",
                "F5",
                "F6",
            ],
            "central": [
                "C3",
                "C4",
                "Cz",
                "C1",
                "C2",
                "C5",
                "C6",
                "FCz",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
            ],
            "temporal": [
                "T3",
                "T4",
                "T5",
                "T6",
                "T7",
                "T8",
                "TP7",
                "TP8",
                "FT7",
                "FT8",
                "FT9",
                "FT10",
            ],
            "parietal": [
                "P3",
                "P4",
                "Pz",
                "P1",
                "P2",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "P10",
                "CPz",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
            ],
            "occipital": ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "POz"],
        }

        # 通道颜色映射
        self.channel_colors = {
            "frontal": "blue",
            "central": "green",
            "temporal": "purple",
            "parietal": "orange",
            "occipital": "red",
        }

        # 用户自定义配置
        self.user_config = {
            "display_channels": [],  # 默认显示所有通道
            "color_map": "RdBu_r",  # 默认颜色映射
            "voltage_scale": (-100, 100),  # 默认电压范围（微伏）
            "time_window": 5.0,  # 默认时间窗口（秒）
            "sliding_step": 1.0,  # 默认滑动步长（秒）
            "freq_range": (0, 50),  # 默认频率范围（Hz）
            "selected_bands": ["alpha"],  # 默认选择的频带
        }

    def _load_standard_montage(self):
        """加载标准10-20系统电极位置"""
        try:
            self.montage = mne.channels.make_standard_montage("standard_1020")
          
        except Exception as e:
            # 如果加载失败，记录错误并使用默认montage
            self.montage = mne.channels.make_standard_montage("biosemi64")

    def _get_channel_group(self, channel_name: str) -> str:
        """获取通道所属的组"""
        for group, channels in self.channel_groups.items():
            if channel_name in channels:
                return group
        return "other"

    def _get_channel_color(self, channel_name: str) -> str:
        """获取通道的颜色"""
        group = self._get_channel_group(channel_name)
        return self.channel_colors.get(group, "gray")

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
        self.data_sources[source_id] = {
            "data": data,
            "channel_names": channel_names,
            "sampling_rate": sampling_rate or self.sampling_rate,
            "metadata": metadata or {},
        }

        # 如果这是第一个数据源，将其设为活动数据源
        if self.active_source is None:
            self.active_source = source_id


    def remove_data_source(self, source_id: str) -> bool:
        """移除数据源

        Args:
            source_id: 要移除的数据源标识符

        Returns:
            bool: 如果成功移除则为True，否则为False
        """
        if source_id in self.data_sources:
            del self.data_sources[source_id]

            # 如果删除的是当前活动数据源，重置活动数据源
            if self.active_source == source_id:
                self.active_source = (
                    next(iter(self.data_sources)) if self.data_sources else None
                )

         
            return True
        else:
         
            return False

    def set_active_source(self, source_id: str) -> bool:
        """设置活动数据源

        Args:
            source_id: 要设置为活动的数据源标识符

        Returns:
            bool: 如果成功设置则为True，否则为False
        """
        if source_id in self.data_sources:
            self.active_source = source_id
            return True
        else:
            return False

    def get_data_source(self, source_id: Optional[str] = None) -> Optional[Dict]:
        """获取数据源

        Args:
            source_id: 要获取的数据源标识符，如果为None，则返回活动数据源

        Returns:
            Optional[Dict]: 数据源信息，如果未找到则为None
        """
        if source_id is None:
            source_id = self.active_source

        if source_id is None:
            return None

        if source_id in self.data_sources:
            return self.data_sources[source_id]
        else:
            return None

    def get_data_source_ids(self) -> List[str]:
        """获取所有数据源标识符

        Returns:
            List[str]: 数据源标识符列表
        """
        return list(self.data_sources.keys())

    def _generate_cache_key(self, func_name: str, **kwargs) -> str:
        """根据函数名和参数生成唯一的缓存键

        Args:
            func_name: 可视化函数的名称
            **kwargs: 用于可视化的参数

        Returns:
            str: 用于缓存的唯一哈希键
        """
        # 创建一个包含所有参数的字典
        key_dict = {"func": func_name}

        # 处理kwargs，确保所有值都是可哈希的
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                # 对于numpy数组，使用其形状和数据的哈希值作为标识
                try:
                    # 使用数组的形状和前1000个元素的哈希值作为标识
                    sample = v.flatten()[:1000] if v.size > 1000 else v.flatten()
                    key_dict[k] = f"array_shape_{v.shape}_hash_{hash(sample.tobytes())}"
                except Exception as e:
                    # 如果哈希失败，使用数组的形状和大小作为标识
                    
                    key_dict[k] = f"array_shape_{v.shape}_size_{v.size}"
            else:
                # 确保其他值也是可哈希的
                try:
                    # 尝试将值转换为字符串
                    key_dict[k] = str(v)
                except Exception:
                    # 如果转换失败，使用类型名称
                    key_dict[k] = f"type_{type(v).__name__}"

        # 将字典转换为字符串并生成哈希值
        try:
            key_str = str(sorted([(k, key_dict[k]) for k in key_dict]))
            return hashlib.md5(key_str.encode()).hexdigest()
        except Exception as e:
            # 如果排序或哈希失败，使用简单的时间戳作为缓存键
           
            return hashlib.md5(f"{func_name}_{time.time()}".encode()).hexdigest()

    def _cache_result(self, key: str, result: str) -> None:
        """缓存可视化结果

        Args:
            key: 缓存键
            result: Base64编码的图像
        """
        # 添加到缓存，记录时间戳
        self.image_cache[key] = {"result": result, "timestamp": time.time()}

        # 如果缓存超过大小限制，删除最旧的项
        if len(self.image_cache) > self.cache_size_limit:
            oldest_key = min(
                self.image_cache.keys(), key=lambda k: self.image_cache[k]["timestamp"]
            )
            del self.image_cache[oldest_key]

    def _get_cached_result(self, key: str) -> Optional[str]:
        """获取缓存的结果（如果可用且未过期）

        Args:
            key: 缓存键

        Returns:
            Optional[str]: 缓存的结果，如果未找到或已过期则为None
        """
        if key in self.image_cache:
            cache_entry = self.image_cache[key]
            # 检查缓存是否过期
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
   
                return cache_entry["result"]
            else:
                # 删除过期缓存
                del self.image_cache[key]
        return None

    def _optimize_figure_for_web(self, fig: Figure) -> None:
        """优化matplotlib图形以便于Web显示

        Args:
            fig: 要优化的Matplotlib图形
        """
        # 减少DPI以减小文件大小，但保持足够的清晰度
        fig.dpi = 80

        # 优化图形布局
        fig.tight_layout()

        # 减少字体大小以节省空间
        for ax in fig.get_axes():
            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                # 确保get_fontsize()返回的是float类型
                fontsize = item.get_fontsize()
                if isinstance(fontsize, (int, float)):
                    item.set_fontsize(fontsize * 0.9)

    def _figure_to_base64(self, fig: Figure) -> str:
        """将matplotlib图形转换为base64编码的字符串

        Args:
            fig: Matplotlib图形

        Returns:
            str: Base64编码的图像
        """
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)
        return img_base64

    def _detect_artifacts(self, data: np.ndarray) -> np.ndarray:
        """检测EEG数据中的伪迹

        Args:
            data: 形状为(n_channels, n_samples)的EEG数据

        Returns:
            np.ndarray: 形状为(n_channels, n_samples)的布尔掩码，True表示伪迹
        """
        n_channels, n_samples = data.shape
        artifacts = np.zeros((n_channels, n_samples), dtype=bool)

        # 幅值异常检测
        amplitude_mask = np.abs(data) > self.artifact_thresholds["amplitude"]
        artifacts = artifacts | amplitude_mask

        # 梯度异常检测（快速变化）
        gradient = np.abs(np.diff(data, axis=1))
        gradient_mask = np.zeros_like(artifacts)
        gradient_mask[:, 1:] = gradient > self.artifact_thresholds["gradient"]
        artifacts = artifacts | gradient_mask

        # 平直线检测（信号变化极小）
        flatline_mask = np.zeros_like(artifacts)
        for i in range(n_channels):
            # 使用滑动窗口检测平直线
            window_size = min(100, n_samples // 10)  # 窗口大小，至少10个窗口
            for j in range(0, n_samples - window_size, window_size // 2):
                window = data[i, j : j + window_size]
                if (
                    np.max(window) - np.min(window)
                    < self.artifact_thresholds["flatline"]
                ):
                    flatline_mask[i, j : j + window_size] = True

        artifacts = artifacts | flatline_mask

        return artifacts
