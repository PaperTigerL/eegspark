import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CLEEGN")


class CLEEGNModel:
    """轻量级CNN模型用于EEG伪迹检测和去除

    CLEEGN (Compact Lightweight EEG Network) 是一个专为实时EEG处理设计的
    轻量级卷积神经网络模型，可以在1-5ms内完成伪迹检测和去除。
    """

    def __init__(
        self,
        sampling_rate: float = 250.0,
        n_channels: int = 32,
        window_size: float = 0.5,
        use_gpu: bool = False,
    ):
        """初始化CLEEGN模型

        Args:
            sampling_rate: 采样率(Hz)
            n_channels: 通道数量
            window_size: 处理窗口大小(秒)
            use_gpu: 是否使用GPU加速
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.window_size = window_size
        self.use_gpu = use_gpu
        self.model_loaded = False
        self.model = None

        # 模型参数
        self.n_samples = int(window_size * sampling_rate)
        self.kernel_size = min(25, self.n_samples // 4)  # 卷积核大小，最大25点
        self.n_filters = 8  # 滤波器数量

        # 初始化模型权重 - 在实际应用中应该从预训练模型加载
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型 - 支持从预训练模型加载权重"""
        try:
            # 尝试导入深度学习库
            try:
                import torch
                import torch.nn as nn
                import os

                self.use_torch = True
                logger.info("使用PyTorch作为后端")

                # 定义一个简单的PyTorch模型
                class CLEEGNTorch(nn.Module):
                    def __init__(self, n_channels, n_samples, n_filters, kernel_size):
                        super(CLEEGNTorch, self).__init__()
                        self.conv1 = nn.Conv1d(
                            n_channels, n_filters, kernel_size, padding=kernel_size // 2
                        )
                        self.pool = nn.MaxPool1d(2)
                        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, 5, padding=2)
                        self.dropout = nn.Dropout(0.25)
                        self.fc1 = nn.Linear(
                            n_filters * 2 * (n_samples // 2), n_channels
                        )
                        self.relu = nn.ReLU()
                        self.sigmoid = nn.Sigmoid()

                    def forward(self, x):
                        # 特征提取
                        x = self.relu(self.conv1(x))
                        x = self.pool(x)
                        x = self.relu(self.conv2(x))
                        x = self.dropout(x)

                        # 展平
                        x = x.view(x.size(0), -1)

                        # 全连接层
                        x = self.sigmoid(self.fc1(x))
                        return x

                # 创建模型实例
                self.model = CLEEGNTorch(
                    self.n_channels, self.n_samples, self.n_filters, self.kernel_size
                )

                # 尝试加载预训练模型权重
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "models",
                    f"cleegn_{self.n_channels}ch_{int(self.sampling_rate)}hz.pth",
                )

                if os.path.exists(model_path):
                    try:
                        # 加载预训练权重
                        state_dict = torch.load(model_path, map_location="cpu")
                        self.model.load_state_dict(state_dict)
                        logger.info(f"成功加载预训练模型: {model_path}")
                    except Exception as e:
                        logger.warning(f"加载预训练模型失败: {e}，使用随机初始化权重")
                else:
                    logger.info(f"未找到预训练模型: {model_path}，使用随机初始化权重")
                    # 创建models目录，以便将来保存模型
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)

                # 如果使用GPU且可用
                if self.use_gpu and torch.cuda.is_available():
                    self.model = self.model.cuda()
                    logger.info("模型已加载到GPU")

                # 设置为评估模式
                self.model.eval()
                self.model_loaded = True

            except ImportError:
                # 如果PyTorch不可用，使用NumPy实现一个简化版本
                self.use_torch = False
                logger.warning("PyTorch不可用，使用NumPy实现简化版CLEEGN")

                # 初始化简化模型参数
                # 尝试加载预训练的NumPy模型参数
                import os
                import pickle

                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "models",
                    f"cleegn_numpy_{self.n_channels}ch_{int(self.sampling_rate)}hz.pkl",
                )

                if os.path.exists(model_path):
                    try:
                        # 加载预训练参数
                        with open(model_path, "rb") as f:
                            params = pickle.load(f)
                        self.W1 = params["W1"]
                        self.b1 = params["b1"]
                        self.W2 = params["W2"]
                        self.b2 = params["b2"]
                        logger.info(f"成功加载NumPy预训练模型: {model_path}")
                    except Exception as e:
                        logger.warning(
                            f"加载NumPy预训练模型失败: {e}，使用随机初始化参数"
                        )
                        self.W1 = (
                            np.random.randn(
                                self.n_filters, self.n_channels, self.kernel_size
                            )
                            * 0.01
                        )
                        self.b1 = np.zeros(self.n_filters)
                        self.W2 = (
                            np.random.randn(self.n_channels, self.n_filters) * 0.01
                        )
                        self.b2 = np.zeros(self.n_channels)
                else:
                    logger.info(
                        f"未找到NumPy预训练模型: {model_path}，使用随机初始化参数"
                    )
                    # 创建models目录，以便将来保存模型
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)

                    self.W1 = (
                        np.random.randn(
                            self.n_filters, self.n_channels, self.kernel_size
                        )
                        * 0.01
                    )
                    self.b1 = np.zeros(self.n_filters)
                    self.W2 = np.random.randn(self.n_channels, self.n_filters) * 0.01
                    self.b2 = np.zeros(self.n_channels)

                self.model_loaded = True

        except Exception as e:
            logger.error(f"初始化CLEEGN模型失败: {e}", exc_info=True)
            self.model_loaded = False

    def detect_artifacts(self, data: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """检测EEG数据中的伪迹

        Args:
            data: EEG数据，形状为(n_channels, n_samples)
            threshold: 伪迹检测阈值

        Returns:
            np.ndarray: 伪迹掩码，形状为(n_channels,)，1表示伪迹，0表示正常
        """
        try:
            start_time = time.time()

            # 确保数据是二维的
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # 确保数据形状正确
            n_channels, n_samples = data.shape
            if n_channels > n_samples:
                # 可能通道和样本点维度颠倒了
                logger.warning(f"输入数据形状可能颠倒: {data.shape}，尝试转置")
                data = data.T
                n_channels, n_samples = data.shape

            # 如果模型未加载，使用传统方法
            if not self.model_loaded:
                logger.warning("模型未加载，使用传统阈值方法检测伪迹")
                return self._threshold_artifact_detection(
                    data, threshold * 100
                )  # 转换为μV阈值

            # 准备数据
            if n_samples != self.n_samples:
                # 如果样本数不匹配，调整数据长度
                if n_samples > self.n_samples:
                    # 截断
                    data = data[:, : self.n_samples]
                else:
                    # 填充
                    pad_width = ((0, 0), (0, self.n_samples - n_samples))
                    data = np.pad(data, pad_width, mode="constant")

            # 数据预处理 - 标准化
            data_mean = np.mean(data, axis=1, keepdims=True)
            data_std = np.std(data, axis=1, keepdims=True)
            data_std[data_std == 0] = 1  # 避免除以零
            data_normalized = (data - data_mean) / data_std

            # 使用模型检测伪迹
            if self.use_torch:
                import torch

                # 转换为PyTorch张量
                data_tensor = torch.from_numpy(data_normalized).float()
                if self.use_gpu and torch.cuda.is_available():
                    data_tensor = data_tensor.cuda()

                # 前向传播
                with torch.no_grad():
                    artifact_scores = (
                        self.model(data_tensor.unsqueeze(0)).cpu().numpy()[0]
                    )
            else:
                # 使用NumPy实现的简化模型
                # 卷积操作 (非常简化版本)
                conv_output = np.zeros((self.n_filters, n_samples))
                for i in range(self.n_filters):
                    for j in range(self.n_channels):
                        # 简化的1D卷积
                        for k in range(self.kernel_size):
                            for s in range(n_samples):
                                if (
                                    s - k + self.kernel_size // 2 >= 0
                                    and s - k + self.kernel_size // 2 < n_samples
                                ):
                                    conv_output[i, s] += (
                                        data_normalized[
                                            j, s - k + self.kernel_size // 2
                                        ]
                                        * self.W1[i, j, k]
                                    )
                    conv_output[i] += self.b1[i]

                # ReLU激活函数
                conv_output = np.maximum(0, conv_output)

                # 平均池化 (简化)
                pooled = np.zeros((self.n_filters, n_samples // 2))
                for i in range(self.n_filters):
                    for j in range(n_samples // 2):
                        pooled[i, j] = np.mean(conv_output[i, j * 2 : (j + 1) * 2])

                # 全连接层
                artifact_scores = np.zeros(self.n_channels)
                for i in range(self.n_channels):
                    for j in range(self.n_filters):
                        artifact_scores[i] += np.mean(pooled[j]) * self.W2[i, j]
                    artifact_scores[i] += self.b2[i]

                # Sigmoid激活函数
                artifact_scores = 1 / (1 + np.exp(-artifact_scores))

            # 应用阈值
            artifacts = (artifact_scores > threshold).astype(int)

            elapsed_time = time.time() - start_time
            logger.debug(f"伪迹检测完成，耗时: {elapsed_time*1000:.2f}ms")

            return artifacts

        except Exception as e:
            logger.error(f"CLEEGN伪迹检测失败: {e}", exc_info=True)
            # 出错时使用传统方法
            return self._threshold_artifact_detection(
                data, threshold * 100
            )  # 转换为μV阈值

    def _threshold_artifact_detection(
        self, data: np.ndarray, threshold: float = 100.0
    ) -> np.ndarray:
        """使用简单阈值方法检测伪迹 (备用方法)

        Args:
            data: EEG数据，形状为(n_channels, n_samples)
            threshold: 伪迹阈值(μV)

        Returns:
            np.ndarray: 伪迹掩码，形状为(n_channels,)，1表示伪迹，0表示正常
        """
        # 计算每个通道的峰峰值
        peak_to_peak = np.max(data, axis=1) - np.min(data, axis=1)

        # 应用阈值
        artifacts = (peak_to_peak > threshold).astype(int)

        return artifacts

    def remove_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测并去除EEG数据中的伪迹

        Args:
            data: EEG数据，形状为(n_channels, n_samples)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - 处理后的数据，形状为(n_channels, n_samples)
                - 伪迹掩码，形状为(n_channels,)，1表示伪迹，0表示正常
        """
        try:
            start_time = time.time()

            # 检测伪迹
            artifacts = self.detect_artifacts(data)

            # 如果没有检测到伪迹，直接返回原始数据
            if np.sum(artifacts) == 0:
                return data, artifacts

            # 创建数据副本
            cleaned_data = data.copy()

            # 对每个被标记为伪迹的通道进行处理
            for ch_idx in range(data.shape[0]):
                if artifacts[ch_idx] == 1:
                    # 方法1: 使用相邻通道的平均值替换
                    # 找出非伪迹通道
                    clean_channels = np.where(artifacts == 0)[0]
                    if len(clean_channels) > 0:
                        # 计算非伪迹通道的平均值
                        clean_avg = np.mean(data[clean_channels], axis=0)
                        cleaned_data[ch_idx] = clean_avg
                    else:
                        # 如果所有通道都是伪迹，使用方法2
                        # 方法2: 使用中值滤波
                        from scipy import signal

                        # 使用较大的窗口进行中值滤波
                        kernel_size = min(51, data.shape[1] // 3)
                        if kernel_size % 2 == 0:  # 确保窗口大小为奇数
                            kernel_size += 1
                        cleaned_data[ch_idx] = signal.medfilt(
                            data[ch_idx], kernel_size=kernel_size
                        )

            elapsed_time = time.time() - start_time
            logger.debug(f"伪迹去除完成，耗时: {elapsed_time*1000:.2f}ms")

            return cleaned_data, artifacts

        except Exception as e:
            logger.error(f"伪迹去除失败: {e}", exc_info=True)
            # 出错时返回原始数据
            return data, np.zeros(data.shape[0], dtype=int)


# 单例模式，确保只创建一个模型实例
_cleegn_instance = None


def get_cleegn_instance(
    sampling_rate: float = 250.0,
    n_channels: int = 32,
    window_size: float = 0.5,
    use_gpu: bool = False,
) -> CLEEGNModel:
    """获取CLEEGN模型实例 (单例模式)

    Args:
        sampling_rate: 采样率(Hz)
        n_channels: 通道数量
        window_size: 处理窗口大小(秒)
        use_gpu: 是否使用GPU加速

    Returns:
        CLEEGNModel: CLEEGN模型实例
    """
    global _cleegn_instance
    if _cleegn_instance is None:
        _cleegn_instance = CLEEGNModel(sampling_rate, n_channels, window_size, use_gpu)
    return _cleegn_instance
