import numpy as np
from .base_processor import BaseProcessor

class QuantumEncoder(BaseProcessor):
    """量子启发式编码压缩处理器
    
    使用量子游走（Quantum Walk）算法对EEG数据进行实时压缩编码，
    可以显著降低数据传输带宽占用
    """
    def __init__(self, use_gpu=False, compression_ratio=0.5, method='hybrid_walk'):
        super().__init__(use_gpu=use_gpu)
        self.compression_ratio = compression_ratio
        self.method = method
        
    def process(self, eegrdd):
        """对EEG数据进行量子启发式压缩
        
        Args:
            eegrdd: EEGRDD实例，包含EEG数据
            
        Returns:
            处理后的EEGRDD实例，添加了压缩数据
        """
        try:
            # 将数据转移到GPU（如果启用）
            voltage_data = self.to_gpu(eegrdd.voltage_data)
            
            # 计算压缩后的数据大小
            original_size = len(voltage_data)
            compressed_size = int(original_size * self.compression_ratio)
            
            if self.method == 'hybrid_walk':
                # 使用混合量子游走算法进行压缩
                compressed_data = self._hybrid_quantum_walk(voltage_data, compressed_size)
            else:
                # 默认使用基础压缩方法
                compressed_data = self._basic_compression(voltage_data, compressed_size)
            
            # 将压缩结果转回CPU（如果需要）
            compressed_features = {
                'compressed_data': self.to_cpu(compressed_data),
                'compression_ratio': self.compression_ratio,
                'original_size': original_size,
                'compressed_size': compressed_size
            }
            
            # 更新EEGRDD的特征
            eegrdd.features.update(compressed_features)
            
        except Exception as e:
            print(f"[错误] 量子编码压缩失败: {e}")
            
        return eegrdd
    
    def _hybrid_quantum_walk(self, data, target_size):
        """混合量子游走压缩算法
        
        使用量子游走的概念对信号进行自适应采样和压缩
        """
        try:
            # 计算信号能量分布
            energy = np.abs(data) ** 2
            total_energy = np.sum(energy)
            
            # 构建量子游走概率分布
            prob_dist = energy / total_energy
            
            # 基于概率分布进行重要性采样
            indices = np.random.choice(
                len(data),
                size=target_size,
                p=prob_dist,
                replace=False
            )
            indices.sort()
            
            # 提取重要样本点
            compressed = data[indices]
            
            return compressed
            
        except Exception as e:
            print(f"[错误] 混合量子游走压缩失败: {e}")
            return data[:target_size]
    
    def _basic_compression(self, data, target_size):
        """基础压缩方法
        
        使用简单的降采样策略
        """
        try:
            # 计算降采样步长
            step = len(data) // target_size
            return data[::step][:target_size]
            
        except Exception as e:
            print(f"[错误] 基础压缩失败: {e}")
            return data[:target_size]
    
    def decompress(self, eegrdd):
        """解压缩数据
        
        Args:
            eegrdd: EEGRDD实例，包含压缩数据
            
        Returns:
            解压缩后的数据
        """
        if 'compressed_data' not in eegrdd.features:
            return None
            
        compressed_data = eegrdd.features['compressed_data']
        original_size = eegrdd.features['original_size']
        
        # 使用线性插值进行重建
        indices = np.linspace(0, len(compressed_data)-1, original_size)
        decompressed = np.interp(
            indices,
            np.arange(len(compressed_data)),
            compressed_data
        )
        
        return decompressed