from processor.spark_processor import SparkProcessor
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger('SparkFeatureExtractor')

class SparkFeatureExtractor(SparkProcessor):
    """EEG特征提取器 - Spark优化版
    
    提取各种EEG信号特征，包括时域和频域特征
    """
    def __init__(self, features=None, use_gpu=False):
        """初始化特征提取器
        
        Args:
            features: 要提取的特征列表，如果为None则使用默认特征
            use_gpu: 是否使用GPU加速
        """
        super().__init__(use_gpu)
        
        # 默认提取的特征
        self.features = features or [
            'mean', 'std', 'kurtosis', 'skewness', 'hjorth_mobility', 
            'hjorth_complexity', 'band_power_ratio', 'connectivity'
        ]
    
    def process(self, eegrdd):
        """提取EEG特征"""
        try:
            # 获取当前活动的SparkSession
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
                
            # 使用经过滤波的数据(如果存在)，否则使用原始数据
            if hasattr(eegrdd, 'filtered_data') and eegrdd.filtered_data is not None:
                data = eegrdd.filtered_data
            else:
                data = eegrdd.voltage_data
            
            n_channels = data.shape[0]
            
            # 创建特征字典
            features = {}
            
            # 准备RDD数据: (通道索引, 通道数据)
            channels = [(i, data[i]) for i in range(n_channels)]
            channels_rdd = spark.sparkContext.parallelize(channels)
            
            # ==== 时域特征 ====
            def extract_time_features(channel_data):
                ch_idx, signal = channel_data
                import numpy as np
                
                # 计算基本统计特征
                mean = np.mean(signal)
                std = np.std(signal)
                
                # 计算峰度和偏度
                signal_norm = signal - mean
                if std > 0:
                    kurtosis = (np.mean(signal_norm**4) / (std**4)) - 3
                    skewness = np.mean(signal_norm**3) / (std**3)
                else:
                    kurtosis = 0
                    skewness = 0
                    
                # 计算Hjorth参数
                diff1 = np.diff(signal)
                diff2 = np.diff(diff1)
                
                var0 = np.var(signal)
                var1 = np.var(diff1)
                var2 = np.var(diff2)
                
                mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
                complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 and var1 > 0 else 0
                
                return (ch_idx, {
                    'mean': mean,
                    'std': std,
                    'kurtosis': kurtosis,
                    'skewness': skewness,
                    'hjorth_mobility': mobility,
                    'hjorth_complexity': complexity
                })
                
            # 提取时域特征
            time_features = channels_rdd.map(extract_time_features).collect()
            
            # 整理时域特征到字典中
            for ch_idx, ch_features in time_features:
                for feat_name, value in ch_features.items():
                    if feat_name not in features:
                        features[feat_name] = {}
                    features[feat_name][ch_idx] = value
            
            # ==== 频域特征 ====
            if hasattr(eegrdd, 'band_powers'):
                band_powers = eegrdd.band_powers
                if all(band in band_powers for band in ['alpha', 'beta', 'theta', 'delta']):
                    # 计算频段比率特征
                    ratios = spark.sparkContext.parallelize(range(n_channels)).map(
                        lambda ch: (ch, {
                            'alpha_theta_ratio': band_powers['alpha'][ch] / band_powers['theta'][ch] 
                                                if band_powers['theta'][ch] > 0 else 0,
                            'beta_alpha_ratio': band_powers['beta'][ch] / band_powers['alpha'][ch]
                                               if band_powers['alpha'][ch] > 0 else 0,
                            'relaxation_index': (band_powers['alpha'][ch] + band_powers['theta'][ch]) / 
                                               (band_powers['beta'][ch] + band_powers['delta'][ch])
                                               if (band_powers['beta'][ch] + band_powers['delta'][ch]) > 0 else 0
                        })
                    ).collect()
                    
                    # 整理频段比率特征
                    for ch_idx, ch_ratios in ratios:
                        for ratio_name, value in ch_ratios.items():
                            if ratio_name not in features:
                                features[ratio_name] = {}
                            features[ratio_name][ch_idx] = value
            
            # ==== 连通性特征 ====
            if 'connectivity' in self.features and n_channels > 1:
                # 计算连接矩阵
                def calc_correlation(ch_pair):
                    i, j = ch_pair
                    import numpy as np
                    return (i, j, np.corrcoef(data[i], data[j])[0, 1])
                
                # 生成所有通道对
                channel_pairs = [(i, j) for i in range(n_channels) for j in range(i+1, n_channels)]
                
                # 并行计算相关性
                corr_values = spark.sparkContext.parallelize(channel_pairs).map(calc_correlation).collect()
                
                # 构建相关矩阵
                corr_matrix = np.zeros((n_channels, n_channels))
                for i, j, corr in corr_values:
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr  # 对称矩阵
                
                features['connectivity_matrix'] = corr_matrix
                
                # 计算平均连通性
                features['avg_connectivity'] = {
                    ch: np.mean([corr_matrix[ch, j] for j in range(n_channels) if j != ch])
                    for ch in range(n_channels)
                }
            
            # 保存提取的特征
            eegrdd.extracted_features = features
            
            # 添加处理历史记录
            if not hasattr(eegrdd, 'processing_history'):
                eegrdd.processing_history = []
            
            eegrdd.processing_history.append({
                'operation': 'feature_extraction',
                'features': self.features,
                'timestamp': datetime.now().isoformat()
            })
            
            return eegrdd
                
        except Exception as e:
            logger.error(f"特征提取错误: {e}", exc_info=True)
            return eegrdd