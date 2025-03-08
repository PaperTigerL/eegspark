from processor.spark_processor import SparkProcessor
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger('SparkPreprocessor')

class SparkPreprocessor(SparkProcessor):
    """EEG预处理器 - Spark优化版
    
    进行滤波、去伪迹等预处理操作
    """
    def __init__(self, bandpass=(0.5, 45), notch=50, use_gpu=False):
        super().__init__(use_gpu)  # 注意这里也要修改
        self.bandpass = bandpass
        self.notch = notch
    
    def process(self, eegrdd):
        """预处理EEG数据"""
        try:
            # 获取当前活动的SparkSession
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
            
            # 获取数据
            data = eegrdd.voltage_data
            fs = eegrdd.sampling_rate
            
            # 设计滤波器
            from scipy import signal
            nyquist = fs / 2
            low, high = self.bandpass
            low = low / nyquist
            high = min(high / nyquist, 0.99)
            
            # 将数据转换为RDD进行并行处理
            channels = [(i, data[i]) for i in range(data.shape[0])]
            channels_rdd = spark.sparkContext.parallelize(channels)
            
            # 定义滤波函数
            def apply_filters(channel_data):
                ch_idx, ch_signal = channel_data
                from scipy import signal
                import numpy as np
                
                # 带通滤波器
                b, a = signal.butter(4, [low, high], btype='band')
                filtered = signal.filtfilt(b, a, ch_signal)
                
                # 陷波滤波器(如果需要)
                if self.notch:
                    notch_freq = self.notch / nyquist
                    q = 30.0
                    b_notch, a_notch = signal.iirnotch(notch_freq, q)
                    filtered = signal.filtfilt(b_notch, a_notch, filtered)
                    
                return (ch_idx, filtered)
            
            # 并行应用滤波器
            filtered_channels = channels_rdd.map(apply_filters).collect()
            
            # 将结果重新组装为数组
            filtered_data = np.zeros_like(data)
            for ch_idx, filtered in filtered_channels:
                filtered_data[ch_idx] = filtered
            
            # 保存滤波后的数据
            eegrdd.filtered_data = filtered_data
            
            # 添加处理历史记录
            if not hasattr(eegrdd, 'processing_history'):
                eegrdd.processing_history = []
            
            eegrdd.processing_history.append({
                'operation': 'filtering',
                'bandpass': self.bandpass,
                'notch': self.notch,
                'timestamp': datetime.now().isoformat()
            })
            
            return eegrdd
            
        except Exception as e:
            logger.error(f"预处理错误: {e}", exc_info=True)
            return eegrdd