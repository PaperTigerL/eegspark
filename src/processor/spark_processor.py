import logging
import numpy as np
from datetime import datetime
from pyspark import SparkContext

logger = logging.getLogger('SparkProcessor')

class SparkProcessor:
    """EEG数据处理器基类 - Spark优化版"""
    
    def __init__(self, use_gpu=False):
        # 移除spark属性，避免序列化问题
        self.use_gpu = use_gpu
        self.start_time = datetime.now()
        
    def process(self, eegrdd):
        """处理EEGRDD数据
        
        Args:
            eegrdd: 输入的EEGRDD对象
            
        Returns:
            处理后的EEGRDD对象
        """
        raise NotImplementedError("子类必须实现process方法")
    
    def process_rdd(self, rdd):
        """处理Spark RDD形式的EEG数据
        
        Args:
            rdd: PySpark RDD，每个元素是(subject_id, eegrdd)元组
        
        Returns:
            处理后的RDD
        """
        return rdd.map(lambda x: (x[0], self.process(x[1])))
        
    def to_gpu(self, data):
        """将数据转移到GPU（如果启用）"""
        if not self.use_gpu:
            return data
        
        try:
            import cupy as cp
            return cp.array(data)
        except ImportError:
            logger.warning("未安装cupy，无法使用GPU加速")
            return data
        except Exception as e:
            logger.warning(f"转移数据到GPU时出错: {e}")
            return data
            
    def to_cpu(self, data):
        """将数据转回CPU（如果需要）"""
        if not self.use_gpu:
            return data
            
        try:
            import cupy as cp
            if isinstance(data, cp.ndarray):
                return data.get()
            return data
        except ImportError:
            return data
        except Exception as e:
            logger.warning(f"转移数据到CPU时出错: {e}")
            return data