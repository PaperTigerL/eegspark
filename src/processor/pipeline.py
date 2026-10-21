import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from pyspark import RDD, SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col
from pyspark.sql.types import *

from common.eegrdd import EEGRDD
from processor.eeg_processor import EEGProcessor, ProcessingConfig

# 配置日志
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EEGProcessingPipeline")


class EEGProcessingPipeline:
    """EEG处理流水线 - 组织和管理EEG处理流程"""

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        config: Optional[ProcessingConfig] = None,
    ):
        """初始化处理流水线

        Args:
            spark: SparkSession对象，必须提供有效的SparkSession
            config: 处理配置，如果为None则使用默认配置
        """
        # 强制要求提供SparkSession，不允许自动创建
        if not spark:
            raise ValueError("必须提供有效的SparkSession对象，不允许自动创建")

        # 新增日志
        try:
            logger.info(
                f"[EEGProcessingPipeline.__init__] 传入SparkSession: id={id(spark)}, appName={getattr(spark.sparkContext, 'appName', '未知')}"
            )
        except Exception as e:
            logger.warning(
                f"[EEGProcessingPipeline.__init__] 记录SparkSession日志失败: {e}"
            )
        self.spark = spark

        self.processor = EEGProcessor(config)
        self.callbacks = []
        logger.info("EEG处理流水线已初始化")

    def add_callback(self, callback: Callable[[EEGRDD], None]):
        """添加处理回调函数

        Args:
            callback: 回调函数，接收处理后的EEGRDD对象
        """
        self.callbacks.append(callback)
        logger.info(f"已添加处理回调函数，当前回调函数数量: {len(self.callbacks)}")

    def process(self, subject_id: str, eegrdd: EEGRDD) -> EEGRDD:
        """处理单个EEGRDD对象

        Args:
            subject_id: 受试者ID
            eegrdd: 输入的EEGRDD对象

        Returns:
            EEGRDD: 处理后的EEGRDD对象
        """
        try:
            logger.info(f"开始处理受试者 {subject_id} 的EEG数据")

            # 验证数据
            if eegrdd.voltage_data.shape[0] != len(eegrdd.channel_names):
                logger.warning(
                    f"数据维度({eegrdd.voltage_data.shape[0]})与通道名称数量({len(eegrdd.channel_names)})不匹配"
                )

            # 处理数据
            processed_eegrdd = self.processor.process_eegrdd(eegrdd)

            # 提取特征
            features = self.processor.extract_features(processed_eegrdd)
            logger.info(f"已提取特征: {list(features['band_powers'].keys())}")

            # 触发回调
            for callback in self.callbacks:
                try:
                    callback(processed_eegrdd)
                except Exception as e:
                    logger.error(f"执行回调函数时出错: {e}")

            return processed_eegrdd

        except Exception as e:
            logger.error(f"处理受试者 {subject_id} 的EEG数据时出错: {e}", exc_info=True)
            return eegrdd  # 出错时返回原始数据

    def process_batch(self, eegrdd_rdd: RDD) -> RDD:
        """批量处理EEGRDD对象

        Args:
            eegrdd_rdd: EEGRDD对象的RDD

        Returns:
            RDD: 处理后的EEGRDD对象RDD
        """
        logger.debug(f"开始批量处理EEGRDD RDD: {type(eegrdd_rdd)}")
        logger.debug(f"输入RDD分区数: {eegrdd_rdd.getNumPartitions()}")
        
        try:
            # 使用map操作处理每个EEGRDD，避免直接调用len()
            result = eegrdd_rdd.map(lambda eegrdd: self.processor.process_eegrdd(eegrdd))
            logger.debug(f"批量处理完成，返回RDD类型: {type(result)}")
            logger.debug(f"输出RDD分区数: {result.getNumPartitions()}")
            return result
        except Exception as e:
            logger.error(f"批量处理EEGRDD时出错: {e}", exc_info=True)
            raise

    def create_features_dataframe(self, eegrdd_rdd: RDD) -> DataFrame:
        """从EEGRDD RDD创建特征DataFrame

        Args:
            eegrdd_rdd: EEGRDD对象的RDD

        Returns:
            DataFrame: 特征DataFrame
        """
        # 提取特征
        features_rdd = eegrdd_rdd.map(self.processor.extract_features)

        # 转换为DataFrame
        features_df = self.spark.createDataFrame(features_rdd)

        # 展开band_powers列
        for band in self.processor.config.freq_bands.keys():
            features_df = features_df.withColumn(
                f"{band}_power", col("band_powers")[band]
            )

        return features_df

    def save_processed_data(self, eegrdd: EEGRDD, output_dir: str):
        """保存处理后的数据

        Args:
            eegrdd: 处理后的EEGRDD对象
            output_dir: 输出目录
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 构建文件名
            filename = f"{eegrdd.subject_id}_{eegrdd.task_name}_{eegrdd.payload_id}.npz"
            filepath = os.path.join(output_dir, filename)

            # 保存数据
            np.savez(
                filepath, eeg_data=eegrdd.voltage_data, metadata=eegrdd.get_metadata()
            )

            logger.info(f"处理后的数据已保存: {filepath}")

        except Exception as e:
            logger.error(f"保存处理后的数据时出错: {e}", exc_info=True)
