#!/usr/bin/env python
import eventlet
eventlet.monkey_patch()
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
import time
import argparse
import logging
import signal
import traceback
import atexit
import threading
import numpy as np
import datetime
import socket
import subprocess
import gc

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取 tests 目录路径
tests_dir = os.path.dirname(current_file_path)
# 获取 src 目录路径（因为 tests 在 src 下）
src_dir = os.path.dirname(tests_dir)

# 将 src 目录加入 Python 路径中
if src_dir not in sys.path:
    sys.path.append(src_dir)
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from src.processor.pipeline import EEGProcessingPipeline

from src.receiver.receiver import EEGReceiver

logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    """信号处理函数，用于捕获各种终止信号"""
    signal_name = (
        signal.Signals(sig).name if hasattr(signal, "Signals") else f"信号 {sig}"
    )
    logger.info(f"接收到{signal_name}信号，准备关闭系统...")
    
    try:
        # 先停止接收器
        if 'receiver' in globals():
            receiver.stop_receiving()
            
        # 再停止Spark会话
        stop_spark_session()
    except Exception as e:
        logger.error(f"关闭资源时出错: {str(e)}")
    finally:
        sys.exit(0)

def setup_spark_session(app_name="EEGSparkApp", master_url="local[*]"):
    """创建并配置Spark会话"""
    try:
        conf = SparkConf()
        conf.setAppName(app_name)
        conf.setMaster(master_url)
        conf.set("spark.driver.memory", "4g")
        conf.set("spark.executor.memory", "4g")
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.jars.ivySettings", "/app/src/config/ivy_settings.conf")
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        sc = spark.sparkContext
        return spark, sc
    except Exception as e:
        return None, None

def stop_spark_session():
    """停止Spark会话并清理资源"""
    def _stop_spark():
        try:
            spark = SparkSession.getActiveSession()
            if spark:     
                spark.stop()
        except Exception as e:
            logger.error(f"停止Spark会话时出错: {str(e)}", exc_info=True)
    
    # 在单独的线程中停止Spark会话以避免阻塞主循环
    threading.Thread(target=_stop_spark, daemon=True).start()

def main():
    """主函数"""
    global pipeline, receiver
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Spark EEG测试程序")
    parser.add_argument("--app-name", default="EEGSparkApp", help="Spark应用名称")
    parser.add_argument("--master", default="local[*]", help="Spark master URL")
    parser.add_argument("--log-dir", default="/app/logs", help="日志目录路径")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 基础日志配置
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join("/app/src/logs", "eegspark.log"))
        ]
    )
    logger.info(f"========== EEGSpark 应用启动 ==========")
    logger.info(f"命令行参数: {sys.argv}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")
    
    
    try:
        # 创建Spark会话
        spark, sc = setup_spark_session(args.app_name, args.master)
        if spark is None or sc is None:
            return 1
        
        # 创建处理流水线
        pipeline = EEGProcessingPipeline(spark)
        
        # 初始化处理器模块
        from src.processor import EEGProcessor, ProcessingConfig
        from src.common.eegrdd import EEGRDD
        
        # 导入可视化模块
        from src.visualization.visualizer import EEGVisualizer
        from src.visualization.run_visualization_server import main as start_visualization_server
        
        # 创建EEG接收器
        receiver = EEGReceiver()
        receiver.sc = sc
        
        # 创建可视化器
        visualizer = EEGVisualizer(sampling_rate=5000.0, reference="FCz")
        
        # 启动可视化服务器
        threading.Thread(target=start_visualization_server, daemon=True).start()
        
        # 设置数据回调处理
        def data_callback(eegrdd_rdd):
            """处理接收到的EEG数据"""
            try:
                # 将数据处理放入线程池异步执行
                def process_data_async(eegrdd_rdd):
                    try:
                        processed_data = pipeline.process_batch(eegrdd_rdd)
                        if processed_data:
                            collected_data = processed_data.collect()
                            if collected_data:
                                # 确保数据格式正确
                                if hasattr(collected_data[0], 'data') and hasattr(collected_data[0], 'channel_names') and hasattr(collected_data[0], 'sampling_rate'):
                                    visualizer.add_data_source(
                                        source_id="eeg_data",
                                        data=collected_data[0].data,
                                        channel_names=collected_data[0].channel_names,
                                        sampling_rate=collected_data[0].sampling_rate
                                    )
                                    # 触发WebSocket数据更新
                                    visualizer.update_visualization("eeg_data")
                    except Exception as e:
                        logger.error(f"异步处理EEG数据时出错: {str(e)}", exc_info=True)
                
                # 启动异步处理线程
                threading.Thread(target=process_data_async, args=(eegrdd_rdd,), daemon=True).start()
                
            except Exception as e:
                logger.error(f"提交数据处理任务时出错: {str(e)}")
        
        # 设置数据处理回调函数
        receiver.set_data_callback(data_callback)
        
        # 启动接收器服务
        receiver.start_receiving()
        
        # 添加默认数据源 - 关键修复！
        receiver.add_data_source({})
        logger.info("EEG接收器服务已启动并添加了默认数据源")
        
        # 主循环
        while True:
            time.sleep(10)
            gc.collect()
            # 主循环运行中（已移除冗余日志）
    
    except Exception as e:  # 捕获所有异常
        error_msg = f"初始化失败: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.error(f"系统错误: {error_msg}", exc_info=True)
        return 1

import sys
def excepthook(type, value, traceback):
    if hasattr(value, '__traceback__'):
        tb = value.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        print(f"错误发生在文件{tb.tb_frame.f_code.co_filename}第{tb.tb_lineno}行")
sys.excepthook = excepthook

if __name__ == "__main__":
    sys.exit(main())
