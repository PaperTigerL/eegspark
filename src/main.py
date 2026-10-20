import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import sys
import time
import argparse
import logging
import signal
import traceback
import py4j.protocol
import atexit
import threading

# 导入Spark相关模块
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

# 导入接收器和处理器模块
from receiver.receiver import SparkEEGReceiver
from processor.pipeline import EEGProcessingPipeline
from processor.realtime_eeg_processor import RealtimeEEGProcessor, RealtimeProcessingConfig

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EEGMain')


def setup_spark_session(app_name="EEGSparkProcessor"):
    """创建SparkSession实例
    
    使用与test_sparkcontext.py相同的方法创建SparkContext，连接到Spark集群
    
    Args:
        app_name: Spark应用名称
        
    Returns:
        SparkSession实例
    """
    from pyspark.sql import SparkSession
    from pyspark import SparkContext, SparkConf
    import os
    import socket
    
    # 首先测试网络连接
    try:
        # 测试与Spark Master的网络连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 设置超时时间为5秒
        result = sock.connect_ex(('spark-master', 7077))
        sock.close()
        
        if result != 0:
            logger.error(f"无法连接到Spark Master, 错误代码: {result}")
            raise ConnectionError(f"无法连接到Spark Master, 错误代码: {result}")
        else:
            logger.info("成功连接到Spark Master")
    except Exception as net_err:
        logger.error(f"网络连接测试失败: {net_err}")
        raise
    
    # 创建SparkContext
    try:
        sc = SparkContext(master="spark://spark-master:7077", appName=app_name)
        logger.info(f"SparkContext创建成功: appName={sc.appName}")
        
        # 基于SparkContext创建SparkSession
        spark = SparkSession(sc)
        logger.info(f"SparkSession创建成功")
        return spark
    except py4j.protocol.Py4JError as py4j_err:
        logger.error(f"Py4J连接错误: {py4j_err}")
        raise
    except Exception as e:
        logger.error(f"SparkContext创建失败: {e}")
        raise




def stop_spark_session():
    """停止SparkContext，确保资源被释放"""
    try:
        from pyspark import SparkContext
        # 检查是否有活跃的SparkContext
        if SparkContext._active_spark_context:
            logger.info("正在停止SparkContext...")
            try:
                SparkContext._active_spark_context.stop()
                logger.info("SparkContext已停止")
            except Exception as stop_err:
                logger.error(f"停止SparkContext失败: {stop_err}")
                # 尝试使用更强制的方式关闭
                if hasattr(SparkContext._active_spark_context, '_jsc'):
                    try:
                        SparkContext._active_spark_context._jsc.sc().stop()
                    except Exception:
                        pass
            
            # 将活跃的SparkContext设为None
            SparkContext._active_spark_context = None
        else:
            logger.info("没有活跃的SparkContext")
    except Exception as e:
        logger.error(f"停止SparkContext失败: {e}")


def signal_handler(sig, frame):
    """信号处理函数，用于捕获各种终止信号"""
    signal_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"信号 {sig}"
    logger.info(f"接收到{signal_name}信号，准备关闭系统...")
    stop_spark_session()
    sys.exit(0)


# 注册退出处理函数，确保程序退出时资源被释放
def exit_handler():
    """程序退出时的清理函数"""
    logger.info("程序正常退出，执行清理操作...")
    stop_spark_session()

# 注册退出处理函数
atexit.register(exit_handler)

def process_eeg_data(subject_id, eegrdd):
    """处理EEG数据的回调函数
    
    Args:
        subject_id: 受试者ID
        eegrdd: EEGRDD对象
    """
    try:
        logger.info(f"接收到受试者 {subject_id} 的EEG数据，准备处理")
        # 全局处理流水线会处理数据
        global pipeline
        if pipeline:
            processed_eegrdd = pipeline.process(subject_id, eegrdd)
            logger.info(f"受试者 {subject_id} 的EEG数据处理完成")
            
            # 保存处理后的数据
            output_dir = os.path.join(os.path.dirname(__file__), 'receiver', 'data', 'output')
            pipeline.save_processed_data(processed_eegrdd, output_dir)
    except Exception as e:
        logger.error(f"处理受试者 {subject_id} 的EEG数据时出错: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Spark EEG测试程序')
    parser.add_argument('--app-name', type=str, default="EEGSparkTest", help='Spark应用名称')
    parser.add_argument('--host', default='0.0.0.0', help='接收器监听地址')
    parser.add_argument('--metadata-port', type=int, default=9000, help='元数据端口')
    parser.add_argument('--data-port', type=int, default=9001, help='数据端口')
    parser.add_argument('--save-dir', default='./received_data', help='数据保存目录')
    args = parser.parse_args()
    
    # 注册信号处理器，处理更多类型的信号
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    # 在Windows上，SIGHUP可能不可用
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)  # 终端关闭
    
    # 创建SparkSession
    try:
        spark = setup_spark_session(args.app_name)
        logger.info(f"SparkSession创建成功: {spark.sparkContext.appName}")
        
        # 创建EEG处理流水线
        global pipeline
        processing_config = RealtimeProcessingConfig()
        pipeline = EEGProcessingPipeline(spark, processing_config)
        logger.info("EEG处理流水线创建成功")
        
        # 创建EEG接收器
        # 配置数据源 - 格式: (subject_id, meta_host, meta_port, data_host, data_port)
        data_sources = [
            ('01', 'data-sender-1', 9000, 'data-sender-1', 9001),
            ('02', 'data-sender-2', 9002, 'data-sender-2', 9003)
        ]
        receiver = SparkEEGReceiver(spark, data_sources, save_dir=args.save_dir)
        
        # 设置数据处理回调
        receiver.set_data_callback(process_eeg_data)
        logger.info("EEG接收器创建成功并设置了数据处理回调")
        
        # 启动接收器
        receiver.start_receiving()
        logger.info("EEG接收器已启动，等待数据...")
        
        # 保持主线程运行，并定期检查SparkContext状态
        logger.info("系统启动成功，按Ctrl+C停止...")
        try:
            check_interval = 30  # 每30秒检查一次
            last_check_time = time.time()
            
            while True:
                current_time = time.time()
                
                # 定期检查SparkContext状态
                if current_time - last_check_time >= check_interval:
                    if SparkContext._active_spark_context is None:
                        logger.warning("检测到SparkContext已失效，尝试重新初始化")
                        try:
                            spark = setup_spark_session(args.app_name)
                            logger.info("SparkContext重新初始化成功")
                            
                            # 重新创建处理流水线和接收器
                            pipeline = EEGProcessingPipeline(spark, processing_config)
                            receiver = SparkEEGReceiver(spark, data_sources, save_dir=args.save_dir)
                            receiver.set_data_callback(process_eeg_data)
                            receiver.start_receiving()
                            logger.info("系统组件已重新初始化")
                        except Exception as reinit_err:
                            logger.error(f"SparkContext重新初始化失败: {reinit_err}")
                    else:
                        # 验证SparkContext是否仍然有效
                        try:
                            is_active = not SparkContext._active_spark_context._jsc.sc().isStopped()
                            if not is_active:
                                logger.warning("SparkContext已停止但引用仍存在，清理资源")
                                SparkContext._active_spark_context = None
                        except Exception:
                            pass
                    
                    # 输出接收器统计信息
                    try:
                        stats = receiver.get_stats()
                        logger.info(f"接收器统计: 已接收 {stats['packets_received']} 个数据包, {stats['bytes_received']/1024/1024:.2f} MB")
                    except Exception:
                        pass
                    
                    last_check_time = current_time
                
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到键盘中断，准备关闭系统")
            stop_spark_session()
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        return 1
    
    return 0


# 全局变量
pipeline = None

if __name__ == "__main__":
    sys.exit(main())