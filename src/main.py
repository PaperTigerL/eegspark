import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import sys
import time
import argparse
import logging
import signal
import numpy as np
from pyspark.sql import SparkSession
from datetime import datetime
# 注释掉处理流水线的导入
# from processor.pipeline import EEGProcessingPipeline
import json
# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EEGMain')

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入必要的模块
from receiver.receiver import SparkEEGReceiver

# 全局变量定义
processing_pipeline = None

def setup_spark_session(app_name="EEGSparkProcessor"):
    """设置SparkSession"""
    return SparkSession.builder \
        .appName(app_name) \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.rpc.message.maxSize", "512") \
        .config("spark.kryoserializer.buffer.max", "256") \
        .getOrCreate()

def process_eegrdd(subject_id, eeg_obj):
    """数据验证函数 - 仅验证数据，不处理"""
    try:
        logger.info(f"数据验证: 受试者 {subject_id}")
        
        # 检查数据形状
        if hasattr(eeg_obj, 'voltage_data'):
            logger.info(f"数据形状: {eeg_obj.voltage_data.shape}")
            
        # 检查通道信息
        if hasattr(eeg_obj, 'channel_names'):
            logger.info(f"通道名称数量: {len(eeg_obj.channel_names)}")
            if len(eeg_obj.channel_names) > 5:
                logger.info(f"通道名称示例: {eeg_obj.channel_names[:5]}...")
        
        # 打印通道名和数据形状的比较
        if hasattr(eeg_obj, 'voltage_data') and hasattr(eeg_obj, 'channel_names'):
            data_channels = eeg_obj.voltage_data.shape[0]
            name_channels = len(eeg_obj.channel_names)
            logger.info(f"通道比较: 数据通道数={data_channels}, 通道名数量={name_channels}")
            
            if data_channels != name_channels:
                logger.warning(f"通道数量不匹配: 数据通道({data_channels}) ≠ 名称数量({name_channels})")
                
        return True
            
    except Exception as e:
        logger.error(f"数据验证出错: {e}", exc_info=True)
        return False

def run_eeg_system(args):
    save_dir = args.save_dir
    app_name = args.app_name
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"创建数据保存目录: {save_dir}")
    
    spark = None
    receiver = None
    
    try:
        # 初始化SparkSession - 必须保留
        spark = setup_spark_session(app_name=app_name)
        logger.info(f"创建SparkSession成功，应用名称: {app_name}")
        
        # 添加必要模块到Spark环境 - 保留但简化
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 添加common模块 - 必须保留
        common_dir = os.path.join(project_root, 'common')
        for module_file in ['eegrdd.py']:
            module_path = os.path.join(common_dir, module_file)
            if os.path.exists(module_path):
                logger.info(f"添加模块到Spark: {module_path}")
                spark.sparkContext.addPyFile(module_path)
        
        # 不要添加processor相关文件
        
        # 配置数据源 - 必须保留
        data_sources = [
            ('01', 'data-sender-1', 9000, 'data-sender-1', 9001),
            ('02', 'data-sender-2', 9002, 'data-sender-2', 9003)
        ]
        
        # 记录测试配置
        system_info = {
            'app_name': app_name,
            'save_dir': save_dir,
            'data_sources': len(data_sources),
            'mode': "数据验证模式",
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(save_dir, 'test_config.json'), 'w') as f:
            json.dump(system_info, f, indent=2)
        
        # 创建接收器 - 必须保留
        receiver = SparkEEGReceiver(
            spark=spark, 
            data_sources=data_sources, 
            save_dir=os.path.join(save_dir, 'received_data')
        )
        
        # 注册数据验证回调 - 必须保留
        receiver.set_data_callback(process_eegrdd)
        
        # 下面是缺少的代码 - 启动接收器和保持程序运行
        # 注册信号处理函数
        def handle_interrupt(sig, frame):
            logger.info("接收到中断信号，正在清理资源...")
            if receiver:
                receiver.stop()
            if spark:
                spark.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_interrupt)
        
        # 启动接收器 - 异步模式 - 这行很关键
        logger.info("正在启动EEG数据接收器...")
        receiver.start_receiving()
        logger.info("接收器已启动，正在等待数据流...")
        
        # 创建一个监控循环，定期显示接收状态 - 这部分保持程序运行
        try:
            while True:
                # 显示当前已处理数据包数量
                if hasattr(receiver, 'get_received_count'):
                    count = receiver.get_received_count()
                    logger.info(f"已处理数据包: {count}")
                
                time.sleep(10)  # 每10秒更新一次状态
        
        except KeyboardInterrupt:
            logger.info("收到用户中断请求，正在停止处理...")
        finally:
            # 停止接收器
            if receiver:
                receiver.stop()
            logger.info("接收器已停止")
    
    except Exception as e:
        logger.error(f"运行EEG数据验证系统时出错: {e}", exc_info=True)
        
        # 确保资源被释放
        if receiver:
            receiver.stop()
        if spark:
            spark.stop()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EEG流式处理系统')
    parser.add_argument('--save-dir', type=str, default='/opt/eegproject/results', 
                        help='数据保存目录')
    parser.add_argument('--app-name', type=str, default='EEGStreamProcessor', 
                        help='Spark应用程序名称')
    parser.add_argument('--use-gpu', action='store_true', 
                        help='启用GPU加速处理')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='日志级别')
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 注释掉处理流水线的初始化
    # global processing_pipeline
    # processing_pipeline = EEGProcessingPipeline()
    
    run_eeg_system(args)

if __name__ == "__main__":
    main()