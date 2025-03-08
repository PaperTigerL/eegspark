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
# 导入处理流水线
from processor.pipeline import EEGProcessingPipeline
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
    """处理EEGRDD对象
    
    此函数使用处理流水线对EEGRDD对象进行处理，支持流式大规模处理
    包括：预处理、频谱分析、事件处理、特征提取等
    """
    global processing_pipeline
    
    try:
        # 简化日志输出，减少I/O开销
        logger.debug(f"开始处理受试者 {subject_id} 的EEG数据")
        
        # 快速验证数据有效性
        if not hasattr(eeg_obj, 'voltage_data') or eeg_obj.voltage_data is None:
            logger.error("无法处理：缺少电压数据")
            return False
            
        # 记录处理开始时间，用于性能监控
        start_time = time.time()
        
        # 使用处理流水线处理数据
        processed_eeg = processing_pipeline.process(eeg_obj)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录处理结果和性能指标
        logger.info(f"EEG数据处理完成：{processed_eeg.subject_id}, 任务：{processed_eeg.task_name}, 处理时间：{process_time:.3f}秒")
        
        # 记录处理的数据维度和特征
        if hasattr(processed_eeg, 'features') and processed_eeg.features:
            feature_count = sum(len(v) for v in processed_eeg.features.values()) if isinstance(processed_eeg.features, dict) else len(processed_eeg.features)
            logger.info(f"提取了 {feature_count} 个特征")
        
        # 检查是否有事件信息并记录
        if hasattr(processed_eeg, 'events') and processed_eeg.events:
            logger.info(f"处理了 {len(processed_eeg.events)} 个事件")
            
        # 检查是否有频谱分析结果
        if hasattr(processed_eeg, 'band_powers') and processed_eeg.band_powers:
            logger.info(f"完成了 {len(processed_eeg.band_powers)} 个频带的能量分析")
        
        return True
            
    except Exception as e:
        logger.error(f"处理EEGRDD对象时出错: {e}", exc_info=True)
        return False

def run_eeg_system(args):
    save_dir = args.save_dir
    app_name = args.app_name
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"创建数据保存目录: {save_dir}")
    
    spark = None
    receiver = None
    global processing_pipeline
    
    try:
        # 初始化SparkSession
        spark = setup_spark_session(app_name=app_name)
        logger.info(f"创建SparkSession成功，应用名称: {app_name}")
        
        # 添加必要模块到Spark环境
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 添加整个processor包到Spark
        processor_dir = os.path.join(project_root, 'processor')
        processor_zip = os.path.join(project_root, 'processor.zip')
        
        # 如果当前工作目录是 /app/src，则processor_dir应该是 /app/src/processor
        logger.info(f"处理器目录: {processor_dir}")
        
        # 打包processor目录
        import shutil
        if os.path.exists(processor_zip):
            os.remove(processor_zip)
        
        # 创建临时处理器包结构
        temp_dir = os.path.join(project_root, 'temp_pkg')
        temp_processor_dir = os.path.join(temp_dir, 'processor')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_processor_dir)
        
        # 递归复制所有处理器文件和子目录到临时目录
        def copy_directory(src, dst):
            if not os.path.exists(dst):
                os.makedirs(dst)
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    copy_directory(s, d)
                elif item.endswith('.py'):
                    shutil.copy2(s, d)
        
        # 复制整个processor目录结构
        copy_directory(processor_dir, temp_processor_dir)
        
        # 创建__init__.py文件
        with open(os.path.join(temp_processor_dir, '__init__.py'), 'w') as f:
            f.write('# processor package\n')
        
        # 打包为zip
        shutil.make_archive(os.path.join(project_root, 'processor'), 'zip', temp_dir)
        logger.info(f"处理器包已打包为: {processor_zip}")
        
        # 添加打包后的处理器到Spark
        spark.sparkContext.addPyFile(processor_zip)
        logger.info("处理器包已添加到Spark")
        
        # 添加common模块
        common_dir = os.path.join(project_root, 'common')
        for module_file in ['eegrdd.py']:
            module_path = os.path.join(common_dir, module_file)
            if os.path.exists(module_path):
                logger.info(f"添加模块到Spark: {module_path}")
                spark.sparkContext.addPyFile(module_path)
        
        # 配置数据源
        data_sources = [
            ('01', 'data-sender-1', 9000, 'data-sender-1', 9001),
            ('02', 'data-sender-2', 9002, 'data-sender-2', 9003)
        ]
        
        # 初始化处理流水线 - 增强版本，支持更多处理选项
        processing_pipeline = EEGProcessingPipeline(
            save_dir=os.path.join(save_dir, 'processed_data'),
            use_gpu=args.use_gpu,  # 支持GPU加速
            visualizer=None  # 可以在需要时添加可视化组件
        )
        
        # 添加性能监控
        processing_pipeline.enable_performance_monitoring = True
        logger.info(f"EEG处理流水线已初始化，GPU加速：{'启用' if args.use_gpu else '禁用'}")
        
        # 记录系统配置信息
        system_info = {
            'app_name': app_name,
            'save_dir': save_dir,
            'use_gpu': args.use_gpu,
            'data_sources': len(data_sources),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存系统配置
        with open(os.path.join(save_dir, 'system_config.json'), 'w') as f:
            json.dump(system_info, f, indent=2)
        
        # 创建接收器
        receiver = SparkEEGReceiver(
            spark=spark, 
            data_sources=data_sources, 
            save_dir=os.path.join(save_dir, 'received_data')
        )
        
        # 注册流式处理回调 - 使用处理函数
        receiver.set_data_callback(process_eegrdd)
        
        # 注册信号处理函数
        def handle_interrupt(sig, frame):
            logger.info("接收到中断信号，正在清理资源...")
            if receiver:
                receiver.stop()
            if spark:
                spark.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_interrupt)
        
        # 启动接收器 - 异步模式
        logger.info("正在启动EEG数据接收器...")
        receiver.start_receiving()
        logger.info("接收器已启动，正在等待数据流...")
        
        # 创建一个监控循环，定期显示接收状态
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
            
            # 关闭Spark会话
            if spark:
                spark.stop()
                logger.info("SparkSession已关闭")
    
    except Exception as e:
        logger.error(f"运行EEG处理系统时出错: {e}", exc_info=True)
        
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
    
    run_eeg_system(args)

if __name__ == "__main__":
    main()
