from pyspark import SparkContext
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TestSparkContext')

try:
    sc = SparkContext(master="spark://spark-master:7077", appName="TestSparkContext")
    logger.info(f"SparkContext创建成功: appName={sc.appName}, master={sc.master}")
    print(f"SparkContext创建成功: appName={sc.appName}, master={sc.master}")
    sc.stop()
    logger.info("SparkContext已关闭")
except Exception as e:
    logger.error(f"SparkContext创建失败: {e}")
    print(f"SparkContext创建失败: {e}")