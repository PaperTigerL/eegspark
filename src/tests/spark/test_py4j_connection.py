import os
import sys
import time
import logging
import traceback

# 导入Ivy配置修复模块
from utils.spark_fixes.ivy_config import fix_ivy_repository_path

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Py4JConnectionTest")


def test_py4j_connection():
    """测试Py4J连接

    测试与Java网关的连接是否正常工作
    使用SparkContextManager确保单例模式，避免多SparkContext冲突

    注意：已移除重试机制，因为重试只会导致多SparkContext冲突

    Returns:
        bool: 连接测试是否成功
    """
    try:
        # 首先修复Ivy配置
        logger.info("正在修复Ivy配置...")
        fix_ivy_repository_path()

        # 强制清理所有现有的SparkContext资源
        logger.info("强制清理所有现有的SparkContext资源...")

        # 使用SparkContextManager进行测试，避免多SparkContext冲突
        try:
            # 尝试导入SparkContextManager
            from common.spark_context_manager import SparkContextManager

            logger.info("使用SparkContextManager进行Py4J连接测试")

            # 首先强制清理所有SparkContext资源
            logger.info("强制清理所有现有的SparkContext资源...")
            SparkContextManager.force_cleanup_spark_context()

            # 等待足够长的时间确保资源完全释放
            logger.info("等待资源完全释放...")
            time.sleep(10)  # 增加等待时间，确保资源完全释放

            # 使用SparkContextManager创建SparkSession
            logger.info("创建SparkSession...")
            spark = SparkContextManager.get_or_create_spark_session(
                app_name="Py4JConnectionTest", master="local[*]"
            )

            if spark is not None:
                # 测试Py4J连接 - 执行一个简单的Spark操作
                logger.info("测试Py4J连接...")
                test_rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
                result = test_rdd.count()

                logger.info(f"Py4J连接测试成功! 测试RDD计数结果: {result}")

                # 停止SparkContext（使用SparkContextManager）
                logger.info("正在停止SparkContext...")
                SparkContextManager.stop_spark_session()

                return True
            else:
                logger.error("无法创建SparkSession")
                return False

        except ImportError:
            logger.warning("无法导入SparkContextManager，使用传统方式进行测试")

            # 导入PySpark相关模块
            from pyspark import SparkContext, SparkConf
            import py4j.protocol
            import gc

            # 强制清理所有现有的SparkContext资源
            logger.info("使用传统方式强制清理所有SparkContext资源...")
            try:
                # 检查是否有活跃的SparkContext
                if SparkContext._active_spark_context:
                    logger.info("检测到已有活跃的SparkContext，强制停止它")
                    try:
                        # 首先尝试正常停止
                        active_sc = SparkContext._active_spark_context
                        active_sc.stop()
                        # 确保引用被清除
                        SparkContext._active_spark_context = None

                        # 等待Java进程完全终止
                        logger.info("等待Java进程完全终止...")
                        time.sleep(10)  # 增加等待时间，确保资源完全释放

                        # 强制垃圾回收
                        gc.collect()
                        time.sleep(2)

                        # 检查是否仍有活跃的SparkContext
                        if SparkContext._active_spark_context:
                            logger.warning("SparkContext仍然活跃，尝试更强制的方式关闭")
                            # 尝试使用更强制的方式关闭
                            if hasattr(active_sc, "_jsc"):
                                try:
                                    active_sc._jsc.sc().stop()
                                except Exception:
                                    pass
                            # 再次确保引用被清除
                            SparkContext._active_spark_context = None
                            # 再次强制垃圾回收
                            gc.collect()
                            time.sleep(2)

                            # 如果仍然存在活跃的SparkContext，尝试重置所有引用
                            if SparkContext._active_spark_context:
                                logger.warning("SparkContext仍然活跃，尝试重置所有引用")
                                # 重置所有SparkContext相关引用
                                SparkContext._gateway = None
                                SparkContext._jvm = None
                                SparkContext._jsc = None
                                SparkContext._active_spark_context = None
                                gc.collect()
                                time.sleep(2)
                    except Exception as stop_err:
                        logger.warning(f"停止已有SparkContext失败: {stop_err}")
                        # 确保引用被清除，即使停止失败
                        SparkContext._active_spark_context = None
                        gc.collect()
                        time.sleep(2)
                else:
                    logger.info("没有检测到活跃的SparkContext")

                # 尝试关闭所有Py4J连接
                try:
                    # 重置Py4J网关
                    import py4j.java_gateway

                    if hasattr(py4j.java_gateway, "GatewayParameters"):
                        py4j.java_gateway._DEFAULT_GATEWAY_PARAMETERS = (
                            py4j.java_gateway.GatewayParameters()
                        )
                except Exception as e:
                    logger.warning(f"重置Py4J网关参数失败: {e}")

                # 最后再次强制垃圾回收
                gc.collect()
                time.sleep(5)  # 等待足够长的时间确保资源完全释放
            except Exception as e:
                logger.error(f"清理SparkContext资源时出错: {e}")

            # 设置Spark配置
            logger.info("正在设置Spark配置...")
            conf = SparkConf()
            # 设置Ivy路径
            ivy_home = os.path.expanduser("~/.ivy2")
            conf.set("spark.driver.extraJavaOptions", f"-Divy.home={ivy_home}")
            conf.set("spark.executor.extraJavaOptions", f"-Divy.home={ivy_home}")
            # 增加Py4J连接超时设置
            conf.set("spark.network.timeout", "800s")  # 增加超时时间
            # 设置Python与Java进程间通信配置
            conf.set("spark.python.use.daemon", "true")
            conf.set("spark.python.worker.reuse", "true")
            # 设置Py4J特定配置
            conf.set("spark.driver.maxResultSize", "2g")
            # 添加防止多SparkContext冲突的配置
            conf.set("spark.driver.allowMultipleContexts", "false")
            # 设置更大的驱动器内存
            conf.set("spark.driver.memory", "4g")

            # 不使用重试机制，直接尝试创建SparkContext
            try:
                # 创建新的SparkContext
                logger.info("正在创建SparkContext...")
                sc = SparkContext(
                    master="local[*]", appName="Py4JConnectionTest", conf=conf
                )

                # 测试Py4J连接 - 执行一个简单的Spark操作
                logger.info("测试Py4J连接...")
                test_rdd = sc.parallelize([1, 2, 3, 4, 5])
                result = test_rdd.count()

                logger.info(f"Py4J连接测试成功! 测试RDD计数结果: {result}")

                # 停止SparkContext
                logger.info("正在停止SparkContext...")
                sc.stop()
                # 确保引用被清除
                SparkContext._active_spark_context = None
                # 强制垃圾回收
                gc.collect()

                return True

            except py4j.protocol.Py4JError as py4j_err:
                error_msg = f"Py4J连接错误: {py4j_err}"
                logger.error(error_msg)

                # 使用专门的Py4J连接错误记录方法
                try:
                    from common.error_logger import error_logger

                    error_logger.log_py4j_connection_error(
                        error_msg,
                        details={
                            "error_type": "py4j_connection",
                            "error_class": py4j_err.__class__.__name__,
                            "function": "test_py4j_connection",
                        },
                    )
                except ImportError:
                    pass

                # 清理资源
                if "sc" in locals() and sc is not None:
                    try:
                        sc.stop()
                    except Exception:
                        pass

                logger.error(f"Py4J连接失败: {py4j_err}")
                return False

            except Exception as e:
                logger.error(f"SparkContext创建失败: {e}")
                logger.error("详细错误信息:", exc_info=True)
                return False

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.error("详细错误信息:", exc_info=True)
        return False

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.error("详细错误信息:", exc_info=True)
        return False


def main():
    logger.info("开始测试Py4J连接...")
    logger.info("注意：已移除重试机制，因为重试只会导致多SparkContext冲突")
    success = test_py4j_connection()

    if success:
        logger.info("Py4J连接测试成功!")
        print("\n✅ Py4J连接测试成功! Java网关连接正常工作。\n")
        return 0
    else:
        logger.error("Py4J连接测试失败!")
        print("\n❌ Py4J连接测试失败! 请检查日志获取详细信息。\n")
        print(
            "\n提示：如果遇到'Only one SparkContext should be running in this JVM'错误，"
        )
        print("请确保没有其他PySpark程序正在运行，或重启Python解释器。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
