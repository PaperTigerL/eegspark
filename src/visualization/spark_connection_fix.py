import os
import sys
import logging
import time
import gc
import py4j.protocol
from pyspark import SparkContext, SparkConf

# 导入Ivy配置修复模块
from utils.spark_fixes.ivy_config import fix_ivy_repository_path

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SparkConnectionFix")


def force_cleanup_spark_context():
    """
    强制清理所有SparkContext资源，确保不会有多个SparkContext同时运行

    解决多SparkContext冲突问题，在启动可视化服务器前调用
    此函数与SparkContextManager.force_cleanup_spark_context保持一致的清理逻辑
    """
    try:
        # 尝试导入SparkContextManager并使用其清理方法
        try:
            # 尝试直接导入
            try:
                from common.spark_context_manager import SparkContextManager

                logger.info(
                    "使用SparkContextManager.force_cleanup_spark_context进行清理"
                )
                SparkContextManager.force_cleanup_spark_context()
                return True
            except ImportError:
                # 如果直接导入失败，尝试添加common目录到Python路径
                sys.path.append(
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")
                )
                try:
                    from common.spark_context_manager import SparkContextManager

                    logger.info(
                        "使用SparkContextManager.force_cleanup_spark_context进行清理"
                    )
                    SparkContextManager.force_cleanup_spark_context()
                    return True
                except ImportError as e:
                    logger.warning(
                        f"导入SparkContextManager失败，将使用本地清理方法: {e}"
                    )
        except Exception as e:
            logger.warning(f"使用SparkContextManager清理失败，将使用本地清理方法: {e}")

        # 如果无法使用SparkContextManager，则使用本地清理方法
        logger.info("使用本地方法强制清理SparkContext资源...")

        # 检查是否有活跃的SparkContext
        if SparkContext._active_spark_context:
            logger.info("检测到已有活跃的SparkContext，正在强制停止它")
            try:
                # 首先尝试正常停止
                active_sc = SparkContext._active_spark_context
                active_sc.stop()
                # 确保引用被清除
                SparkContext._active_spark_context = None

                # 等待Java进程完全终止
                logger.info("等待Java进程完全终止...")
                time.sleep(15)  # 显著增加等待时间，确保资源完全释放

                # 尝试强制垃圾回收
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
                    time.sleep(5)

                    # 如果仍然存在活跃的SparkContext，尝试更极端的方法
                    if SparkContext._active_spark_context:
                        logger.warning("SparkContext仍然活跃，尝试重置所有引用")
                        # 重置所有SparkContext相关引用
                        SparkContext._gateway = None
                        SparkContext._jvm = None
                        SparkContext._jsc = None
                        SparkContext._active_spark_context = None
                        gc.collect()
                        time.sleep(5)
            except Exception as stop_err:
                logger.warning(f"停止已有SparkContext失败: {stop_err}")
                # 确保引用被清除，即使停止失败
                SparkContext._active_spark_context = None
                # 重置所有SparkContext相关引用
                SparkContext._gateway = None
                SparkContext._jvm = None
                SparkContext._jsc = None
                gc.collect()
                time.sleep(5)
        else:
            logger.info("没有检测到活跃的SparkContext")

        # 尝试通过JVM系统属性强制停止所有SparkContext
        try:
            import jpype

            if jpype.isJVMStarted():
                jpype.java.lang.System.setProperty(
                    "spark.driver.allowMultipleContexts", "false"
                )
                # 尝试强制JVM垃圾回收
                jpype.java.lang.System.gc()
                time.sleep(3)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"设置JVM系统属性失败: {e}")

        # 检查JVM中是否有SparkContext实例
        try:
            from py4j.java_gateway import JavaGateway

            gateway = JavaGateway()
            if (
                hasattr(gateway, "_gateway_client")
                and gateway._gateway_client is not None
            ):
                logger.info("正在关闭现有的JavaGateway连接")
                gateway.shutdown()
                time.sleep(3)  # 等待连接完全关闭
        except Exception as e:
            logger.warning(f"关闭JavaGateway失败: {e}")

        # 尝试关闭所有Py4J连接
        try:
            import py4j.java_gateway

            # 重置Py4J网关工厂
            if hasattr(py4j.java_gateway, "GatewayParameters"):
                py4j.java_gateway._DEFAULT_GATEWAY_PARAMETERS = (
                    py4j.java_gateway.GatewayParameters()
                )
            # 尝试关闭所有已知的网关连接
            if hasattr(py4j.java_gateway, "_gateway_connection_pool"):
                logger.info("尝试关闭所有Py4J网关连接池")
                try:
                    py4j.java_gateway._gateway_connection_pool.close()
                except:
                    pass
        except Exception as e:
            logger.warning(f"重置Py4J网关参数失败: {e}")

        # 最后的清理：设置环境变量，防止多SparkContext
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            "--conf spark.driver.allowMultipleContexts=false pyspark-shell"
        )

        # 确保最终没有活跃的SparkContext
        if (
            hasattr(SparkContext, "_active_spark_context")
            and SparkContext._active_spark_context is not None
        ):
            logger.warning("所有清理尝试后仍有活跃的SparkContext，强制设置为None")
            SparkContext._active_spark_context = None

        # 最终垃圾回收
        gc.collect()
        time.sleep(2)

        logger.info("SparkContext资源清理完成")
        return True
    except Exception as e:
        logger.error(f"清理SparkContext资源时出错: {e}")
        # 确保即使出错，也重置SparkContext引用
        try:
            SparkContext._active_spark_context = None
        except:
            pass
        return False


def ensure_spark_connection(app_name="EEGVisualization"):
    """
    确保Py4J连接正常工作，在可视化服务器启动前调用

    使用SparkContextManager类管理SparkContext实例，解决多SparkContext冲突问题
    不再使用重试机制，避免创建多个SparkContext实例

    Args:
        app_name: Spark应用名称

    Returns:
        bool: 连接是否正常
    """
    # 确保Ivy配置正确 - 这是解决Py4J连接问题的关键
    logger.info("正在修复Ivy配置...")
    fix_ivy_repository_path()

    # 导入SparkContextManager类
    try:
        from common.spark_context_manager import SparkContextManager
    except ImportError:
        # 如果导入失败，尝试添加common目录到Python路径
        sys.path.append(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")
        )
        try:
            from common.spark_context_manager import SparkContextManager
        except ImportError as e:
            logger.error(f"导入SparkContextManager失败: {e}")
            return False

    # 使用SparkContextManager强制清理现有的SparkContext资源
    logger.info("使用SparkContextManager强制清理现有的SparkContext资源...")
    SparkContextManager.force_cleanup_spark_context()

    # 等待足够长的时间确保资源完全释放
    time.sleep(15)  # 显著增加等待时间，确保资源完全释放

    # 检查是否有活跃的SparkContext
    from pyspark import SparkContext

    if SparkContext._active_spark_context:
        logger.warning("清理后仍检测到活跃的SparkContext，无法继续")
        return False

    logger.info("SparkContext资源清理完成，可视化服务器可以安全启动")
    return True


def fix_visualization_spark_connection():
    """
    修复可视化模块的Spark连接问题

    在启动可视化服务器前调用此函数，使用SparkContextManager类强制清理现有SparkContext资源
    解决多SparkContext冲突问题

    特别处理Docker容器环境中的Py4J连接问题
    """
    # 确保Ivy配置正确
    fix_ivy_repository_path()

    # 设置额外的环境变量，解决可视化服务器与Spark集成的问题
    os.environ["PYSPARK_SUBMIT_ARGS"] = (
        "--driver-memory 4g --conf spark.network.timeout=800s --conf spark.driver.allowMultipleContexts=false pyspark-shell"
    )

    # 确保Java临时目录存在并有正确权限
    java_tmp_dir = "/tmp/java_tmpdir"
    os.makedirs(java_tmp_dir, exist_ok=True)

    # 检查是否在Docker容器内运行
    in_docker = os.path.exists("/.dockerenv")
    if in_docker:
        logger.info("检测到在Docker容器内运行，应用特定的Docker环境配置...")
        try:
            # 设置Java临时目录权限
            import subprocess

            subprocess.run(["chmod", "777", java_tmp_dir], check=False)
            logger.info(f"已设置Docker容器中Java临时目录的权限: {java_tmp_dir}")

            # 设置额外的Docker环境变量
            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

            # 检查网络连接
            logger.info("检查与spark-master的网络连接...")
            try:
                ping_result = subprocess.run(
                    ["ping", "-c", "2", "spark-master"], capture_output=True, text=True
                )
                if ping_result.returncode == 0:
                    logger.info("成功ping通spark-master")
                else:
                    logger.warning(
                        f"无法ping通spark-master，这可能导致连接问题: {ping_result.stderr}"
                    )
            except Exception as e:
                logger.warning(f"检查网络连接时出错: {e}")
        except Exception as e:
            logger.warning(f"设置Docker特定配置时出错: {e}")

    # 设置Java临时目录环境变量
    os.environ["JAVA_OPTS"] = (
        f"{os.environ.get('JAVA_OPTS', '')} -Djava.io.tmpdir={java_tmp_dir}"
    )
    os.environ["SPARK_JAVA_OPTS"] = (
        f"{os.environ.get('SPARK_JAVA_OPTS', '')} -Djava.io.tmpdir={java_tmp_dir}"
    )

    # 导入SparkContextManager类并使用其清理方法
    try:
        # 尝试直接导入
        try:
            from common.spark_context_manager import SparkContextManager

            # 使用SparkContextManager强制清理现有的SparkContext资源
            logger.info("使用SparkContextManager强制清理现有的SparkContext资源...")
            SparkContextManager.force_cleanup_spark_context()
        except ImportError:
            # 如果导入失败，尝试添加common目录到Python路径
            sys.path.append(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "common")
            )
            try:
                from common.spark_context_manager import SparkContextManager

                # 使用SparkContextManager强制清理现有的SparkContext资源
                logger.info("使用SparkContextManager强制清理现有的SparkContext资源...")
                SparkContextManager.force_cleanup_spark_context()
            except ImportError as e:
                # 如果导入失败，使用本地的force_cleanup_spark_context函数
                logger.warning(
                    f"导入SparkContextManager失败，使用本地的force_cleanup_spark_context函数: {e}"
                )
                force_cleanup_spark_context()
    except Exception as e:
        # 如果使用SparkContextManager失败，使用本地的force_cleanup_spark_context函数
        logger.warning(
            f"使用SparkContextManager清理失败，使用本地的force_cleanup_spark_context函数: {e}"
        )
        force_cleanup_spark_context()

    # 等待足够长的时间确保资源完全释放
    logger.info("等待资源完全释放...")
    time.sleep(15)  # 显著增加等待时间，确保资源完全释放

    # 测试Py4J连接 - 不再使用重试机制
    connection_ok = ensure_spark_connection()

    if connection_ok:
        logger.info("可视化模块的Spark连接修复成功")
    else:
        logger.warning("可视化模块的Spark连接修复失败，但将继续尝试启动服务器")

        # 如果在Docker容器内运行且连接失败，提供更详细的诊断信息
        if in_docker:
            logger.info("在Docker容器内检测到Py4J连接问题，尝试进一步诊断...")
            try:
                # 检查Java进程
                import subprocess

                java_proc = subprocess.run(
                    ["ps", "-ef", "|", "grep", "java"],
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                if java_proc.stdout.strip():
                    logger.info(f"发现Java进程:\n{java_proc.stdout}")
                else:
                    logger.warning("未发现Java进程，这可能是Py4J连接失败的原因")

                # 检查网络端口
                net_ports = subprocess.run(
                    ["netstat", "-tulpn"], shell=True, capture_output=True, text=True
                )
                logger.info(f"网络端口状态:\n{net_ports.stdout}")

                logger.info(
                    "建议使用 docker_py4j_fix.py 脚本修复Docker容器内的Py4J连接问题"
                )
            except Exception as e:
                logger.warning(f"诊断Docker容器内Py4J连接问题时出错: {e}")

    return connection_ok
