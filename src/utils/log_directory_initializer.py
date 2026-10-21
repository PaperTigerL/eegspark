import os
import logging
import sys
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LogDirectoryInitializer")


class LogDirectoryInitializer:
    """日志目录初始化器

    确保日志目录存在并具有正确的权限，创建必要的日志文件
    """

    def __init__(self):
        """初始化日志目录初始化器"""
        # 可能的日志目录路径
        self.possible_log_dirs = [
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
            ),  # src/logs
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "logs",
            ),  # 项目根目录/logs
            "/app/src/logs",  # Docker容器内的应用程序目录
            "/tmp/logs",  # 临时目录
        ]

        # 默认日志文件列表
        self.default_log_files = [
            "app.log",  # 应用程序主日志
            "error.log",  # 错误日志
            "api_server.log",  # API服务器日志
            "websocket.log",  # WebSocket服务器日志
            "processor.log",  # 处理器日志
            "visualization.log",  # 可视化日志
            "svd_errors.log",  # SVD错误日志
            "mne_connectivity_errors.log",  # MNE连接性错误日志
            "py4j_connection_errors.log",  # Py4J连接错误日志
        ]

    def initialize(self):
        """初始化日志目录

        Returns:
            str: 成功初始化的日志目录路径，如果所有尝试都失败则返回None
        """
        initialized_dir = None

        for log_dir in self.possible_log_dirs:
            try:
                # 确保目录存在
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                    logger.info(f"已创建日志目录: {log_dir}")

                # 设置目录权限
                try:
                    os.chmod(log_dir, 0o777)  # 设置最宽松的权限
                    logger.info(f"已设置目录权限: {log_dir}")
                except (PermissionError, OSError) as e:
                    logger.warning(f"无法设置目录权限: {log_dir}, 错误: {e}")

                # 创建默认日志文件
                for log_file in self.default_log_files:
                    file_path = os.path.join(log_dir, log_file)
                    if not os.path.exists(file_path):
                        try:
                            # 创建空日志文件
                            with open(file_path, "w") as f:
                                pass

                            # 设置文件权限
                            os.chmod(file_path, 0o666)  # 设置最宽松的文件权限
                            logger.info(f"已创建日志文件: {file_path}")
                        except (PermissionError, OSError) as e:
                            logger.warning(f"无法创建日志文件: {file_path}, 错误: {e}")
                            continue

                # 测试目录是否可写
                test_file = os.path.join(log_dir, ".write_test")
                try:
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)

                    logger.info(f"成功初始化日志目录: {log_dir}")
                    initialized_dir = log_dir
                    break
                except (PermissionError, OSError) as e:
                    logger.warning(f"目录不可写: {log_dir}, 错误: {e}")
                    continue

            except (PermissionError, OSError) as e:
                logger.warning(f"无法初始化日志目录: {log_dir}, 错误: {e}")

        if initialized_dir is None:
            logger.error("所有日志目录初始化尝试均失败")

        return initialized_dir

    def ensure_log_file_exists(self, log_file_name):
        """确保指定的日志文件存在

        Args:
            log_file_name: 日志文件名

        Returns:
            str: 日志文件的完整路径，如果创建失败则返回None
        """
        for log_dir in self.possible_log_dirs:
            if os.path.exists(log_dir) and os.access(log_dir, os.W_OK):
                file_path = os.path.join(log_dir, log_file_name)
                try:
                    # 如果文件不存在，创建它
                    if not os.path.exists(file_path):
                        with open(file_path, "w") as f:
                            pass
                        os.chmod(file_path, 0o666)  # 设置最宽松的文件权限

                    # 测试文件是否可写
                    with open(file_path, "a") as f:
                        pass

                    return file_path
                except (PermissionError, OSError) as e:
                    logger.warning(f"无法创建或访问日志文件: {file_path}, 错误: {e}")
                    continue

        logger.error(f"无法为 {log_file_name} 创建日志文件")
        return None


# 如果直接运行此脚本，初始化日志目录
if __name__ == "__main__":
    initializer = LogDirectoryInitializer()
    log_dir = initializer.initialize()

    if log_dir:
        print(f"日志目录初始化成功: {log_dir}")
        sys.exit(0)
    else:
        print("日志目录初始化失败")
        sys.exit(1)
