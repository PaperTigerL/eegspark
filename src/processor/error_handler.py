# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet

eventlet.monkey_patch()

import os
import logging
import traceback
import numpy as np
from typing import Dict, Any, Optional
import sys

# 添加common目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "common"))
from common.error_logger import error_logger


class EEGErrorHandler:
    """EEG处理错误处理器

    专门用于处理EEG数据处理过程中的常见错误，如SVD不收敛和MNE连接性错误，
    并提供详细的错误日志记录和恢复策略。
    """

    def __init__(self, log_dir: str = None):
        """初始化错误处理器

        Args:
            log_dir: 日志保存目录，如果为None，则使用默认目录
        """
        # 使用全局错误日志记录器
        self.error_logger = error_logger

        # 设置日志
        self.logger = logging.getLogger("EEGErrorHandler")

        # 尝试多个可能的日志目录位置
        if log_dir is None:
            # 可能的日志目录路径
            possible_log_dirs = [
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    ),
                    "logs",
                ),  # 项目根目录/logs
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
                ),  # src/logs
                "/app/src/logs",  # Docker容器内的应用程序目录
                "/tmp/logs",  # 临时目录
            ]

            # 尝试每个可能的目录
            for dir_path in possible_log_dirs:
                try:
                    # 确保目录存在
                    os.makedirs(dir_path, exist_ok=True)

                    # 尝试设置目录权限
                    try:
                        os.chmod(dir_path, 0o777)  # 设置最宽松的权限
                    except (PermissionError, OSError) as e:
                        self.logger.warning(f"无法设置目录权限: {dir_path}, 错误: {e}")

                    # 测试目录是否可写
                    test_file = os.path.join(dir_path, ".write_test")
                    try:
                        with open(test_file, "w") as f:
                            f.write("test")
                        os.remove(test_file)

                        self.log_dir = dir_path
                        self.logger.info(f"成功初始化日志目录: {dir_path}")
                        break
                    except (PermissionError, OSError) as e:
                        self.logger.warning(f"目录不可写: {dir_path}, 错误: {e}")
                        continue
                except (PermissionError, OSError) as e:
                    self.logger.warning(f"无法初始化日志目录: {dir_path}, 错误: {e}")
            else:  # 如果所有目录都失败
                # 使用当前目录作为最后的备选
                self.log_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "logs"
                )
                try:
                    os.makedirs(self.log_dir, exist_ok=True)
                    self.logger.warning(f"使用备选日志目录: {self.log_dir}")
                except (PermissionError, OSError) as e:
                    self.logger.error(f"无法创建任何日志目录，将使用内存日志: {e}")
        else:
            self.log_dir = log_dir
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                # 尝试设置目录权限
                try:
                    os.chmod(self.log_dir, 0o777)  # 设置最宽松的权限
                except (PermissionError, OSError) as e:
                    self.logger.warning(f"无法设置目录权限: {self.log_dir}, 错误: {e}")
            except (PermissionError, OSError) as e:
                self.logger.error(f"无法创建指定的日志目录: {self.log_dir}, 错误: {e}")
                # 使用备选目录
                self.log_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "logs"
                )
                os.makedirs(self.log_dir, exist_ok=True)

    def handle_svd_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理SVD不收敛错误

        Args:
            error: 捕获的异常
            context: 错误发生时的上下文信息，如数据形状、处理方法等

        Returns:
            Dict: 处理结果，包含成功/失败标志和备用结果
        """
        error_msg = str(error)
        self.logger.error(f"SVD不收敛错误: {error_msg}")

        # 记录详细错误信息
        details = {
            "error_type": "SVD did not converge",
            "error_message": error_msg,
            "traceback": traceback.format_exc(),
        }

        # 添加上下文信息
        if context:
            details.update(context)

        # 记录到专门的SVD错误日志
        self.error_logger.log_svd_error(error_msg, details)

        # 返回错误处理结果
        return {
            "success": False,
            "error": error_msg,
            "details": details,
            "recovery_attempted": True,
        }

    def handle_mne_connectivity_error(
        self, error: Exception, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理MNE连接性分析错误

        Args:
            error: 捕获的异常
            context: 错误发生时的上下文信息，如连接方法、频段等

        Returns:
            Dict: 处理结果，包含成功/失败标志和备用结果
        """
        error_msg = str(error)
        self.logger.error(f"MNE连接性分析错误: {error_msg}")

        # 记录详细错误信息
        details = {
            "error_type": "MNE Connectivity Error",
            "error_message": error_msg,
            "traceback": traceback.format_exc(),
        }

        # 添加上下文信息
        if context:
            details.update(context)

        # 记录到专门的MNE连接性错误日志
        self.error_logger.log_mne_connectivity_error(error_msg, details)

        # 返回错误处理结果
        return {
            "success": False,
            "error": error_msg,
            "details": details,
            "recovery_attempted": True,
        }

    def handle_general_error(
        self, error: Exception, error_type: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """处理一般性错误

        Args:
            error: 捕获的异常
            error_type: 错误类型标识
            context: 错误发生时的上下文信息

        Returns:
            Dict: 处理结果，包含成功/失败标志
        """
        error_msg = str(error)
        self.logger.error(f"{error_type}错误: {error_msg}")

        # 记录详细错误信息
        details = {
            "error_type": error_type,
            "error_message": error_msg,
            "traceback": traceback.format_exc(),
        }

        # 添加上下文信息
        if context:
            details.update(context)

        # 记录到通用错误日志
        self.error_logger.log_error(error_type.lower(), error_msg, details)

        # 返回错误处理结果
        return {"success": False, "error": error_msg, "details": details}


# 导入日志目录初始化器
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from utils.log_directory_initializer import LogDirectoryInitializer

# 初始化日志目录
log_initializer = LogDirectoryInitializer()
initialized_log_dir = log_initializer.initialize()

# 创建全局错误处理器实例
error_handler = EEGErrorHandler(log_dir=initialized_log_dir)


# 辅助函数：检测是否为SVD不收敛错误
def is_svd_convergence_error(error: Exception) -> bool:
    """检测是否为SVD不收敛错误

    Args:
        error: 捕获的异常

    Returns:
        bool: 是否为SVD不收敛错误
    """
    error_str = str(error)
    return (
        isinstance(error, np.linalg.LinAlgError) and "SVD did not converge" in error_str
    )


# 辅助函数：检测是否为MNE连接性错误
def is_mne_connectivity_error(error: Exception) -> bool:
    """检测是否为MNE连接性错误

    Args:
        error: 捕获的异常

    Returns:
        bool: 是否为MNE连接性错误
    """
    error_str = str(error)
    return (
        "mne_connectivity" in error_str.lower()
        or "connectivity" in error_str.lower()
        and "mne" in error_str.lower()
    )
