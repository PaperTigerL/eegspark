# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet

eventlet.monkey_patch()

import os
import logging
import traceback
import datetime
from typing import Dict, Any, Optional


class ErrorLogger:
    """专门用于记录和管理EEG处理过程中的错误

    这个类提供了一种统一的方式来记录、保存和查询处理过程中的错误，
    特别是针对SVD不收敛和MNE连接性分析错误等常见问题。

    特性：
    - 使用覆盖模式(mode='w')创建日志文件，每次运行时创建新的日志文件
    - 可选创建带时间戳的备份日志文件，保留历史记录
    - 支持多种错误类型的专门日志文件
    - 自动处理权限和路径问题
    """

    def __init__(
        self,
        log_dir: str = None,
        create_backup: bool = True,
        overwrite_mode: bool = True,
    ):
        """初始化错误日志记录器

        Args:
            log_dir: 日志保存目录，如果为None，则使用默认目录
            create_backup: 是否创建带时间戳的备份日志文件，默认为True
            overwrite_mode: 是否使用覆盖模式创建日志文件，默认为True
        """
        # 是否创建带时间戳的备份日志文件
        self.create_backup = create_backup
        # 强制使用覆盖模式
        self.overwrite_mode = True

        # 导入日志目录初始化器
        try:
            import sys
            import os

            sys.path.append(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils")
            )
            from utils.log_directory_initializer import LogDirectoryInitializer

            # 使用日志目录初始化器确保日志目录存在
            log_initializer = LogDirectoryInitializer()
            initialized_log_dir = log_initializer.initialize()

            # 如果初始化成功，使用初始化后的目录
            if initialized_log_dir is not None:
                log_dir = initialized_log_dir
        except ImportError:
            print("警告: 无法导入日志目录初始化器，将尝试直接使用指定的日志目录")
        except Exception as e:
            print(f"警告: 日志目录初始化失败: {e}")

        # 设置日志目录
        if log_dir is None:
            # 优先使用指定的目录
            host_log_dir = "d:\\file\\eeg\\EEGSpark-patch-1\\temp_project\\src\\logs"
            if os.path.exists(host_log_dir) or os.path.exists(
                os.path.dirname(host_log_dir)
            ):
                possible_dirs = [host_log_dir]
            else:
                # 尝试在几个可能有写入权限的位置创建日志目录
                possible_dirs = [
                    os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "logs",
                    ),  # 优先使用src/logs目录
                    os.path.join(
                        os.path.dirname(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        ),
                        "logs",
                    ),  # 项目根目录下的logs
                    "/tmp/logs",  # 临时目录，通常有写入权限
                    "/app/src/logs",  # 应用程序目录下
                    os.path.join(os.getcwd(), "logs"),  # 当前工作目录
                ]

            for dir_path in possible_dirs:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    # 测试目录是否可写
                    test_file_path = os.path.join(dir_path, ".write_test")
                    try:
                        with open(test_file_path, "w") as f:
                            f.write("test")
                        os.remove(test_file_path)  # 清理测试文件
                        self.log_dir = dir_path
                        print(f"成功创建日志目录并验证可写: {dir_path}")
                        break
                    except (PermissionError, OSError) as write_err:
                        print(f"日志目录 {dir_path} 存在但不可写: {write_err}")
                        continue
                except (PermissionError, OSError) as e:
                    print(f"无法创建日志目录 {dir_path}: {e}")
                    continue
            else:  # 如果所有目录都失败
                # 使用当前目录作为最后的备选
                self.log_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "logs"
                )
                try:
                    os.makedirs(self.log_dir, exist_ok=True)
                    # 测试目录是否可写
                    test_file_path = os.path.join(self.log_dir, ".write_test")
                    try:
                        with open(test_file_path, "w") as f:
                            f.write("test")
                        os.remove(test_file_path)  # 清理测试文件
                        print(f"成功创建备选日志目录并验证可写: {self.log_dir}")
                    except (PermissionError, OSError):
                        # 如果不可写，使用内存日志
                        print(f"警告: 备选日志目录不可写，将使用内存日志")
                        self.log_dir = None
                except (PermissionError, OSError) as e:
                    # 如果仍然失败，使用内存日志（不写入文件）
                    print(f"警告: 无法创建日志目录，将使用内存日志: {e}")
                    self.log_dir = None
        else:
            self.log_dir = log_dir
            try:
                # 确保日志目录存在
                os.makedirs(self.log_dir, exist_ok=True)
                # 测试目录是否可写
                test_file_path = os.path.join(self.log_dir, ".write_test")
                try:
                    with open(test_file_path, "w") as f:
                        f.write("test")
                    os.remove(test_file_path)  # 清理测试文件
                except (PermissionError, OSError):
                    print(f"警告: 指定的日志目录 {self.log_dir} 不可写，将使用内存日志")
                    self.log_dir = None
            except (PermissionError, OSError) as e:
                print(f"警告: 无法创建指定的日志目录 {self.log_dir}: {e}")
                self.log_dir = None

        # 创建专门的错误日志记录器
        self.logger = logging.getLogger("ErrorLogger")
        self.logger.setLevel(logging.DEBUG)

        # 设置格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 检查是否有可用的日志目录
        if self.log_dir is not None:
            try:
                # 创建文件处理器，根据配置使用覆盖模式或追加模式
                error_log_path = os.path.join(self.log_dir, "error.log")

                # 如果需要创建备份，先将现有日志文件复制到带时间戳的备份文件
                if (
                    self.create_backup
                    and os.path.exists(error_log_path)
                    and os.path.getsize(error_log_path) > 0
                ):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = os.path.join(self.log_dir, f"error_{timestamp}.log")
                    try:
                        with open(error_log_path, "r", encoding="utf-8") as src, open(
                            backup_path, "w", encoding="utf-8"
                        ) as dst:
                            dst.write(src.read())
                        print(f"已创建错误日志备份: {backup_path}")
                    except Exception as e:
                        print(f"创建错误日志备份失败: {e}")

                # 根据配置使用覆盖模式或追加模式
                file_mode = "w" if self.overwrite_mode else "a"
                file_handler = logging.FileHandler(
                    error_log_path, mode=file_mode, encoding="utf-8"
                )
                file_handler.setLevel(logging.ERROR)
                file_handler.setFormatter(formatter)

                # 添加处理器
                if not self.logger.handlers:
                    self.logger.addHandler(file_handler)

                # 创建专门的SVD错误日志文件
                self.svd_log_path = os.path.join(self.log_dir, "svd_errors.log")
                # 初始化SVD错误日志文件（根据配置使用覆盖模式或追加模式）
                try:
                    # 根据配置使用覆盖模式或追加模式
                    file_mode = "w" if self.overwrite_mode else "a"
                    with open(self.svd_log_path, file_mode, encoding="utf-8") as f:
                        f.write(
                            f"# SVD错误日志 - 创建于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        )
                    print(f"成功创建SVD错误日志文件: {self.svd_log_path}")
                except (PermissionError, OSError, IOError) as e:
                    print(f"警告: 无法创建SVD错误日志文件: {e}")

                # 创建专门的MNE连接性错误日志文件
                self.mne_connectivity_log_path = os.path.join(
                    self.log_dir, "mne_connectivity_errors.log"
                )
                # 初始化MNE连接性错误日志文件（根据配置使用覆盖模式或追加模式）
                try:
                    # 根据配置使用覆盖模式或追加模式
                    file_mode = "w" if self.overwrite_mode else "a"
                    with open(
                        self.mne_connectivity_log_path, file_mode, encoding="utf-8"
                    ) as f:
                        f.write(
                            f"# MNE连接性错误日志 - 创建于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        )
                    print(
                        f"成功创建MNE连接性错误日志文件: {self.mne_connectivity_log_path}"
                    )
                except (PermissionError, OSError, IOError) as e:
                    print(f"警告: 无法创建MNE连接性错误日志文件: {e}")
            except (PermissionError, OSError) as e:
                print(f"警告: 无法创建日志文件: {e}")
                self.log_dir = None

        # 如果没有可用的日志目录，使用控制台日志
        if self.log_dir is None:
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(formatter)

            # 添加处理器
            if not self.logger.handlers:
                self.logger.addHandler(console_handler)

            # 设置内存日志路径
            self.svd_log_path = None
            self.mne_connectivity_log_path = None
            print("警告: 将使用控制台输出代替文件日志")

        # 错误计数器
        self.error_counts = {
            "svd": 0,
            "mne_connectivity": 0,
            "py4j_connection": 0,
            "other": 0,
        }

        # 创建专门的Py4J连接错误日志文件
        if self.log_dir is not None:
            try:
                self.py4j_log_path = os.path.join(
                    self.log_dir, "py4j_connection_errors.log"
                )
                # 初始化Py4J连接错误日志文件（根据配置使用覆盖模式或追加模式）
                # 根据配置使用覆盖模式或追加模式
                file_mode = "w" if self.overwrite_mode else "a"
                with open(self.py4j_log_path, file_mode, encoding="utf-8") as f:
                    f.write(
                        f"# Py4J连接错误日志 - 创建于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    )
                print(f"成功创建Py4J连接错误日志文件: {self.py4j_log_path}")
            except (PermissionError, OSError, IOError) as e:
                print(f"警告: 无法创建Py4J连接错误日志文件: {e}")
                self.py4j_log_path = None
        else:
            self.py4j_log_path = None

    def log_error(
        self,
        error_type: str,
        error_message: str,
        details: Dict[str, Any] = None,
        stack_trace: bool = True,
    ):
        """记录错误信息

        Args:
            error_type: 错误类型，如'svd'、'mne_connectivity'等
            error_message: 错误消息
            details: 错误详情，包含上下文信息
            stack_trace: 是否记录堆栈跟踪信息
        """
        # 更新错误计数
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
        else:
            self.error_counts["other"] += 1

        # 获取当前时间
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建错误日志内容
        log_content = f"[{timestamp}] {error_type.upper()} ERROR: {error_message}\n"

        # 添加详细信息
        if details:
            log_content += "详细信息:\n"
            for key, value in details.items():
                log_content += f"  {key}: {value}\n"

        # 添加堆栈跟踪
        if stack_trace:
            log_content += "堆栈跟踪:\n"
            # 检查是否有实际的异常堆栈
            stack_info = traceback.format_exc()
            if stack_info and stack_info.strip() != "NoneType: None":
                log_content += stack_info
            else:
                log_content += "[无异常堆栈信息]\n"

        log_content += "\n" + "-" * 80 + "\n"

        # 记录到通用错误日志
        self.logger.error(error_message)

        # 只有当日志目录存在时才写入文件
        if self.log_dir is not None:
            try:
                # 根据错误类型记录到专门的日志文件
                if error_type == "svd" and self.svd_log_path is not None:
                    try:
                        with open(
                            self.svd_log_path, "a", encoding="utf-8"
                        ) as f:  # 使用追加模式
                            f.write(log_content)
                            f.flush()  # 立即刷新缓冲区，确保内容写入文件
                            os.fsync(f.fileno())  # 强制操作系统将数据写入磁盘
                        print(f"已记录SVD错误到日志文件: {self.svd_log_path}")
                    except (PermissionError, OSError, IOError) as e:
                        print(f"警告: 无法写入SVD错误日志文件: {e}")
                        print(log_content)
                elif (
                    error_type == "mne_connectivity"
                    and self.mne_connectivity_log_path is not None
                ):
                    try:
                        with open(
                            self.mne_connectivity_log_path, "a", encoding="utf-8"
                        ) as f:  # 使用追加模式
                            f.write(log_content)
                            f.flush()  # 立即刷新缓冲区，确保内容写入文件
                            os.fsync(f.fileno())  # 强制操作系统将数据写入磁盘
                        print(
                            f"已记录MNE连接性错误到日志文件: {self.mne_connectivity_log_path}"
                        )
                    except (PermissionError, OSError, IOError) as e:
                        print(f"警告: 无法写入MNE连接性错误日志文件: {e}")
                        print(log_content)
            except (PermissionError, OSError, IOError) as e:
                # 如果写入失败，记录到控制台
                print(f"警告: 无法写入错误日志文件: {e}")
                print(log_content)

    def log_svd_error(self, error_message: str, details: Dict[str, Any] = None):
        """记录SVD计算错误

        Args:
            error_message: 错误消息
            details: 错误详情，如数据形状、方法等
        """
        self.log_error("svd", error_message, details)

    def log_mne_connectivity_error(
        self, error_message: str, details: Dict[str, Any] = None
    ):
        """记录MNE连接性分析错误

        Args:
            error_message: 错误消息
            details: 错误详情，如连接方法、频段等
        """
        self.log_error("mne_connectivity", error_message, details)

    def log_py4j_connection_error(
        self, error_message: str, details: Dict[str, Any] = None
    ):
        """记录Py4J连接错误

        Args:
            error_message: 错误消息
            details: 错误详情，如连接尝试次数、错误类型等
        """
        # 记录到通用错误日志和专门的Py4J连接错误日志
        self.log_error("py4j_connection", error_message, details)

        # 确保错误也被记录到主错误日志文件
        if self.log_dir is not None:
            try:
                error_log_path = os.path.join(self.log_dir, "error.log")
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 构建错误日志内容
                log_content = f"{timestamp} - ErrorLogger - ERROR - Py4J连接错误: {error_message}\n"

                # 添加详细信息
                if details:
                    for key, value in details.items():
                        log_content += f"  {key}: {value}\n"

                # 添加堆栈跟踪
                stack_info = traceback.format_exc()
                if stack_info and stack_info.strip() != "NoneType: None":
                    log_content += stack_info

                # 写入到错误日志文件（如果文件不存在则创建，如果已存在则追加）
                # 这里使用追加模式是因为主错误日志文件已经在初始化时被覆盖
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(log_content + "\n")
                    f.flush()  # 立即刷新缓冲区，确保内容写入文件
                    os.fsync(f.fileno())  # 强制操作系统将数据写入磁盘
            except Exception as e:
                print(f"警告: 无法将Py4J连接错误写入主错误日志文件: {e}")

    def get_error_summary(self) -> Dict[str, int]:
        """获取错误统计摘要

        Returns:
            Dict: 各类错误的计数
        """
        return self.error_counts

    def get_recent_errors(
        self, error_type: Optional[str] = None, count: int = 10
    ) -> str:
        """获取最近的错误记录

        Args:
            error_type: 错误类型，如果为None则获取所有类型
            count: 返回的错误数量

        Returns:
            str: 最近的错误记录
        """
        # 检查是否有日志目录
        if self.log_dir is None:
            return "日志系统使用内存模式，无法获取历史错误记录"

        # 确定日志文件路径
        if error_type == "svd":
            log_path = self.svd_log_path
        elif error_type == "mne_connectivity":
            log_path = self.mne_connectivity_log_path
        elif error_type == "py4j_connection":
            log_path = self.py4j_log_path
        else:
            # 通用错误日志
            log_path = os.path.join(self.log_dir, "error.log")

        # 检查日志文件是否存在
        if log_path is None or not os.path.exists(log_path):
            return "没有找到错误日志文件"

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return "错误日志文件为空"

            # 按错误分隔
            errors = content.split("-" * 80)

            # 获取最近的count个错误
            recent_errors = errors[-count - 1 : -1] if len(errors) > count else errors

            return "\n".join(recent_errors)
        except (PermissionError, OSError, IOError) as e:
            return f"无法访问错误日志文件: {e}"
        except Exception as e:
            return f"读取错误日志失败: {e}"


# 创建全局错误日志记录器实例
error_logger = ErrorLogger()
