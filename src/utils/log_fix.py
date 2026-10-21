#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志文件修复工具

此脚本用于检查和修复日志文件的权限问题，确保Docker容器内的进程有足够权限写入日志文件。
同时添加会话标识符到日志文件，以便区分不同运行时的日志。
"""

import os
import sys
import time
import datetime
import logging
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 设置日志格式
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LogFix")


def fix_log_permissions():
    """修复日志文件权限

    确保日志目录和文件具有正确的权限，使Docker容器内的进程能够写入日志
    """
    # 可能的日志目录路径
    possible_log_dirs = [
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

    fixed_dirs = []

    for log_dir in possible_log_dirs:
        if os.path.exists(log_dir):
            try:
                logger.info(f"检查日志目录: {log_dir}")

                # 确保目录有正确的权限
                os.chmod(log_dir, 0o777)  # 设置最宽松的权限

                # 检查目录中的所有日志文件
                for file_name in os.listdir(log_dir):
                    if file_name.endswith(".log"):
                        file_path = os.path.join(log_dir, file_name)
                        try:
                            # 设置文件权限
                            os.chmod(file_path, 0o666)  # 设置最宽松的文件权限
                            logger.info(f"已设置文件权限: {file_path}")

                            # 检查文件是否可写
                            with open(file_path, "a") as f:
                                pass
                        except (PermissionError, OSError) as e:
                            logger.warning(f"无法设置文件权限: {file_path}, 错误: {e}")

                # 测试目录是否可写
                test_file = os.path.join(log_dir, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)

                logger.info(f"成功修复日志目录权限: {log_dir}")
                fixed_dirs.append(log_dir)
            except (PermissionError, OSError) as e:
                logger.warning(f"无法修复日志目录权限: {log_dir}, 错误: {e}")

    return fixed_dirs


def add_session_identifier():
    """添加会话标识符到日志文件

    在每次运行时添加唯一的会话标识符，以便区分不同运行时的日志
    """
    # 可能的日志目录路径
    possible_log_dirs = [
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

    # 生成会话标识符
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_marker = f"\n\n{'='*80}\n会话开始: {session_id}\n{'='*80}\n\n"

    for log_dir in possible_log_dirs:
        if os.path.exists(log_dir):
            try:
                # 检查目录中的所有日志文件
                for file_name in os.listdir(log_dir):
                    if file_name.endswith(".log"):
                        file_path = os.path.join(log_dir, file_name)
                        try:
                            # 添加会话标识符
                            with open(file_path, "a", encoding="utf-8") as f:
                                f.write(session_marker)
                                f.flush()  # 立即刷新缓冲区
                                os.fsync(f.fileno())  # 强制操作系统将数据写入磁盘
                            logger.info(f"已添加会话标识符到: {file_path}")
                        except (PermissionError, OSError) as e:
                            logger.warning(
                                f"无法添加会话标识符: {file_path}, 错误: {e}"
                            )
            except (PermissionError, OSError) as e:
                logger.warning(f"无法访问日志目录: {log_dir}, 错误: {e}")


def check_docker_environment():
    """检查是否在Docker容器内运行

    Returns:
        bool: 是否在Docker容器内运行
    """
    # 检查是否存在Docker环境标志文件
    docker_env_files = ["/.dockerenv", "/proc/1/cgroup"]
    for file_path in docker_env_files:
        if os.path.exists(file_path):
            if file_path == "/proc/1/cgroup":
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                        if "docker" in content:
                            return True
                except:
                    pass
            else:
                return True

    return False


def fix_docker_log_permissions():
    """修复Docker容器内的日志权限问题

    在Docker容器内运行时，使用特殊的权限修复方法
    """
    if not check_docker_environment():
        logger.info("未检测到Docker环境，跳过Docker特定的权限修复")
        return

    logger.info("检测到Docker环境，应用特定的权限修复")

    # Docker容器内的日志目录
    docker_log_dirs = ["/app/src/logs", "/tmp/logs"]

    for log_dir in docker_log_dirs:
        if os.path.exists(log_dir):
            try:
                # 在Docker容器内使用更强的权限设置
                subprocess.run(["chmod", "-R", "777", log_dir], check=True)
                logger.info(f"已设置Docker容器内日志目录权限: {log_dir}")

                # 确保目录所有者正确
                subprocess.run(["chown", "-R", "root:root", log_dir], check=True)
                logger.info(f"已设置Docker容器内日志目录所有者: {log_dir}")
            except (subprocess.SubprocessError, OSError) as e:
                logger.warning(
                    f"无法设置Docker容器内日志目录权限: {log_dir}, 错误: {e}"
                )


def ensure_log_directories():
    """确保日志目录存在

    创建必要的日志目录，如果它们不存在
    """
    # 可能的日志目录路径
    possible_log_dirs = [
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

    created_dirs = []

    for log_dir in possible_log_dirs:
        try:
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"确保日志目录存在: {log_dir}")
            created_dirs.append(log_dir)
        except (PermissionError, OSError) as e:
            logger.warning(f"无法创建日志目录: {log_dir}, 错误: {e}")

    return created_dirs


def main():
    """主函数

    执行日志修复流程
    """
    logger.info("开始日志文件修复流程")

    # 确保日志目录存在
    ensure_log_directories()

    # 修复日志文件权限
    fixed_dirs = fix_log_permissions()

    # 如果在Docker容器内运行，应用特定的权限修复
    fix_docker_log_permissions()

    # 添加会话标识符
    add_session_identifier()

    if fixed_dirs:
        logger.info(f"日志文件修复完成，已修复目录: {', '.join(fixed_dirs)}")
    else:
        logger.warning("未能修复任何日志目录，请检查权限和路径")


if __name__ == "__main__":
    main()
