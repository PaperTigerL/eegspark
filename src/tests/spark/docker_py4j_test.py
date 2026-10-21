#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docker容器内Py4J连接测试脚本

此脚本专门用于在Docker容器内测试Py4J连接问题
并提供详细的诊断信息和修复步骤
"""

import os
import sys
import time
import logging
import subprocess
import traceback
import socket

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DockerPy4JTest")

# 导入修复模块
from utils.spark_fixes.ivy_config import fix_ivy_repository_path
from visualization.spark_connection_fix import (
    fix_visualization_spark_connection,
    ensure_spark_connection,
)


def check_java_process():
    """检查Java进程状态

    检查Docker容器内是否有Java进程在运行，以及它们的监听端口

    Returns:
        bool: 是否有Java进程在运行
    """
    try:
        logger.info("检查Java进程状态...")

        # 使用ps命令检查Java进程
        result = subprocess.run(
            ["ps", "-ef", "|", "grep", "java"],
            shell=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warning(f"检查Java进程失败: {result.stderr}")

        java_processes = result.stdout.strip()
        if java_processes:
            logger.info(f"发现Java进程:\n{java_processes}")

            # 检查Java进程的网络连接
            logger.info("检查Java进程的网络连接...")
            net_result = subprocess.run(
                ["netstat", "-tulpn", "|", "grep", "java"],
                shell=True,
                capture_output=True,
                text=True,
            )

            if net_result.stdout.strip():
                logger.info(f"Java进程的网络连接:\n{net_result.stdout}")
            else:
                logger.warning("未发现Java进程的网络连接")

            return True
        else:
            logger.warning("未发现Java进程")
            return False
    except Exception as e:
        logger.error(f"检查Java进程时出错: {e}")
        return False


def check_py4j_gateway():
    """检查Py4J网关连接

    尝试直接连接Py4J网关端口，检查网络连接是否正常

    Returns:
        bool: 连接是否成功
    """
    try:
        logger.info("检查Py4J网关连接...")

        # 尝试连接常见的Py4J网关端口
        gateway_ports = [25333, 25334, 25335, 8880, 8881, 8882, 8883, 8884, 8885]

        for port in gateway_ports:
            try:
                logger.info(f"尝试连接Py4J网关端口 {port}...")
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)  # 设置超时时间
                result = s.connect_ex(("localhost", port))
                s.close()

                if result == 0:
                    logger.info(f"成功连接到Py4J网关端口 {port}")
                    return True
            except Exception as e:
                logger.warning(f"连接端口 {port} 失败: {e}")

        logger.warning("未能连接到任何Py4J网关端口")
        return False
    except Exception as e:
        logger.error(f"检查Py4J网关连接时出错: {e}")
        return False


def test_docker_py4j_connection():
    """在Docker容器内测试Py4J连接

    全面测试Docker容器内的Py4J连接问题，并提供详细的诊断信息

    Returns:
        bool: 测试是否成功
    """
    try:
        logger.info("开始在Docker容器内测试Py4J连接...")

        # 检查是否在Docker容器内运行
        if not os.path.exists("/.dockerenv"):
            logger.warning("此脚本应在Docker容器内运行!")
            print(
                "\n⚠️ 此脚本应在Docker容器内运行! 请使用 'docker exec -it spark-worker python /app/src/tests/spark/docker_py4j_test.py'\n"
            )
            return False

        logger.info("确认在Docker容器内运行")

        # 首先修复Ivy配置
        logger.info("正在修复Ivy配置...")
        ivy_result = fix_ivy_repository_path()
        if not ivy_result:
            logger.error("Ivy配置修复失败")
        else:
            logger.info("Ivy配置修复成功")

        # 检查Java进程状态
        java_running = check_java_process()
        if not java_running:
            logger.warning("Java进程未运行，这可能导致Py4J连接问题")

        # 检查Py4J网关连接
        gateway_ok = check_py4j_gateway()
        if not gateway_ok:
            logger.warning("Py4J网关连接失败，这可能导致Py4J连接问题")

        # 然后修复可视化模块的Spark连接
        logger.info("正在修复可视化模块的Spark连接...")
        connection_result = fix_visualization_spark_connection()
        if not connection_result:
            logger.error("可视化模块的Spark连接修复失败")
            return False
        logger.info("可视化模块的Spark连接修复成功")

        # 最后测试Spark连接
        logger.info("正在测试Spark连接...")
        connection_test = ensure_spark_connection()
        if not connection_test:
            logger.error("Spark连接测试失败")
            return False
        logger.info("Spark连接测试成功")

        return True
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.error("详细错误信息:", exc_info=True)
        return False


def main():
    logger.info("开始Docker容器内Py4J连接测试...")
    success = test_docker_py4j_connection()

    if success:
        logger.info("Docker容器内Py4J连接测试成功!")
        print("\n✅ Docker容器内Py4J连接测试成功! 可视化服务器应能正常工作。\n")
        return 0
    else:
        logger.error("Docker容器内Py4J连接测试失败!")
        print("\n❌ Docker容器内Py4J连接测试失败! 请检查日志获取详细信息。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
