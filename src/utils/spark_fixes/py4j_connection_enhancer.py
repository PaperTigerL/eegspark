#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Py4J连接增强脚本

此脚本用于增强Py4J连接的稳定性，解决ConnectionRefusedError问题
"""

import os
import sys
import time
import logging
import subprocess
import socket
import gc

# 全局变量，用于存储成功连接的Java网关服务器主机地址
_gateway_host = "java-gateway"  # 默认值为java-gateway容器

# 在模块加载时自动设置环境变量
os.environ["JAVA_GATEWAY_HOST"] = _gateway_host
os.environ["SPARK_DRIVER_HOST"] = os.environ.get("SPARK_DRIVER_HOST", "spark-worker")
os.environ["SPARK_LOCAL_IP"] = os.environ.get("SPARK_LOCAL_IP", "spark-worker")

# 配置日志
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

# 创建日志记录器
logger = logging.getLogger("Py4JConnectionEnhancer")
logger.setLevel(logging.INFO)

# 移除所有现有处理器，防止重复日志
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 添加文件处理器
try:
    os.chmod(log_dir, 0o777)
    log_file_path = os.path.join(log_dir, "py4j_connection_enhancer.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
except Exception as e:
    # 使用标准输出
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(stream_handler)


def check_java_gateway_server(port=None, max_retries=3):
    """
    检查Java网关服务器是否可访问

    Args:
        port: Java网关服务器端口，如果为None则使用环境变量中的配置
        max_retries: 最大重试次数

    Returns:
        bool: 服务器是否可访问
    """
    global _gateway_host

    # 如果未指定端口，尝试从环境变量获取
    if port is None:
        port = int(os.environ.get("SPARK_PY4J_GATEWAY_PORT", "25333"))

    # 检查Java网关服务器是否可访问
    host = _gateway_host
    for i in range(max_retries):
        try:
            # 尝试连接到Java网关服务器
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            result = s.connect_ex((host, port))
            s.close()

            if result == 0:
                logger.info(f"Java网关服务器 {host}:{port} 可用")
                return True

            # 如果连接失败，等待一段时间后重试
            logger.warning(
                f"Java网关服务器 {host}:{port} 不可用，尝试重试 {i+1}/{max_retries}"
            )
            time.sleep(1)
        except Exception as e:
            logger.error(f"检查Java网关服务器时出错: {e}")
            time.sleep(1)

    logger.error(
        f"Java网关服务器 {host}:{port} 不可用，请确保java-gateway容器已启动并运行"
    )
    return False


def enhance_py4j_connection():
    """
    增强Py4J连接，确保在Docker环境中正确设置网关主机

    Returns:
        bool: 增强是否成功
    """
    try:
        # 检查Java网关服务器是否可访问
        if not check_java_gateway_server():
            return False

        # 应用Py4J连接补丁
        patch_result = patch_py4j_connection()
        if not patch_result:
            return False

        return True
    except Exception as e:
        logger.error(f"增强Py4J连接时出错: {e}")
        return False


def patch_py4j_connection():
    """
    修补Py4J连接，确保所有连接使用正确的主机地址

    Returns:
        bool: 修补是否成功
    """
    try:
        global _gateway_host
        host = _gateway_host
        port = int(os.environ.get("SPARK_PY4J_GATEWAY_PORT", "25333"))

        # 修补py4j库的内部连接创建函数
        import py4j.java_gateway
        from py4j.java_gateway import JavaGateway, GatewayClient, GatewayParameters

        # 设置默认网关参数
        py4j.java_gateway._DEFAULT_GATEWAY_PARAMETERS = GatewayParameters(
            address=host, port=port
        )

        logger.info(f"已设置Py4J默认网关参数: 主机={host}, 端口={port}")
        return True
    except Exception as e:
        logger.error(f"修补Py4J连接时出错: {e}")
        return False


if __name__ == "__main__":
    # 增强Py4J连接
    success = enhance_py4j_connection()

    if success:
        print("\n✅ Py4J连接增强成功! Java网关服务器应能正常工作。")
    else:
        print("\n❌ Py4J连接增强失败! 请检查日志获取详细信息。")
