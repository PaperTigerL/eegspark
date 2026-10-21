#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docker环境中的Py4J连接修复

此脚本用于检查Java网关服务器连接，确保能正确连接到java-gateway容器
"""

import os
import sys
import time
import socket
import logging

# 配置日志
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

# 创建日志记录器
logger = logging.getLogger("DockerPy4JFix")
logger.setLevel(logging.INFO)

# 移除所有现有处理器，防止重复日志
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 添加文件处理器
try:
    os.chmod(log_dir, 0o777)
    log_file_path = os.path.join(log_dir, "docker_py4j_fix.log")
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
    # 设置Java临时目录
    java_tmpdir = "/tmp/java_tmpdir"
    os.makedirs(java_tmpdir, exist_ok=True)
    os.environ["JAVA_IO_TMPDIR"] = java_tmpdir
    os.environ["_JAVA_OPTIONS"] = f"-Djava.io.tmpdir={java_tmpdir}"

    # 设置Py4J连接超时
    os.environ["PY4J_GATEWAY_TIMEOUT"] = "60"

    # 如果未指定端口，尝试从环境变量获取
    if port is None:
        port = int(os.environ.get("SPARK_PY4J_GATEWAY_PORT", "25333"))

    # 检查Java网关服务器是否可访问
    host = "java-gateway"  # 使用Docker容器名称作为主机名
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


if __name__ == "__main__":
    # 检查Java网关服务器
    success = check_java_gateway_server()

    if success:
        print("\n✅ Java网关服务器检查成功! 可以正常连接。")
    else:
        print("\n❌ Java网关服务器检查失败! 请检查日志获取详细信息。")
