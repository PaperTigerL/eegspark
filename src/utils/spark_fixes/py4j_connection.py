# -*- coding: utf-8 -*-
"""
Py4J连接修复模块

提供用于修复Py4J连接问题的工具函数，特别是在Docker容器环境中
"""

import os
import subprocess
import logging
from typing import Optional, List, Dict, Any

# 配置日志记录器
logger = logging.getLogger("Py4JConnectionFix")


def start_java_gateway_with_root(
    java_command: str, java_opts: Optional[List[str]] = None
) -> Optional[subprocess.Popen]:
    """
    使用root权限启动Java网关进程

    Args:
        java_command: Java命令路径
        java_opts: Java选项列表

    Returns:
        subprocess.Popen对象，如果启动失败则返回None
    """
    if java_opts is None:
        java_opts = []

    # 构建完整的命令
    full_command = ["sudo", java_command] + java_opts

    try:
        # 使用subprocess启动进程
        process = subprocess.Popen(
            full_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        logger.info(f"已使用root权限启动Java网关进程: PID={process.pid}")
        return process
    except Exception as e:
        logger.error(f"启动Java网关进程失败: {e}")
        return None


def configure_py4j_connection(gateway_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    配置Py4J连接参数，增强连接稳定性

    Args:
        gateway_params: 原始网关参数字典

    Returns:
        更新后的网关参数字典
    """
    # 复制原始参数
    updated_params = gateway_params.copy()

    # 增加连接超时时间
    updated_params["connect_timeout"] = 60.0

    # 增加读取超时时间
    if "read_timeout" in updated_params:
        updated_params["read_timeout"] = max(updated_params["read_timeout"], 300.0)
    else:
        updated_params["read_timeout"] = 300.0

    # 增加重试次数
    updated_params["auto_convert"] = True

    # 返回更新后的参数
    return updated_params
