#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试可视化模块的Py4J连接修复

此脚本用于测试可视化模块的Py4J连接修复是否有效
"""

import os
import sys
import logging
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VisualizationPy4JTest")

# 导入修复模块
from utils.spark_fixes.ivy_config import fix_ivy_repository_path
from visualization.spark_connection_fix import (
    fix_visualization_spark_connection,
    ensure_spark_connection,
)


def test_visualization_py4j_fix():
    """
    测试可视化模块的Py4J连接修复

    Returns:
        bool: 测试是否成功
    """
    try:
        logger.info("开始测试可视化模块的Py4J连接修复...")

        # 首先修复Ivy配置
        logger.info("正在修复Ivy配置...")
        ivy_result = fix_ivy_repository_path()
        if not ivy_result:
            logger.error("Ivy配置修复失败")
            return False
        logger.info("Ivy配置修复成功")

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
    logger.info("开始测试可视化模块的Py4J连接修复...")
    success = test_visualization_py4j_fix()

    if success:
        logger.info("可视化模块的Py4J连接修复测试成功!")
        print("\n✅ 可视化模块的Py4J连接修复测试成功! 可视化服务器应能正常工作。\n")
        return 0
    else:
        logger.error("可视化模块的Py4J连接修复测试失败!")
        print("\n❌ 可视化模块的Py4J连接修复测试失败! 请检查日志获取详细信息。\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
