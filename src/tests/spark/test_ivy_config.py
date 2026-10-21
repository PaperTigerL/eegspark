import os
import sys
import logging

# 导入Ivy配置修复模块
from utils.spark_fixes.ivy_config import fix_ivy_repository_path

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IvyConfigTest")


def test_ivy_config():
    """测试Ivy配置是否正确"""
    # 修复Ivy配置
    fixed = fix_ivy_repository_path()
    if not fixed:
        logger.error("Ivy配置修复失败")
        return False

    # 检查Ivy目录是否存在
    ivy_dir = os.path.expanduser("~/.ivy2")
    local_repo_dir = os.path.join(ivy_dir, "local")
    properties_file = os.path.join(ivy_dir, "ivy.properties")

    if not os.path.exists(ivy_dir):
        logger.error(f"Ivy目录不存在: {ivy_dir}")
        return False

    if not os.path.exists(local_repo_dir):
        logger.error(f"Ivy本地仓库目录不存在: {local_repo_dir}")
        return False

    if not os.path.exists(properties_file):
        logger.error(f"Ivy属性文件不存在: {properties_file}")
        return False

    # 检查环境变量
    if "IVY_HOME" not in os.environ:
        logger.warning("IVY_HOME环境变量未设置")
    else:
        logger.info(f"IVY_HOME环境变量: {os.environ['IVY_HOME']}")

    # 检查ivy.properties文件内容
    try:
        with open(properties_file, "r") as f:
            content = f.read()
            logger.info(f"ivy.properties内容:\n{content}")
    except Exception as e:
        logger.error(f"无法读取ivy.properties文件: {e}")
        return False

    logger.info("Ivy配置测试通过")
    return True


if __name__ == "__main__":
    logger.info("开始测试Ivy配置...")
    result = test_ivy_config()
    if result:
        logger.info("Ivy配置测试成功")
        sys.exit(0)
    else:
        logger.error("Ivy配置测试失败")
        sys.exit(1)
