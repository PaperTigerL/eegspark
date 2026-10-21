# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet

eventlet.monkey_patch()

import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IvyConfigFixer")


def fix_ivy_repository_path():
    """
    修复Ivy仓库路径配置，确保使用绝对路径

    这个函数解决了Spark启动时出现的Ivy仓库路径问题和Py4J连接问题
    """
    # 获取当前用户的主目录
    # home_dir = os.path.expanduser('~') # 使用固定路径代替，避免 HOME 环境变量问题

    # 确保.ivy2目录存在
    ivy_dir = "/app/src/ivy_cache"  # 使用固定的绝对路径 /app/src/ivy_cache 作为 Ivy 的基础目录
    local_repo_dir = os.path.join(
        ivy_dir, "local"
    )  # 本地仓库路径将是 /app/src/ivy_cache/local

    try:
        # 创建目录（如果不存在）
        os.makedirs(ivy_dir, exist_ok=True)
        os.makedirs(local_repo_dir, exist_ok=True)

        # 设置环境变量 - 确保Java进程能找到Ivy目录
        os.environ["IVY_HOME"] = ivy_dir
        # 设置Java系统属性 - 这对Py4J连接很重要
        os.environ["JAVA_OPTS"] = (
            f"-Divy.home={ivy_dir} {os.environ.get('JAVA_OPTS', '')}"
        )
        # 设置Spark Java选项 - 确保Spark的Java进程也能找到Ivy目录
        os.environ["SPARK_JAVA_OPTS"] = (
            f"-Divy.home={ivy_dir} {os.environ.get('SPARK_JAVA_OPTS', '')}"
        )

        # 创建ivysettings.xml文件
        ivysettings_xml_path = os.path.join(ivy_dir, "ivysettings.xml")
        ivysettings_content = f"""<ivysettings>
    <settings defaultResolver="default"/>
    <caches defaultCacheDir=\"{ivy_dir}\"/>
    <resolvers>
        <chain name=\"default\">
            <filesystem name=\"local\">
                <ivy pattern=\"{local_repo_dir}/[organisation]/[module]/[revision]/[type]s/[artifact].[ext]\"/>
                <artifact pattern=\"{local_repo_dir}/[organisation]/[module]/[revision]/[type]s/[artifact].[ext]\"/>
            </filesystem>
            <ibiblio name=\"public\" m2compatible=\"true\"/>
        </chain>
    </resolvers>
    <modules>
        <module organisation=\".*\" name=\".*\" resolver=\"default\"/>
    </modules>
</ivysettings>
"""
        with open(ivysettings_xml_path, "w") as f:
            f.write(ivysettings_content)

        logger.info(f"已创建Ivy配置文件: {ivysettings_xml_path}")

        # 设置Python与Java进程通信的环境变量
        # 这些设置对解决Py4J连接问题很重要
        # 更新 PYSPARK_SUBMIT_ARGS 以包含 spark.jars.ivySettings
        base_opts = "--driver-memory 2g --conf spark.network.timeout=300s"
        ivy_settings_opt = f"--conf spark.jars.ivySettings={ivysettings_xml_path}"
        # 假设 'pyspark-shell' 是 PYSPARK_SUBMIT_ARGS 的一部分，并应保留在末尾
        current_pyspark_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS", "")
        if "pyspark-shell" in current_pyspark_submit_args:
            # 移除旧的 'pyspark-shell' 以避免重复，并确保它在最后
            args_list = (
                current_pyspark_submit_args.replace("pyspark-shell", "").strip().split()
            )
        else:
            args_list = current_pyspark_submit_args.strip().split()

        # 添加或更新配置
        final_args = []
        ivy_settings_updated = False
        for arg in args_list:
            if arg.startswith("--conf") and "spark.jars.ivySettings" in arg:
                final_args.append(ivy_settings_opt)  # 更新现有的 spark.jars.ivySettings
                ivy_settings_updated = True
            elif arg.startswith("--driver-memory"):
                if not any(
                    opt.startswith("--driver-memory") for opt in base_opts.split()
                ):  # 避免重复
                    final_args.append(arg)
            elif arg.startswith("--conf") and "spark.network.timeout" in arg:
                if not any(
                    opt.startswith("--conf") and "spark.network.timeout" in opt
                    for opt in base_opts.split()
                ):  # 避免重复
                    final_args.append(arg)
            elif arg:  # 保留其他非空参数
                final_args.append(arg)

        # 添加基础选项（如果尚未通过现有参数处理）
        if not any(opt.startswith("--driver-memory") for opt in final_args):
            final_args.append("--driver-memory 2g")
        if not any(
            opt.startswith("--conf") and "spark.network.timeout" in opt
            for opt in final_args
        ):
            final_args.append("--conf spark.network.timeout=300s")
        if not ivy_settings_updated:
            final_args.append(ivy_settings_opt)

        # 确保pyspark-shell在最后（如果最初存在或作为默认行为）
        if (
            "pyspark-shell" in current_pyspark_submit_args
            or not current_pyspark_submit_args
        ):
            final_args.append("pyspark-shell")

        os.environ["PYSPARK_SUBMIT_ARGS"] = " ".join(final_args)
        logger.info(f"PYSPARK_SUBMIT_ARGS set to: {os.environ['PYSPARK_SUBMIT_ARGS']}")

        # 设置Py4J连接超时
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        # 如果在Docker容器中运行，确保Java可以访问这些目录
        if os.path.exists("/.dockerenv"):
            try:
                import subprocess

                # 设置Ivy目录权限
                subprocess.run(["chmod", "-R", "777", ivy_dir], check=False)
                logger.info(f"已设置Docker容器中Ivy目录的权限")

                # 确保Java临时目录存在并有正确权限
                java_tmp_dir = "/tmp/java_tmpdir"
                os.makedirs(java_tmp_dir, exist_ok=True)
                subprocess.run(["chmod", "777", java_tmp_dir], check=False)

                # 设置Java临时目录环境变量
                os.environ["JAVA_OPTS"] = (
                    f"{os.environ.get('JAVA_OPTS', '')} -Djava.io.tmpdir={java_tmp_dir}"
                )
                os.environ["SPARK_JAVA_OPTS"] = (
                    f"{os.environ.get('SPARK_JAVA_OPTS', '')} -Djava.io.tmpdir={java_tmp_dir}"
                )

                logger.info(f"已设置Java临时目录: {java_tmp_dir}")
            except Exception as e:
                logger.warning(f"设置Docker容器权限失败: {e}")

        logger.info("Ivy配置和Py4J连接环境已设置完成")
        return True
    except Exception as e:
        logger.error(f"修复Ivy配置失败: {e}")
        return False


# 导出函数供其他模块使用
__all__ = ["fix_ivy_repository_path"]
