#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docker容器网络连接修复脚本

此脚本用于修复Docker容器中的网络连接问题，特别是Py4J连接被拒绝的问题
确保eventlet的monkey_patch在所有导入之前执行
"""

# 确保eventlet的monkey_patch在所有导入之前执行
import eventlet

eventlet.monkey_patch()

import os
import sys
import time
import logging
import subprocess
import socket
import gc

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("NetworkConnectionFixer")


def install_network_tools():
    """
    安装网络工具，如ping、ip等

    Returns:
        bool: 安装是否成功
    """
    try:
        logger.info("检查并安装必要的网络工具...")

        # 检查是否有apt-get或apk命令
        has_apt = (
            subprocess.run(
                ["which", "apt-get"], capture_output=True, text=True
            ).returncode
            == 0
        )
        has_apk = (
            subprocess.run(["which", "apk"], capture_output=True, text=True).returncode
            == 0
        )

        if has_apt:
            logger.info("使用apt-get安装网络工具...")
            # 更新软件包列表
            subprocess.run(["apt-get", "update", "-y"], check=False)
            # 安装网络工具
            subprocess.run(
                ["apt-get", "install", "-y", "iputils-ping", "iproute2", "net-tools"],
                check=False,
            )
            return True
        elif has_apk:
            logger.info("使用apk安装网络工具...")
            # 更新软件包列表
            subprocess.run(["apk", "update"], check=False)
            # 安装网络工具
            subprocess.run(
                ["apk", "add", "iputils", "iproute2", "net-tools"], check=False
            )
            return True
        else:
            logger.warning("无法找到适合的包管理器，无法安装网络工具")
            return False
    except Exception as e:
        logger.warning(f"安装网络工具时出错: {e}")
        return False


def start_java_gateway_server():
    """
    启动Java网关服务器

    Returns:
        bool: 启动是否成功
    """
    try:
        logger.info("尝试启动Java网关服务器...")

        # 检查Java进程
        java_process = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        if "org.apache.spark" in java_process.stdout:
            logger.info("检测到Spark Java进程正在运行")
            return True

        # 尝试使用spark-submit启动一个后台Java进程
        logger.info("尝试使用spark-submit启动Java网关服务器...")
        try:
            # 查找spark-examples.jar的确切路径
            logger.info("查找spark-examples.jar的确切路径...")
            try:
                # 尝试使用find命令查找jar文件
                find_cmd = (
                    "find $SPARK_HOME -name 'spark-examples*.jar' -type f | head -1"
                )
                jar_path_result = subprocess.run(
                    find_cmd, shell=True, capture_output=True, text=True
                )

                if jar_path_result.returncode == 0 and jar_path_result.stdout.strip():
                    jar_path = jar_path_result.stdout.strip()
                    logger.info(f"找到spark-examples.jar: {jar_path}")
                else:
                    # 如果find命令失败，尝试直接使用常见路径
                    jar_path = "$SPARK_HOME/examples/jars/spark-examples_*.jar"
                    logger.info(f"未找到确切路径，使用默认路径: {jar_path}")
            except Exception as e:
                logger.warning(f"查找jar文件失败: {e}")
                jar_path = "$SPARK_HOME/examples/jars/spark-examples_*.jar"

            # 创建正确的ivy设置文件
            ivy_settings_path = "/tmp/ivysettings.xml"
            try:
                # 确保使用绝对路径
                absolute_ivy_home = os.path.abspath(os.path.expanduser("~/.ivy2"))
                absolute_local_repo = os.path.join(absolute_ivy_home, "local")
                # 确保目录存在
                os.makedirs(absolute_local_repo, exist_ok=True)

                with open(ivy_settings_path, "w") as f:
                    f.write("<ivysettings>\n")
                    f.write('  <settings defaultResolver="local-maven2" />\n')
                    f.write("  <resolvers>\n")
                    f.write('    <chain name="local-maven2">\n')
                    f.write('      <filesystem name="local">\n')
                    f.write(
                        f'        <ivy pattern="{absolute_local_repo}/[organisation]/[module]/[revision]/[type]s/[artifact].[ext]" />\n'
                    )
                    f.write(
                        f'        <artifact pattern="{absolute_local_repo}/[organisation]/[module]/[revision]/[type]s/[artifact].[ext]" />\n'
                    )
                    f.write("      </filesystem>\n")
                    f.write("    </chain>\n")
                    f.write("  </resolvers>\n")
                    f.write("</ivysettings>\n")
                logger.info(f"已创建Ivy设置文件: {ivy_settings_path}")
                os.chmod(ivy_settings_path, 0o644)
            except Exception as e:
                logger.warning(f"创建Ivy设置文件失败: {e}")

            # 使用nohup启动一个后台Java进程
            # 确保Ivy设置作为Java选项的一部分，而不是单独的命令行参数
            # 使用引号包裹Java选项，确保它们被正确解析为一个整体
            # 修复Ivy路径问题，确保使用绝对路径
            absolute_ivy_home = os.path.abspath(os.path.expanduser("~/.ivy2"))
            java_opts = f"-Djava.io.tmpdir=/tmp/java_tmpdir -Divy.home={absolute_ivy_home} -Divy.settings={ivy_settings_path}"
            start_cmd = [
                "nohup",
                "spark-submit",
                "--master",
                "local[*]",
                "--conf",
                "spark.driver.bindAddress=0.0.0.0",
                "--conf",
                "spark.driver.host=spark-worker",
                "--conf",
                f'spark.driver.extraJavaOptions="{java_opts}"',
                "--class",
                "org.apache.spark.examples.SparkPi",
                jar_path,
                "10",
                "&",
            ]
            subprocess.Popen(
                " ".join(start_cmd),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("已尝试启动Java网关服务器")
            # time.sleep(10)  # 等待进程启动 (已移除)

            # 检查是否成功启动
            java_process = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            if "org.apache.spark" in java_process.stdout:
                logger.info("Java网关服务器启动成功")
                return True
            else:
                logger.warning("Java网关服务器启动失败")
                return False
        except Exception as e:
            logger.warning(f"启动Java网关服务器失败: {e}")
            return False
    except Exception as e:
        logger.warning(f"启动Java网关服务器时出错: {e}")
        return False


def fix_network_connection():
    """
    修复Docker容器中的网络连接问题

    Returns:
        bool: 修复是否成功
    """
    try:
        logger.info("开始修复Docker容器中的网络连接问题...")

        # 检查是否在Docker容器内运行
        if not os.path.exists("/.dockerenv"):
            logger.warning("此脚本应在Docker容器内运行!")
            print(
                "\n⚠️ 此脚本应在Docker容器内运行! 请使用 'docker exec -it --user root spark-worker python /app/src/utils/spark_fixes/fix_network_connection.py'\n"
            )
            return False

        # 安装必要的网络工具
        install_result = install_network_tools()
        if install_result:
            logger.info("网络工具安装成功")
        else:
            logger.warning("网络工具安装失败，部分网络诊断功能可能不可用")

        # 设置环境变量，确保使用正确的网络配置
        os.environ["SPARK_DRIVER_BIND_ADDRESS"] = "0.0.0.0"
        logger.info("已设置SPARK_DRIVER_BIND_ADDRESS=0.0.0.0")

        # 获取容器主机名并设置环境变量
        try:
            container_hostname = socket.gethostname()
            os.environ["SPARK_DRIVER_HOST"] = container_hostname
            logger.info(f"已设置SPARK_DRIVER_HOST={container_hostname}")
        except Exception as e:
            logger.warning(f"获取容器主机名失败: {e}")

        # 设置Py4J端口
        os.environ["SPARK_PY4J_GATEWAY_PORT"] = "25333"
        os.environ["SPARK_PY4J_CALLBACK_PORT"] = "25334"
        logger.info("已设置Py4J端口: gateway=25333, callback=25334")

        # 检查网络接口
        logger.info("检查网络接口...")
        try:
            # 使用ip命令检查网络接口
            try:
                result = subprocess.run(["ip", "addr"], capture_output=True, text=True)

                if result.returncode != 0:
                    logger.warning(f"检查网络接口失败: {result.stderr}")
                else:
                    logger.info(f"网络接口配置:\n{result.stdout}")
            except Exception as e:
                logger.warning(f"执行ip命令失败: {e}")
                # 尝试使用ifconfig作为备选
                try:
                    ifconfig_result = subprocess.run(
                        ["ifconfig"], capture_output=True, text=True
                    )
                    logger.info(f"网络接口配置(ifconfig):\n{ifconfig_result.stdout}")
                except Exception:
                    logger.warning("无法使用ip或ifconfig命令检查网络接口")

            # 检查Docker网络配置
            try:
                docker_network = subprocess.run(
                    ["cat", "/etc/hosts"], capture_output=True, text=True
                )
                logger.info(f"Docker主机配置:\n{docker_network.stdout}")

                # 尝试从hosts文件获取spark-master的IP
                import re

                ip_match = re.search(
                    r"(\d+\.\d+\.\d+\.\d+)\s+spark-master", docker_network.stdout
                )
                if ip_match:
                    spark_master_ip = ip_match.group(1)
                    logger.info(
                        f"从hosts文件中获取到spark-master IP: {spark_master_ip}"
                    )
                    os.environ["SPARK_MASTER_IP"] = spark_master_ip
            except Exception as e:
                logger.warning(f"检查Docker主机配置失败: {e}")

            # 检查是否能ping通spark-master
            logger.info("检查是否能ping通spark-master...")
            try:
                ping_result = subprocess.run(
                    ["ping", "-c", "3", "spark-master"], capture_output=True, text=True
                )

                if ping_result.returncode == 0:
                    logger.info("成功ping通spark-master")
                else:
                    logger.warning(f"无法ping通spark-master: {ping_result.stderr}")
                    # 如果有spark-master IP，尝试直接ping IP
                    if "spark_master_ip" in locals():
                        logger.info(f"尝试ping spark-master IP: {spark_master_ip}")
                        ip_ping_result = subprocess.run(
                            ["ping", "-c", "3", spark_master_ip],
                            capture_output=True,
                            text=True,
                        )
                        if ip_ping_result.returncode == 0:
                            logger.info(f"成功ping通spark-master IP: {spark_master_ip}")
                        else:
                            logger.warning(
                                f"无法ping通spark-master IP: {ip_ping_result.stderr}"
                            )
            except Exception as e:
                logger.warning(f"执行ping命令失败: {e}")
                # 尝试使用nc或telnet检查连接
                try:
                    logger.info("尝试使用nc检查spark-master连接...")
                    nc_result = subprocess.run(
                        ["nc", "-zv", "spark-master", "7077"],
                        capture_output=True,
                        text=True,
                    )
                    if nc_result.returncode == 0:
                        logger.info("成功使用nc连接到spark-master:7077")
                    else:
                        logger.warning(
                            f"无法使用nc连接到spark-master:7077: {nc_result.stderr}"
                        )
                except Exception:
                    logger.warning("无法使用nc命令检查连接")
        except Exception as e:
            logger.warning(f"检查网络接口时出错: {e}")

        # 检查Java网关服务器端口
        logger.info("检查Java网关服务器端口...")
        gateway_ports = [25333, 25334, 4040, 7077, 8080, 8081]
        accessible_ports = []

        for port in gateway_ports:
            try:
                # 检查本地端口
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    result = s.connect_ex(("127.0.0.1", port))
                    if result == 0:
                        logger.info(f"本地端口 {port} 可访问")
                        accessible_ports.append(f"127.0.0.1:{port}")
                    else:
                        logger.warning(f"本地端口 {port} 不可访问 (错误码: {result})")

                # 检查spark-master端口
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    result = s.connect_ex(("spark-master", port))
                    if result == 0:
                        logger.info(f"spark-master端口 {port} 可访问")
                        accessible_ports.append(f"spark-master:{port}")
                    else:
                        logger.warning(
                            f"spark-master端口 {port} 不可访问 (错误码: {result})"
                        )
            except Exception as e:
                logger.warning(f"检查端口 {port} 时出错: {e}")

        if accessible_ports:
            logger.info(f"发现可访问的端口: {accessible_ports}")
        else:
            logger.warning("未发现任何可访问的Java网关服务器端口，这可能导致连接问题")

        # 尝试重启网络服务
        try:
            logger.info("尝试重启网络服务...")
            # 尝试多种方式重启网络服务
            restart_methods = [
                ["service", "networking", "restart"],
                ["systemctl", "restart", "network"],
                ["ip", "link", "set", "eth0", "down"],
                ["ip", "link", "set", "eth0", "up"],
            ]

            network_restarted = False
            for method in restart_methods:
                try:
                    logger.info(f"尝试使用 {' '.join(method)} 重启网络...")
                    result = subprocess.run(
                        method, check=False, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        logger.info(f"使用 {' '.join(method)} 重启网络成功")
                        network_restarted = True
                        break
                    else:
                        logger.warning(
                            f"使用 {' '.join(method)} 重启网络失败: {result.stderr}"
                        )
                except Exception as e:
                    logger.warning(f"执行 {' '.join(method)} 命令失败: {e}")

            if not network_restarted:
                logger.warning("所有网络重启方法都失败，尝试使用Docker内部网络修复方法")
                # 尝试刷新DNS缓存
                try:
                    subprocess.run(["nscd", "-i", "hosts"], check=False)
                    logger.info("已尝试刷新DNS缓存")
                except Exception:
                    logger.warning("刷新DNS缓存失败")

            # time.sleep(5)  # 等待网络服务重启 (已移除)
            logger.info("网络服务重启尝试完成")
        except Exception as e:
            logger.warning(f"重启网络服务失败: {e}")

        # 设置Java临时目录
        java_tmp_dir = "/tmp/java_tmpdir"
        os.makedirs(java_tmp_dir, exist_ok=True)

        # 设置Java临时目录权限
        try:
            # 首先尝试使用chmod
            os.chmod(java_tmp_dir, 0o777)
            logger.info(f"已设置Java临时目录权限: {java_tmp_dir}")

            # 尝试使用sudo确保权限正确设置
            try:
                subprocess.run(["sudo", "chmod", "777", java_tmp_dir], check=False)
                logger.info(f"已使用sudo设置Java临时目录权限: {java_tmp_dir}")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"设置Java临时目录权限失败: {e}")

        # 设置环境变量
        os.environ["JAVA_OPTS"] = (
            f"{os.environ.get('JAVA_OPTS', '')} -Djava.io.tmpdir={java_tmp_dir}"
        )
        os.environ["SPARK_JAVA_OPTS"] = (
            f"{os.environ.get('SPARK_JAVA_OPTS', '')} -Djava.io.tmpdir={java_tmp_dir}"
        )
        os.environ["PYSPARK_SUBMIT_ARGS"] = (
            "--driver-memory 4g --conf spark.network.timeout=1200s pyspark-shell"
        )
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        # 启动Java网关服务器
        logger.info("尝试启动Java网关服务器...")
        gateway_started = start_java_gateway_server()
        if gateway_started:
            logger.info("Java网关服务器启动成功")
        else:
            logger.warning("Java网关服务器启动失败，这可能导致Py4J连接问题")

        # 强制垃圾回收
        gc.collect()

        # 等待资源完全释放
        # wait_time = 15 # (已移除)
        # logger.info(f"等待资源完全释放... ({wait_time}秒)") # (已移除)
        # time.sleep(wait_time) # (已移除)

        # 检查Java网关服务器端口是否可访问
        logger.info("最终检查Java网关服务器端口...")
        gateway_port_accessible = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex(("127.0.0.1", 25333))
            if result == 0:
                logger.info("Java网关服务器端口25333可访问，修复成功!")
                gateway_port_accessible = True
            else:
                logger.warning(
                    f"Java网关服务器端口25333不可访问 (错误码: {result})，修复可能不完全"
                )

        logger.info("网络连接修复完成，请重新启动应用程序")
        if gateway_port_accessible:
            print("\n✅ 网络连接修复成功! 请重新启动应用程序:\n")
        else:
            print("\n⚠️ 网络连接部分修复! 请重新启动应用程序:\n")
        print("docker exec -it --user root spark-worker python /app/src/main.py")

        return True
    except Exception as e:
        logger.error(f"修复网络连接时出错: {e}")
        return False


def test_py4j_connection():
    """
    测试Py4J连接是否正常

    Returns:
        bool: 连接是否成功
    """
    try:
        logger.info("测试Py4J连接...")

        # 尝试导入Py4J模块
        try:
            from py4j.java_gateway import JavaGateway, GatewayParameters

            logger.info("成功导入Py4J模块")
        except ImportError as e:
            logger.warning(f"导入Py4J模块失败: {e}")
            return False

        # 尝试连接Java网关服务器
        try:
            logger.info("尝试连接Java网关服务器...")
            gateway = JavaGateway(
                gateway_parameters=GatewayParameters(
                    address="127.0.0.1", port=25333, auto_convert=True, auto_field=True
                )
            )

            # 尝试获取Java版本信息
            java_version = gateway.jvm.System.getProperty("java.version")
            logger.info(f"成功连接Java网关服务器，Java版本: {java_version}")

            # 关闭连接
            gateway.shutdown()
            logger.info("成功关闭Java网关服务器连接")

            return True
        except Exception as e:
            logger.warning(f"连接Java网关服务器失败: {e}")
            return False
    except Exception as e:
        logger.error(f"测试Py4J连接时出错: {e}")
        return False


if __name__ == "__main__":
    # 执行网络连接修复
    fix_result = fix_network_connection()

    # 如果修复成功，测试Py4J连接
    if fix_result:
        logger.info("网络连接修复完成，测试Py4J连接...")
        connection_ok = test_py4j_connection()
        if connection_ok:
            logger.info("Py4J连接测试成功!")
            print("\n🎉 Py4J连接测试成功! 网络连接和Java网关服务器已完全修复!\n")
        else:
            logger.warning("Py4J连接测试失败，可能需要重启应用程序")
            print("\n⚠️ Py4J连接测试失败，请重启应用程序后再试:\n")
            print("docker exec -it --user root spark-worker python /app/src/main.py")
    else:
        logger.error("网络连接修复失败")
        print("\n❌ 网络连接修复失败，请联系管理员\n")
