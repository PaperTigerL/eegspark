import os
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CONFIG = {
    # 服务发现配置
    "service_discovery": {
        "enabled": True,
        "registry_host": "localhost",
        "registry_port": 8500,
    },
    # 数据源配置
    "data_source": {
        "host": "0.0.0.0",
        "metadata_port": 9000,
        "data_port": 9001,
        "dataset_root": "/app/data/ds005779",
    },
    # 接收端配置
    "receiver": {
        "host": "0.0.0.0",
        "metadata_port": 9000,
        "data_port": 9001,
        "save_dir": "./received_data",
        "buffer_duration": 3.0,
        "min_data_duration": 2.0,
    },
    # Java网关配置
    "java_gateway": {"host": "localhost", "port": 25333, "callback_port": 25334},
}


class ServiceConfig:
    """服务配置管理类，用于统一管理连接信息"""

    _instance = None
    _config = None

    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(ServiceConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """加载配置信息"""
        # 首先加载默认配置
        self._config = DEFAULT_CONFIG.copy()

        # 尝试从配置文件加载
        config_file = os.environ.get("EEG_CONFIG_FILE", "config/service_config.json")
        try:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                    # 递归更新配置
                    self._update_config(self._config, file_config)
                logger.info(f"已从配置文件 {config_file} 加载配置")
        except Exception as e:
            logger.warning(f"加载配置文件 {config_file} 失败: {e}")

        # 从环境变量加载配置
        self._load_from_env()

    def _update_config(self, target: Dict, source: Dict):
        """递归更新配置字典"""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._update_config(target[key], value)
            else:
                target[key] = value

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 服务发现配置
        if os.environ.get("EEG_SERVICE_DISCOVERY_ENABLED"):
            self._config["service_discovery"]["enabled"] = (
                os.environ.get("EEG_SERVICE_DISCOVERY_ENABLED").lower() == "true"
            )
        if os.environ.get("EEG_SERVICE_DISCOVERY_HOST"):
            self._config["service_discovery"]["registry_host"] = os.environ.get(
                "EEG_SERVICE_DISCOVERY_HOST"
            )
        if os.environ.get("EEG_SERVICE_DISCOVERY_PORT"):
            self._config["service_discovery"]["registry_port"] = int(
                os.environ.get("EEG_SERVICE_DISCOVERY_PORT")
            )

        # 数据源配置
        if os.environ.get("EEG_DATA_SOURCE_HOST"):
            self._config["data_source"]["host"] = os.environ.get("EEG_DATA_SOURCE_HOST")
        if os.environ.get("EEG_DATA_SOURCE_METADATA_PORT"):
            self._config["data_source"]["metadata_port"] = int(
                os.environ.get("EEG_DATA_SOURCE_METADATA_PORT")
            )
        if os.environ.get("EEG_DATA_SOURCE_DATA_PORT"):
            self._config["data_source"]["data_port"] = int(
                os.environ.get("EEG_DATA_SOURCE_DATA_PORT")
            )
        if os.environ.get("EEG_DATA_SOURCE_DATASET_ROOT"):
            self._config["data_source"]["dataset_root"] = os.environ.get(
                "EEG_DATA_SOURCE_DATASET_ROOT"
            )

        # 接收端配置
        if os.environ.get("EEG_RECEIVER_HOST"):
            self._config["receiver"]["host"] = os.environ.get("EEG_RECEIVER_HOST")
        if os.environ.get("EEG_RECEIVER_METADATA_PORT"):
            self._config["receiver"]["metadata_port"] = int(
                os.environ.get("EEG_RECEIVER_METADATA_PORT")
            )
        if os.environ.get("EEG_RECEIVER_DATA_PORT"):
            self._config["receiver"]["data_port"] = int(
                os.environ.get("EEG_RECEIVER_DATA_PORT")
            )
        if os.environ.get("EEG_RECEIVER_SAVE_DIR"):
            self._config["receiver"]["save_dir"] = os.environ.get(
                "EEG_RECEIVER_SAVE_DIR"
            )

        # Java网关配置
        if os.environ.get("EEG_JAVA_GATEWAY_HOST"):
            self._config["java_gateway"]["host"] = os.environ.get(
                "EEG_JAVA_GATEWAY_HOST"
            )
        if os.environ.get("EEG_JAVA_GATEWAY_PORT"):
            self._config["java_gateway"]["port"] = int(
                os.environ.get("EEG_JAVA_GATEWAY_PORT")
            )
        if os.environ.get("EEG_JAVA_GATEWAY_CALLBACK_PORT"):
            self._config["java_gateway"]["callback_port"] = int(
                os.environ.get("EEG_JAVA_GATEWAY_CALLBACK_PORT")
            )

    def get_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config

    def get_service_discovery_config(self) -> Dict[str, Any]:
        """获取服务发现配置"""
        return self._config["service_discovery"]

    def get_data_source_config(self) -> Dict[str, Any]:
        """获取数据源配置"""
        return self._config["data_source"]

    def get_receiver_config(self) -> Dict[str, Any]:
        """获取接收端配置"""
        return self._config["receiver"]

    def get_java_gateway_config(self) -> Dict[str, Any]:
        """获取Java网关配置"""
        return self._config["java_gateway"]


# 全局访问点
def get_config() -> ServiceConfig:
    """获取配置实例"""
    return ServiceConfig()
