# Py4J连接问题修复 - Root权限方案

## 问题描述

在Docker容器环境中运行Spark应用时，经常遇到Py4J连接问题，主要表现为：

1. 连接被拒绝错误：`ConnectionRefusedError: [Errno 111] ECONNREFUSED`
2. Java网关服务器无法启动
3. 重试机制无效，只是浪费时间

## 解决方案

本修复方案通过以下方式解决问题：

1. **移除无效的重试机制**：之前的实现中，重试机制只是重复失败的操作，没有解决根本问题
2. **使用root权限启动Java进程**：确保Java进程有足够的权限创建和访问所需资源
3. **优化Py4J连接参数**：增加超时时间，提高连接稳定性

## 实现细节

### 1. 新增Py4J连接修复模块

创建了新的`py4j_connection.py`模块，提供以下功能：

- `start_java_gateway_with_root`：使用root权限启动Java网关进程
- `configure_py4j_connection`：优化Py4J连接参数

### 2. 修改SparkContextManager类

- 移除了重试循环，直接尝试创建SparkSession
- 使用root权限启动Java进程
- 简化错误处理逻辑

### 3. 更新main.py

- 更新`setup_spark_session`函数，移除不必要的参数

## 使用方法

```python
# 导入Py4J连接修复模块
from utils.spark_fixes.py4j_connection import start_java_gateway_with_root, configure_py4j_connection

# 使用SparkContextManager创建SparkSession（已集成root权限启动）
from common.spark_context_manager import SparkContextManager
spark = SparkContextManager.get_or_create_spark_session(app_name="MyApp")
```

## 注意事项

1. 此修复方案需要在Docker容器中有sudo权限
2. 如果遇到权限问题，请确保容器启动时已授予足够的权限
3. 此方案主要针对Docker环境中的Py4J连接问题，在其他环境中可能需要调整