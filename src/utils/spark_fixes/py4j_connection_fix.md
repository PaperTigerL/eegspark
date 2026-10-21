# Py4J连接问题分析与解决方案

## 问题描述

根据错误日志，系统出现了Py4J连接被拒绝的错误：

```
ConnectionRefusedError: [Errno 111] ECONNREFUSED
```

这表明Python无法连接到Java网关服务器（127.0.0.1:25333），导致Py4J通信失败。

## 问题原因

通过分析错误日志和代码，可能的原因包括：

1. **Java网关服务器未启动**：Spark的Java进程可能没有正确启动或已经崩溃
2. **网络配置问题**：Docker容器内的网络接口配置可能有问题
3. **权限问题**：临时目录或Ivy配置文件的权限设置不正确
4. **资源冲突**：可能存在多个SparkContext实例导致的资源冲突
5. **eventlet monkey_patch顺序问题**：eventlet的monkey_patch可能没有在所有导入之前执行

## 解决方案

### 1. 运行Docker Py4J修复脚本

使用root权限在Docker容器内运行修复脚本：

```bash
docker exec -it --user root spark-worker python /app/src/utils/spark_fixes/docker_py4j_fix.py
```

这个脚本会：
- 检查网络接口配置
- 修复Java网关服务器
- 清理SparkContext资源
- 修复权限问题
- 测试Spark连接

### 2. 确保eventlet.monkey_patch()正确执行

在任何使用Py4J的Python脚本中，确保在所有其他导入之前执行eventlet的monkey_patch：

```python
import eventlet
eventlet.monkey_patch()

# 其他导入
import os
import sys
# ...
```

### 3. 检查Java进程

在Docker容器内检查Java进程是否正在运行：

```bash
docker exec -it spark-worker ps aux | grep java
```

如果没有Java进程，可以尝试手动启动：

```bash
docker exec -it spark-worker spark-submit --master local[*] --class org.apache.spark.examples.SparkPi $SPARK_HOME/examples/jars/spark-examples*.jar
```

### 4. 检查网络端口

检查Java网关服务器端口是否可访问：

```bash
docker exec -it spark-worker netstat -tulpn | grep 25333
```

### 5. 重启Docker容器

如果以上方法都不能解决问题，尝试重启Docker容器：

```bash
docker-compose restart spark-worker
```

然后再次运行修复脚本。

## 预防措施

为防止此类问题再次发生，建议：

1. 在所有使用Py4J的Python脚本中，确保eventlet.monkey_patch()在所有导入之前执行
2. 在创建SparkContext之前，先检查是否已有活跃的SparkContext
3. 设置适当的超时参数，如：
   ```python
   conf.set("spark.network.timeout", "800s")
   ```
4. 确保Docker容器有足够的资源（内存、CPU）运行Spark和Java进程

## 相关文件

- `/app/src/utils/spark_fixes/docker_py4j_fix.py` - Docker Py4J连接修复脚本
- `/app/src/utils/spark_fixes/ivy_config.py` - Ivy配置修复脚本
- `/app/src/PY4J_CONNECTION_FIX.md` - Py4J连接问题修复指南