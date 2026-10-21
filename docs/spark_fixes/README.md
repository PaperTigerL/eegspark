# Spark和Py4J连接修复模块

## 概述

本模块包含用于修复Spark和Py4J连接问题的工具和测试脚本。这些工具主要解决以下问题：

1. Ivy仓库路径配置问题
2. Docker容器内的Py4J连接问题
3. 多SparkContext冲突问题
4. 可视化模块与Spark集成的连接问题

## 目录结构

项目重构后的目录结构如下：

```
src/
├── utils/
│   └── spark_fixes/
│       ├── __init__.py
│       ├── ivy_config.py        # Ivy配置修复工具
│       └── docker_py4j_fix.py   # Docker容器内Py4J连接修复工具
├── tests/
│   └── spark/
│       ├── test_py4j_connection.py        # Py4J连接测试
│       ├── test_ivy_config.py             # Ivy配置测试
│       ├── test_sparkcontext.py           # SparkContext测试
│       ├── test_visualization_py4j_fix.py # 可视化模块Py4J连接修复测试
│       └── docker_py4j_test.py            # Docker容器内Py4J连接测试
└── visualization/
    └── spark_connection_fix.py  # 可视化模块Spark连接修复工具
```

## 使用方法

### 1. Ivy配置修复

```python
from utils.spark_fixes.ivy_config import fix_ivy_repository_path

# 修复Ivy配置
fix_ivy_repository_path()
```

### 2. Docker容器内Py4J连接修复

在Docker容器内执行：

```bash
docker exec -it spark-worker python /app/src/utils/spark_fixes/docker_py4j_fix.py
```

### 3. 测试Py4J连接

```bash
python src/tests/spark/test_py4j_connection.py
```

### 4. 测试可视化模块的Py4J连接修复

```bash
python src/tests/spark/test_visualization_py4j_fix.py
```

## 注意事项

1. 所有修复工具都已更新导入路径，确保可以正确执行
2. 测试文件已移动到tests/spark目录，与实现文件分离
3. 如果在Docker容器内运行，请使用正确的路径
4. 修复工具会自动处理多SparkContext冲突问题

## 常见问题

### Q: 为什么会出现Py4J连接问题？

A: Py4J连接问题通常由以下原因导致：
- Ivy仓库路径配置不正确
- Java网关服务器没有正确启动
- 多个SparkContext同时运行
- 网络连接问题

### Q: 如何解决多SparkContext冲突？

A: 使用SparkContextManager类管理SparkContext实例，确保在整个应用程序中只有一个SparkContext实例。

```python
from common.spark_context_manager import SparkContextManager

# 获取SparkSession实例
spark = SparkContextManager.get_or_create_spark_session(app_name="MyApp")

# 使用完毕后停止SparkSession
SparkContextManager.stop_spark_session()
```

### Q: 如何在Docker容器内测试Py4J连接？

A: 使用docker_py4j_test.py脚本测试Docker容器内的Py4J连接：

```bash
docker exec -it spark-worker python /app/src/tests/spark/docker_py4j_test.py
```