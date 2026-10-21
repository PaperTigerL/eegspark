# Py4J和Ivy修复文件重构计划

## 当前问题

根据项目代码分析，我们发现Spark连接和Py4J修复相关的文件分散在不同位置，导致代码组织混乱，不便于维护。具体问题包括：

1. 修复脚本直接放在src目录根部
2. 测试文件与源代码混合
3. 文档分散在不同位置

## 文件移动计划

### 1. 修复脚本移动

将以下修复脚本移动到 `src/utils/spark_fixes/` 目录：

| 当前位置 | 目标位置 |
|---------|----------|
| `src/fix_ivy_config.py` | `src/utils/spark_fixes/fix_ivy_config.py` |
| `src/docker_py4j_fix.py` | `src/utils/spark_fixes/docker_py4j_fix.py` |
| `src/docker_entrypoint.sh` | `src/utils/spark_fixes/docker_entrypoint.sh` |
| `src/visualization/spark_connection_fix.py` | `src/utils/spark_fixes/spark_connection_fix.py` |

### 2. 测试文件移动

将以下测试文件移动到 `src/tests/` 目录：

| 当前位置 | 目标位置 |
|---------|----------|
| `src/test_ivy_config.py` | `src/tests/test_ivy_config.py` |
| `src/test_py4j_connection.py` | `src/tests/test_py4j_connection.py` |
| `src/test_sparkcontext.py` | `src/tests/test_sparkcontext.py` |
| `src/test_visualization_py4j_fix.py` | `src/tests/test_visualization_py4j_fix.py` |
| `src/docker_py4j_test.py` | `src/tests/docker_py4j_test.py` |

### 3. 文档移动

将以下文档移动到 `docs/spark_fixes/` 目录：

| 当前位置 | 目标位置 |
|---------|----------|
| `src/DOCKER_PY4J_FIX_README.md` | `docs/spark_fixes/DOCKER_PY4J_FIX_README.md` |
| `src/IVY_FIX_README.md` | `docs/spark_fixes/IVY_FIX_README.md` |
| `src/PY4J_CONNECTION_FIX.md` | `docs/spark_fixes/PY4J_CONNECTION_FIX.md` |
| `src/visualization/README_PY4J_FIX.md` | `docs/spark_fixes/VISUALIZATION_PY4J_FIX.md` |

## 导入路径更新

移动文件后，需要更新以下导入路径：

1. 在测试文件中，将 `from fix_ivy_config import fix_ivy_repository_path` 更新为 `from utils.spark_fixes.fix_ivy_config import fix_ivy_repository_path`

2. 在使用可视化模块Spark连接修复的文件中，将 `from visualization.spark_connection_fix import ...` 更新为 `from utils.spark_fixes.spark_connection_fix import ...`

## 实施步骤

1. 创建新的目录结构：
   - `src/utils/spark_fixes/`
   - `src/tests/`
   - `docs/spark_fixes/`

2. 移动修复脚本到 `src/utils/spark_fixes/`

3. 移动测试文件到 `src/tests/`

4. 移动文档到 `docs/spark_fixes/`

5. 更新导入路径

6. 测试确保功能正常

## 预期收益

1. **提高代码可维护性**：相关功能的文件集中在一起，便于维护
2. **清晰的模块边界**：明确区分工具函数、测试和文档
3. **简化开发流程**：开发人员可以更容易找到需要的文件
4. **便于扩展**：未来添加新的修复脚本或测试可以直接放在对应目录

## 注意事项

1. 移动文件前先备份
2. 更新导入路径时要全面检查所有依赖
3. 移动后进行充分测试，确保功能正常

通过这个重构计划，我们可以使Py4J和Ivy修复相关的文件组织更加清晰，提高代码的可维护性和可扩展性。