#!/bin/bash

# 确保Ivy目录存在并设置正确权限
mkdir -p /root/.ivy2/local
chmod -R 777 /root/.ivy2

# 设置环境变量
export IVY_HOME=/root/.ivy2

# 创建ivy.properties文件
cat > /root/.ivy2/ivy.properties << EOF
ivy.local.default.root=/root/.ivy2/local
ivy.local.default.ivy.pattern=[organisation]/[module]/[revision]/[type]s/[artifact].[ext]
ivy.local.default.artifact.pattern=[organisation]/[module]/[revision]/[type]s/[artifact].[ext]
EOF

# 设置Java系统属性
export SPARK_JAVA_OPTS="$SPARK_JAVA_OPTS -Divy.home=/root/.ivy2"

# 执行原始的entrypoint脚本
exec "$@"